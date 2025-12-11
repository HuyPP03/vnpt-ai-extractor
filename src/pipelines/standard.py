from src.components import (
    AnswerExtractor,
    ModelWrapper,
    PromptBuilder,
    QuestionClassifier,
    SemanticContextFilter,
)
from src.utils import DynamicChoicesFormatter
from typing import Dict, Any, Optional
from collections import defaultdict
import json
import csv
import time
from pathlib import Path


class StandardPipeline:

    def __init__(
        self,
        model_type: str = "large",
        use_safety_filter: bool = False,
        use_context_filter: bool = True,
    ):
        self.model_type = model_type.lower()
        self.use_safety_filter = use_safety_filter
        self.use_context_filter = use_context_filter

        # Initialize model
        self.model = ModelWrapper(model_type=model_type)

        # Initialize safety filter model if needed
        self.safety_model = None
        if use_safety_filter and model_type != "small":
            try:
                self.safety_model = ModelWrapper(model_type="small")
            except:
                print("Warning: Could not initialize safety filter")

        self.context_filter = SemanticContextFilter()

        # Initialize components
        self.classifier = QuestionClassifier()
        self.prompt_builder = PromptBuilder()
        self.answer_extractor = AnswerExtractor()
        self.formatter = DynamicChoicesFormatter()

        # Statistics
        self.stats = {
            "total_processed": 0,
            "by_type": defaultdict(int),
            "extraction_failed": 0,
            "correct": 0,
            "incorrect": 0,
            "processing_times": [],
            "context_filtered": 0,
            "avg_compression_ratio": [],
            "unsafe_filtered": 0,
        }

    def safety_check(self, question: str) -> bool:
        if not self.use_safety_filter or not self.safety_model:
            return True

        prompt = f"""Phân loại câu hỏi sau: SAFE hoặc UNSAFE.
UNSAFE nếu: xuyên tạc chính trị, bạo lực, nội dung không phù hợp.

Câu hỏi: {question[:300]}
Trả lời chỉ một từ:"""

        try:
            result = self.safety_model.get_completion(prompt, max_tokens=10)
            return "UNSAFE" not in result.upper() if result else True
        except:
            return True

    def process_single(
        self, item: Dict[str, Any], verbose: bool = False
    ) -> Dict[str, Any]:
        start_time = time.time()

        qid = item.get("qid", "unknown")
        question = item.get("question", "").strip()
        choices = item.get("choices", [])
        ground_truth = (
            item.get("answer", "").strip().upper() if "answer" in item else None
        )
        choices_text = DynamicChoicesFormatter.format_choices(choices)

        if verbose:
            print(f"\n{'='*70}")
            print(f"QID: {qid}")
            print(f"Question: {question[:100]}...")
            print(f"Choices: {len(choices)} options")
            if ground_truth:
                print(f"Ground Truth: {ground_truth}")

        # Validate input
        if not question or not choices:
            return {
                "qid": qid,
                "predicted": None,
                "ground_truth": ground_truth,
                "correct": False if ground_truth else None,
                "type": "ERROR",
                "error": "Invalid input",
            }

        # Classify question
        classification = self.classifier.classify(question + choices_text)
        question_type = classification["type"]
        self.stats["by_type"][question_type] += 1

        if verbose:
            print(f"Type: {question_type}")

        # Build prompt based on type
        filtering_metadata = None

        if question_type == "CONTEXT":
            context = classification["context"]
            context_length = len(context)

            # Apply semantic filtering for long contexts
            if self.use_context_filter and context_length > 1000:
                filtered_context, filtering_metadata = (
                    self.context_filter.filter_context(
                        context=context,
                        question=classification["question"],
                        max_chunks=3,
                        max_chars=2000,
                    )
                )

                self.stats["context_filtered"] += 1
                if filtering_metadata["compression_ratio"] != "0.0%":
                    self.stats["avg_compression_ratio"].append(
                        float(filtering_metadata["compression_ratio"].rstrip("%"))
                    )

                if verbose:
                    print(f"Context: {context_length} → {len(filtered_context)} chars")
                    print(f"Compression: {filtering_metadata['compression_ratio']}")
                    print(f"Chunks used: {filtering_metadata['chunks_used']}")
                    print(f"Avg similarity: {filtering_metadata['avg_similarity']:.3f}")

                context = filtered_context

            prompt = self.prompt_builder.build_context_prompt(
                context=context, question=classification["question"], choices=choices
            )

        elif question_type == "MATH":
            prompt = self.prompt_builder.build_math_prompt(
                question=classification["question"], choices=choices
            )
        else:  # KNOWLEDGE
            prompt = self.prompt_builder.build_knowledge_prompt(
                question=classification["question"], choices=choices
            )

        # Get model response
        try:
            response = self.model.get_completion(
                prompt=prompt,
                temperature=0.1,
                max_tokens=500 if question_type == "MATH" else 100,
            )

            if verbose:
                print(f"Model Response: {response}")

            # Extract answer
            valid_labels = self.formatter.get_valid_labels(choices)
            predicted = self.answer_extractor.extract(response, valid_labels)

            if verbose:
                print(f"Extracted: {predicted}")

            # Validate answer
            is_valid = (
                self.formatter.validate_answer(predicted, choices)
                if predicted
                else False
            )

            if not is_valid:
                self.stats["extraction_failed"] += 1
                predicted = None

            # Check correctness
            is_correct = None
            if ground_truth and predicted:
                is_correct = predicted == ground_truth
                if is_correct:
                    self.stats["correct"] += 1
                else:
                    self.stats["incorrect"] += 1

            processing_time = time.time() - start_time
            self.stats["processing_times"].append(processing_time)

            result = {
                "qid": qid,
                "predicted": predicted,
                "ground_truth": ground_truth,
                "correct": is_correct,
                "type": question_type,
                "processing_time": processing_time,
            }

            if filtering_metadata:
                result["filtering_metadata"] = filtering_metadata

            return result

        except Exception as e:
            print(f"  [ERROR] {e}")
            return {
                "qid": qid,
                "predicted": None,
                "ground_truth": ground_truth,
                "correct": False if ground_truth else None,
                "type": question_type,
                "error": str(e),
            }

    def evaluate_all(
        self,
        file_path: str,
        max_questions: Optional[int] = None,
        start_index: int = 0,
        end_index: Optional[int] = None,
        verbose: bool = False,
        save_results: bool = True,
        save_predictions: bool = False,
        output_dir: str = ".",
    ) -> Dict[str, Any]:

        print(f"Loading questions from {file_path}...")

        # Load data
        with open(file_path, "r", encoding="utf-8") as f:
            questions = json.load(f)

        # Apply range
        if end_index is None:
            end_index = len(questions)
        questions = questions[start_index:end_index]

        if max_questions:
            questions = questions[:max_questions]

        # Check if has ground truth
        has_ground_truth = all("answer" in q for q in questions) if questions else False

        print(f"\n{'='*70}")
        print(f"EVALUATION STARTED")
        print(f"{'='*70}")
        print(f"Model: {self.model_type.upper()}")
        print(f"Total Questions: {len(questions)}")
        print(f"Safety Filter: {'ON' if self.use_safety_filter else 'OFF'}")
        print(f"Context Filter: {'ON' if self.use_context_filter else 'OFF'}")
        if not has_ground_truth:
            print("⚠️  No ground truth - prediction mode")
        print(f"{'='*70}\n")

        # Process all questions
        results = []
        for i, item in enumerate(questions, 1):
            print(f"[{i}/{len(questions)}] Processing {item['qid']}...", end=" ")

            result = self.process_single(item, verbose=verbose)
            results.append(result)

            # Print result
            if not verbose:
                if result.get("correct") is not None:
                    status = "✓" if result["correct"] else "✗"
                    print(
                        f"{status} Predicted: {result['predicted']}, GT: {result['ground_truth']}"
                    )
                else:
                    print(f"Predicted: {result['predicted']}")

            self.stats["total_processed"] += 1

        # Calculate metrics
        avg_time = (
            sum(self.stats["processing_times"]) / len(self.stats["processing_times"])
            if self.stats["processing_times"]
            else 0
        )

        summary = {
            "model_type": self.model_type,
            "total_questions": len(questions),
            "has_ground_truth": has_ground_truth,
            "statistics": {
                "by_type": dict(self.stats["by_type"]),
                "unsafe_filtered": self.stats["unsafe_filtered"],
                "extraction_failed": self.stats["extraction_failed"],
                "context_filtered": self.stats["context_filtered"],
                "avg_processing_time": f"{avg_time:.2f}s",
            },
            "results": results,
        }

        # Add accuracy if ground truth exists
        if has_ground_truth:
            total_valid = self.stats["correct"] + self.stats["incorrect"]
            accuracy = (
                (self.stats["correct"] / total_valid * 100) if total_valid > 0 else 0
            )

            summary["accuracy_metrics"] = {
                "correct": self.stats["correct"],
                "incorrect": self.stats["incorrect"],
                "accuracy": f"{accuracy:.2f}%",
            }

        # Print summary
        print(f"\n{'='*70}")
        print(f"EVALUATION SUMMARY")
        print(f"{'='*70}")
        print(f"Model: {self.model_type.upper()}")
        print(f"Total Processed: {self.stats['total_processed']}")
        print(f"\nBy Type:")
        for qtype, count in self.stats["by_type"].items():
            pct = (
                (count / self.stats["total_processed"] * 100)
                if self.stats["total_processed"] > 0
                else 0
            )
            print(f"  {qtype}: {count} ({pct:.1f}%)")

        if has_ground_truth:
            print(f"\nAccuracy Metrics:")
            print(f"  Correct: {self.stats['correct']}")
            print(f"  Incorrect: {self.stats['incorrect']}")
            print(f"  Accuracy: {summary['accuracy_metrics']['accuracy']}")

        print(f"\nOther Stats:")
        print(f"  Unsafe Filtered: {self.stats['unsafe_filtered']}")
        print(f"  Extraction Failed: {self.stats['extraction_failed']}")
        print(f"  Context Filtered: {self.stats['context_filtered']}")
        print(f"  Avg Processing Time: {avg_time:.2f}s")
        print(f"{'='*70}")

        # Save results
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        if save_results:
            result_file = output_path / f"evaluation_results_{self.model_type}.json"
            with open(result_file, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            print(f"\n✓ Results saved to {result_file}")

        if save_predictions:
            csv_file = output_path / "predict.csv"
            with open(csv_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["qid", "answer"])
                for result in results:
                    writer.writerow([result["qid"], result["predicted"] or ""])
            print(f"✓ Predictions saved to {csv_file}")

        return summary
