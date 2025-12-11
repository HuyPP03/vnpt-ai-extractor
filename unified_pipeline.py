import argparse
import json
import time
import csv
from pathlib import Path
from typing import Dict, Any, List, Optional
from collections import defaultdict
from utils.pipeline import OptimizedModelEvaluator
from utils.hybrid_pipeline import OptimizedHybridPipeline


class UnifiedPipeline:

    STRATEGIES = ["baseline", "hybrid", "cost-optimized", "quality-optimized"]
    MODEL_PROVIDERS = ["vnpt", "openai"]

    def __init__(
        self,
        strategy: str = "baseline",
        model_provider: str = "vnpt",
        large_model: str = "large",
        small_model: str = "small",
        use_improved_prompts: bool = True,
    ):

        self.strategy = strategy
        self.model_provider = model_provider
        self.large_model = large_model
        self.small_model = small_model
        self.use_improved_prompts = use_improved_prompts

        # Initialize pipeline
        self.pipeline = self._create_pipeline()

        # Statistics
        self.stats = {
            "total_processed": 0,
            "correct": 0,
            "incorrect": 0,
            "by_type": defaultdict(lambda: {"correct": 0, "total": 0}),
            "processing_times": [],
        }

    def _create_pipeline(self):
        if self.strategy == "baseline":
            return OptimizedModelEvaluator(
                model_type=self._resolve_model_name(self.large_model),
                use_context_filter=True,
            )
        else:
            strategy_map = {
                "hybrid": "hybrid",
                "cost-optimized": "cost-optimized",
                "quality-optimized": "quality-optimized",
            }
            return OptimizedHybridPipeline(
                strategy=strategy_map.get(self.strategy, "hybrid"),
                use_improved_prompts=self.use_improved_prompts,
                large_model_name=self._resolve_model_name(self.large_model),
                small_model_name=self._resolve_model_name(self.small_model),
            )

    def _resolve_model_name(self, model: str) -> str:
        if self.model_provider == "openai":
            if model == "large":
                return "gpt-4o-mini"
            elif model == "small":
                return "gpt-3.5-turbo"
            else:
                return model
        else:
            return model

    def evaluate(
        self,
        file_path: str,
        max_questions: Optional[int] = None,
        start_index: int = 0,
        end_index: Optional[int] = None,
        verbose: bool = False,
        save_results: bool = False,
        save_predictions: bool = False,
        output_dir: str = ".",
    ) -> Dict[str, Any]:

        print(f"\n{'='*70}")
        print(f"UNIFIED PIPELINE EVALUATION")
        print(f"{'='*70}")
        print(f"Strategy: {self.strategy.upper()}")
        print(f"Model Provider: {self.model_provider.upper()}")
        if self.strategy != "baseline":
            print(f"Large Model: {self.large_model}")
            print(f"Small Model: {self.small_model}")
        else:
            print(f"Model: {self.large_model}")
        print(f"Improved Prompts: {'ON' if self.use_improved_prompts else 'OFF'}")
        print(f"{'='*70}\n")

        # Load questions
        with open(file_path, "r", encoding="utf-8") as f:
            questions = json.load(f)

        # Apply range
        if end_index is None:
            end_index = len(questions)
        questions = questions[start_index:end_index]

        if max_questions:
            questions = questions[:max_questions]

        print(f"Total questions to process: {len(questions)}\n")

        # Process questions
        results = []
        start_time = time.time()

        for i, item in enumerate(questions, 1):
            qid = item.get("qid", f"q_{i}")
            print(f"[{i}/{len(questions)}] Processing {qid}...", end=" ")

            # Process based on strategy
            if self.strategy == "baseline":
                result = self._process_baseline(item, verbose)
            else:
                result = self._process_hybrid(item, verbose)

            results.append(result)

            # Print result
            if not verbose:
                self._print_result(result)

            # Update stats
            self._update_stats(result)

        total_time = time.time() - start_time

        # Create summary
        summary = self._create_summary(
            questions=questions, results=results, total_time=total_time
        )

        # Print summary
        self._print_summary(summary)

        # Save results
        if save_results or save_predictions:
            self._save_results(
                summary=summary,
                save_results=save_results,
                save_predictions=save_predictions,
                output_dir=output_dir,
            )

        return summary

    def _process_baseline(self, item: Dict[str, Any], verbose: bool) -> Dict[str, Any]:

        result = self.pipeline.process_single(item, verbose=verbose)

        # Add model info
        result["model_used"] = self.large_model
        result["strategy"] = "baseline"

        return result

    def _process_hybrid(self, item: Dict[str, Any], verbose: bool) -> Dict[str, Any]:

        result = self.pipeline.process_single(item, verbose=verbose)

        # Add strategy info
        result["strategy"] = self.strategy

        return result

    def _print_result(self, result: Dict[str, Any]):
        """In kết quả một câu hỏi"""
        if result.get("correct") is not None:
            status = "✓" if result["correct"] else "✗"
            model_info = f"[{result.get('model_used', 'unknown')}]"

            if result.get("fallback_used"):
                model_info += " (fallback)"

            print(
                f"{status} {model_info} Predicted: {result['predicted']}, GT: {result['ground_truth']}"
            )
        else:
            print(
                f"Predicted: {result['predicted']} [{result.get('model_used', 'unknown')}]"
            )

    def _update_stats(self, result: Dict[str, Any]):
        """Cập nhật statistics"""
        self.stats["total_processed"] += 1

        if result.get("correct") is not None:
            if result["correct"]:
                self.stats["correct"] += 1
            else:
                self.stats["incorrect"] += 1

        # By type
        qtype = result.get("type", "UNKNOWN")
        if result.get("correct") is not None:
            self.stats["by_type"][qtype]["total"] += 1
            if result["correct"]:
                self.stats["by_type"][qtype]["correct"] += 1

        # Processing time
        if "processing_time" in result:
            self.stats["processing_times"].append(result["processing_time"])

    def _create_summary(
        self, questions: List[Dict], results: List[Dict], total_time: float
    ) -> Dict[str, Any]:
        """Tạo summary"""
        has_ground_truth = all("answer" in q for q in questions)

        summary = {
            "strategy": self.strategy,
            "model_provider": self.model_provider,
            "total_questions": len(questions),
            "has_ground_truth": has_ground_truth,
            "total_time": f"{total_time:.2f}s",
            "avg_time_per_question": (
                f"{total_time/len(questions):.2f}s" if questions else "0s"
            ),
            "results": results,
        }

        # Add accuracy metrics
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

            # By type
            summary["accuracy_by_type"] = {}
            for qtype, stats in self.stats["by_type"].items():
                if stats["total"] > 0:
                    acc = stats["correct"] / stats["total"] * 100
                    summary["accuracy_by_type"][qtype] = {
                        "correct": stats["correct"],
                        "total": stats["total"],
                        "accuracy": f"{acc:.1f}%",
                    }

        # Add model usage for hybrid
        if self.strategy != "baseline" and hasattr(self.pipeline, "get_statistics"):
            summary["model_usage"] = self.pipeline.get_statistics()

        return summary

    def _print_summary(self, summary: Dict[str, Any]):
        """In summary"""
        print(f"\n{'='*70}")
        print(f"EVALUATION SUMMARY - {self.strategy.upper()}")
        print(f"{'='*70}")
        print(f"Total Questions: {summary['total_questions']}")
        print(f"Total Time: {summary['total_time']}")
        print(f"Avg Time/Question: {summary['avg_time_per_question']}")

        # Model usage
        if "model_usage" in summary:
            usage = summary["model_usage"]
            print(f"\nModel Usage:")
            print(
                f"  Small: {usage.get('small_used', 0)} ({usage.get('small_percentage', '0%')})"
            )
            print(
                f"  Large: {usage.get('large_used', 0)} ({usage.get('large_percentage', '0%')})"
            )
            print(
                f"  Fallback triggered: {usage.get('fallback_triggered', 0)} ({usage.get('fallback_rate', '0%')})"
            )

        # Accuracy
        if "accuracy_metrics" in summary:
            print(f"\nAccuracy:")
            print(f"  Correct: {summary['accuracy_metrics']['correct']}")
            print(f"  Incorrect: {summary['accuracy_metrics']['incorrect']}")
            print(f"  Overall: {summary['accuracy_metrics']['accuracy']}")

            if "accuracy_by_type" in summary:
                print(f"\nAccuracy by Type:")
                for qtype, stats in summary["accuracy_by_type"].items():
                    print(
                        f"  {qtype}: {stats['accuracy']} ({stats['correct']}/{stats['total']})"
                    )

        print(f"{'='*70}\n")

    def _save_results(
        self,
        summary: Dict[str, Any],
        save_results: bool,
        save_predictions: bool,
        output_dir: str,
    ):
        """Lưu kết quả"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Save full results
        if save_results:
            result_file = output_path / f"evaluation_results_{self.strategy}.json"
            with open(result_file, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            print(f"✓ Results saved to {result_file}")

        # Save predictions CSV
        if save_predictions:
            csv_file = output_path / f"predictions_{self.strategy}.csv"
            with open(csv_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["qid", "answer"])
                for result in summary["results"]:
                    writer.writerow([result["qid"], result["predicted"] or ""])
            print(f"✓ Predictions saved to {csv_file}")


def main():
    """Main function với argument parsing"""
    parser = argparse.ArgumentParser(
        description="Unified Pipeline - Evaluate with multiple strategies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Baseline với VNPT large model
  python unified_pipeline.py --strategy baseline --provider vnpt --large-model large
  
  # Hybrid với VNPT models
  python unified_pipeline.py --strategy hybrid --provider vnpt
  
  # Hybrid với OpenAI models
  python unified_pipeline.py --strategy hybrid --provider openai --large-model gpt-4o-mini --small-model gpt-3.5-turbo
  
  # Cost-optimized strategy
  python unified_pipeline.py --strategy cost-optimized --max-questions 50
  
  # Quality-optimized strategy với verbose
  python unified_pipeline.py --strategy quality-optimized --verbose
  
  # Save predictions
  python unified_pipeline.py --strategy hybrid --save-predictions
        """,
    )

    # Strategy arguments
    parser.add_argument(
        "--strategy",
        type=str,
        default="baseline",
        choices=["baseline", "hybrid", "cost-optimized", "quality-optimized"],
        help="Evaluation strategy (default: baseline)",
    )

    # Model provider arguments
    parser.add_argument(
        "--provider",
        type=str,
        default="vnpt",
        choices=["vnpt", "openai"],
        help="Model provider (default: vnpt)",
    )

    parser.add_argument(
        "--large-model",
        type=str,
        default="large",
        help="Large model name (default: large for VNPT, gpt-4o-mini for OpenAI)",
    )

    parser.add_argument(
        "--small-model",
        type=str,
        default="small",
        help="Small model name (default: small for VNPT, gpt-3.5-turbo for OpenAI)",
    )

    # Data arguments
    parser.add_argument(
        "--input",
        type=str,
        default="val.json",
        help="Input JSON file (default: val.json)",
    )

    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Maximum number of questions (default: all)",
    )

    parser.add_argument(
        "--start-index", type=int, default=0, help="Start index (default: 0)"
    )

    parser.add_argument(
        "--end-index", type=int, default=None, help="End index (default: end of list)"
    )

    # Output arguments
    parser.add_argument(
        "--verbose", action="store_true", help="Print detailed information"
    )

    parser.add_argument("--no-save", action="store_true", help="Do not save results")

    parser.add_argument(
        "--save-predictions", action="store_true", help="Save predictions to CSV"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Output directory (default: current directory)",
    )

    # Prompt arguments
    parser.add_argument(
        "--no-improved-prompts", action="store_true", help="Disable improved prompts"
    )

    args = parser.parse_args()

    # Set default models for OpenAI
    if args.provider == "openai":
        if args.large_model == "large":
            args.large_model = "gpt-4o-mini"
        if args.small_model == "small":
            args.small_model = "gpt-3.5-turbo"

    # Create pipeline
    pipeline = UnifiedPipeline(
        strategy=args.strategy,
        model_provider=args.provider,
        large_model=args.large_model,
        small_model=args.small_model,
        use_improved_prompts=not args.no_improved_prompts,
    )

    # Run evaluation
    pipeline.evaluate(
        file_path=args.input,
        max_questions=args.max_questions,
        start_index=args.start_index,
        end_index=args.end_index,
        verbose=args.verbose,
        save_results=not args.no_save,
        save_predictions=args.save_predictions,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
