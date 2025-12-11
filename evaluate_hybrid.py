import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any
from utils.hybrid_pipeline import OptimizedHybridPipeline


def evaluate_with_strategy(
    file_path: str,
    strategy: str = "hybrid",
    max_questions: int = None,
    verbose: bool = False,
) -> Dict[str, Any]:

    print(f"\n{'='*70}")
    print(f"EVALUATING WITH STRATEGY: {strategy.upper()}")
    print(f"{'='*70}\n")

    # Load questions
    with open(file_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    if max_questions:
        questions = questions[:max_questions]

    # Create pipeline
    pipeline = OptimizedHybridPipeline(strategy=strategy)

    # Process all questions
    results = []
    correct_count = 0
    incorrect_count = 0
    start_time = time.time()

    for i, item in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] Processing {item['qid']}...", end=" ")

        result = pipeline.process_single(item, verbose=verbose)
        results.append(result)

        # Print result
        if not verbose:
            if result.get("correct") is not None:
                status = "✓" if result["correct"] else "✗"
                model_info = f"[{result['model_used']}]"
                if result.get("fallback_used"):
                    model_info += " (fallback)"
                print(
                    f"{status} {model_info} Predicted: {result['predicted']}, GT: {result['ground_truth']}"
                )

                if result["correct"]:
                    correct_count += 1
                else:
                    incorrect_count += 1
            else:
                print(f"Predicted: {result['predicted']} [{result['model_used']}]")

    total_time = time.time() - start_time

    # Get statistics
    stats = pipeline.get_statistics()

    # Calculate metrics
    has_ground_truth = all("answer" in q for q in questions)

    summary = {
        "strategy": strategy,
        "total_questions": len(questions),
        "has_ground_truth": has_ground_truth,
        "total_time": f"{total_time:.2f}s",
        "avg_time_per_question": f"{total_time/len(questions):.2f}s",
        "model_usage": {
            "small_used": stats["small_used"],
            "large_used": stats["large_used"],
            "small_percentage": stats["small_percentage"],
            "large_percentage": stats["large_percentage"],
            "fallback_triggered": stats["fallback_triggered"],
            "fallback_rate": stats["fallback_rate"],
        },
        "results": results,
    }

    if has_ground_truth:
        total_valid = correct_count + incorrect_count
        accuracy = (correct_count / total_valid * 100) if total_valid > 0 else 0

        summary["accuracy_metrics"] = {
            "correct": correct_count,
            "incorrect": incorrect_count,
            "accuracy": f"{accuracy:.2f}%",
        }

        # Accuracy by type
        by_type = {}
        for result in results:
            qtype = result["type"]
            if qtype not in by_type:
                by_type[qtype] = {"correct": 0, "total": 0}

            if result.get("correct") is not None:
                by_type[qtype]["total"] += 1
                if result["correct"]:
                    by_type[qtype]["correct"] += 1

        summary["accuracy_by_type"] = {
            qtype: {
                "correct": stats["correct"],
                "total": stats["total"],
                "accuracy": f"{stats['correct']/stats['total']*100:.1f}%",
            }
            for qtype, stats in by_type.items()
        }

    # Print summary
    print(f"\n{'='*70}")
    print(f"EVALUATION SUMMARY - {strategy.upper()}")
    print(f"{'='*70}")
    print(f"Total Questions: {len(questions)}")
    print(f"Total Time: {summary['total_time']}")
    print(f"Avg Time/Question: {summary['avg_time_per_question']}")

    print(f"\nModel Usage:")
    print(f"  Small: {stats['small_used']} ({stats['small_percentage']})")
    print(f"  Large: {stats['large_used']} ({stats['large_percentage']})")
    print(
        f"  Fallback triggered: {stats['fallback_triggered']} ({stats['fallback_rate']})"
    )

    if has_ground_truth:
        print(f"\nAccuracy:")
        print(f"  Correct: {correct_count}")
        print(f"  Incorrect: {incorrect_count}")
        print(f"  Overall: {summary['accuracy_metrics']['accuracy']}")

        print(f"\nAccuracy by Type:")
        for qtype, stats in summary["accuracy_by_type"].items():
            print(
                f"  {qtype}: {stats['accuracy']} ({stats['correct']}/{stats['total']})"
            )

    print(f"{'='*70}\n")

    return summary


def compare_strategies(
    file_path: str,
    max_questions: int = None,
    verbose: bool = False,
    save_results: bool = True,
) -> Dict[str, Any]:

    strategies = ["cost-optimized", "quality-optimized", "hybrid"]

    results = {}
    for strategy in strategies:
        results[strategy] = evaluate_with_strategy(
            file_path=file_path,
            strategy=strategy,
            max_questions=max_questions,
            verbose=verbose,
        )

        # Pause between strategies
        time.sleep(2)

    # Comparison summary
    print(f"\n{'='*70}")
    print(f"STRATEGY COMPARISON")
    print(f"{'='*70}\n")

    # Create comparison table
    print(
        f"{'Strategy':<20} {'Accuracy':<12} {'Small%':<10} {'Large%':<10} {'Fallback%':<12} {'Avg Time':<10}"
    )
    print("-" * 80)

    for strategy in strategies:
        result = results[strategy]
        accuracy = result.get("accuracy_metrics", {}).get("accuracy", "N/A")
        small_pct = result["model_usage"]["small_percentage"]
        large_pct = result["model_usage"]["large_percentage"]
        fallback_rate = result["model_usage"]["fallback_rate"]
        avg_time = result["avg_time_per_question"]

        print(
            f"{strategy:<20} {accuracy:<12} {small_pct:<10} {large_pct:<10} {fallback_rate:<12} {avg_time:<10}"
        )

    print(f"\n{'='*70}\n")

    # Save results
    if save_results:
        output_file = Path("evaluation_comparison.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"✓ Comparison results saved to {output_file}\n")

    return results


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Evaluate with Hybrid Pipeline and compare strategies"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="val.json",
        help="Input JSON file (default: val.json)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="hybrid",
        choices=["cost-optimized", "quality-optimized", "hybrid", "compare"],
        help="Strategy to use (default: hybrid). Use 'compare' to test all strategies",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Maximum number of questions (default: all)",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print detailed information"
    )
    parser.add_argument("--no-save", action="store_true", help="Do not save results")

    args = parser.parse_args()

    if args.strategy == "compare":
        # Compare all strategies
        compare_strategies(
            file_path=args.input,
            max_questions=args.max_questions,
            verbose=args.verbose,
            save_results=not args.no_save,
        )
    else:
        # Evaluate with single strategy
        result = evaluate_with_strategy(
            file_path=args.input,
            strategy=args.strategy,
            max_questions=args.max_questions,
            verbose=args.verbose,
        )

        # Save results
        if not args.no_save:
            output_file = Path(f"evaluation_results_{args.strategy}.json")
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"✓ Results saved to {output_file}\n")


if __name__ == "__main__":
    """
    Usage examples:

    # Evaluate with hybrid strategy (recommended)
    python evaluate_hybrid.py --strategy hybrid

    # Evaluate with cost-optimized strategy
    python evaluate_hybrid.py --strategy cost-optimized

    # Compare all strategies
    python evaluate_hybrid.py --strategy compare --max-questions 50

    # Verbose mode
    python evaluate_hybrid.py --strategy hybrid --verbose

    # Custom input file
    python evaluate_hybrid.py --input test.json --strategy hybrid
    """
    main()
