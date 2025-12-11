import argparse

from src.pipelines import UnifiedPipeline


def main():
    """Main function với argument parsing"""
    parser = argparse.ArgumentParser(
        description="Unified Pipeline - Evaluate with multiple strategies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Baseline với VNPT large model
  python main.py --strategy baseline --provider vnpt --large-model large --input data/val.json
  
  # Hybrid với VNPT models
  python main.py --strategy hybrid --provider vnpt --input data/val.json
  
  # Hybrid với OpenAI models
  python main.py --strategy hybrid --provider openai --large-model gpt-4o-mini --small-model gpt-3.5-turbo
  
  # Cost-optimized strategy
  python main.py --strategy cost-optimized --max-questions 50
  
  # Quality-optimized strategy với verbose
  python main.py --strategy quality-optimized --verbose --input data/val.json
  
  # Save predictions
  python main.py --strategy hybrid --save-predictions
  
  # Enable safety classifier (keyword mode - fast)
  python main.py --strategy hybrid --safety-mode keyword
  
  # Enable safety classifier (model mode - accurate)
  python main.py --strategy hybrid --safety-mode model
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
        default="data/val.json",
        help="Input JSON file (default: data/val.json)",
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

    # Safety arguments
    parser.add_argument(
        "--safety-mode",
        type=str,
        default="none",
        choices=["none", "keyword", "model"],
        help="Safety check mode: none (disabled), keyword (fast), model (accurate) (default: none)",
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
        safety_mode=args.safety_mode,
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
