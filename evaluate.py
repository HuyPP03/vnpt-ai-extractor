from utils.pipeline import OptimizedModelEvaluator

def main():
    """Main evaluation function"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate VNPT AI model on Q&A dataset"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="large",
        choices=["small", "large", "openai"],
        help="Model type to use (default: large)",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="val.json",
        help="Input JSON file with questions (default: val.json)",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Maximum number of questions to evaluate (default: all)",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start index for question range (0-based, default: 0)",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        default=None,
        help="End index for question range (exclusive, default: end of list)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed information for each question",
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Do not save results to file"
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save predictions to predict.csv file",
    )

    args = parser.parse_args()

    # Create evaluator
    evaluator = OptimizedModelEvaluator(model_type=args.model)

    # Run evaluation
    evaluator.evaluate_all(
        file_path=args.input,
        max_questions=args.max_questions,
        start_index=args.start_index,
        end_index=args.end_index,
        verbose=args.verbose,
        save_results=not args.no_save,
        save_predictions=args.save_predictions,
    )


if __name__ == "__main__":
    ###
    # To use this file, run from command line:
    # python evaluate.py --model large --input val.json --max-questions 100
    # python evaluate.py --model small --input val.json --verbose
    # python evaluate.py --model openai --input test.json --save-predictions
    # python evaluate.py --model large --input val.json --start-index 10 --end-index 20

    main()
