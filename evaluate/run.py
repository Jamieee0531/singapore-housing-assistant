"""
Entry point for RAG evaluation.

Usage:
    python -m evaluate.run                  # Run both retrieval + response eval
    python -m evaluate.run --retrieval      # Run retrieval eval only
    python -m evaluate.run --response       # Run response eval only

Results are saved as Markdown reports in evaluate/results/
"""

import argparse
import logging

from src.config import setup_logging

from evaluate.retrieval_eval import run_retrieval_eval
from evaluate.response_eval import run_response_eval
from evaluate.report import generate_report

logger = logging.getLogger(__name__)


def main():
    setup_logging()

    parser = argparse.ArgumentParser(description="Run RAG evaluation")
    parser.add_argument("--retrieval", action="store_true", help="Run retrieval evaluation only")
    parser.add_argument("--response", action="store_true", help="Run response evaluation only")
    parser.add_argument("--dataset", default="evaluate/dataset.json", help="Path to dataset")
    args = parser.parse_args()

    # If neither flag is set, run both
    run_both = not args.retrieval and not args.response

    retrieval_results = None
    response_results = None

    if args.retrieval or run_both:
        print("\n" + "=" * 50)
        print("Running Retrieval Evaluation...")
        print("=" * 50)
        retrieval_results = run_retrieval_eval(args.dataset)
        r = retrieval_results["aggregate"]
        print(f"\nPrecision@k: {r['mean_precision_at_k']:.3f}")
        print(f"MRR:         {r['mrr']:.3f}")

    if args.response or run_both:
        print("\n" + "=" * 50)
        print("Running Response Evaluation (LLM-as-Judge)...")
        print("=" * 50)
        response_results = run_response_eval(args.dataset)
        r = response_results["aggregate"]
        print(f"\nFaithfulness:    {r['mean_faithfulness']:.2f} / 5")
        print(f"Answer Relevance: {r['mean_relevance']:.2f} / 5")

    # Generate report if we have both results
    if retrieval_results and response_results:
        report_path = generate_report(retrieval_results, response_results)
        print(f"\nReport saved to: {report_path}")
    elif retrieval_results:
        print("\nSkipped report generation (response eval not run)")
    elif response_results:
        print("\nSkipped report generation (retrieval eval not run)")


if __name__ == "__main__":
    main()
