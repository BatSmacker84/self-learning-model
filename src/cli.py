import argparse
import sys
from typing import List, Optional, Sequence
import json
from datetime import datetime
from pathlib import Path

from src.common.session import LearningSession
from src.common.types import Problem, StudentResponse, Feedback

def format_session_results(history: List[tuple[Problem, StudentResponse, Feedback]]) -> dict:
    """Format session results for display and storage."""
    results = {
        "timestamp": datetime.now().isoformat(),
        "total_problems": len(history),
        "correct_answers": sum(1 for _, _, f in history if f.is_correct),
        "problems": []
    }

    for problem, response, feedback in history:
        results["problems"].append({
            "question": problem.question,
            "difficulty": problem.difficulty,
            "student_answer": response.answer,
            "correct_answer": problem.solution,
            "confidence": response.confidence,
            "is_correct": feedback.is_correct,
            "score": feedback.score
        })

    return results

def save_results(results: dict, output_dir: Path) -> Path:
    """Save session results to a JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"session_{timestamp}.json"

    with output_file.open('w') as f:
        json.dump(results, f, indent=2)

    return output_file

def display_results(results: dict) -> None:
    """Display session results in a readable format."""
    print("\n=== Learning Session Results ===")
    print(f"Total Problems: {results['total_problems']}")
    print(f"Correct Answers: {results['correct_answers']}")
    print(f"Success Rate: {(results['correct_answers'] / results['total_problems']) * 100:.1f}%")
    print("\nProblem Details:")

    for i, problem in enumerate(results['problems'], 1):
        print(f"\nProblem {i}:")
        print(f"  Question: {problem['question']}")
        print(f"  Student Answer: {problem['student_answer']}")
        print(f"  Correct Answer: {problem['correct_answer']}")
        print(f"  Confidence: {problem['confidence']:.2f}")
        print(f"  Result: {'✓' if problem['is_correct'] else '✗'}")

def main(args: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run an AI self-learning session"
    )
    parser.add_argument(
        "-n", "--num-problems",
        type=int,
        default=10,
        help="Number of problems to generate (default: 10)"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory to save session results (default: ./results)"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to file"
    )

    parsed_args = parser.parse_args(args)

    try:
        # Run learning session
        print(f"Starting learning session with {parsed_args.num_problems} problems...")
        session = LearningSession(max_problems=parsed_args.num_problems)
        history = session.run_session()

        # Process results
        results = format_session_results(history)

        # Display results
        display_results(results)

        # Save results if requested
        if not parsed_args.no_save:
            output_file = save_results(results, parsed_args.output_dir)
            print(f"\nResults saved to: {output_file}")

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())
