from typing import Tuple, Optional
from src.common.types import EvalResult
import re

class StudentEvaluator:
    def __init__(self, format_weight: float = 0.3, correct_weight: float = 0.7):
        self.format_weight = format_weight
        self.correct_weight = correct_weight

    def _extract_thinking_and_solution(self, response: str) -> Tuple[Optional[str], Optional[str]]:
        think_pattern = r'<think>(.*?)</think>'
        think_match = re.search(think_pattern, response, re.DOTALL)
        thinking = think_match.group(1).strip() if think_match else None

        solution = None
        if think_match:
            solution = response[think_match.end():].strip()

        return thinking, solution

    def _extract_numeric_answer(self, solution: str) -> Optional[float]:
        boxed_pattern = r'\\boxed{([-\d.]+)}'
        boxed_match = re.search(boxed_pattern, solution)
        if boxed_match:
            return float(boxed_match.group(1))

        solution_pattern = r'Solution:\s*([-\d.]+)'
        solution_match = re.search(solution_pattern, solution)
        if solution_match:
            return float(solution_match.group(1))

        number_pattern = r'([-\d.]+)'
        number_match = re.search(number_pattern, solution)
        if number_match:
            return float(number_match.group(1))

        return None

    def evaluate(self, response: str, ground_truth: float, tolerance: float = 1e-6) -> EvalResult:
        thinking, solution = self._extract_thinking_and_solution(response)

        has_thinking = thinking is not None
        has_solution = solution is not None
        formatted_properly = has_thinking and has_solution

        format_score = 0.0
        if has_thinking:
            format_score += 0.5
        if has_solution:
            format_score += 0.5

        extracted_answer = self._extract_numeric_answer(solution) if solution else None
        correct_score = 0.0

        if extracted_answer is not None:
            if abs(extracted_answer - ground_truth) <= tolerance:
                correct_score = 1.0
            else:
                relative_error = abs(extracted_answer - ground_truth) / (abs(ground_truth) + tolerance)
                correct_score = max(0.0, 1.0 - relative_error)

        total_score = (
            self.format_weight * format_score +
            self.correct_weight * correct_score
        )

        return EvalResult(
            format_score=format_score,
            correct_score=correct_score,
            total_score=total_score,
            extracted_answer=extracted_answer,
            formatted_properly=formatted_properly,
            has_thinking=has_thinking,
            has_solution=has_solution,
        )
