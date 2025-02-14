import random
from typing import Tuple
from ..common.types import Problem

class MathProblemGenerator:
    """Generates simple arithmetic problems."""

    def __init__(self, difficulty_range: Tuple[float, float] = (0.0, 1.0)):
        self.difficulty_range = difficulty_range
        self.current_difficulty = difficulty_range[0]

    def generate_addition_problem(self) -> Problem:
        """Generates an simple addition problem based on the current difficulty."""
        max_number = int(10 ** (1 + self.current_difficulty))
        a = random.randint(1, max_number)
        b = random.randint(1, max_number)

        question = f"What is {a} + {b}?"
        solution = a + b

        return Problem(
            question=question,
            difficulty=self.current_difficulty,
            topic="math",
            solution=solution,
            metadata={
                "operands": (a, b),
                "operation": "addition",
            },
        )
