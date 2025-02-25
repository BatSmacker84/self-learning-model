import random
from typing import List, Optional
from src.common.types import MathOperation, MathProblem
import numpy as np

class MathProblemGenerator:
    def __init__(
        self,
        min_val: float = 0,
        max_val: float = 100,
        int_only: bool = True,
        max_operands: int = 2,
    ):
        self.min_val = min_val
        self.max_val = max_val
        self.int_only = int_only
        self.max_operands = max_operands

    def _gen_operands(self, num_operands: int) -> List[float]:
        if self.int_only:
            return [float(random.randint(int(self.min_val), int(self.max_val)))
                   for _ in range(num_operands)]
        else:
            return [random.uniform(self.min_val, self.max_val)
                   for _ in range(num_operands)]

    def _calc_solution(
        self,
        operation: MathOperation,
        operands: List[float],
    ) -> Optional[float]:
        try:
            if operation == MathOperation.ADD:
                return sum(operands)
            elif operation == MathOperation.SUB:
                return operands[0] - sum(operands[1:])
            elif operation == MathOperation.MUL:
                return np.prod(operands).astype(float)
            elif operation == MathOperation.DIV:
                result = operands[0]
                for operand in operands[1:]:
                    if operand == 0:
                        return None
                    result /= operand
                return result
        except Exception:
            return None

    def _calc_difficulty(
        self,
        operation: MathOperation,
        operands: List[float],
    ) -> int:
        difficulty = 1

        difficulty = len(operands) - 2

        max_operand = max(abs(x) for x in operands)
        difficulty += int(np.log10(max_operand)) if max_operand > 0 else 0

        if operation == MathOperation.MUL:
            difficulty += 2
        elif operation == MathOperation.DIV:
            difficulty += 3

        if not self.int_only:
            difficulty += 2

        return min(max(difficulty, 1), 10)

    def gen_problem(
        self,
        operation: Optional[MathOperation] = None,
        num_operands: Optional[int] = None,
    ) -> Optional[MathProblem]:
        if operation is None:
            operation = random.choice(list(MathOperation))

        if num_operands is None:
            num_operands = random.randint(2, self.max_operands)

        operands = self._gen_operands(num_operands)

        solution = self._calc_solution(operation, operands)
        if solution is None:
            return None

        difficulty = self._calc_difficulty(operation, operands)

        metadata = {
            "integer_only": self.int_only,
            "num_operands": num_operands,
        }

        return MathProblem(operation, operands, solution, difficulty, metadata)

    def gen_problem_batch(
        self,
        batch_size: int,
        operation: Optional[MathOperation] = None,
        num_operands: Optional[int] = None,
    ) -> List[MathProblem]:
        problems = []
        while len(problems) < batch_size:
            problem = self.gen_problem(operation, num_operands)
            if problem is not None:
                problems.append(problem)
        return problems

if __name__ == "__main__":
    generator = MathProblemGenerator(
        min_val=1,
        max_val=100,
        int_only=True,
        max_operands=3,
    )

    problem = generator.gen_problem(MathOperation.ADD)
    if problem is not None:
        print(f"Operation: {problem.operation.value}")
        print(f"Operands: {problem.operands}")
        print(f"Solution: {problem.solution}")
        print(f"Difficulty: {problem.difficulty}")
        print(f"Metadata: {problem.metadata}")

    problems = generator.gen_problem_batch(5)
    for i, problem in enumerate(problems):
        print(f"\nProblem {i+1}:")
        print(f"Operation: {problem.operation.value}")
        print(f"Operands: {problem.operands}")
        print(f"Solution: {problem.solution}")
