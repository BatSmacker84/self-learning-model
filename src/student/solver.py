from ..common.types import Problem, StudentResponse
import random

class StudentSolver:
    """Basic student model that attempts to solve math problems."""

    def __init__(self, learning_rate: float = 0.1):
        self.learning_rate = learning_rate
        self.performance_history = []
        self.knowledge_level = 0.0 # Range from 0 to 1

    def solve_problem(self, problem: Problem) -> StudentResponse:
        """Attempt to solve the given problem."""
        if problem.topic != "math":
            raise ValueError(f"Unsupported topic: {problem.topic} (StudentSolver can only solve math problems)")

        # Extract numbers from the question using more robust parsing
        try:
            # Remove any trailing punctuation and split into words
            words = problem.question.rstrip('?!.').split()

            # Find the numbers in the question
            numbers = []
            for word in words:
                # Remove any commas from numbers
                cleaned = word.replace(',', '')
                try:
                    num = int(cleaned)
                    numbers.append(num)
                except ValueError:
                    continue

            if len(numbers) != 2:
                raise ValueError("Could not find exactly two numbers in the problem")

            a, b = numbers

            # Simulate knowledge level influence on accuracy
            correct_answer = a + b
            confidence = random.uniform(self.knowledge_level, 1.0)

            # Introduce possible errors based on knowledge level
            if random.random() > self.knowledge_level:
                # Simulate making a mistake
                answer = correct_answer + random.randint(-2, 2)
            else:
                answer = correct_answer

            return StudentResponse(
                problem_id=str(id(problem)),
                answer=answer,
                reasoning=f"I added {a} and {b} to get {answer}",
                confidence=confidence
            )

        except (IndexError, ValueError) as e:
            raise ValueError(f"Failed to parse problem: {e}")

    def update_knowledge(self, was_correct: bool):
        """Update the knowledge level based on correctness."""
        if was_correct:
            self.knowledge_level = min(1.0,
                self.knowledge_level + self.learning_rate)
        else:
            self.knowledge_level = max(0.0,
                self.knowledge_level - self.learning_rate * 0.5)

        self.performance_history.append(was_correct)
