from typing import List, Tuple
from .types import Problem, StudentResponse, Feedback
from ..teacher.problem_gen import MathProblemGenerator
from ..student.solver import StudentSolver

class LearningSession:
    """Manages the interaction between teacher and student components."""

    def __init__(self, max_problems: int = 10):
        self.teacher = MathProblemGenerator()
        self.student = StudentSolver()
        self.max_problems = max_problems
        self.history: List[Tuple[Problem, StudentResponse, Feedback]] = []

    def run_session(self) -> List[Tuple[Problem, StudentResponse, Feedback]]:
        """Run a complete learning session."""
        for _ in range(self.max_problems):
            # Generate problem
            problem = self.teacher.generate_addition_problem()

            # Get student's solution
            response = self.student.solve_problem(problem)

            # Evaluate response
            is_correct = response.answer == problem.solution
            feedback = Feedback(
                is_correct=is_correct,
                explanation=f"Correct answer was {problem.solution}",
                score=1.0 if is_correct else 0.0,
                suggestions=None if is_correct else "Check your calculation carefully"
            )

            # Update student's knowledge
            self.student.update_knowledge(is_correct)

            # Store interaction
            self.history.append((problem, response, feedback))

            # Adjust difficulty based on performance
            if len(self.history) >= 3:
                recent_performance = sum(f.is_correct for _, _, f in self.history[-3:]) / 3
                self.teacher.current_difficulty = min(1.0,
                    self.teacher.current_difficulty + (0.1 if recent_performance > 0.7 else -0.1))

        return self.history
