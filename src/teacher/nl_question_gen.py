from typing import Optional, List

from src.models.llm import LLM
from src.common.types import MathProblem,  TeacherPrompt

class TeacherModel(LLM):
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B",
        prompt: Optional[TeacherPrompt] = None,
    ):
        super().__init__(model_name)
        self.prompt = prompt if prompt is not None else TeacherPrompt()

    def _format_numbers(self, operands: List[float]) -> str:
        str_nums = [
            str(int(x)) if x.is_integer() else f"{x:.2f}"
            for x in operands
        ]

        if len(str_nums) > 2:
            return ", ".join(str_nums[:-1]) + f", and {str_nums[-1]}"
        elif len(str_nums) == 2:
            return f"{str_nums[0]} and {str_nums[1]}"
        return str_nums[0]

    def _create_prompt(self, problem: MathProblem) -> str:
        fmt_nums = self._format_numbers(problem.operands)

        message = self.prompt.template.format(
            operation=problem.operation.value,
            numbers=fmt_nums,
            difficulty=problem.difficulty,
        )

        return f"{self.prompt.system}\n\n{message}"

    def _extract_question(self, response: str) -> str:
        response = response.replace(self.prompt.system, "").strip()

        if "Question:" in response:
            response = response.split("Question:")[-1].strip()

        response = response.replace("Generated question:", "").strip()
        response = response.replace("Answer:", "").split("Answer:")[0].strip()

        return response

    def gen_question(self, problem: MathProblem) -> str:
        prompt = self._create_prompt(problem)
        response = self.generate(prompt).strip()
        return self._extract_question(response)

    def gen_question_batch(self, problems: List[MathProblem]) -> List[str]:
        return [self.gen_question(problem) for problem in problems]

if __name__ == "__main__":
    from src.teacher.gt_problem_gen import MathProblemGenerator

    generator = MathProblemGenerator()
    teacher = TeacherModel()
    teacher.load()

    problems = generator.gen_problem_batch(3)

    print("Generated questions:")
    for problem in problems:
        question = teacher.gen_question(problem)
        print(f"\nOperation: {problem.operation.value}")
        print(f"Numbers: {problem.operands}")
        print(f"Question: {question}")
        print(f"Solution: {problem.solution}")
