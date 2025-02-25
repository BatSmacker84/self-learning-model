from typing import List, Tuple
import torch
from src.models.llm import LLM

class StudentModel(LLM):
    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B"):
        super().__init__(model_name)

    def _clean_response(self, response: str, question: str) -> str:
        return response.replace(question, "").strip()

    def gen_with_probs(self, question: str) -> Tuple[str, torch.Tensor]:
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded first")

        self.model.eval()

        inputs = self.tokenize(question)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                output_scores=True,
                return_dict_in_generate=True,
                output_hidden_states=True,
            )

        self.last_hidden_states = outputs.hidden_states[-1][-1]

        generated_tokens = outputs.sequences[0]
        scores = torch.stack(outputs.scores, dim=0)
        log_probs = torch.log_softmax(scores, dim=-1)

        seq_length = generated_tokens.size(0) - inputs["input_ids"].size(1)
        token_log_probs = torch.zeros(
            seq_length,
            device=self.device,
            dtype=self.dtype,
        )

        for i in range(seq_length):
            token_idx = generated_tokens[inputs["input_ids"].size(1) + i]
            token_log_probs[i] = log_probs[i, 0, token_idx]

        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        generated_text = self._clean_response(question, generated_text)

        return generated_text, token_log_probs

    def get_last_hidden_states(self):
        if self.last_hidden_states is None:
            raise ValueError("No hidden states available. Run generate_with_probs first.")
        return self.last_hidden_states

    def solve(self, question: str) -> str:
        response = self.generate(question).strip()
        return self._clean_response(response, question)

    def solve_batch(self, questions: List[str]) -> List[str]:
        return [self.solve(question) for question in questions]

if __name__ == "__main__":
    from src.teacher.gt_problem_gen import MathProblemGenerator
    from src.teacher.nl_question_gen import TeacherModel

    generator = MathProblemGenerator()
    problems = generator.gen_problem_batch(3)

    teacher = TeacherModel()
    teacher.load()
    questions = teacher.gen_question_batch(problems)
    teacher = None

    student = StudentModel()
    student.load()

    print("Problem Solving Session:")
    for problem, question in zip(problems, questions):
        print("\n" + "="*50)
        print(f"Operation: {problem.operation.value}")
        print(f"Numbers: {problem.operands}")
        print(f"GT Solution: {problem.solution}")
        print(f"\nQuestion: {question}")

        print(f"\nStudent's Answer: {student.solve(question)}")
