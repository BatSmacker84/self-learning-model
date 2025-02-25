from typing import List, Tuple
import torch
import random
from tqdm import tqdm
from torch.optim import AdamW
from src.common.types import EvalResult
from src.student.evaluator import StudentEvaluator
from src.student.solver import StudentModel

class StudentTrainer:
    def __init__(
        self,
        student_model: StudentModel,
        evaluator: StudentEvaluator,
        learning_rate: float = 1e-5,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        grad_acc_steps: int = 4,
    ):
        self.student = student_model
        self.evaluator = evaluator
        self.optimizer = AdamW(student_model.model.parameters(), lr=learning_rate)

        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.grad_acc_steps = grad_acc_steps

        self.value_head = torch.nn.Linear(
            student_model.model.config.hidden_size,
            1
        ).to(student_model.device).to(student_model.dtype)
        self.value_optimizer = AdamW(self.value_head.parameters(), lr=learning_rate)

        self.steps = 0

    def compute_reward(self, eval_result: EvalResult) -> torch.Tensor:
        return torch.tensor(
            eval_result.total_score,
            dtype=self.student.dtype,
            device=self.student.device,
            requires_grad=True,
        )

    def estimate_value(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.value_head(hidden_states.mean(dim=1))

    def compute_entropy(self, log_probs: torch.Tensor) -> torch.Tensor:
        probs = torch.exp(log_probs)
        return -torch.sum(probs * log_probs, dim=-1).mean()

    def train_step(
        self,
        question: str,
        ground_truth: float,
    ) -> Tuple[float, EvalResult]:
        self.student.model.train()
        self.value_head.train()

        old_response, old_log_probs = self.student.gen_with_probs(question)
        old_hidden_states = self.student.get_last_hidden_states().to(self.student.dtype)
        old_value = self.estimate_value(old_hidden_states)

        eval_result = self.evaluator.evaluate(old_response, ground_truth)
        reward = torch.tensor(
            self.compute_reward(eval_result),
            device=self.student.device
        )

        new_response, new_log_probs = self.student.gen_with_probs(question)
        new_hidden_states = self.student.get_last_hidden_states().to(self.student.dtype)
        new_value = self.estimate_value(new_hidden_states)

        new_log_probs = new_log_probs.to(self.student.dtype).requires_grad_(True)
        old_log_probs = old_log_probs.to(self.student.dtype).detach()

        advantage = (reward - old_value.detach()).to(self.student.dtype).requires_grad_(True)

        ratio = torch.exp(new_log_probs.sum() - old_log_probs.sum()).requires_grad_(True)

        surr1 = ratio * advantage
        surr2 = torch.clamp(
            ratio,
            1 - self.clip_epsilon,
            1 + self.clip_epsilon,
        ) * advantage
        policy_loss = -torch.min(surr1, surr2).mean()

        value_loss = self.value_coef * (advantage ** 2).mean()

        entropy = self.entropy_coef * self.compute_entropy(new_log_probs)

        total_loss = policy_loss + value_loss - entropy
        total_loss /= self.grad_acc_steps

        self.optimizer.zero_grad()
        self.value_optimizer.zero_grad()
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.student.model.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.value_head.parameters(), max_norm=1.0)

        if (self.steps % self.grad_acc_steps == 0):
            self.optimizer.step()
            self.value_optimizer.step()

        self.steps += 1

        return reward.item(), eval_result

    def train(
        self,
        problems: List[Tuple[str, float]],
        n_epochs: int,
        batch_size: int,
    ):
        for epoch in range(n_epochs):
            epoch_rewards = []
            self.student.model.train()

            for i in range(0, len(problems), batch_size):
                batch = problems[i:i+batch_size]
                batch_rewards = []

                for question, ground_truth in batch:
                    try:
                        reward, eval_result = self.train_step(question, ground_truth)
                        batch_rewards.append(reward)

                    except Exception as e:
                        print(f"Error during training step: {e}")
                        continue

                epoch_rewards.extend(batch_rewards)

if __name__ == "__main__":
    from src.teacher.gt_problem_gen import MathProblemGenerator
    from src.teacher.nl_question_gen import TeacherModel
    from src.student.evaluator import StudentEvaluator
    from src.student.solver import StudentModel

    # Training config
    num_problems = 100
    num_epochs = 10
    batch_size = 4

    # Generate problems
    generator = MathProblemGenerator(
        min_val=1,
        max_val=100,
        int_only=True,
        max_operands=5,
    )
    problems = generator.gen_problem_batch(num_problems)

    # Prepare training data
    print("Generating questions from teacher model...")
    teacher = TeacherModel()
    teacher.load()
    questions = teacher.gen_question_batch(problems)
    train_problems = [(question, problem.solution) for question, problem in zip(questions, problems)]
    teacher = None  # Free up memory

    # Initialize models
    student = StudentModel()
    student.load()
    evaluator = StudentEvaluator()

    # Initialize trainer
    trainer = StudentTrainer(
        student,
        evaluator,
        learning_rate=2e-5,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
    )

    # Training loop with detailed logging
    print(f"Starting training with {num_problems} problems for {num_epochs} epochs")
    print(f"Batch size: {batch_size}")

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        epoch_rewards = []
        epoch_format_scores = []
        epoch_correct_scores = []

        # Shuffle problems at the start of each epoch
        random.shuffle(train_problems)

        # Use tqdm for progress bar
        progress_bar = tqdm(range(0, len(train_problems), batch_size))

        for i in progress_bar:
            batch = train_problems[i:i+batch_size]
            batch_rewards = []
            batch_format_scores = []
            batch_correct_scores = []

            for question, ground_truth in batch:
                try:
                    reward, eval_result = trainer.train_step(question, ground_truth)
                    batch_rewards.append(reward)
                    batch_format_scores.append(eval_result.format_score)
                    batch_correct_scores.append(eval_result.correct_score)
                except Exception as e:
                    print(f"\nError during training step: {e}")
                    continue

            epoch_rewards.extend(batch_rewards)
            epoch_format_scores.extend(batch_format_scores)
            epoch_correct_scores.extend(batch_correct_scores)

            # Update progress bar with current batch metrics
            if batch_rewards:  # Only update if we have rewards
                progress_bar.set_description(
                    f"Reward: {sum(batch_rewards)/len(batch_rewards):.3f} "
                    f"Format: {sum(batch_format_scores)/len(batch_format_scores):.3f} "
                    f"Correct: {sum(batch_correct_scores)/len(batch_correct_scores):.3f}"
                )

        # Calculate and display epoch metrics
        if epoch_rewards:
            avg_reward = sum(epoch_rewards) / len(epoch_rewards)
            avg_format = sum(epoch_format_scores) / len(epoch_format_scores)
            avg_correct = sum(epoch_correct_scores) / len(epoch_correct_scores)

            print(f"\nEpoch {epoch+1} Summary:")
            print(f"Average Reward: {avg_reward:.4f}")
            print(f"Average Format Score: {avg_format:.4f}")
            print(f"Average Correctness Score: {avg_correct:.4f}")

            # Display some example problems and solutions
            print("\nExample Solutions:")
            sample_idx = random.randint(0, len(train_problems)-1)
            question, ground_truth = train_problems[sample_idx]
            response = student.solve(question)
            print(f"\nQuestion: {question}")
            print(f"Ground Truth: {ground_truth}")
            print(f"Student's Answer: {response}")

    print("\nTraining completed!")
