from dataclasses import dataclass
from enum import Enum
from typing import List, Any, Dict, Optional

class MathOperation(Enum):
    ADD = "addition"
    SUB = "subtraction"
    MUL = "multiplication"
    DIV = "division"

@dataclass
class MathProblem:
    operation: MathOperation
    operands: List[float]
    solution: float
    difficulty: int
    metadata: Dict[str, Any]

@dataclass
class TeacherPrompt:
    system: str = """You are a mathematics teacher creating clear, engaging questions for students.
Your task is to convert mathematical problems into natural language questions.
Keep your questions clear, concise, and appropriate for the difficulty level.
Do not provide solutions or hints in your questions.
Focus on making the mathematical concept understandable in real-world contexts when appropriate."""

    template: str = """Given the following mathematical problem, create a natural language question:

Operation: {operation}
Numbers: {numbers}
Difficulty Level: {difficulty}/10

Generate a clear and engaging question that tests this mathematical concept.
Your response should only contain the question itself, without any additional explanation or solution.

Question:"""

@dataclass
class EvalResult:
    format_score: float
    correct_score: float
    total_score: float
    extracted_answer: Optional[float]
    formatted_properly: bool
    has_thinking: bool
    has_solution: bool
