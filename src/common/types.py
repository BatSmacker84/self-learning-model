from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class Problem:
    """Represents a learning problem."""
    question: str
    difficulty: float
    topic: str
    solution: Any
    metadata: Dict[str, Any]

@dataclass
class StudentResponse:
    """Represents a student's answer to a problem."""
    problem_id: str
    answer: Any
    reasoning: str
    confidence: float

@dataclass
class Feedback:
    """Represents teacher feedback on a student response."""
    is_correct: bool
    explanation: str
    score: float
    suggestions: Optional[str] = None
