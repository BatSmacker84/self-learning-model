from src.teacher.problem_gen import MathProblemGenerator
from src.student.solver import StudentSolver

def test_student_solver_initialization():
    solver = StudentSolver()
    assert solver.learning_rate == 0.1
    assert solver.knowledge_level == 0.0
    assert len(solver.performance_history) == 0

def test_student_problem_solving():
    generator = MathProblemGenerator()
    solver = StudentSolver()

    problem = generator.generate_addition_problem()
    response = solver.solve_problem(problem)

    assert isinstance(response.answer, int)
    assert isinstance(response.confidence, float)
    assert 0.0 <= response.confidence <= 1.0
    assert isinstance(response.reasoning, str)

def test_knowledge_update():
    solver = StudentSolver()
    initial_knowledge = solver.knowledge_level

    # Test correct answer
    solver.update_knowledge(True)
    assert solver.knowledge_level > initial_knowledge

    # Test incorrect answer
    current_knowledge = solver.knowledge_level
    solver.update_knowledge(False)
    assert solver.knowledge_level < current_knowledge
