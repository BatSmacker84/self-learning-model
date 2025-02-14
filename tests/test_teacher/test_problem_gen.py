from src.teacher.problem_gen import MathProblemGenerator

def test_problem_generator_init():
    generator = MathProblemGenerator()
    assert generator.difficulty_range == (0.0, 1.0)
    assert generator.current_difficulty == 0.0

def test_addition_problem_gen():
    generator = MathProblemGenerator()
    problem = generator.generate_addition_problem()

    assert isinstance(problem.question, str)
    assert '+' in problem.question
    assert isinstance(problem.solution, int)
    assert problem.topic == "math"
    assert 0.0 <= problem.difficulty <= 1.0
