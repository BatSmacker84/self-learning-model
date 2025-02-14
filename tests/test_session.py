from src.common.session import LearningSession

def test_learning_session():
    session = LearningSession(max_problems=5)
    history = session.run_session()

    assert len(history) == 5
    for problem, response, feedback in history:
        assert hasattr(problem, 'solution')
        assert hasattr(response, 'answer')
        assert hasattr(feedback, 'is_correct')
        assert feedback.is_correct == (problem.solution == response.answer)
