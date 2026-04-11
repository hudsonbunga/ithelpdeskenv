"""
Pytest test suite for IT Helpdesk OpenEnv.

Tests cover:
  - Environment initialisation
  - reset() / step() / state() contracts
  - Reward bounds [0.0, 1.0]
  - All 3 task difficulties
  - Customer simulator logic
  - Max-step truncation
"""

import pytest
from src.env import ITHelpdeskEnv
from src.customer_sim import CustomerSimulator
from src.tasks import TASK_DEFINITIONS


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

@pytest.fixture(params=["easy", "medium", "hard"])
def env_all(request):
    """Parametrised fixture — runs each test for all 3 difficulties."""
    e = ITHelpdeskEnv(request.param)
    e.reset()
    return e


@pytest.fixture
def env_easy():
    e = ITHelpdeskEnv("easy")
    e.reset()
    return e


# --------------------------------------------------------------------------
# Initialisation
# --------------------------------------------------------------------------

def test_env_initialization():
    env = ITHelpdeskEnv("easy")
    assert env.task_level == "easy"
    assert env.max_steps == TASK_DEFINITIONS["easy"]["max_steps"]


def test_env_invalid_task_level():
    with pytest.raises(AssertionError):
        ITHelpdeskEnv("impossible")


# --------------------------------------------------------------------------
# reset()
# --------------------------------------------------------------------------

def test_reset_returns_obs_and_info(env_easy):
    env = ITHelpdeskEnv("easy")
    obs, info = env.reset()
    assert isinstance(obs, dict)
    assert isinstance(info, dict)


def test_reset_obs_keys(env_easy):
    env = ITHelpdeskEnv("easy")
    obs, _ = env.reset()
    required_keys = [
        "task_name", "conversation_history", "customer_state",
        "diagnosed_info", "step_count", "max_steps",
        "resolution_criteria", "completed_steps",
    ]
    for key in required_keys:
        assert key in obs, f"Missing key: {key}"


def test_reset_clears_state(env_easy):
    # Take a step, then reset
    env_easy.step({"type": "ask", "content": "What is your employee ID?"})
    assert env_easy.step_count == 1

    obs, _ = env_easy.reset()
    assert obs["step_count"] == 0
    assert obs["conversation_history"] == []
    assert obs["diagnosed_info"] == {}


# --------------------------------------------------------------------------
# state()
# --------------------------------------------------------------------------

def test_state_method(env_easy):
    state = env_easy.state()
    assert isinstance(state, dict)
    assert state["step_count"] == 0


# --------------------------------------------------------------------------
# step()
# --------------------------------------------------------------------------

def test_step_returns_5_tuple(env_easy):
    result = env_easy.step({"type": "ask", "content": "hello"})
    assert len(result) == 5
    obs, reward, terminated, truncated, info = result
    assert isinstance(obs, dict)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_step_increments_step_count(env_easy):
    env_easy.step({"type": "ask", "content": "test"})
    assert env_easy.step_count == 1
    env_easy.step({"type": "ask", "content": "test"})
    assert env_easy.step_count == 2


def test_step_adds_to_history(env_easy):
    env_easy.step({"type": "ask", "content": "What is your employee ID?"})
    assert len(env_easy.conversation_history) == 1


# --------------------------------------------------------------------------
# Reward bounds
# --------------------------------------------------------------------------

def test_reward_bounds_all_tasks(env_all):
    for _ in range(5):
        _, reward, _, _, _ = env_all.step({"type": "ask", "content": "test"})
        assert 0.0 < reward < 1.0, f"Reward out of bounds: {reward}"


def test_reward_on_resolution(env_easy):
    """Full resolution should push reward toward high values."""
    # Unlock all steps
    env_easy.step({"type": "action", "content": "verify identity"})
    env_easy.step({"type": "action", "content": "reset password"})
    _, reward, terminated, _, info = env_easy.step({"type": "close", "content": "resolved"})
    # If all steps are done before close, terminated should be true
    if terminated:
        assert reward >= 0.5


# --------------------------------------------------------------------------
# All difficulty levels
# --------------------------------------------------------------------------

def test_all_difficulties_init():
    for level in ["easy", "medium", "hard"]:
        env = ITHelpdeskEnv(level)
        obs, _ = env.reset()
        assert obs["task_name"] == level
        assert obs["max_steps"] == TASK_DEFINITIONS[level]["max_steps"]


def test_all_difficulties_step():
    for level in ["easy", "medium", "hard"]:
        env = ITHelpdeskEnv(level)
        env.reset()
        obs, reward, terminated, truncated, info = env.step(
            {"type": "ask", "content": "Can you describe the issue?"}
        )
        assert 0.0 < reward < 1.0


# --------------------------------------------------------------------------
# Truncation
# --------------------------------------------------------------------------

def test_max_steps_truncation():
    env = ITHelpdeskEnv("easy")
    env.reset()
    env.max_steps = 3

    last_truncated = False
    for i in range(4):
        _, _, terminated, truncated, _ = env.step(
            {"type": "ask", "content": f"step {i}"}
        )
        if truncated:
            last_truncated = True

    assert last_truncated, "Episode should have been truncated after max_steps"


# --------------------------------------------------------------------------
# Customer simulator
# --------------------------------------------------------------------------

def test_customer_respond_ask():
    sim = CustomerSimulator("easy")
    sim.reset()
    result = sim.respond({"type": "ask", "content": "what is your employee id"})
    assert "response" in result
    assert isinstance(result["response"], str)


def test_customer_respond_premature_close():
    sim = CustomerSimulator("easy")
    sim.reset()
    result = sim.respond({"type": "close", "content": "done"})
    # Premature close should return negative satisfaction delta
    assert result["satisfaction_delta"] < 0


def test_customer_progress_updates():
    sim = CustomerSimulator("easy")
    sim.reset()
    assert sim.progress == 0.01
    sim.respond({"type": "action", "content": "verify identity"})
    assert sim.progress > 0.0


def test_customer_get_state():
    sim = CustomerSimulator("easy")
    sim.reset()
    state = sim.get_state()
    assert "issue_type" in state
    assert "progress" in state
    assert "satisfaction" in state
    assert 0.0 < state["satisfaction"] < 1.0
    assert 0.0 < state["progress"] < 1.0


# --------------------------------------------------------------------------
# Invalid actions
# --------------------------------------------------------------------------

def test_invalid_action_type(env_easy):
    obs, reward, terminated, truncated, info = env_easy.step(
        {"type": "invalid_type", "content": "what?"}
    )
    assert 0.0 < reward < 1.0
    assert not terminated


if __name__ == "__main__":
    pytest.main([__file__, "-v"])