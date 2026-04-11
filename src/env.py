"""
IT Helpdesk OpenEnv — Core Environment

Implements the OpenEnv specification:
  - step(action)  → obs, reward, done, info
  - reset()       → obs, info
  - state()       → current observation dict

Reward is bounded [0.0, 1.0] and is computed from:
  - Resolution bonus  (did the agent solve the ticket?)
  - Efficiency bonus  (did the agent solve it in few steps?)
  - Satisfaction score (was the customer happy with the interaction?)
"""

from typing import Any, Dict, List, Optional, Tuple
import gymnasium as gym
from gymnasium import spaces

from .customer_sim import CustomerSimulator
from .tasks import TASK_DEFINITIONS


class ITHelpdeskEnv(gym.Env):
    """
    OpenEnv-compliant IT Helpdesk environment.

    The agent takes the role of an IT support specialist interacting
    with a simulated customer through structured actions.

    Action Space (dict):
        type    : str  — one of ["ask", "action", "close"]
        content : str  — natural-language instruction / technical command

    Observation Space (dict):
        task_name            : str
        conversation_history : list[dict]  — last 10 turns
        customer_state       : dict        — progress, satisfaction, resolved flags
        diagnosed_info       : dict        — facts the agent has discovered
        step_count           : int
        max_steps            : int
        resolution_criteria  : list[str]   — what needs to happen to close the ticket
        completed_steps      : list[str]   — criteria already met

    Reward is in [0.0, 1.0] per step, shaped to encourage:
        - Completing resolution criteria (0.60 weight for "easy")
        - Acting efficiently within max_steps
        - Keeping customer satisfaction high
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(self, task_level: str = "easy"):
        super().__init__()
        assert task_level in TASK_DEFINITIONS, (
            f"task_level must be one of {list(TASK_DEFINITIONS.keys())}"
        )
        self.task_level = task_level
        self.task_def = TASK_DEFINITIONS[task_level]
        self.max_steps = self.task_def["max_steps"]

        # Gymnasium spaces (kept for spec compliance; env uses dict actions/obs)
        self.action_space = spaces.Dict({
            "type":    spaces.Text(max_length=20),
            "content": spaces.Text(max_length=500),
        })
        self.observation_space = spaces.Dict({
            "task_name":            spaces.Text(max_length=50),
            "conversation_history": spaces.Sequence(spaces.Text(max_length=2000)),
            "step_count":           spaces.Discrete(self.max_steps + 1),
            "max_steps":            spaces.Discrete(self.max_steps + 1),
        })

        self.customer: Optional[CustomerSimulator] = None
        self.conversation_history: List[Dict] = []
        self.step_count: int = 0
        self.diagnosed_info: Dict[str, Any] = {}

        # Initialise — IMPORTANT: do NOT call reset() in __init__ to avoid
        # double-initialisation when the caller does env = ...; obs, _ = env.reset()
        self._initialised = False

    # ------------------------------------------------------------------
    # OpenEnv API
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """Reset the environment and return (observation, info)."""
        super().reset(seed=seed)

        self.customer = CustomerSimulator(self.task_level)
        self.customer.reset()
        self.conversation_history = []
        self.step_count = 0
        self.diagnosed_info = {}
        self._initialised = True

        return self._get_obs(), {}

    def step(self, action: Dict[str, Any]) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Execute one action and return (obs, reward, terminated, truncated, info).

        Compatible with both 4-return (OpenEnv) and 5-return (Gymnasium) callers:
          OpenEnv callers receive the 5-tuple and can ignore `truncated`.
        """
        if not self._initialised:
            self.reset()

        self.step_count += 1

        # Execute action against customer simulator
        response = self.customer.respond(action)

        self.conversation_history.append({
            "agent":    action,
            "customer": {
                "response":          response.get("response", ""),
                "satisfaction_delta": response.get("satisfaction_delta", 0.0),
            },
        })

        # Accumulate discovered facts
        self.diagnosed_info.update(response.get("facts_discovered", {}))

        # Compute termination
        terminated = self.customer.is_resolved()
        truncated  = self.step_count >= self.max_steps and not terminated

        # Reward signal
        reward = self._compute_reward(terminated)

        obs = self._get_obs()
        info = {
            "resolution_status":  self.customer.resolution_status,
            "completed_steps":    list(self.customer.completed_steps),
            "facts_known":        len(self.diagnosed_info),
            "satisfaction":       round(max(0.01, min(0.99, self.customer.satisfaction)), 3),
            "step_count":         self.step_count,
        }

        return obs, reward, terminated, truncated, info

    def state(self) -> Dict[str, Any]:
        """Return the current full state (OpenEnv spec method)."""
        if not self._initialised:
            obs, _ = self.reset()
            return obs
        return self._get_obs()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_obs(self) -> Dict[str, Any]:
        """Build the observation dictionary."""
        customer_state = self.customer.get_state() if self.customer else {}
        return {
            "task_name":            self.task_level,
            "task_description":     self.task_def["description"],
            "conversation_history": self.conversation_history[-10:],
            "customer_state":       customer_state,
            "diagnosed_info":       dict(self.diagnosed_info),
            "step_count":           self.step_count,
            "max_steps":            self.max_steps,
            "resolution_criteria":  self.task_def["resolution_criteria"],
            "completed_steps":      list(self.customer.completed_steps) if self.customer else [],
            "initial_message":      self.task_def["initial_customer_message"],
        }

    def _compute_reward(self, resolved: bool) -> float:
        """
        Compute a shaped reward in [0.0, 1.0].

        Weights come from task_def["scoring_weights"] so each difficulty
        level can tune what matters most.
        """
        weights = self.task_def["scoring_weights"]
        customer_state = self.customer.get_state()

        # Resolution component: fraction of criteria completed
        total = len(self.task_def["resolution_criteria"])
        done  = len(self.customer.completed_steps)
        resolution_score = done / total if total > 0 else 0.0

        # Full-resolution bonus
        if resolved:
            resolution_score = 1.0

        # Efficiency: how much of the step budget is left
        steps_remaining = max(0, self.max_steps - self.step_count)
        efficiency_score = steps_remaining / self.max_steps

        # Satisfaction: directly from customer state
        satisfaction_score = float(customer_state.get("satisfaction", 1.0))

        reward = (
            weights["resolution"]  * resolution_score  +
            weights["efficiency"]  * efficiency_score   +
            weights["satisfaction"] * satisfaction_score
        )

        return round(min(0.99, max(0.01, reward)), 4)

    def render(self, mode: str = "human") -> Optional[str]:
        """Render the current conversation state."""
        lines = [
            f"=== IT Helpdesk | Task: {self.task_level.upper()} ===",
            f"Step: {self.step_count}/{self.max_steps}",
            f"Satisfaction: {self.customer.satisfaction:.2f}",
            f"Progress: {self.customer.progress:.2f}",
            "--- Conversation ---",
        ]
        for turn in self.conversation_history[-5:]:
            lines.append(f"  AGENT:    {turn['agent'].get('content', '')}")
            lines.append(f"  CUSTOMER: {turn['customer'].get('response', '')}")
        output = "\n".join(lines)
        if mode == "human":
            print(output)
        return output