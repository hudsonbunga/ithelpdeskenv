"""
Customer Simulator for IT Helpdesk OpenEnv.

Simulates a realistic customer who has a specific IT problem and responds
to agent actions with natural-language replies. Tracks resolution progress,
satisfaction, and the facts the agent has discovered.
"""

import random
from typing import Dict, Any, List


# ---------------------------------------------------------------------------
# Scripted response banks per issue type
# (Used for deterministic, realistic dialogue without requiring LLM calls)
# ---------------------------------------------------------------------------

_CUSTOMER_RESPONSES = {
    "password_reset": {
        # What does the customer say when asked about different things?
        "identity_ask": [
            "My employee ID is EMP-4872. My email is sarah.jones@company.com.",
            "Sure — ID is EMP-4872 and email is sarah.jones@company.com.",
        ],
        "reset_ack": [
            "Okay, I followed the link. I've set a new password.",
            "Done! I've just reset it using the link you sent.",
        ],
        "access_confirm": [
            "Yes! I'm in now. Everything is working perfectly, thank you!",
            "It works! I'm logged in. Thanks so much for the quick help.",
        ],
        "generic": [
            "I'm not sure what you mean. Can you explain?",
            "Hmm, what should I do next?",
            "Okay, I'll try that. What should I do?",
        ],
        "irrelevant": [
            "I already told you that. Can we move on?",
            "That doesn't seem related to my issue.",
        ],
    },

    "software_crash": {
        "error_logs": [
            "I found an error: 'msvcp140.dll not found'. Is that the problem?",
            "The log shows: Application Error 0xc0000005 at ntdll.dll offset 0x0002d36b.",
        ],
        "recent_changes": [
            "Actually yes, IT pushed a Windows update last night.",
            "Now that you mention it — there was a Windows Defender update this morning.",
        ],
        "disk_space": [
            "It shows 2.1 GB free on C drive. Is that enough?",
            "Only 3% free space left on the drive! Could that be it?",
        ],
        "fix_ack": [
            "I ran the repair. Restarting now... IT'S WORKING! The file opens!",
            "Office repair is done. Just tested — no more crash! You're a lifesaver!",
        ],
        "verify_stable": [
            "I've had it open for 5 minutes and it's still fine. Presentation is saved!",
            "Working perfectly! I even opened three files at once — no crashes.",
        ],
        "generic": [
            "Okay, let me try that...",
            "Done. What should I check next?",
            "All right. Anything else I should do?",
        ],
    },

    "network_issue": {
        "hardware_check": [
            "The switch in the server room has a flashing amber light on port 12.",
            "All cables look secure, but the patch panel light for our section is off.",
        ],
        "ping_result": [
            "Ping to 192.168.1.1: packets lost 30%. Latency spikes to 800ms.",
            "Only 2 out of 10 pings to the gateway came back. Average 450ms.",
        ],
        "dns_check": [
            "DNS shows 10.0.0.1 as primary, but I think it should be 10.0.0.50?",
            "The DNS entry looks the same as last week. Not sure if it's correct.",
        ],
        "vpn_config": [
            "VPN client version is 4.6.2. The server address is vpn.company.com.",
            "It shows 'UDP timeout' in the VPN logs every 12-14 minutes.",
        ],
        "fix_ack": [
            "I changed the MTU to 1400 and updated the VPN client. VPN is stable now!",
            "After switching to TCP mode and flushing DNS, we've had 20 minutes of uptime!",
        ],
        "verify_stable": [
            "It's been 25 minutes — all 8 users are still connected. Fixed!",
            "Rock solid! Everyone is back online and productivity is restored.",
        ],
        "generic": [
            "Let me check that... okay, what do you need to know?",
            "I'll check with the team. Give me a moment.",
            "Done. Should I tell the others to do anything?",
        ],
    },
}


class CustomerSimulator:
    """
    Simulates a customer interacting with an IT helpdesk agent.

    Tracks:
    - resolution_criteria progress (which gate steps are unlocked)
    - satisfaction score (decreases with irrelevant/repeated questions)
    - progress score (0→1 as resolution steps are completed)
    - facts discovered by the agent
    """

    def __init__(self, task_level: str):
        self.task_level = task_level
        self.reset()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self):
        from src.tasks import TASK_DEFINITIONS  # avoid circular import at module level

        task = TASK_DEFINITIONS[self.task_level]
        self.issue_type: str = task["issue_type"]
        self.required_steps: List[str] = list(task["resolution_criteria"])
        self.completed_steps: List[str] = []
        self.progress: float = 0.0
        self.satisfaction: float = 1.0
        self.resolved: bool = False
        self.resolution_status: str = "open"
        self._step_unlock_map = self._build_unlock_map()
        self._question_count: int = 0

    def respond(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an agent action and return a customer response dict with:
          - response: str          (natural-language customer reply)
          - facts_discovered: dict (new KV facts the agent now knows)
          - satisfaction_delta: float
        """
        action_type = action.get("type", "unknown").lower()
        content = action.get("content", "").lower()

        if action_type == "ask":
            return self._handle_ask(content)
        elif action_type == "action":
            return self._handle_action(content)
        elif action_type == "close":
            return self._handle_close(content)
        else:
            self.satisfaction = max(0.1, self.satisfaction - 0.05)
            return {
                "response": "I'm not sure what you're trying to do. Can you clarify?",
                "facts_discovered": {},
                "satisfaction_delta": -0.05,
            }

    def get_state(self) -> Dict[str, Any]:
        return {
            "issue_type": self.issue_type,
            "progress": self.progress,
            "satisfaction": self.satisfaction,
            "completed_steps": self.completed_steps,
            "required_steps": self.required_steps,
            "resolved": self.resolved,
        }

    def is_resolved(self) -> bool:
        return self.resolved

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _handle_ask(self, content: str) -> Dict[str, Any]:
        self._question_count += 1
        resp = _CUSTOMER_RESPONSES.get(self.issue_type, {})

        # Match content to known question categories
        matched_key, response_text, facts = self._match_question_to_key(content, resp)

        # Slight satisfaction penalty for excessive questions
        delta = 0.0
        if self._question_count > 5:
            delta = -0.05
            self.satisfaction = max(0.1, self.satisfaction + delta)

        return {
            "response": response_text,
            "facts_discovered": facts,
            "satisfaction_delta": delta,
        }

    def _handle_action(self, content: str) -> Dict[str, Any]:
        """Apply a technical action — may unlock resolution steps."""
        facts_discovered = {}
        delta = 0.0

        # Check whether this action maps to a resolution criterion
        for step in self.required_steps:
            if step in self.completed_steps:
                continue
            hints = self._step_unlock_map.get(step, [])
            if any(h in content for h in hints):
                self.completed_steps.append(step)
                self._update_progress()
                facts_discovered[step] = True
                response = self._pick_response_for_step(step)
                delta = 0.05
                self.satisfaction = min(1.0, self.satisfaction + delta)
                return {
                    "response": response,
                    "facts_discovered": facts_discovered,
                    "satisfaction_delta": delta,
                }

        # No matching step — mild penalty
        delta = -0.05
        self.satisfaction = max(0.1, self.satisfaction + delta)
        resp = _CUSTOMER_RESPONSES.get(self.issue_type, {})
        generic = resp.get("generic", ["Okay, let me try that..."])
        return {
            "response": random.choice(generic),
            "facts_discovered": facts_discovered,
            "satisfaction_delta": delta,
        }

    def _handle_close(self, content: str) -> Dict[str, Any]:
        """Agent closes the ticket — only valid if all steps resolved."""
        all_done = all(s in self.completed_steps for s in self.required_steps)

        if all_done:
            self.resolved = True
            self.resolution_status = "resolved"
            self.satisfaction = min(1.0, self.satisfaction + 0.1)
            return {
                "response": "Thank you so much! My issue is completely resolved. Great support!",
                "facts_discovered": {"ticket_closed": True},
                "satisfaction_delta": 0.1,
            }
        else:
            # Premature close — big satisfaction penalty
            delta = -0.2
            self.satisfaction = max(0.1, self.satisfaction + delta)
            pending = [s for s in self.required_steps if s not in self.completed_steps]
            return {
                "response": (
                    f"Wait, my issue isn't solved yet! "
                    f"Still need help with: {', '.join(pending).replace('_', ' ')}."
                ),
                "facts_discovered": {},
                "satisfaction_delta": delta,
            }

    # ------------------------------------------------------------------
    # Question matching
    # ------------------------------------------------------------------

    def _match_question_to_key(
        self, content: str, resp: Dict
    ):
        """Route the agent's question to a response bank key."""
        from src.tasks import TASK_DEFINITIONS

        task = TASK_DEFINITIONS[self.task_level]

        # Map question content to response keys based on issue type
        routing = {
            "password_reset": {
                "identity_ask": ["employee id", "id", "email", "name", "verify", "identity", "who are you"],
                "reset_ack":    ["password reset", "reset link", "new password", "set password"],
                "access_confirm": ["working", "logged in", "access", "can you log", "try logging"],
            },
            "software_crash": {
                "error_logs":      ["error log", "event viewer", "crash log", "log", "error message"],
                "recent_changes":  ["recent changes", "update", "installed", "changed", "upgrade"],
                "disk_space":      ["disk space", "storage", "free space", "drive"],
                "fix_ack":         ["repair", "reinstall", "cache", "clear", "fix", "run repair"],
                "verify_stable":   ["working", "stable", "confirmed", "test", "open the file"],
            },
            "network_issue": {
                "hardware_check": ["cable", "switch", "physical", "hardware", "port", "patch"],
                "ping_result":    ["ping", "latency", "packet loss", "gateway"],
                "dns_check":      ["dns", "name server", "lookup", "resolve"],
                "vpn_config":     ["vpn", "tunnel", "client version", "udp", "tcp", "mtu"],
                "fix_ack":        ["change setting", "flush dns", "update client", "mtu", "tcp mode"],
                "verify_stable":  ["stable", "connected", "uptime", "everyone online"],
            },
        }

        issue_routing = routing.get(self.issue_type, {})
        for key, keywords in issue_routing.items():
            if any(kw in content for kw in keywords):
                texts = resp.get(key, resp.get("generic", ["Okay."]))
                return (
                    key,
                    random.choice(texts),
                    self._facts_for_key(key),
                )

        generic = resp.get("generic", ["Okay, I'll try that."])
        return ("generic", random.choice(generic), {})

    def _facts_for_key(self, key: str) -> Dict:
        """Return newly discovered facts for a matched key."""
        facts_map = {
            # password_reset
            "identity_ask":    {"employee_id": "EMP-4872", "email": "sarah.jones@company.com"},
            "reset_ack":       {"password_reset_done": True},
            "access_confirm":  {"access_confirmed": True},
            # software_crash
            "error_logs":      {"crash_log_reviewed": True, "error_code": "0xc0000005"},
            "recent_changes":  {"recent_update": "Windows Defender"},
            "disk_space":      {"disk_space_gb": 2.1},
            "fix_ack":         {"fix_applied": True},
            "verify_stable":   {"app_stable": True},
            # network_issue
            "hardware_check":  {"switch_amber_light": True, "port": 12},
            "ping_result":     {"packet_loss_pct": 30, "avg_latency_ms": 450},
            "dns_check":       {"dns_server": "10.0.0.1"},
            "vpn_config":      {"vpn_version": "4.6.2", "vpn_error": "UDP timeout"},
        }
        return facts_map.get(key, {})

    # ------------------------------------------------------------------
    # Resolution progress
    # ------------------------------------------------------------------

    def _build_unlock_map(self) -> Dict[str, List[str]]:
        """Map each resolution criterion to action keywords that unlock it."""
        maps = {
            "password_reset": {
                "identity_verified": ["verify identity", "check identity", "validate employee"],
                "password_reset":    ["reset password", "send reset", "unlock account", "reset_password"],
                "access_confirmed":  ["confirm access", "verify login", "check_access"],
            },
            "software_crash": {
                "crash_cause_identified": [
                    "check error logs", "run diagnostics", "check_logs",
                    "check disk space", "diagnose"
                ],
                "fix_applied": [
                    "repair office", "clear cache", "reinstall", "update drivers",
                    "fix_app", "repair_office", "kill process"
                ],
                "app_running": ["confirm app works", "test file", "verify stable", "confirm_running"],
            },
            "network_issue": {
                "hardware_checked": [
                    "check cables", "check switch", "inspect physical",
                    "check_hardware", "ping gateway"
                ],
                "config_verified": [
                    "check dns", "check ip", "verify vpn", "review config",
                    "check_config", "flush dns", "check_firewall"
                ],
                "connection_stable": [
                    "run ping test", "confirm stable", "test connection",
                    "verify_stable", "confirm_stable"
                ],
            },
        }
        return maps.get(self.issue_type, {})

    def _update_progress(self):
        done = len(self.completed_steps)
        total = len(self.required_steps)
        self.progress = done / total if total > 0 else 0.0

    def _pick_response_for_step(self, step: str) -> str:
        resp = _CUSTOMER_RESPONSES.get(self.issue_type, {})
        # Map resolution steps to response keys
        step_to_response = {
            # password_reset
            "identity_verified": "identity_ask",
            "password_reset":    "reset_ack",
            "access_confirmed":  "access_confirm",
            # software_crash
            "crash_cause_identified": "error_logs",
            "fix_applied":            "fix_ack",
            "app_running":            "verify_stable",
            # network_issue
            "hardware_checked":   "hardware_check",
            "config_verified":    "vpn_config",
            "connection_stable":  "verify_stable",
        }
        key = step_to_response.get(step, "generic")
        texts = resp.get(key, resp.get("generic", ["Great, that seemed to work!"]))
        return random.choice(texts)