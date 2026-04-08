"""
Task definitions for the IT Helpdesk OpenEnv.

Each task represents a real-world IT support scenario at a different
difficulty level, with resolution criteria, optimal step counts, and
contextual metadata for the LLM agent.
"""

TASK_DEFINITIONS = {
    "easy": {
        "name": "Password Reset",
        "issue_type": "password_reset",
        "description": (
            "A company employee is locked out of their account after too many "
            "failed login attempts. The agent must verify the employee's identity "
            "using their employee ID and registered email, then initiate a secure "
            "password reset, and finally confirm that the employee can log in again."
        ),
        "max_steps": 8,
        "optimal_steps": 4,
        "resolution_criteria": ["identity_verified", "password_reset", "access_confirmed"],
        "initial_customer_message": (
            "Hi, I can't log into my computer. I tried my password a few times "
            "and now it says my account is locked."
        ),
        "action_hints": {
            "identity_verified": ["ask employee ID", "ask email", "verify identity"],
            "password_reset":    ["reset password", "unlock account", "send reset link"],
            "access_confirmed":  ["confirm access", "verify login", "close ticket"],
        },
        "scoring_weights": {
            "resolution": 0.60,
            "efficiency": 0.25,
            "satisfaction": 0.15,
        },
    },

    "medium": {
        "name": "Software Crash",
        "issue_type": "software_crash",
        "description": (
            "An employee's critical business application crashes repeatedly "
            "whenever they try to open a specific file. The agent must diagnose "
            "the root cause (corrupted file, missing DLL, outdated driver, or "
            "conflicting process), apply the correct fix, and verify the application "
            "runs stably."
        ),
        "max_steps": 12,
        "optimal_steps": 7,
        "resolution_criteria": ["crash_cause_identified", "fix_applied", "app_running"],
        "initial_customer_message": (
            "My Excel keeps crashing every time I open the Q4 report. "
            "It was fine yesterday. This is urgent — I have a presentation in 2 hours!"
        ),
        "action_hints": {
            "crash_cause_identified": [
                "check error logs", "ask about recent changes",
                "check disk space", "run diagnostics"
            ],
            "fix_applied": [
                "repair office", "clear cache", "reinstall", "update drivers",
                "kill conflicting process"
            ],
            "app_running": ["confirm app works", "test with file", "verify stable"],
        },
        "scoring_weights": {
            "resolution": 0.55,
            "efficiency": 0.30,
            "satisfaction": 0.15,
        },
    },

    "hard": {
        "name": "Network Connectivity Issue",
        "issue_type": "network_issue",
        "description": (
            "An entire department is experiencing intermittent VPN drops and slow "
            "network speeds. The agent must systematically diagnose the issue across "
            "hardware (cables, switch), OS configuration (DNS, IP settings), and "
            "VPN client settings, then apply targeted fixes and confirm stable "
            "connectivity for all affected users."
        ),
        "max_steps": 15,
        "optimal_steps": 10,
        "resolution_criteria": ["hardware_checked", "config_verified", "connection_stable"],
        "initial_customer_message": (
            "Our whole team in Building B keeps getting disconnected from the VPN "
            "every 10-15 minutes. It's been happening since this morning and "
            "we can't get any work done. We have 8 people affected."
        ),
        "action_hints": {
            "hardware_checked": [
                "check cables", "check switch lights", "ping gateway",
                "check physical connection"
            ],
            "config_verified": [
                "check DNS", "check IP settings", "review VPN config",
                "check firewall rules", "verify DHCP"
            ],
            "connection_stable": [
                "run ping test", "confirm stable VPN", "test for 5 minutes",
                "verify all users connected"
            ],
        },
        "scoring_weights": {
            "resolution": 0.50,
            "efficiency": 0.30,
            "satisfaction": 0.20,
        },
    },
}