"""
inference.py - IT Helpdesk OpenEnv Baseline Agent
Scaler OpenEnv Hackathon Submission

Architecture:
  FastAPI backend:
    GET  /          <- serves ui.html (theme selector + puter.js agent)
    POST /env/reset <- initialise episode, return session_id + obs
    POST /env/step  <- step env with action, return obs+reward+done
    GET  /health    <- health check
  Browser (puter.js) <- FREE GPT-4o, no API key required

Usage:
    python inference.py                         # Launch web UI (default)
    python inference.py --mode demo --task easy # CLI heuristic demo
    python inference.py --mode llm  --task hard # CLI OpenAI demo (needs key)
"""

from __future__ import annotations

import os
import sys
import json
import time
import uuid
import argparse
import traceback
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, HTMLResponse
import uvicorn

from src.env import ITHelpdeskEnv
from src.tasks import TASK_DEFINITIONS

# --------------------------------------------------------------------------
# Session store
# --------------------------------------------------------------------------

_sessions: Dict[str, Any] = {}

OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
MODEL:          str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
MAX_STEPS:      int = int(os.getenv("MAX_STEPS", "20"))


# --------------------------------------------------------------------------
# Structured Logger
# --------------------------------------------------------------------------

class StructuredLogger:
    def start(self, task_level: str, obs: Dict) -> None:
        print("[START]")
        print(json.dumps({
            "event":      "start",
            "task_level": task_level,
            "timestamp":  time.time(),
            "initial_obs": {
                "task_name":   obs.get("task_name"),
                "max_steps":   obs.get("max_steps"),
                "criteria":    obs.get("resolution_criteria"),
                "description": obs.get("task_description"),
            },
        }, indent=2))

    def step(self, step_num: int, action: Dict, obs: Dict,
             reward: float, done: bool, info: Dict) -> None:
        hist = obs.get("conversation_history", [])
        last_resp = hist[-1].get("customer", {}).get("response", "") if hist else ""
        print("[STEP]")
        print(json.dumps({
            "event":        "step",
            "step":         step_num,
            "timestamp":    time.time(),
            "action":       action,
            "reward":       round(reward, 4),
            "done":         done,
            "completed":    info.get("completed_steps", []),
            "satisfaction": info.get("satisfaction", 1.0),
            "facts_known":  info.get("facts_known", 0),
            "last_response": last_resp,
        }, indent=2))

    def end(self, steps: int, final_reward: float, success: bool) -> None:
        print("[END]")
        print(json.dumps({
            "event":        "end",
            "timestamp":    time.time(),
            "total_steps":  steps,
            "final_reward": round(final_reward, 4),
            "success":      success,
            "grade":        _grade(final_reward),
        }, indent=2))


logger = StructuredLogger()


def _grade(r: float) -> str:
    return ("EXCELLENT" if r >= 0.85 else
            "GOOD"      if r >= 0.70 else
            "PARTIAL"   if r >= 0.40 else "FAILED")


def _serialisable(obj: Any) -> Any:
    if isinstance(obj, dict):          return {k: _serialisable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [_serialisable(v) for v in obj]
    try:    json.dumps(obj);           return obj
    except: return str(obj)


# --------------------------------------------------------------------------
# Environment helpers
# --------------------------------------------------------------------------

def env_reset(task_level: str) -> Dict:
    session_id = str(uuid.uuid4())[:8]
    env = ITHelpdeskEnv(task_level)
    obs, _ = env.reset()
    logger.start(task_level, obs)
    _sessions[session_id] = {"env": env, "task_level": task_level}
    if len(_sessions) > 50:
        del _sessions[next(iter(_sessions))]
    return {"session_id": session_id, "obs": _serialisable(obs)}


def env_step(session_id: str, action: Dict) -> Dict:
    session = _sessions.get(session_id)
    if not session:
        return {"error": "Session not found or expired. Please reset."}
    env = session["env"]
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    logger.step(env.step_count, action, obs, reward, done, info)
    if done:
        logger.end(env.step_count, reward, terminated)
    return {
        "obs":        _serialisable(obs),
        "reward":     reward,
        "done":       done,
        "terminated": terminated,
        "grade":      _grade(reward),
        "info":       info,
    }


# --------------------------------------------------------------------------
# FastAPI app
# --------------------------------------------------------------------------

fast_app = FastAPI(title="IT Helpdesk OpenEnv API", version="1.0.0")


@fast_app.post("/env/reset")
async def api_reset(request: Request):
    body   = await request.json()
    result = env_reset(body.get("task_level", "easy"))
    return JSONResponse(content=result)


@fast_app.post("/env/step")
async def api_step(request: Request):
    body   = await request.json()
    result = env_step(body.get("session_id", ""), body.get("action", {}))
    return JSONResponse(content=result)


@fast_app.get("/health")
async def health():
    return {"status": "ok", "sessions": len(_sessions)}


# --------------------------------------------------------------------------
# UI page — loaded from ui.html (keeps JS clean, no Python escaping issues)
# --------------------------------------------------------------------------

import pathlib as _pathlib

_UI_FILE = _pathlib.Path(__file__).parent / "ui.html"
_PAGE_HTML: str = _UI_FILE.read_text(encoding="utf-8")


@fast_app.get("/")
async def serve_ui():
    from fastapi.responses import HTMLResponse as _HR
    # Re-read on each request in dev so edits are reflected without restart
    html = _UI_FILE.read_text(encoding="utf-8")
    return _HR(content=html)







# --------------------------------------------------------------------------
# Heuristic baseline agent  (CLI demo)
# --------------------------------------------------------------------------

class HeuristicAgent:
    _SCRIPTS = {
        "password_reset": [
            {"type": "ask",    "content": "Can you provide your employee ID and registered email?"},
            {"type": "action", "content": "verify identity"},
            {"type": "action", "content": "reset password"},
            {"type": "ask",    "content": "Please try logging in now - does it work?"},
            {"type": "action", "content": "confirm access"},
            {"type": "close",  "content": "Password reset complete. You should be able to log in now."},
        ],
        "software_crash": [
            {"type": "ask",    "content": "Can you open Event Viewer and tell me the exact error?"},
            {"type": "action", "content": "check error logs"},
            {"type": "action", "content": "repair office"},
            {"type": "action", "content": "confirm app works"},
            {"type": "close",  "content": "Application repaired successfully. Ticket closed."},
        ],
        "network_issue": [
            {"type": "ask",    "content": "Can you check the switch port lights and all cable connections?"},
            {"type": "action", "content": "check cables"},
            {"type": "ask",    "content": "What does your DNS show? Can you pull up the VPN client log?"},
            {"type": "action", "content": "check dns"},
            {"type": "action", "content": "run ping test"},
            {"type": "close",  "content": "Network stabilised. All users should be reconnected."},
        ],
    }

    def __init__(self):
        self._script: List[Dict] = []
        self._idx: int = 0

    def start(self, issue_type: str) -> None:
        self._script = list(self._SCRIPTS.get(issue_type, self._SCRIPTS["password_reset"]))
        self._idx = 0

    def next_action(self, obs: Dict) -> Dict:
        if self._idx < len(self._script):
            action = self._script[self._idx]
            self._idx += 1
            return action
        return {"type": "close", "content": "All issues resolved. Ticket closed."}


# --------------------------------------------------------------------------
# Optional OpenAI LLM agent (CLI --mode llm)
# --------------------------------------------------------------------------

_LLM_SYSTEM = """You are an IT Helpdesk specialist resolving a customer ticket.
RESPOND WITH RAW JSON ONLY: {"type": "ask|action|close", "content": "..."}
- ask: ONE focused question
- action: technical command (e.g. "verify identity", "reset password", "check error logs")
- close: ONLY when STILL NEEDED list is empty
Example: {"type":"action","content":"verify identity"}"""


class LLMAgent:
    def __init__(self, model: str = MODEL):
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set.")
        from openai import OpenAI as _OAI
        self.client = _OAI(api_key=OPENAI_API_KEY)
        self.model  = model

    def next_action(self, obs: Dict) -> Dict:
        criteria  = obs.get("resolution_criteria", [])
        completed = obs.get("completed_steps", [])
        pending   = [c for c in criteria if c not in completed]
        history   = obs.get("conversation_history", [])
        facts     = obs.get("diagnosed_info", {})
        prompt = (
            f"TASK: {obs.get('task_name','').upper()}\n"
            f"CRITERIA: {criteria}\nCOMPLETED: {completed}\nSTILL NEEDED: {pending}\n"
            f"FACTS: {json.dumps(facts)}\n"
            f"CUSTOMER (opening): {obs.get('initial_message','')}\n"
        )
        for t in history[-4:]:
            prompt += f"AGENT [{t['agent'].get('type')}]: {t['agent'].get('content','')}\n"
            prompt += f"CUSTOMER: {t['customer'].get('response','')}\n"
        prompt += "\nYOUR ACTION (raw JSON):"
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": _LLM_SYSTEM},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.1, max_tokens=120,
            )
            raw = resp.choices[0].message.content.strip()
            raw = raw.strip("`").strip()
            if raw.startswith("json"): raw = raw[4:].strip()
            s, e = raw.find("{"), raw.rfind("}") + 1
            if s != -1 and e > s:
                return json.loads(raw[s:e])
        except Exception as err:
            print(f"[WARN] LLM: {err}", file=sys.stderr)
        return {"type": "ask", "content": "Can you describe the issue?"}


# --------------------------------------------------------------------------
# CLI demo runner
# --------------------------------------------------------------------------

def run_demo(task_level: str, max_steps: int = MAX_STEPS,
             use_llm: bool = False) -> List[Dict]:
    env = ITHelpdeskEnv(task_level)
    obs, _ = env.reset()
    logger.start(task_level, obs)

    issue_type = TASK_DEFINITIONS[task_level]["issue_type"]

    if use_llm:
        agent: Any = LLMAgent()
        print(f"[LLM Agent] model={MODEL}")
    else:
        h = HeuristicAgent()
        h.start(issue_type)
        agent = h
        print(f"[Heuristic Agent] issue={issue_type}")

    print(f"Customer: {obs['initial_message']}\n")
    history = [{"step": 0, "action": None, "reward": 0.0}]

    for step_num in range(1, max_steps + 1):
        action = agent.next_action(obs)
        print(f"Step {step_num} | [{action['type']}] {action['content']}")

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        logger.step(step_num, action, obs, reward, done, info)

        hist = obs.get("conversation_history", [])
        if hist:
            print(f"         Customer: {hist[-1]['customer'].get('response','')}")
        print(f"         Reward: {reward:.4f} | Completed: {info.get('completed_steps',[])}")

        history.append({"step": step_num, "action": action, "reward": reward, "done": done})
        if done:
            break

    final   = history[-1]
    success = env.customer.is_resolved()
    logger.end(final["step"], final["reward"], success)

    print(f"\n{'-'*50}")
    print(f"Steps:  {final['step']}   Reward: {final['reward']:.4f}   Grade: {_grade(final['reward'])}   Success: {success}")
    return history


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="IT Helpdesk OpenEnv")
    parser.add_argument("--mode",  choices=["ui", "demo", "llm"], default="ui",
                        help="ui=web UI  demo=heuristic CLI  llm=OpenAI CLI")
    parser.add_argument("--task",  choices=["easy", "medium", "hard"], default="easy")
    parser.add_argument("--steps", type=int, default=MAX_STEPS)
    args = parser.parse_args()

    if args.mode == "demo":
        run_demo(args.task, args.steps, use_llm=False)

    elif args.mode == "llm":
        run_demo(args.task, args.steps, use_llm=True)

    else:  # ui
        print("Launching IT Helpdesk OpenEnv on http://localhost:7860")
        print("Puter.js provides free GPT-4o - no API key required.")
        uvicorn.run(fast_app, host="0.0.0.0", port=7860, log_level="warning")


if __name__ == "__main__":
    main()