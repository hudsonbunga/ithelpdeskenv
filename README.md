---
title: IT Helpdesk OpenEnv
emoji: 🛠️
colorFrom: blue
colorTo: indigo
sdk: docker
app_file: inference.py
pinned: false
license: mit
---

# 🛠️ IT Helpdesk OpenEnv

[![Scaler OpenEnv Hackathon](https://img.shields.io/badge/Scaler-OpenEnv%20Hackathon-blueviolet?style=flat-square)](https://scaler.com)
[![HF Spaces](https://img.shields.io/badge/HuggingFace-Spaces-blue?style=flat-square)](https://huggingface.co/spaces/hudsonbunga/it-helpdesk-env)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python)](https://python.org)
[![Tests](https://img.shields.io/badge/Tests-21%20passed-brightgreen?style=flat-square)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

> **Real-world IT Support Ticket Resolution** — An OpenEnv-compliant RL environment where an LLM agent resolves IT helpdesk tickets across 3 difficulty levels. Uses [puter.js](https://puter.com) for **free GPT-4o** access — no API key required.

---

## Problem Overview

Corporate IT helpdesks handle thousands of tickets daily. Each ticket requires an agent to:

1. **Diagnose** — ask targeted questions to understand the issue
2. **Act** — apply the correct technical fix
3. **Confirm** — verify resolution and close the ticket efficiently

This environment trains LLM agents to do exactly that — with measurable reward signals, customer satisfaction tracking, and step efficiency penalties.

---

## Architecture

```
Browser (puter.js)
  └── puter.ai.chat(prompt)       ← FREE GPT-4o, no API key needed
        └── POST /env/step        ← FastAPI Python backend
              └── ITHelpdeskEnv   ← [START] / [STEP] / [END] logs
```

```
ithelpdeskenv/
├── inference.py          ← Entrypoint: FastAPI + puter.js Gradio UI
├── src/
│   ├── env.py            ← ITHelpdeskEnv  (step / reset / state)
│   ├── tasks.py          ← Task definitions (easy / medium / hard)
│   ├── customer_sim.py   ← Deterministic customer simulator
│   └── __init__.py
├── tests/
│   ├── test_env.py       ← 21-test pytest suite
│   └── __init__.py
├── Dockerfile
├── requirements.txt
└── pyproject.toml
```

---

## Task Definitions

| Level | Task | Max Steps | Optimal | Key Challenge |
|-------|------|-----------|---------|---------------|
| Easy | Password Reset | 8 | 4–6 | Identity verification flow |
| Medium | Software Crash | 12 | 6–8 | Root-cause diagnosis |
| Hard | Network Connectivity | 15 | 9–11 | Multi-layer system diagnosis |

### Resolution Criteria

**Easy — Password Reset**
1. `identity_verified` — Confirm employee ID + registered email
2. `password_reset` — Initiate and complete the password reset
3. `access_confirmed` — Customer confirms they can log in

**Medium — Software Crash**
1. `crash_cause_identified` — Review error logs / diagnose root cause
2. `fix_applied` — Apply the correct fix (repair, update, reinstall)
3. `app_running` — Verify application runs stably

**Hard — Network Connectivity**
1. `hardware_checked` — Inspect physical connections and switch status
2. `config_verified` — Validate DNS, IP, VPN, and firewall settings
3. `connection_stable` — Confirm all users have stable connectivity

---

## Reward Function

Reward is bounded in `[0.0, 1.0]` and computed per step as a weighted sum:

```
reward = w_resolution  × (criteria_completed / total_criteria)
       + w_efficiency  × (steps_remaining / max_steps)
       + w_satisfaction × customer_satisfaction
```

| Component | Easy | Medium | Hard | Description |
|-----------|------|--------|------|-------------|
| Resolution | 60% | 55% | 50% | Fraction of resolution criteria met |
| Efficiency | 25% | 30% | 30% | Steps remaining as fraction of budget |
| Satisfaction | 15% | 15% | 20% | Customer happiness score (0.0–1.0) |

**Grades:** EXCELLENT ≥ 0.85 · GOOD ≥ 0.70 · PARTIAL ≥ 0.40 · FAILED < 0.40

---

## Action Space

```json
{ "type": "ask | action | close", "content": "<message or command>" }
```

| Type | Purpose | Example |
|------|---------|---------|
| `ask` | Ask the customer a diagnostic question | `"What is your employee ID?"` |
| `action` | Execute a technical fix | `"reset password"`, `"check dns"` |
| `close` | Close ticket — only when all criteria are done | `"Issue resolved."` |

---

## Observation Space

```json
{
  "task_name":            "easy | medium | hard",
  "task_description":     "...",
  "conversation_history": [{ "agent": {...}, "customer": {...} }],
  "customer_state": {
    "issue_type":      "password_reset",
    "progress":        0.67,
    "satisfaction":    0.95,
    "completed_steps": ["identity_verified", "password_reset"],
    "resolved":        false
  },
  "diagnosed_info":       { "employee_id": "EMP-4872" },
  "step_count":           3,
  "max_steps":            8,
  "resolution_criteria":  ["identity_verified", "password_reset", "access_confirmed"],
  "completed_steps":      ["identity_verified", "password_reset"],
  "initial_message":      "Hi, I can't log into my computer..."
}
```

---

## Structured Log Format

```
[START]
{ "event": "start", "task_level": "easy", "timestamp": 1712345678.9, ... }

[STEP]
{ "event": "step", "step": 2, "action": { "type": "action", "content": "verify identity" },
  "reward": 0.5375, "done": false, "completed": ["identity_verified"] }

[END]
{ "event": "end", "total_steps": 6, "final_reward": 0.8125, "success": true, "grade": "GOOD" }
```

---

## Environment API

The FastAPI backend exposes these endpoints (called by the puter.js frontend):

```
POST /env/reset   { "task_level": "easy" }
               →  { "session_id": "a1b2c3", "obs": {...} }

POST /env/step    { "session_id": "a1b2c3", "action": {...} }
               →  { "obs": {...}, "reward": 0.54, "done": false, "grade": "PARTIAL" }

GET  /health   →  { "status": "ok", "sessions": 1 }
```

---

## Infrastructure

| Requirement | Value |
|-------------|-------|
| Runtime | < 5 min per episode |
| CPU | 2 vCPU |
| RAM | < 250 MB |
| Python | 3.11+ |
| API key | Not required (puter.js handles LLM calls for free) |

---

## Running the Project

> All commands must be run from the **project root directory**:
> `c:\Users\hudso\Private\OpenEnvRL\ithelpdeskenv\`

### 1. Install dependencies
```bash
# Run from: ithelpdeskenv/
pip install -r requirements.txt
```

### 2. Web UI — free GPT-4o via puter.js (no API key needed)
```bash
# Run from: ithelpdeskenv/
python inference.py

# Then open in browser: http://localhost:7860
```

### 3. CLI demos — heuristic agent (zero cost, offline)
```bash
# Run from: ithelpdeskenv/
python inference.py --mode demo --task easy
python inference.py --mode demo --task medium
python inference.py --mode demo --task hard
```

### 4. Run tests
```bash
# Run from: ithelpdeskenv/
python -m pytest tests/ -v

# Expected output:
# tests/test_env.py .....................  [100%]
# 21 passed in 0.30s
```

### 5. Docker
```bash
# Run from: ithelpdeskenv/  (where Dockerfile is located)
docker build -t ithelpdeskenv .
docker run -p 7860:7860 ithelpdeskenv

# Then open in browser: http://localhost:7860
```

---

## License

REC © 2024 Hudson Bunga
