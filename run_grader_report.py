#!/usr/bin/env python3
"""Run all tasks with baseline agent and print step-by-step + final scores.

Usage:
    python run_grader_report.py          # random seed each run
    python run_grader_report.py 99       # fixed seed=99
"""
import random
import sys
from env.environment import TicketTriageEnvironment, _SESSIONS
from env.schemas import TriageAction
from graders.easy_grader import EasyGrader
from graders.medium_grader import MediumGrader
from graders.hard_grader import HardGrader
from baseline.rule_based_agent import baseline_action

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else random.randint(1, 100000)
print(f"  [Seed: {SEED}]")

graders = {"easy": EasyGrader(), "medium": MediumGrader(), "hard": HardGrader()}

for task_id in ["easy", "medium", "hard"]:
    env = TicketTriageEnvironment()
    sid = f"grade-{task_id}"
    obs = env.reset(task_id=task_id, session_id=sid, seed=SEED)

    print()
    print("=" * 85)
    print(f"  TASK: {task_id.upper()}")
    print("=" * 85)
    header = f'  {"Step":<5} {"Action":<35} {"Reward":>7} {"Score":>7} {"Queue":>6} {"SLA-B":>6} {"Correct":>8}'
    print(header)
    print("-" * 85)

    steps = 0
    while not obs.done:
        action = baseline_action(obs.model_dump())
        obs = env.step(TriageAction(action=action, session_id=sid))
        steps += 1
        reward = getattr(obs, "reward", 0.0) or 0.0
        print(
            f"  {steps:<5} {action:<35} {reward:>+7.3f} {obs.score_estimate:>7.4f}"
            f" {obs.queue_size:>6} {obs.sla_breaches:>6} {obs.correct_assignments:>8}"
        )

    state = _SESSIONS[sid]["state"]
    traj = state.trajectory
    score = graders[task_id].grade(traj)

    print("-" * 85)
    acc = state.correct_assignments / max(
        state.correct_assignments + state.incorrect_assignments, 1
    )
    total_t = len(state.all_tickets)
    sla_comp = 1.0 - state.sla_breaches / max(total_t, 1)
    print(f"  FINAL SCORE:         {score:.4f}")
    print(f"  Tickets assigned:    {state.tickets_assigned}")
    print(f"  Tickets resolved:    {state.tickets_resolved}")
    print(f"  Tickets escalated:   {state.tickets_escalated}")
    print(f"  Tickets deferred:    {state.tickets_deferred}")
    print(f"  Correct / Wrong:     {state.correct_assignments} / {state.incorrect_assignments}")
    print(f"  Accuracy:            {acc:.2%}")
    print(f"  SLA breaches:        {state.sla_breaches}")
    print(f"  SLA compliance:      {sla_comp:.2%}")
    print(f"  VIP breaches:        {state.vip_breaches}")
    print(f"  Invalid actions:     {state.invalid_actions}")
    print(f"  Max wait seen:       {state.max_wait_seen}")
    print(f"  Dept loads:          {dict(state.department_load)}")

print()
print("=" * 85)
print("  FINAL SUMMARY  (baseline rule-based agent)")
print("=" * 85)
for task_id in ["easy", "medium", "hard"]:
    state = _SESSIONS[f"grade-{task_id}"]["state"]
    score = graders[task_id].grade(state.trajectory)
    acc = state.correct_assignments / max(
        state.correct_assignments + state.incorrect_assignments, 1
    )
    print(
        f"  {task_id.upper():<8}  Score: {score:.4f}  |  Accuracy: {acc:.2%}"
        f"  |  Resolved: {state.tickets_resolved}  |  SLA breaches: {state.sla_breaches}"
    )
print("=" * 85)
