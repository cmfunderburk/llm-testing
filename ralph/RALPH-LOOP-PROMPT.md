# Ralph Loop Prompt

Copy the prompt below to start the ralph-loop in a new session.

---

## Pre-flight Checklist

Before starting:
- [ ] Review `PRD-LLM-LEARNING-LAB.json` and make any edits
- [ ] Decide on iteration limit (conservative: 10-20 for first run)
- [ ] Ensure you have time to monitor initial iterations

---

## Prompt

```
@ralph/DESIGN-PRINCIPLES.md @PRD-LLM-LEARNING-LAB.json

Execute the ralph-loop on this PRD.

**Iteration limit**: [INSERT NUMBER, e.g., 15]

**Instructions**:
1. Read the PRD and DESIGN-PRINCIPLES thoroughly before starting
2. Work through phases sequentially (Phase 1 first, etc.)
3. For each feature:
   - Check if entry criteria are met
   - Implement to satisfy all criteria
   - Run verification commands
   - Update feature `passes` to `true` when complete
   - Update phase `status` when all phase features pass
4. Track progress via:
   - Git commits (commit after meaningful progress)
   - Update `progress.txt` with current state
   - Update PRD status fields
5. When a phase completes, STOP and ask before continuing to next phase
6. Prioritize thoroughness over speed — this is a learning project
7. If you hit an escalation trigger, STOP and report

**Do NOT emit the completion promise until ALL features have `passes: true`.**

Begin with Phase 1: Infrastructure & Foundation.
```

---

## Resuming an Interrupted Loop

If the loop was interrupted mid-execution:

```
@ralph/DESIGN-PRINCIPLES.md @PRD-LLM-LEARNING-LAB.json

Resume the ralph-loop on this PRD.

**Iteration limit**: [INSERT NUMBER]

Check the current state:
1. Read `progress.txt` for last known state
2. Check PRD for feature `passes` and phase `status` values
3. Review recent git commits

Continue from where execution stopped. Ask before proceeding to a new phase.
```

---

## Phase Continuation Prompt

After reviewing a completed phase and approving continuation:

```
@PRD-LLM-LEARNING-LAB.json

Continue the ralph-loop to Phase [N]: [Phase Name].

**Iteration limit**: [INSERT NUMBER]

Previous phase verified complete. Proceed with next phase features.
```

---

## Notes

- The iteration limit is the total number of loop cycles, not per-phase
- Each iteration should make meaningful progress on one or more features
- The loop should commit frequently to preserve progress
- If the loop runs out of iterations before completing, it will stop — you can resume later
- The completion promise `<promise>LLM-LEARNING-LAB-COMPLETE</promise>` signals the entire PRD is done
