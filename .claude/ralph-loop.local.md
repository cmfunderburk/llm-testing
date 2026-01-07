---
active: true
iteration: 6
max_iterations: 10
completion_promise: "LLM-LEARNING-LAB-COMPLETE"
started_at: "2026-01-07T16:42:18Z"
---

@ralph/DESIGN-PRINCIPLES.md @PRD-LLM-LEARNING-LAB.json

Execute the ralph-loop on this PRD.

**Instructions**:
1. Read the PRD and DESIGN-PRINCIPLES thoroughly before starting
2. Work through phases sequentially (Phase 1 first, etc.)
3. For each feature:
   - Check if entry criteria are met
   - Implement to satisfy all criteria
   - Run verification commands
   - Update feature passes to true when complete
   - Update phase status when all phase features pass
4. Track progress via:
   - Git commits (commit after meaningful progress)
   - Update progress.txt with current state
   - Update PRD status fields
5. When a phase completes, STOP and ask before continuing to next phase
6. Prioritize thoroughness over speed — this is a learning project
7. If you hit an escalation trigger, STOP and report

**Do NOT emit the completion promise until ALL features have passes: true.**
