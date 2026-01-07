# Ralph Interview Protocol v2

A vision-anchored interview process for transforming project goals into ralph-ready PRDs.

---

## Purpose

This protocol guides an LLM agent through a structured process that translates an existing project vision into a comprehensive PRD (Product Requirements Document) suitable for autonomous ralph-loop execution.

**Critical framing**: The interview does not discover what to build. The vision document tells us what to build. The interview clarifies, operationalizes, and structures that vision into executable specifications.

**When to invoke this protocol:**
- `@ralph/RALPH-INTERVIEW.md` - mention this file to begin
- Before starting any ralph-loop on the project

**What this protocol produces:**
1. Structured extraction of the vision document
2. Gap analysis: vision requirements vs. current codebase state
3. Clarified understanding through targeted questions
4. A comprehensive PRD covering the full project vision
5. Phase decomposition for execution

---

## Prerequisite: Vision Document

This protocol requires a vision document. The agent must locate one of:

1. **VISION.md** - Explicit project vision file
2. **README.md** - If it contains vision/goals/purpose sections
3. **Explicit user statement** - User provides vision verbally if no doc exists

**If no vision document exists and user cannot articulate one:**
STOP. A ralph-loop cannot pursue a goal that doesn't exist. Help the user create a VISION.md first, then restart this protocol.

---

## Interview Phases

### Phase 0: Vision Anchoring

**Objective:** Parse and internalize the vision document before asking any questions.

**Agent actions (autonomous, no questions yet):**

1. Locate the vision document (VISION.md, README.md, or request from user)
2. Read the document completely
3. Extract and present to user in structured format:

```
## Vision Extraction

### Primary Goal
[One sentence: what is this project trying to achieve?]

### Success Criteria
[List: how will we know the project succeeded?]

### Explicit Deliverables
[List: concrete things the vision says should exist]

### Open Questions in Vision
[List: ambiguities, "TBD" items, or unclear specifications]

### Assumptions I'm Making
[List: interpretations that might be wrong]
```

4. Present this extraction and ask: "Does this accurately capture the vision?"

**Phase exit:** User confirms extraction is accurate (or provides corrections).

---

### Phase 1: Gap Analysis

**Objective:** Map the vision to current codebase state and identify what work needs to be done.

**Agent actions (autonomous exploration):**

1. Examine codebase structure relevant to vision goals
2. Identify what already exists vs. what the vision requires
3. Present to user:

```
## Gap Analysis

### Vision Requirement → Current State → Gap

| Requirement | Exists? | Current State | Work Needed |
|-------------|---------|---------------|-------------|
| [from vision] | Yes/Partial/No | [what's there] | [what's missing] |
```

4. Ask clarifying questions ONLY about gaps where:
   - Multiple implementation approaches exist
   - Priority/ordering is unclear
   - Technical constraints affect feasibility

**Questions to ask (examples):**

```
Q1.1: "The vision mentions [X]. I see [Y] exists. Should I extend Y or build X separately?"

Q1.2: "The vision lists multiple goals. For the PRD, should I:
- Cover all goals with phased execution
- Focus on [specific goal] first
- Something else?"

Q1.3: "The vision is silent on [technical decision]. Should I:
- Make a reasonable choice and document as ADR
- Ask you to decide now
- Flag it as pending?"
```

**Phase exit:** Gap analysis presented, clarifying questions resolved.

---

### Phase 2: Clarification Interview

**Objective:** Resolve ambiguities in the vision to produce unambiguous specifications.

**Key principle:** This phase is NOT "what do you want to build?" It IS "help me understand the vision precisely enough to specify it."

**Questions emerge from:**
1. Open questions identified in Phase 0
2. Gaps identified in Phase 1
3. Ambiguities that would cause different implementations

**Question patterns:**

```
Ambiguity resolution:
"The vision says [X]. This could mean [interpretation A] or [interpretation B]. Which is intended?"

Operationalization:
"The vision's success criterion is [abstract goal]. What would be a concrete, verifiable version of this?"

Priority clarification:
"The vision mentions [A, B, C]. If these can't all be done in one phase, what's the priority order?"

Constraint surfacing:
"To achieve [vision goal], I'd typically do [approach]. Are there constraints that would rule this out?"
```

**Do NOT ask:**
- "What do you want to build?" (vision tells us)
- "What type of work is this?" (vision tells us)
- Generic questions not anchored to specific vision content

**Phase exit:** Convergence check - agent summarizes understanding, user confirms.

---

### Phase 3: Verification Design

**Objective:** Establish how we'll know each vision goal is achieved.

**For each goal/deliverable from the vision:**

```
Q3.x: "[Goal] - how should completion be verified?"
- Automated tests (specify what kind)
- Manual verification (specify criteria)
- Benchmarks/metrics (specify thresholds)
- Theory verification (specify equations/properties)
- Human review (specify what reviewer checks)
```

**For research/learning projects (like LLM learning labs):**

```
Q3.x: "The vision goal is [learning outcome]. How should we verify this?"
- Documented experiments with analysis
- Working code that demonstrates understanding
- Ability to explain/predict outcomes
- Comparison against theoretical predictions
```

**Phase exit:** Every vision goal has associated verification criteria.

---

### Phase 4: Scope Confirmation

**Objective:** Confirm the PRD will cover the full vision (default) or document intentional scoping.

**Default assumption:** PRD covers the entire vision, decomposed into phases.

**Only ask if vision is very large:**

```
Q4.1: "The vision encompasses [large scope]. The PRD will cover all of this with phased execution. Confirm this is correct, or specify if you want to scope to a subset."
- Full vision (phased execution)
- Scope to [specific phase/goal] only
- Other
```

**Phase exit:** Scope confirmed.

---

### Phase 5: Loop Parameters

**Objective:** Set execution parameters for the ralph-loop.

```
Q5.1: "What iteration budget is appropriate?"
- Conservative (10-20) - for well-specified phases
- Standard (20-50) - for moderate complexity
- Extended (50-100) - for exploratory or large work
- Specify number

Q5.2: "Progress tracking preferences?"
- Git commits only
- Git commits + progress.txt
- All methods (git + progress + PRD status updates)

Q5.3: "If a phase completes early, should the loop:"
- Stop and await next phase approval
- Continue to next phase automatically
- Ask before continuing
```

**Phase exit:** Parameters set.

---

## Convergence Check

Before generating the PRD, the agent must present a summary and receive confirmation:

```
## Understanding Summary

**Project**: [name from vision]

**Vision in my words**: [1-2 sentence restatement]

**Scope**: [full vision / specific subset]

**Phases I'll define**:
1. [Phase name]: [what it accomplishes]
2. [Phase name]: [what it accomplishes]
...

**Key clarifications received**:
- [Decision 1]
- [Decision 2]

**Verification approach**: [summary]

Does this accurately capture what the PRD should specify?
```

**Only proceed to PRD generation after user confirms.**

---

## PRD Generation

Generate PRD in this JSON format:

```json
{
  "task": {
    "name": "Project name from vision",
    "description": "Full vision description",
    "created": "ISO timestamp",
    "vision_source": "path/to/VISION.md",
    "completion_promise": "<promise>PROJECT-NAME-COMPLETE</promise>"
  },

  "context": {
    "codebase_state": "greenfield | active | maintenance | legacy",
    "primary_language": "language",
    "frameworks": ["list"],
    "vision_extraction": {
      "primary_goal": "one sentence",
      "success_criteria": ["list from vision"],
      "deliverables": ["list from vision"]
    },
    "gap_analysis_summary": "what exists vs what's needed"
  },

  "phases": [
    {
      "phase": 1,
      "name": "Phase name",
      "description": "What this phase accomplishes",
      "features": ["FEAT-001", "FEAT-002"],
      "entry_criteria": "what must be true to start",
      "exit_criteria": "what must be true to complete",
      "estimated_iterations": 20
    }
  ],

  "features": [
    {
      "id": "FEAT-001",
      "name": "Feature name",
      "phase": 1,
      "description": "What this feature does toward vision goals",
      "priority": 1,
      "passes": false,
      "criteria": {
        "functional": ["criterion 1"],
        "property": ["invariant 1"],
        "theory": ["theoretical requirement if applicable"]
      },
      "verification_commands": ["command 1"]
    }
  ],

  "architectural_decisions": {
    "resolved": [
      {
        "decision": "description",
        "choice": "what was decided",
        "rationale": "why",
        "source": "vision | interview | inferred"
      }
    ],
    "pending_adrs": []
  },

  "escalation_triggers": [
    "Trigger 1: condition that should pause the loop"
  ],

  "loop_parameters": {
    "max_iterations": 30,
    "progress_tracking": ["git_commits", "progress_file"],
    "on_phase_complete": "stop | continue | ask"
  }
}
```

---

## Completion Promise

The completion promise signals the ralph-loop should terminate.

**For phased PRDs:** Each phase may have its own intermediate promise.

```
Phase 1 complete: <promise>PROJECT-NAME-PHASE-1-COMPLETE</promise>
Phase 2 complete: <promise>PROJECT-NAME-PHASE-2-COMPLETE</promise>
Full project: <promise>PROJECT-NAME-COMPLETE</promise>
```

**Emit only when:**
- All features in scope have `passes: true`
- All verification commands pass
- No escalation triggers have fired

---

## Interview Checklist

Before concluding, verify:

- [ ] Vision document located and extracted
- [ ] Gap analysis completed
- [ ] All ambiguities from vision clarified
- [ ] Every goal has verification criteria
- [ ] Scope confirmed (full vision or explicit subset)
- [ ] Loop parameters set
- [ ] Convergence check passed
- [ ] PRD generated and presented

---

## Foundational Context

This protocol operationalizes principles from **DESIGN-PRINCIPLES.md**:

- **Completion as contract** - PRD promises are measurable and verifiable
- **Theory-primary verification** - For research projects, theory tests define correctness
- **External memory as primitive** - Progress persists via filesystem and git
- **Predictable failure beats unpredictable success** - Specs make failure informative

If DESIGN-PRINCIPLES.md exists, read it for deeper context.

---

## Usage Instructions for Agents

When this file is mentioned (`@ralph/RALPH-INTERVIEW.md`):

1. **Locate vision document** - VISION.md, README.md, or ask user
2. **Execute Phase 0** - Extract and present vision understanding
3. **Execute Phase 1** - Gap analysis against codebase
4. **Execute Phase 2** - Clarification interview for ambiguities
5. **Execute Phase 3** - Design verification criteria
6. **Execute Phase 4** - Confirm scope (default: full vision)
7. **Execute Phase 5** - Set loop parameters
8. **Convergence check** - Summarize, get confirmation
9. **Generate PRD** - Comprehensive, phased, verifiable
10. **Await approval** - User must confirm PRD before loop execution
