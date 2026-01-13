# Ralph Interview Protocol v5

A vision-anchored interview process for transforming project goals into ralph-ready PRDs and operational artifacts.

---

## Purpose

This protocol guides an LLM agent through a structured process that translates an existing project vision into comprehensive documentation suitable for autonomous ralph-loop execution.

**Critical framing**: The interview does not discover what to build. The vision document tells us what to build. The interview clarifies, operationalizes, and structures that vision into executable specifications.

**When to invoke this protocol:**
- `@ralph/RALPH-INTERVIEW.md` - mention this file to begin
- Before starting any ralph-loop on the project

**What this protocol produces:**
1. Structured extraction of the vision document
2. Project type determination (research software, learning, implementation)
3. Gap analysis: vision requirements vs. current codebase state
4. Resource constraint analysis (if applicable)
5. Clarified understanding through targeted questions
6. ADR drafts for major architectural decisions
7. A comprehensive PRD covering the full project vision
8. Phase decomposition for execution
9. Verification design appropriate to project type
10. Theory divergence tracking template (for theory-grounded work)
11. **Loop startup document** (START-RALPH-LOOP.md) with copy-paste prompts
12. Retrospective template (for learning projects)

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

### Phase 0: Vision Anchoring & Project Type

**Objective:** Parse and internalize the vision document, then determine project type.

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

4. **Determine project type:**

```
## Project Type Assessment

Based on the vision, this appears to be:

[ ] **Research Software** - Theory defines correctness
    - Examples: economic simulations, scientific computing, ML research
    - Verification: Theory tests primary, property tests derived
    - Key artifacts: ADRs with literature references, THEORY-DIVERGENCES.md
    - Canonical sources: [identify textbooks/papers that define correctness]

[ ] **Learning/Exploration Project** - Understanding is the goal
    - Examples: learning labs, paper reproductions, skill-building
    - Verification: Design alignment, behavioral verification, retrospective
    - Key artifacts: RETROSPECTIVE.md, paper analysis documents
    - Success = mental model formation, not just working code

[ ] **Implementation Project** - Features define completeness
    - Examples: standard software, tools, applications
    - Verification: Functional tests, integration tests
    - Key artifacts: Standard PRD, progress tracking
```

5. Present extraction and type assessment, ask: "Does this accurately capture the vision and project type?"

**Phase exit:** User confirms extraction is accurate and project type is correct.

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

### Phase 1.5: Resource Constraint Analysis (Optional)

**Objective:** Identify constraints that affect scope and execution approach.

**When to include this phase:**
- Compute-intensive work (ML training, simulations)
- Time-bounded projects
- Infrastructure dependencies (cloud, GPUs, external services)

**Questions:**

```
Q1.5.1: "What resources are available for this work?"
- Compute: [local specs / cloud budget / none needed]
- Time budget: [hours / days / weeks / open-ended]
- Dependencies requiring setup: [list]

Q1.5.2: "Do resource constraints affect scope?"
- Full scope feasible with current resources
- Scope should be reduced to fit constraints
- Phased approach with resource checkpoints between phases
- Hybrid approach (local for X, cloud burst for Y)
```

**If constraints identified, produce:**

```
## Resource Constraint Analysis

### Available Resources
| Resource | Available | Required | Gap |
|----------|-----------|----------|-----|
| GPU VRAM | 16GB | [estimate] | [none/mild/severe] |
| Time | [budget] | [estimate] | [none/mild/severe] |
| [other] | ... | ... | ... |

### Scope Implications
[How constraints affect what's feasible in each phase]

### Mitigation Strategy
[Cloud burst, phased approach, scope reduction, etc.]
```

**Phase exit:** Resource constraints documented, scope implications understood.

---

### Phase 1.7: Operational Model (Conditional)

**Objective:** Establish how the system will be used in practice, preventing architectural mismatches discovered during implementation.

**When to include this phase:**
- Multi-user or networked systems
- Systems with shared state across components
- Real-time or event-driven applications
- Anything involving concurrency, sessions, or persistence

**Why this phase exists:**
The microecon web frontend assumed single-user operation. Phase 5 (Comparison Mode) discovered the need for per-client simulation instances, requiring architectural changes that could have been prevented by explicit operational model discussion upfront.

**Agent actions:**

1. Based on vision and gap analysis, infer the likely operational model
2. Present assumptions and ask for confirmation/correction:

```
## Operational Model

### Usage Pattern
| Aspect | Assumption | Alternatives |
|--------|------------|--------------|
| Users | Single user | Multiple concurrent users |
| Sessions | Stateless requests | Persistent sessions |
| State sharing | Shared instance | Per-user isolation |
| Execution | Synchronous | Async / event-driven |

### Concurrency Model
- [ ] Single-threaded, single-user (simplest)
- [ ] Single-threaded, request-queued (web server pattern)
- [ ] Multi-threaded with shared state (requires synchronization)
- [ ] Process-per-user isolation (highest isolation, highest overhead)

### State Lifecycle
- **Creation**: [when/how state is initialized]
- **Persistence**: [ephemeral / session / permanent]
- **Reset semantics**: [what "reset" means - clear all? preserve config?]
- **Cleanup**: [when/how state is destroyed]
```

**Questions to ask:**

```
Q1.7.1: "Will multiple users/clients interact with this simultaneously?"
- Single user only
- Multiple users, independent sessions
- Multiple users, shared state (collaboration)
- Multiple users, comparison mode (same inputs, different views)

Q1.7.2: "What happens on 'reset' or 'restart'?"
- Clear all state, fresh start
- Preserve configuration, clear data
- Preserve history, reset to checkpoint
- [Other - specify]

Q1.7.3: "How should state be shared between components?"
- Single source of truth (one component owns state)
- Event-driven synchronization (pub/sub)
- Polling/refresh on demand
- Real-time sync (WebSocket, SSE)
```

**Phase exit:** Operational model documented, concurrency and state semantics explicit.

---

### Phase 1.8: Integration Inventory (Conditional)

**Objective:** Map all data flows between components with explicit type specifications, preventing mid-execution consistency bugs.

**When to include this phase:**
- Multi-component systems (frontend/backend, services)
- Systems integrating with existing codebases
- Migration or porting projects
- Any work crossing module boundaries with shared data structures

**Why this phase exists:**
The microecon project had two `TradeEvent` classes (in simulation.py and logging/events.py) with divergent fields. This was discovered only when the visualization layer tried to display pre/post trade state. Explicit integration inventory would have caught this upfront.

**Agent actions:**

1. Identify all component boundaries the work will cross
2. Map data structures that flow between components
3. Flag inconsistencies or underspecified types
4. Present inventory for validation:

```
## Integration Inventory

### Component Boundaries
| From | To | Transport | Data Format |
|------|-----|-----------|-------------|
| [component A] | [component B] | [HTTP/WS/IPC/import] | [JSON/Protobuf/Python objects] |

### Shared Data Structures

#### [Structure Name] (e.g., TradeEvent)
| Field | Source Component | Consumer Component | Type | Notes |
|-------|------------------|-------------------|------|-------|
| [field_name] | [where defined] | [where used] | [type] | [mismatches?] |

### Identified Inconsistencies
| Issue | Components | Impact | Resolution Needed |
|-------|------------|--------|-------------------|
| [e.g., field missing] | [A, B] | [data loss / runtime error] | [add field / create adapter] |

### Type Specifications Required
For each boundary, explicit contracts:
- [ ] [Component A → B]: [schema or interface definition]
- [ ] [Component B → C]: [schema or interface definition]
```

**Questions to ask:**

```
Q1.8.1: "I found [N] data structures crossing component boundaries.
        Should I document type contracts for all of them, or focus on [subset]?"

Q1.8.2: "There's inconsistency in [structure]: [component A] has [fields],
        [component B] expects [different fields]. Should I:
        - Unify to single source of truth
        - Create explicit adapter/transformer
        - Document as known divergence with rationale"

Q1.8.3: "For [boundary], should type contracts be:
        - Informal (documented in PRD only)
        - Semi-formal (TypeScript interfaces / Python dataclasses)
        - Formal (JSON Schema / Protobuf / OpenAPI spec)"
```

**For migration/porting projects, add:**

```
### Feature Parity Matrix
| Feature | Source System | Target System | Parity Status |
|---------|---------------|---------------|---------------|
| [feature] | ✅ [how implemented] | ⚠️ [partial/missing] | [gap description] |

### Field Mapping
| Source Field | Source Type | Target Field | Target Type | Transform |
|--------------|-------------|--------------|-------------|-----------|
| [name] | [type] | [name] | [type] | [none/conversion/derived] |
```

**Phase exit:** All component boundaries mapped, type contracts specified, inconsistencies flagged for resolution.

---

### Phase 2: Clarification Interview & ADR Identification

**Objective:** Resolve ambiguities in the vision and identify architectural decisions requiring ADRs.

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

**ADR Identification:**

During clarification, identify decisions that warrant Architecture Decision Records:
- Choices between fundamentally different approaches
- Decisions that constrain future work
- Tradeoffs between theoretical purity and practical constraints
- Integration patterns with existing systems

**For each ADR candidate, draft:**

```
## ADR-XXX: [Decision Title]

**Status:** Proposed (awaiting approval)
**Created:** [date]
**Theoretical Basis:** [specific reference if applicable, e.g., "Kreps I Ch 5-7"]

### Context
[Why this decision is needed]

### Options Considered
1. **Option A**: [description]
   - Pro: [advantage]
   - Con: [disadvantage]

2. **Option B**: [description]
   - Pro: [advantage]
   - Con: [disadvantage]

### Recommended Decision
[Which option and why]

### Consequences
[What this decision enables/constrains]

---
*Awaiting approval before implementation proceeds.*
```

**When to create separate ADR files vs inline in PRD:**

| Separate ADR File | Inline in PRD |
|-------------------|---------------|
| Architectural decisions affecting multiple PRDs | Decisions scoped to this PRD only |
| Theory-grounded decisions with literature references | Implementation choices without theoretical basis |
| Decisions that should be discoverable across project | Routine design choices |
| Decisions requiring detailed justification (>1 page) | Brief decisions (few paragraphs) |

**Separate ADR file naming:** `ADR-[TOPIC]-[SHORT-TITLE].md` (e.g., `ADR-BELIEF-ARCHITECTURE.md`)

**Inline ADR format:** Use the `architectural_decisions.resolved` array in the PRD JSON, with abbreviated fields:
```json
{
  "id": "ADR-XXX-001",
  "decision": "brief description",
  "choice": "what was decided",
  "rationale": "why",
  "source": "vision | interview | inferred"
}
```

For research software projects, major theory-grounded decisions should almost always be separate files to capture the full theoretical justification and literature references.

**Do NOT ask:**
- "What do you want to build?" (vision tells us)
- "What type of work is this?" (Phase 0 determined this)
- Generic questions not anchored to specific vision content

**Phase exit:**
- Convergence check - agent summarizes understanding, user confirms
- ADR drafts presented for decisions requiring them

---

### Phase 2.5: Design Sketch (UI/Visualization Projects Only)

**Objective:** Validate visual/interaction design before committing to implementation.

**When to include this phase:**
- The PRD includes significant UI or visualization work
- The design involves novel visual encodings or interaction patterns
- Previous implementations have required mid-execution redesign

**Why this phase exists:**
UI/visualization work often reveals design problems only when implemented. The session log showed a trade network overlay that "didn't work well" on the simulation grid, requiring a complete redesign as a separate panel. Early design validation prevents wasted implementation effort.

**Agent actions:**

1. For each major visual/UI feature, present a design sketch:

```
## Design Sketch: [Feature Name]

### Visual Concept
[ASCII art or description of what it will look like]

### Information Encoded
| Visual Property | Data Property | Why This Encoding |
|-----------------|---------------|-------------------|
| Color | [what] | [rationale] |
| Position | [what] | [rationale] |
| Size | [what] | [rationale] |

### Interaction Model
- Click: [behavior]
- Hover: [behavior]
- Drag: [behavior if any]

### Potential Issues
- [Issue 1]: [why it might not work]
- [Issue 2]: [why it might not work]

### Alternatives Considered
1. [Alternative A]: [why rejected]
2. [Alternative B]: [why rejected]
```

2. Ask user to validate design before proceeding:
   - "Does this design approach make sense?"
   - "Are there obvious issues I'm missing?"
   - "Should I prototype this before committing to full PRD?"

**Design validation options:**
- **Approve** - Proceed to PRD generation with this design
- **Revise** - Iterate on the design sketch
- **Prototype first** - Create a minimal implementation spike before PRD
- **Split** - Design is too uncertain; create separate discovery PRD first

**Phase exit:** User approves design sketches or authorizes proceeding despite uncertainty.

---

### Phase 3: Verification Design

**Objective:** Establish how we'll know each vision goal is achieved, appropriate to project type.

#### For Research Software Projects

**Verification Hierarchy** (per DESIGN-PRINCIPLES.md):

```
Theory Verification (Primary)
    ↓ defines what "correct" means
Property Verification (Derived)
    ↓ operationalizes theoretical invariants
Functional Verification (Instrumental)
    ↓ ensures the machinery runs
```

**For each goal/deliverable:**

```
Q3.x: "[Goal] - what theory defines correctness?"

Theory verification:
- Canonical source: [textbook/paper reference]
- Testable prediction: [specific equation or property]
- Tolerance: [strict (1e-6) for formulas, moderate (1e-3) for optimization]

Property verification (derived from theory):
- [Invariant that must hold]

Functional verification:
- [Basic "it runs" criteria]
```

#### For Learning/Exploration Projects

**Verification Hierarchy:**

```
Design Alignment (Primary)
    ↓ embodies learning science / research principles
Behavioral Verification (Derived)
    ↓ system behaves as design requires
Outcome Measurement (Ultimate but Deferred)
    ↓ requires longitudinal data
```

**For each learning goal:**

```
Q3.x: "[Learning goal] - how should understanding be verified?"

Design alignment:
- Principle embodied: [learning science / research principle]
- Observable indicator: [what demonstrates alignment]

Behavioral verification:
- [Specific measurable behavior]

Self-assessment criteria:
- Can explain [concept] without referencing materials
- Can predict [outcome] before running experiment
- Can identify [pattern] in novel examples
```

**Additionally for learning projects:**

```
Q3.x: "What retrospective questions should be answered at completion?"
- What mental model was formed?
- What worked in the learning approach?
- What gaps remain?
- What questions emerged for future exploration?
```

#### For Implementation Projects

**Standard verification:**

```
Q3.x: "[Goal] - how should completion be verified?"
- Automated tests (specify what kind)
- Manual verification (specify criteria)
- Integration tests (specify scenarios)
- Performance benchmarks (specify thresholds)
```

**Phase exit:** Every vision goal has appropriate verification criteria for project type.

---

### Phase 3.2: Testability Planning (Conditional)

**Objective:** Identify what cannot be manually tested and establish automation requirements before implementation begins.

**When to include this phase:**
- Systems with concurrency, race conditions, or timing-dependent behavior
- Features requiring load/stress testing
- Output formats requiring automated validation (exports, generated files)
- State consistency that can't be visually verified
- Any feature where "it looks right" is insufficient verification

**Why this phase exists:**
The microecon web frontend assumed visual testing was sufficient. WebSocket race conditions, export file validation, and state consistency issues were discovered post-implementation because no testability planning identified them as requiring automation.

**Agent actions:**

1. Review all features and verification criteria from Phase 3
2. For each feature, classify testability:

```
## Testability Analysis

### Manual Testing Sufficient
| Feature | Manual Test | Why Sufficient |
|---------|-------------|----------------|
| [feature] | [test description] | [visual output / simple interaction / etc.] |

### Automation Required
| Feature | Why Manual Insufficient | Automation Type | Priority |
|---------|------------------------|-----------------|----------|
| [feature] | [race condition / scale / precision] | [unit / integration / load / property] | [P0-P2] |

### Untestable Without Infrastructure
| Feature | Missing Infrastructure | Mitigation |
|---------|----------------------|------------|
| [feature] | [GPU / external service / etc.] | [mock / defer / conditional] |
```

3. For automation-required features, specify what the test must verify:

```
### Automation Specifications

#### [Feature]: [Test Name]
**Type:** [unit / integration / load / property / visual regression]
**Verifies:** [specific behavior or invariant]
**Inputs:** [test data / scenarios]
**Expected output:** [success criteria]
**Tolerance:** [exact match / threshold / statistical]
```

**Questions to ask:**

```
Q3.2.1: "These features have concurrency or timing concerns: [list].
        Should I specify:
        - Load tests with specific concurrency levels
        - Race condition property tests
        - Timeout/retry behavior tests
        - Manual testing is acceptable for MVP"

Q3.2.2: "These features produce file outputs: [list].
        Validation approach:
        - Checksum/hash comparison against golden files
        - Structural validation (parse and verify schema)
        - Content spot-checks only
        - Manual inspection sufficient"

Q3.2.3: "These features have state consistency requirements: [list].
        Should state assertions be:
        - Inline in implementation (debug mode)
        - Separate property tests
        - Manual verification via UI inspection"
```

**Phase exit:** Features classified by testability, automation requirements specified for non-manual-testable features.

---

### Phase 3.5: Divergence Tracking Setup (Research Software Only)

**Objective:** Establish tracking for discovered divergences between implementation and theory.

**When to include:** Research software projects where theory defines correctness.

**Create THEORY-DIVERGENCES.md template:**

```markdown
# Theory Divergences Report

This document records divergences discovered between implementation and canonical theory during development.

## Summary

| ID | Theory | Status | Resolution |
|----|--------|--------|------------|
| DIV-001 | [description] | Pending/Fixed/Accepted | [approach] |

---

## Template for New Divergences

### DIV-XXX: [Title]

**Discovery Date:** [date]
**Test:** [test file::test_method]
**Canonical Source:** [textbook/paper reference]

#### Issue
[What was expected vs. what was observed]

#### Root Cause
[Why the divergence occurred]

#### Resolution
[How it was fixed, or why it's acceptable]

#### Theoretical Justification
[Why the resolution is consistent with theory]

---
```

**Divergence Decision Tree:**

When a divergence is discovered, use this decision tree to determine resolution:

```
## Divergence Resolution Decision Tree

┌─────────────────────────────────────────────────────────────┐
│ DIVERGENCE DISCOVERED: Implementation ≠ Theory              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │ Is the theory source correct? │
              │ (check citation, edition,     │
              │  transcription errors)        │
              └───────────────────────────────┘
                     │              │
                    YES            NO → Fix theory reference, re-verify
                     │
                     ▼
              ┌───────────────────────────────┐
              │ Is the implementation wrong?  │
              │ (bug, formula error, sign     │
              │  flip, off-by-one)            │
              └───────────────────────────────┘
                     │              │
                    YES            NO
                     │              │
                     ▼              ▼
              ┌──────────┐  ┌───────────────────────────┐
              │ FIX BUG  │  │ Is divergence due to      │
              │ Status:  │  │ acceptable approximation? │
              │ Fixed    │  │ (numerical precision,     │
              └──────────┘  │  convergence tolerance)   │
                            └───────────────────────────┘
                                   │              │
                                  YES            NO
                                   │              │
                                   ▼              ▼
                            ┌──────────┐  ┌───────────────────────────┐
                            │ DOCUMENT │  │ Is divergence due to      │
                            │ tolerance│  │ intentional simplification│
                            │ Status:  │  │ documented in ADR?        │
                            │ Accepted │  └───────────────────────────┘
                            └──────────┘         │              │
                                                YES            NO
                                                 │              │
                                                 ▼              ▼
                                          ┌──────────┐  ┌──────────────┐
                                          │ REFERENCE│  │ ESCALATE     │
                                          │ ADR in   │  │ Create ADR or│
                                          │ resolution│ │ fix impl     │
                                          │ Status:  │  │ Status:      │
                                          │ Accepted │  │ Pending      │
                                          └──────────┘  └──────────────┘

## Resolution Categories

| Status | Meaning | Required Documentation |
|--------|---------|----------------------|
| **Fixed** | Bug corrected, now matches theory | Root cause, fix description |
| **Accepted (Tolerance)** | Numerical precision acceptable | Tolerance threshold, justification |
| **Accepted (ADR)** | Intentional deviation per design decision | Reference to ADR |
| **Pending** | Requires decision before proceeding | Escalation to user |

## Tolerance Guidelines

| Context | Acceptable Tolerance | Example |
|---------|---------------------|---------|
| Closed-form formulas | 1e-10 to 1e-6 | Nash bargaining solution |
| Numerical optimization | 1e-6 to 1e-3 | Utility maximization |
| Iterative algorithms | 1e-3 to 1e-2 | Convergence to equilibrium |
| Statistical properties | Problem-dependent | Distribution comparisons |

## Red Flags (Always Escalate)

- Sign errors (positive should be negative or vice versa)
- Order of magnitude errors (off by 10x or more)
- Qualitative differences (monotonic should be non-monotonic)
- Boundary condition failures (edge cases violate theory)
```

**Questions to ask:**

```
Q3.5.1: "What tolerance thresholds are appropriate for this project?
        - Strict (1e-10): Mathematical identities, closed-form solutions
        - Standard (1e-6): Numerical methods, optimization results
        - Relaxed (1e-3): Iterative convergence, Monte Carlo
        - Custom: [specify per-feature tolerances]"

Q3.5.2: "Should divergence discovery during execution:
        - Always stop the loop (conservative)
        - Stop only for red-flag divergences (balanced)
        - Document and continue, review at phase end (aggressive)"
```

**Phase exit:** THEORY-DIVERGENCES.md template created, decision tree reviewed, tolerance thresholds established, escalation behavior set.

---

### Phase 4: Scope Confirmation

**Objective:** Confirm the PRD will cover the full vision (default) or document intentional scoping.

**Default assumption:** PRD covers the entire vision, decomposed into phases.

**Only ask if vision is very large or resources are constrained:**

```
Q4.1: "The vision encompasses [large scope]. Given [resource constraints if any], the PRD will cover:
- Full vision (phased execution)
- Scope to [specific phase/goal] only
- Scope adjusted per resource analysis
- Other"
```

**Phase exit:** Scope confirmed, aligned with resource constraints if applicable.

---

### Phase 4.5: Phase Dependency Mapping

**Objective:** Establish explicit dependency graph between phases to prevent silent assumption failures during execution.

**Why this phase exists:**
Multi-phase PRDs in the microecon project ran sequentially without verifying that earlier phases were sufficient for later ones. Phase 2 needed toggle infrastructure from Phase 1, but no gate verified it existed before Phase 2 began. Explicit dependency mapping prevents cascading failures.

**Agent actions:**

1. Review all phases defined so far
2. For each phase, identify:
   - What must exist before it can start (depends_on)
   - What it enables for later phases (gates_phases)
   - Specific verification that predecessor phases are sufficient (gate_checks)
3. Present dependency graph for validation:

```
## Phase Dependencies

### Dependency Matrix
| Phase | Depends On | Enables | Gate Check |
|-------|------------|---------|------------|
| 1 | (none) | 2, 4 | - |
| 2 | 1 | 5 | Phase 1 toggle infrastructure renders |
| 3 | (none) | (none) | - |
| 4 | 1, 2 | 5, 6 | Phase 1 rendering + Phase 2 data access |
| 5 | 2, 4 | (none) | Phase 2 beliefs + Phase 4 overlays |

### Critical Path
[Longest dependency chain - determines minimum sequential work]
Example: 1 → 2 → 4 → 5 (4 phases minimum)

### Parallel Opportunities
[Phases with no dependencies between them]
Example: Phases 1 and 3 can run in parallel

### Dependency Visualization

    ┌───┐
    │ 1 │ Infrastructure
    └─┬─┘
      │
  ┌───┴───┐
  ▼       ▼
┌───┐   ┌───┐
│ 2 │   │ 3 │  (parallel eligible)
└─┬─┘   └───┘
  │
  ▼
┌───┐
│ 4 │ ← also depends on 1
└─┬─┘
  │
  ▼
┌───┐
│ 5 │ ← depends on 2 and 4
└───┘
```

**Questions to ask:**

```
Q4.5.1: "Phase [N] depends on infrastructure from Phase [M].
        What specific check confirms Phase [M] is sufficient?
        - [Suggested gate check based on features]
        - Custom verification
        - No explicit gate needed (exit_criteria sufficient)"

Q4.5.2: "Phases [A] and [B] appear independent. Should they:
        - Run sequentially anyway (reduces complexity)
        - Be flagged as parallel-eligible (HOTL can parallelize)
        - Have explicit ordering preference documented"
```

**For simple PRDs (≤3 phases, linear dependencies):**
Skip the full DAG and just note: "Linear execution: 1 → 2 → 3, no parallel opportunities."

**Phase exit:** Dependency DAG complete, gate checks specified for non-trivial dependencies, critical path identified.

---

### Phase 5: Loop Parameters

**Objective:** Set execution parameters for the ralph-loop.

**Critical clarification on iterations:**
The `--max-iterations` parameter sets the budget for the **entire** ralph-loop invocation, not per-phase. The loop should complete the **entire PRD** within the iteration budget, not just one phase. Do not specify "iterations per phase" - phases are logical groupings, not separate loop invocations.

- A loop with `--max-iterations 50` should complete all phases if possible
- The loop terminates when the completion promise is emitted OR iterations exhausted
- If interrupted mid-PRD, use a Resume Prompt to continue with fresh iterations

```
Q5.1: "What iteration budget is appropriate for the ENTIRE PRD?"
- Conservative (20-40) - for small PRDs (5-10 features)
- Standard (40-80) - for medium PRDs (10-20 features)
- Extended (80-150) - for large PRDs (20+ features) or exploratory work
- Specify number

Q5.2: "Progress tracking preferences?"
- Git commits only
- Git commits + progress.txt
- All methods (git + progress + PRD status updates)

Q5.3: "Phase transition behavior - when a phase completes, should the loop:"
- **continue** - Proceed to next phase automatically (recommended for HOTL mode)
- **ask** - Use AskUserQuestion to request approval before next phase
- **stop** - Emit phase-specific promise, require new loop invocation for next phase (recommended for AFK mode)

See "HOTL vs AFK Mode Considerations" section below for guidance.

Q5.4: "Escalation triggers - loop should STOP if:"
- [Default: changes to core architecture without ADR]
- [Default: test failures that suggest theory divergence]
- [Custom triggers based on project]
```

**Phase exit:** Parameters set.

---

### Phase 6: Loop Startup Document Creation

**Objective:** Produce copy-paste ready prompts for executing the ralph-loop.

**Generate START-RALPH-LOOP.md:**

```markdown
# Ralph Loop Startup: [PRD-NAME]

Copy and paste the command below to start the ralph-loop.

---

## Pre-flight Checklist

Before starting:
- [ ] All existing tests pass: `[test command]`
- [ ] On correct branch: `git branch`
- [ ] Working directory clean: `git status`
- [ ] PRD reviewed and any edits made
- [ ] ADRs approved (if applicable)

---

## Startup Prompt

```
/ralph-loop:ralph-loop "Execute [PRD-NAME].json systematically.

[PHASE-SPECIFIC INSTRUCTIONS - customized per PRD]

CONSTRAINTS:
- [Phase ordering constraints]
- [Escalation triggers]
- [Approval requirements between phases]

Track progress via git commits, update PRD status fields as features complete." --max-iterations [N] --completion-promise "[PROMISE]"
```

### Completion Promise Escaping

**Important:** The completion promise must be properly escaped for shell execution. Angle brackets (`<` and `>`) can trigger shell interpretation issues.

**Safe patterns:**
```bash
# Option 1: Use quotes (recommended)
--completion-promise "PROJECT-NAME-COMPLETE"

# Option 2: Escape angle brackets if using XML-style tags
--completion-promise "\<promise\>PROJECT-NAME-COMPLETE\</promise\>"

# Option 3: Use alternative delimiters
--completion-promise "[[PROMISE:PROJECT-NAME-COMPLETE]]"
```

**Avoid:**
```bash
# This may fail - unescaped angle brackets
--completion-promise "<promise>PROJECT-NAME-COMPLETE</promise>"
```

**Recommendation:** Use simple alphanumeric completion promises without special characters (e.g., `PROJECT-NAME-COMPLETE`) unless XML-style tags are specifically required.

---

## Resume Prompt (for interrupted loops)

```
/ralph-loop:ralph-loop "Resume [PRD-NAME].json execution.

Check current state:
1. Read progress.txt for last known state
2. Check PRD for feature passes and phase status values
3. Review recent git commits

Continue from where execution stopped. Ask before proceeding to a new phase." --max-iterations [N] --completion-promise "[PROMISE]"
```

---

## Phase Continuation Prompt

After reviewing a completed phase and approving continuation:

```
/ralph-loop:ralph-loop "Continue [PRD-NAME].json to Phase [N]: [Phase Name].

Previous phase verified complete. Proceed with next phase features." --max-iterations [N] --completion-promise "[PROMISE]"
```

---

## Key Files

| File | Purpose |
|------|---------|
| `[PRD-NAME].json` | Full PRD specification |
| `VISION.md` | Authoritative project vision |
| `[ADR files]` | Architectural decisions |
| `[THEORY-DIVERGENCES.md]` | Theory divergence tracking (if applicable) |
| `[other relevant docs]` | [purposes] |

---

## Expected Outputs

**Phase [N]:**
- [Expected files/artifacts]
- [Expected test files]
- [Expected documentation updates]

[Repeat for each phase]

---

## Progress Tracking Template

Create `progress.txt` with this structure for append-only progress logging:

```markdown
# Progress Log: [PRD-NAME]

## Format
Each entry follows: [timestamp] [phase] [status] [description]

---

## Log

### Phase 1: [Phase Name]

#### Completed
- [timestamp] FEAT-001: [feature name] - [brief description of what was done]
- [timestamp] FEAT-002: [feature name] - [brief description]

#### Issues Encountered
- [timestamp] ISSUE: [description of problem]
  - Root cause: [what caused it]
  - Resolution: [how it was fixed]
  - Time impact: [minimal/moderate/significant]

#### Workarounds Applied
- [timestamp] WORKAROUND: [what was done differently than planned]
  - Reason: [why the workaround was needed]
  - Technical debt: [yes/no - needs revisiting?]

#### Skipped/Deferred
- [timestamp] SKIPPED: FEAT-003 - [reason for skipping]
  - Blocker: [what blocks this]
  - Revisit: [condition for revisiting]

---

### Phase 2: [Phase Name]
[same structure]

---

## Summary Statistics
| Metric | Count |
|--------|-------|
| Features completed | X |
| Features skipped | X |
| Issues encountered | X |
| Workarounds applied | X |
| Phases completed | X/Y |
```

**Why issues tracking matters:**
The microecon progress.txt only recorded successes. Bugs, workarounds, and skipped features weren't documented, making it impossible to learn from problems or understand why certain features were deferred. The issues section creates institutional memory.
```

**Phase exit:** START-RALPH-LOOP.md created and reviewed.

---

## Convergence Check

Before generating the PRD, the agent must present a summary and receive confirmation:

```
## Understanding Summary

**Project**: [name from vision]
**Project Type**: [Research Software / Learning / Implementation]

**Vision in my words**: [1-2 sentence restatement]

**Scope**: [full vision / specific subset]
**Resource Constraints**: [none / summary of constraints]

**Phases I'll define**:
1. [Phase name]: [what it accomplishes]
2. [Phase name]: [what it accomplishes]
...

**Key clarifications received**:
- [Decision 1]
- [Decision 2]

**ADRs to include**: [list or "none"]

**Verification approach**: [summary appropriate to project type]

**Divergence tracking**: [Yes with template / Not applicable]

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
    "project_type": "research_software | learning | implementation",
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
    "gap_analysis_summary": "what exists vs what's needed",
    "resource_constraints": {
      "compute": "description or null",
      "time_budget": "description or null",
      "other": "description or null"
    },
    "canonical_sources": ["textbook/paper references for research software"]
  },

  "phases": [
    {
      "phase": 1,
      "name": "Phase name",
      "description": "What this phase accomplishes",
      "features": ["FEAT-001", "FEAT-002"],
      "depends_on": [],
      "gates_phases": [2],
      "entry_criteria": "what must be true to start",
      "exit_criteria": "what must be true to complete",
      "gate_verification": "specific check that this phase is sufficient for dependents"
    }
  ],

  "phase_dependencies": {
    "dag": {
      "1": [],
      "2": [1],
      "3": [1, 2]
    },
    "critical_path": [1, 2, 3],
    "parallel_eligible": [],
    "gate_checks": {
      "1→2": "Phase 1 infrastructure verified: [specific check]",
      "2→3": "Phase 2 features verified: [specific check]"
    }
  },

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
        "id": "ADR-001",
        "decision": "description",
        "choice": "what was decided",
        "rationale": "why",
        "source": "vision | interview | inferred",
        "canonical_reference": "textbook/paper if applicable"
      }
    ],
    "pending_adrs": []
  },

  "tracking": {
    "theory_divergences_file": "path or null",
    "retrospective_template": "path or null (for learning projects)"
  },

  "escalation_triggers": [
    "Trigger 1: condition that should pause the loop"
  ],

  "loop_parameters": {
    "max_iterations": null,
    "progress_tracking": ["git_commits", "progress_file"],
    "on_phase_complete": "stop | continue | ask"
  }
  // Note: max_iterations is null in PRD because it's set at ralph-loop invocation time
  // via --max-iterations flag. The PRD records the user's preference but doesn't
  // constrain the actual invocation.
}
```

---

## Additional Artifacts by Project Type

### For Research Software Projects

**THEORY-DIVERGENCES.md** - Created in Phase 3.5

**ADR files** - One per major architectural decision, format from Phase 2

### For Learning/Exploration Projects

**RETROSPECTIVE-TEMPLATE.md:**

```markdown
# [Project Name]: Retrospective

## Self-Assessment Against Original Success Criteria

| Criterion | Assessment | Evidence |
|-----------|------------|----------|
| [from vision] | Achieved / Partial / Gap | [specific evidence] |

---

## Track-by-Track Analysis

### [Track/Phase Name]

**What was built:**
- [list of artifacts, code, documents]

**Key learnings:**
1. [learning 1]
2. [learning 2]

**What worked:**
- [effective approach 1]

**What didn't work:**
- [ineffective approach 1]

---

## Mental Model Formed

### Before
[What you understood before starting]

### After
[What you understand now]

### Visual representation (if applicable)
```
[diagram or structured representation of mental model]
```

---

## Open Questions for Future Exploration

### High Priority
1. [question that blocks next work]

### Medium Priority
1. [question worth investigating]

### Lower Priority
1. [question for eventual exploration]

---

## Next Steps

### Immediate
1. [next action]

### Short-term
1. [near-future work]

### Long-term
1. [eventual direction]
```

**Paper Analysis Template** (if vision involves paper reproduction):

```markdown
# [Paper Title]

**Paper**: [arXiv/DOI link]
**Authors**: [names]
**Date**: [publication date]

## Reading Notes

### The Central Question
[What problem does this paper address?]

### The Key Innovation
[What's novel about the approach?]

### Core Claims (numbered for reference)
1. [Claim 1]
2. [Claim 2]
...

### My Interpretation
[Understanding in own words]

---

## Claims Identified for Reproduction

### High Priority (feasible with current setup)
1. **[Claim reference]**: [description]
   - Method: [how to test]
   - Connection: [how it relates to our work]

### Medium Priority (requires more setup)
1. ...

### Lower Priority (may exceed scope)
1. ...

---

## Questions This Paper Raises
1. [question 1]
2. [question 2]

## Connection to Our Work
| Our Component | Paper's Claim | Connection |
|---------------|---------------|------------|
| [component] | [claim] | [how they relate] |

---

## Sources
- [link to paper]
- [link to supplementary materials]
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
- Theory divergences resolved or documented as acceptable (research software)

---

## Interview Checklist

Before concluding, verify:

### Core Deliverables
- [ ] Vision document located and extracted
- [ ] Project type determined and confirmed
- [ ] Gap analysis completed
- [ ] All ambiguities from vision clarified
- [ ] Design sketches validated (UI/visualization projects)
- [ ] Every goal has verification criteria appropriate to project type
- [ ] Scope confirmed (full vision or explicit subset)
- [ ] Loop parameters set (including HOTL vs AFK mode)
- [ ] Convergence check passed
- [ ] PRD generated and presented

### Operational Artifacts
- [ ] START-RALPH-LOOP.md created with copy-paste prompts
- [ ] Pre-flight checklist included
- [ ] Resume and continuation prompts included

### Project-Type Specific
- [ ] **Research Software**: ADR drafts for major decisions
- [ ] **Research Software**: THEORY-DIVERGENCES.md template created
- [ ] **Research Software**: Canonical sources identified in PRD
- [ ] **Learning Projects**: RETROSPECTIVE-TEMPLATE.md created
- [ ] **Learning Projects**: Paper analysis template (if applicable)

### Conditional Phases (if applicable)
- [ ] Resource constraint analysis completed (Phase 1.5)
- [ ] Operational model documented (Phase 1.7) - for multi-user/networked systems
- [ ] Integration inventory mapped (Phase 1.8) - for multi-component/migration projects
- [ ] Testability analysis completed (Phase 3.2) - for features needing automation
- [ ] Divergence decision tree reviewed, tolerances set (Phase 3.5) - for research software
- [ ] Phase dependencies mapped (Phase 4.5) - for multi-phase PRDs
- [ ] Scope adjusted for constraints

---

## HOTL vs AFK Mode Considerations

The interview should establish which execution mode the user intends, as this affects phase transition behavior and iteration budgets.

### HOTL (Human On The Loop)

User is actively monitoring the session and can intervene.

**Recommended settings:**
- `on_phase_complete: "continue"` - User can interrupt if needed
- Moderate iteration budgets - can always restart if exhausted
- More aggressive escalation triggers - user can resolve quickly
- AskUserQuestion for design decisions as they arise

**Best for:**
- Exploratory work where direction may shift
- UI/visualization work requiring visual feedback
- First-time PRD execution on a codebase
- Complex multi-phase work where inter-phase review adds value

### AFK (Away From Keyboard)

User will leave the loop running unattended (overnight, etc.).

**Recommended settings:**
- `on_phase_complete: "stop"` or `"ask"` - Creates natural checkpoints
- Generous iteration budgets with safety margin
- Conservative escalation triggers - prefer false stops over runaway loops
- All decisions pre-resolved in ADRs before loop starts

**Best for:**
- Well-specified implementation tasks
- Repetitive work (test coverage, documentation)
- Tasks where partial progress is still valuable

### Mixed-Mode Strategy

For multi-phase PRDs, consider:
1. **HOTL for Phase 1** - Establish patterns, catch early issues
2. **AFK for middle phases** - Execute established patterns overnight
3. **HOTL for final phase** - Polish and integration review

The START-RALPH-LOOP document should include prompts for both modes:
- Full-PRD prompt for HOTL (expects to complete everything)
- Per-phase prompts for AFK (natural stopping points)

---

## Emergent PRD Pattern

Sometimes during PRD execution, issues are discovered that require a fundamentally different approach. Rather than forcing the original PRD to accommodate the change, it may be appropriate to spawn a new PRD.

### When to Spawn a New PRD

**Spawn a new PRD when:**
- The original design assumption was fundamentally flawed (e.g., "grid overlay for trade network" → "dedicated panel needed")
- The scope expansion would more than double the original PRD
- The new work has different success criteria than the original
- The new work could be useful independent of the original PRD

**Do NOT spawn a new PRD when:**
- It's just a bug fix or design refinement
- The scope creep is modest and related to original goals
- Splitting would create artificial boundaries

### Process for Mid-Execution PRD Spawning

1. **Complete or checkpoint the current PRD** - Don't leave it in a broken state
2. **Document the discovery** - What was learned? Why is the original approach inadequate?
3. **Invoke the interview protocol** - Use this protocol (RALPH-INTERVIEW.md) to create the new PRD
4. **Cross-reference** - Both PRDs should reference each other in their context sections
5. **Decide execution order** - Does the new PRD block the original? Can they proceed in parallel?

### Example: Trade Network Panel

From the session log:
```
Original PRD: VIZ-018 "Trade Network Overlay" - draw lines on simulation grid
Problem discovered: Lines connect agents at current positions, not trade locations
                   Makes overlay meaningless as agents move
Solution: Spawn new PRD for dedicated Trade Network Panel with proper graph layout
```

The new PRD (Trade Network Panel) was created using this interview protocol, producing:
- Gap analysis showing existing code that could be reused
- Design decisions captured as ADRs (dockable window, both layouts, encoding scheme)
- Properly scoped features (11 features across 4 phases)
- START-RALPH-LOOP document for execution

### Updating the Original PRD

When spawning a new PRD obsoletes features in the original:
- Mark obsolete features as `"passes": true` with note: `"superseded by [NEW-PRD]"`
- Or remove them and update phase structure
- Document the change in the original PRD's context or a changelog field

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
2. **Execute Phase 0** - Extract vision AND determine project type
3. **Execute Phase 1** - Gap analysis against codebase
4. **Execute Phase 1.5** - Resource constraint analysis (if applicable)
5. **Execute Phase 1.7** - Operational model (if multi-user/networked/stateful)
6. **Execute Phase 1.8** - Integration inventory (if multi-component/migration)
7. **Execute Phase 2** - Clarification interview AND ADR identification
8. **Execute Phase 2.5** - Design sketch validation (UI/visualization projects only)
9. **Execute Phase 3** - Design verification criteria (per project type)
10. **Execute Phase 3.2** - Testability planning (if features need automation)
11. **Execute Phase 3.5** - Divergence tracking setup with decision tree (research software only)
12. **Execute Phase 4** - Confirm scope (default: full vision)
13. **Execute Phase 4.5** - Phase dependency mapping (if multi-phase PRD)
14. **Execute Phase 5** - Set loop parameters (including HOTL vs AFK mode)
15. **Convergence check** - Summarize, get confirmation
16. **Generate PRD** - Comprehensive, phased, verifiable, with dependency DAG
17. **Execute Phase 6** - Create START-RALPH-LOOP.md with progress template
18. **Generate additional artifacts** - Per project type (retrospective template, paper analysis)
19. **Await approval** - User must confirm all artifacts before loop execution

---

## Appendix: Example Interview Snippets

These examples from actual sessions demonstrate effective interview patterns.

### Good Gap Analysis Presentation

```
## Gap Analysis

### Vision Requirement → Current State → Gap

| Requirement | Exists? | Current State | Work Needed |
|-------------|---------|---------------|-------------|
| Trade network data structure | Yes | analysis/emergence.py: TradeEdge, build_trade_network() | Can reuse for panel |
| Network statistics | Yes | trade_network_stats(): density, clustering, degrees | Can reuse for panel |
| Network visualization | Partial | render_trade_network() draws lines on grid (doesn't work well) | Need dedicated panel with graph layout |
| Left-side panel | No | Only right-side metrics panel exists | Need new panel infrastructure |
| Graph layout algorithm | No | No layout code exists | Need force-directed or circular layout |
```

Note: The table clearly shows what exists vs. what's needed, making the scope of work visible.

### Good Clarifying Questions

```
Q1.1: Panel location and sizing
The vision mentions a left-side panel. Should it be:
- Same width as right metrics panel (~300px)?
- Resizable/collapsible?
- Or a separate dockable window?

Q1.2: Layout algorithm preference
For the network graph layout:
- Force-directed - Naturally clusters connected agents, more organic
- Circular - All agents on a ring, edges cross through center, cleaner
- Both - Toggle between layouts?

Q1.3: Node encoding
What should node appearance encode?
- Color = alpha (same as grid view) - maintains visual consistency
- Size = trade count (total trades) or degree (unique partners)?
```

Note: Questions are specific, provide options, and explain tradeoffs.

### Good User Responses

```
User: 1.1 - separate dockable window
       1.2 - both, toggle
       1.3 - color = alpha, let's not resize them as that may be overly cluttered
       1.4 - thickness = trade count, color = recency
```

Note: Concise, decisive, addresses all questions. This is ideal input for the interview.

### Good Convergence Check

```
## Understanding Summary

**Project**: Trade Network Panel
**Project Type**: Implementation

**Vision in my words**: Create a dedicated dockable window that visualizes
trade relationships as a network graph, replacing the ineffective grid-overlay
approach. The panel shows who traded with whom using force-directed or circular
layouts, with nodes colored by alpha and edges encoding frequency (thickness)
and recency (color).

**Scope**: Full feature as discussed

**Phases I'll define**:
1. Window Infrastructure: Dockable window, menu/button to open, basic layout
2. Graph Rendering: Node drawing, edge drawing, layout algorithms
3. Visual Encoding: Alpha colors, edge thickness/color, layout toggle
4. Interaction & Integration: Click-to-select, live/replay modes, metrics display

**Key clarifications received**:
- Separate dockable window (not embedded panel)
- Both layout algorithms with toggle
- Node color = alpha, fixed size
- Edge thickness = trade count, color = recency

**ADRs to include**: ADR-NET-001 (window architecture)

**Verification approach**: Functional testing, visual inspection

Does this accurately capture what the PRD should specify?
```

Note: Restates understanding in agent's own words, lists all decisions, asks for confirmation.

### Good Phase Transition Request

```
Phase 5 is complete. Let me ask for review before proceeding to Phase 6:

[AskUserQuestion]
Question: "Phase 5 (Agent Perspective Mode) is complete. Should I proceed
          with Phase 6 (Advanced Overlays)?"
Options:
- Yes, proceed to Phase 6
- Review implementation first
- Stop here
```

Note: Clear summary of what completed, explicit options for user.

---

## Changelog

### v5 (2026-01-08)

**High-impact additions:**
- Added Phase 1.7: Operational Model - captures usage patterns, concurrency model, state lifecycle
- Added Phase 1.8: Integration Inventory - maps component boundaries, shared data structures, type contracts
- Added Phase 4.5: Phase Dependency Mapping - explicit DAG, gate checks, critical path identification
- Added `phase_dependencies` section to PRD schema with DAG, critical path, parallel eligibility
- Added `depends_on`, `gates_phases`, `gate_verification` fields to phase schema

**Medium-impact additions:**
- Added Phase 3.2: Testability Planning - classifies features by testability, specifies automation requirements
- Enhanced Phase 3.5 with Divergence Decision Tree - systematic fix-vs-accept criteria, tolerance guidelines, red flags
- Added completion promise escaping guidance to Phase 6 - prevents shell interpretation issues with angle brackets
- Added progress.txt template with issues/workarounds/skipped sections - tracks failures, not just successes

**Documentation updates:**
- Updated interview checklist with conditional phases section
- Updated usage instructions to include new phases (now 19 steps)

**Lessons incorporated from microecon/llm-testing workflow analysis:**
- Operational model prevents architectural mismatches (frontend concurrency bugs)
- Integration inventory catches data structure divergence (TradeEvent classes)
- Phase dependencies prevent silent assumption failures (Phase 2 needing Phase 1 infrastructure)
- Feature parity matrix for migration projects
- Testability planning catches automation gaps (WebSocket race conditions, export validation)
- Theory divergence decision tree provides clear fix-vs-accept criteria
- Progress tracking with issues creates institutional memory

### v4 (2026-01-07)
- Clarified iteration budget semantics (entire loop, not per-phase)
- Added HOTL vs AFK Mode Considerations section
- Added Phase 2.5: Design Sketch for UI/visualization projects
- Added ADR file creation guidance (separate files vs inline)
- Added Emergent PRD Pattern section
- Added example interview snippets appendix
- Updated PRD schema to remove per-phase estimated_iterations
- Updated Phase 5 questions for clarity on phase transitions

### v3 (2026-01-07)
- Added project type determination (research software / learning / implementation)
- Added Phase 1.5: Resource Constraint Analysis
- Enhanced Phase 2 with ADR identification and drafting
- Added Phase 3.5: Divergence Tracking Setup for research software
- Added Phase 6: Loop Startup Document Creation
- Added RETROSPECTIVE-TEMPLATE.md for learning projects
- Added Paper Analysis Template for research/learning projects
- Expanded PRD schema with project_type, resource_constraints, tracking fields
- Updated interview checklist with project-type-specific items

### v2 (2026-01-06)
- Initial vision-anchored protocol
- Phased interview structure
- PRD generation with completion promises
