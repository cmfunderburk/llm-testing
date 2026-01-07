# Design Principles for Ralph-Loop-Appropriate Documentation

A conceptual framework for designing documentation and prompts that enable effective autonomous iteration.

---

## Foundational Philosophy

### The Paradox of Predictable Failure

Ralph-loop documentation embodies a counterintuitive insight: **systems that fail predictably outperform systems that succeed unpredictably**. The goal is not to write documentation so perfect that the loop never fails—that's impossible and counterproductive. The goal is to write documentation that makes failure *informative* and *recoverable*.

Traditional documentation assumes a reader who understands context, fills gaps with judgment, and adapts to ambiguity. Ralph-loop documentation assumes a reader who will:
- Interpret literally and act immediately
- Lose context between iterations (memory may refresh)
- Need explicit signals to know when work is complete
- Learn from failure only if failure produces legible feedback

This is not a limitation to work around—it's a design constraint that produces better documentation for humans too. Documentation clear enough for a ralph-loop is documentation clear enough for a tired developer at 2am, a new team member on day one, or your future self six months later.

**Anti-pattern**: Writing documentation that assumes the reader "will figure out" implicit requirements. If the loop can't figure it out, neither will many humans.

### External Memory as Architectural Primitive

A ralph-loop's "intelligence" across iterations comes not from an expanding context window but from the filesystem and git history. The loop sees:
- Modified files from previous iterations
- Commit history and messages
- State files (TODO.md, progress.txt, status markers)
- Test results and verification outputs

This means documentation must be designed for **stateful accumulation**. Each iteration should leave traces that inform the next. Progress isn't just made—it's *recorded* in forms the next iteration can read.

**Anti-pattern**: Designing workflows where progress is implicit (in the code changes themselves) without explicit progress markers the loop can parse.

### Completion as Contract

The completion promise is not a convenience—it's a contract between the human designer and the autonomous loop. When you specify `<promise>COMPLETE</promise>`, you're asserting:

1. **Measurability**: There exists a state of the world that unambiguously satisfies this promise
2. **Verifiability**: The loop can determine whether that state has been reached
3. **Achievability**: The task is actually completable within the iteration budget
4. **Atomicity**: The promise represents a coherent unit of work, not a grab-bag of unrelated criteria

A vague promise ("make it good") produces infinite loops. An unverifiable promise ("users will find it intuitive") produces false completions. An unachievable promise produces exhausted iteration budgets. A non-atomic promise produces partial completions mistaken for success.

**Anti-pattern**: Completion promises that require human judgment to evaluate. If you can't write a test or check for it, the loop can't verify it.

---

## Core Principles

### Principle 1: Specification Precedes Implementation

Before a ralph-loop can implement something, that something must be specified with sufficient precision that completion is recognizable. This applies at multiple levels:

**Interface Level**: What does the component do? What are its inputs, outputs, and contracts with other components? A loop implementing a `BargainingProtocol` needs to know the ABC it must satisfy, the method signatures it must implement, and the invariants it must maintain.

**Behavioral Level**: How does the component behave in specific scenarios? Edge cases, error conditions, and boundary behaviors must be enumerated. A loop implementing `TIOLI` (take-it-or-leave-it) bargaining needs to know: Who proposes? How is the offer calculated? What happens on rejection?

**Verification Level**: How do we know the implementation is correct? This includes functional tests (does it run?), property tests (does it maintain invariants?), and—crucially for research software—theory tests (does it match the mathematical model?).

**Anti-pattern**: Asking a loop to "implement TIOLI bargaining" without specifying proposer assignment rules, offer strategies, and acceptance criteria. The loop will make choices—possibly inconsistent with your theoretical framework.

### Principle 2: Architectural Decisions Must Be Resolved or Escalated

When a ralph-loop encounters a decision point that affects system architecture, one of two patterns must apply:

**Pre-Resolution Pattern**: The decision is made before the loop runs and captured in an Architecture Decision Record (ADR). The loop sees the decision as settled fact and implements accordingly.

```markdown
# ADR-001: Multi-Agent Mechanism Architecture

## Decision
Mechanisms own the matching call site. Matching logic is injected as MatchingStrategy.

## Consequences
- MatchingProtocol becomes a strategy pattern
- Mechanisms can customize matching without modifying tick loop
- Venues become spatial containers for mechanism participation

## Status: ACCEPTED
```

A loop implementing a new mechanism reads this ADR and knows the architectural context. It doesn't need to re-litigate the decision.

**Escalation Pattern**: The loop recognizes it has encountered a decision point outside its authority and pauses for human input. This requires:
- Explicit markers for what constitutes a decision point
- A mechanism for the loop to signal it has stopped (not completed)
- Documentation of the decision boundary

```markdown
## Implementation Notes
If you encounter any of the following, STOP and request human input:
- Changes to the tick loop phase structure
- New abstract base classes
- Modifications to the Agent state/observable type separation
```

**Anti-pattern**: Leaving architectural decisions implicit, allowing the loop to make major structural choices without human awareness. You wake up to a working feature built on an incompatible architecture.

### Principle 3: Theory Verification Is Primary

For research software, the verification hierarchy inverts the typical software engineering approach. Theory verification isn't the final polish—it's the primary constraint that gives meaning to all other tests.

**The Research Software Verification Hierarchy:**

```
Theory Verification (Primary)
    ↓ defines what "correct" means
Property Verification (Derived)
    ↓ operationalizes theoretical invariants
Functional Verification (Instrumental)
    ↓ ensures the machinery runs
```

**Theory Verification (Primary Constraint)**

Does the implementation match the mathematical model? For research software, this is the question. Everything else exists to answer it.

```markdown
## Theory Requirements (PRIMARY)
- [ ] With standard Cobb-Douglas preferences, equilibrium prices match analytical solution
- [ ] Rubinstein allocation converges to Nash as δ → 1
- [ ] TIOLI proposer captures full surplus minus responder's reservation utility
- [ ] First-mover advantage magnitude: (1-δ₂)/(1-δ₁δ₂) matches implementation
```

Theory tests are not "nice to have"—they are the definition of correctness. An implementation that passes all functional tests but fails theory verification is *wrong*, even if it runs beautifully.

**Property Verification (Derived from Theory)**

Properties are theoretical invariants translated into testable assertions. They derive their authority from theory.

```markdown
## Property Requirements (derived from theory)
- [ ] Allocations are Pareto efficient (no trades improve both parties)
- [ ] Budget constraints satisfied (agents don't receive more than endowment)
- [ ] Symmetry: Nash bargaining with identical agents produces equal splits
- [ ] Individual rationality: no agent worse off than initial endowment
```

Property tests serve theory verification by catching violations of known invariants quickly, before expensive theory tests run.

**Functional Verification (Instrumental)**

Functional tests verify the machinery works. They have no authority to declare correctness—only to declare that the code runs.

```markdown
## Functional Requirements (instrumental)
- [ ] `compute_allocation()` returns valid Bundle objects
- [ ] Handles zero-endowment agents without division errors
- [ ] Works with any valid alpha in (0, 1)
- [ ] Integrates with existing tick loop without error
```

Functional tests passing is necessary but carries no weight regarding research validity.

**Completion Promises Must Reference Theory**

For research software, a completion promise that references only functional tests is inadequate:

- ❌ "All tests pass" — which tests? Functional tests passing proves nothing about theory.
- ❌ "Implementation complete" — complete relative to what standard?
- ✓ "Theory verification suite passes with <1% tolerance" — references the primary constraint.
- ✓ "Equilibrium predictions match analytical solution" — directly states theoretical criterion.

**Anti-pattern**: Treating theory verification as optional or "Phase 3" work. If theory tests don't exist, you don't know if your implementation is correct—you only know it runs. For research software, that's not knowledge; it's hope.

### Principle 4: Progress Must Be Legible Across Iterations

Each iteration should leave the project in a state where the next iteration can understand:
- What has been accomplished
- What remains to be done
- What obstacles were encountered
- What decisions were made (and why)

This requires explicit progress tracking mechanisms:

**Append-Only Progress Files**: Following Matt Pocock's pattern, use files that accumulate rather than overwrite:

```markdown
## progress.txt (append-only)

### Iteration 3 - 2026-01-06T14:32
- Implemented TIOLI proposer assignment (random selection)
- Tests passing for basic accept/reject paths
- BLOCKED: Need clarification on multi-round TIOLI (is offer updated or fixed?)

### Iteration 4 - 2026-01-06T14:58
- Proceeding with fixed-offer interpretation per ADR-002
- Added rejection handling
- All functional tests passing
```

**Structured State Files**: For complex tasks, use parseable state:

```json
{
  "current_phase": "implementation",
  "completed": ["proposer_assignment", "offer_calculation", "acceptance_logic"],
  "in_progress": "rejection_handling",
  "blocked": [],
  "decisions_made": ["fixed_offer_interpretation"]
}
```

**Git Commit Discipline**: Each meaningful unit of progress should be a commit. The loop should commit frequently with descriptive messages. The git log becomes a progress record.

**Anti-pattern**: Loops that work for 20 iterations, make many changes, but don't commit until the end. If the loop fails, you can't easily recover partial progress or understand what was attempted.

### Principle 5: Task Scope Must Match Iteration Budget

A ralph-loop with `--max-iterations 20` has a finite budget. The task must be:
- **Decomposable** into steps achievable within that budget
- **Recoverable** if the budget is exhausted (partial progress saved)
- **Adjustable** based on observed iteration consumption

This requires thinking about task granularity:

**Too Broad**: "Implement all bilateral bargaining protocols" — unbounded scope, no single completion point

**Too Narrow**: "Add a docstring to compute_allocation" — trivial, doesn't need a loop

**Appropriate**: "Implement TIOLI bargaining protocol with full test coverage" — bounded, verifiable, meaningful

For larger initiatives, use a PRD-based decomposition (following Pocock's pattern):

```json
{
  "features": [
    {"id": "TIOLI-001", "name": "Basic TIOLI protocol", "passes": false},
    {"id": "TIOLI-002", "name": "Multi-agent TIOLI venue", "passes": false},
    {"id": "TIOLI-003", "name": "Theory verification suite", "passes": false}
  ]
}
```

The loop picks the highest-priority incomplete feature and works only on that. Completion of one feature doesn't complete the whole initiative—it marks that feature as done and the loop can pick the next.

**Anti-pattern**: Monolithic tasks that either succeed entirely or waste the iteration budget. Design for incremental, committable progress.

---

## Application: Protocol Implementation (Research Software)

For implementing new protocols in research simulation software, ralph-loop documentation should address:

### Theoretical Grounding

Every protocol must trace to its theoretical foundations. The loop needs to know not just *what* to implement but *why* this specific design:

```markdown
# TIOLI Bargaining Protocol

## Theoretical Foundation
Take-it-or-leave-it bargaining represents the extreme of asymmetric bargaining power.
Per Osborne & Rubinstein (1990), Chapter 3: when one party has all proposal power
and the other has only veto power, the proposer captures the entire surplus.

## Key Theoretical Predictions
1. Proposer receives their ideal allocation subject to responder's participation constraint
2. Responder receives exactly their reservation utility (indifference point)
3. No inefficiency from rejection in equilibrium (responder always accepts)
```

### Interface Specification

The protocol must satisfy existing abstractions. The loop needs the interface documented explicitly:

```markdown
## Interface Requirements (BargainingProtocol ABC)

Must implement:
- `compute_allocation(agent1: Agent, agent2: Agent) -> tuple[Bundle, Bundle]`
- Must respect both agents' endowment constraints
- Must return valid Bundle objects (non-negative quantities)

May implement:
- `get_surplus_split() -> tuple[float, float]` for analysis
- `requires_information_environment() -> bool` if protocol needs specific information

Must NOT:
- Modify agent state directly
- Access global simulation state
- Make network calls or I/O
```

### Edge Cases and Boundaries

The loop will encounter edge cases. Document them proactively:

```markdown
## Edge Cases

1. **Identical Agents**: If agents have identical preferences and endowments, any allocation
   on the Pareto frontier is valid. Implement: give proposer their preference-weighted optimum.

2. **Zero Surplus**: If no mutually beneficial trade exists, return initial endowments unchanged.

3. **Corner Solutions**: If optimal allocation gives one good entirely to one agent, ensure
   bundle representation handles zero quantities correctly.

4. **Numerical Precision**: Use `np.isclose()` for equality checks, not `==`.
```

### Verification Criteria

The completion promise must be tied to specific, testable criteria:

```markdown
## Completion Criteria

This protocol is complete when ALL of the following pass:

### Functional Tests
- [ ] Protocol runs without error for standard agent configurations
- [ ] Handles edge cases (identical agents, zero surplus, corners) correctly
- [ ] Integrates with existing tick loop

### Property Tests
- [ ] Allocations are Pareto efficient (property-based testing)
- [ ] Budget constraints satisfied for all random seeds
- [ ] Consistent results across repeated calls (deterministic)

### Theory Tests
- [ ] Proposer surplus matches theoretical prediction (±1% tolerance)
- [ ] Responder receives exactly reservation utility
- [ ] No rejected offers in equilibrium runs (responder always accepts)

When all boxes are checked and tests pass: `<promise>TIOLI-COMPLETE</promise>`
```

---

## Application: Experimentation and Feature Development (Learning Systems)

For systems where the goal is prompt refinement, A/B testing, or iterative feature development, ralph-loop documentation needs different emphases.

### Theory-Primary Verification for Pedagogical Systems

The "theory-primary" principle applies to learning systems, but the nature of theory differs fundamentally from mathematical economics. Learning science provides *empirical regularities* and *design principles* rather than *analytical solutions*.

**Mathematical Theory (Economics):**
- "Rubinstein allocation converges to Nash as δ → 1"
- Verification: Compute both values, compare within tolerance
- Outcome: Binary pass/fail against known answer

**Empirical Theory (Learning Science):**
- "Spaced repetition produces better long-term retention than massed practice"
- Verification: Design must *embody* the principle; outcomes require longitudinal measurement
- Outcome: Alignment with principle (design-level) or statistical comparison (outcome-level)

This distinction produces a different verification hierarchy for pedagogical systems:

**Design Alignment (Primary Constraint)**

Does the implementation embody learning science principles? This is checkable at design-time:

```markdown
## Design Alignment Requirements (PRIMARY)
- [ ] Leitner scheduling enforces spacing (intervals increase with mastery)
- [ ] Strict grading creates desirable difficulty (partial credit → reset)
- [ ] Socratic mode asks questions rather than providing answers (retrieval practice)
- [ ] Card generation targets "aha" moments, not passive mentions (generation effect)
```

Design alignment is the pedagogical equivalent of theory verification. If the design violates learning science (e.g., allowing partial credit everywhere), the implementation is *wrong* regardless of whether it runs.

**Behavioral Verification (Derived)**

Does the system behave as design alignment requires? This is checkable at runtime:

```markdown
## Behavioral Requirements (derived from design)
- [ ] Cards actually advance through Leitner boxes on strict criteria
- [ ] Socratic mode produces >80% questions, <20% statements
- [ ] Generated cards meet atomicity constraint (2-15 lines)
- [ ] Difficulty calibration adjusts based on session outcomes
```

Behavioral tests verify that the machinery implements the design. A bug that allows cards to advance without meeting criteria violates design alignment.

**Outcome Measurement (Ultimate but Deferred)**

Does the system produce learning outcomes? This requires longitudinal data:

```markdown
## Outcome Measures (ultimate validation, requires data collection)
- [ ] Retention curves follow expected spacing-effect patterns
- [ ] Learners using Socratic mode show deeper comprehension (measured how?)
- [ ] Difficulty calibration reduces struggle without reducing learning
```

Outcome verification is the ultimate test but cannot be the completion criterion for a ralph-loop—it requires human learners over time. Ralph-loops for learning systems complete at the design-alignment or behavioral level, with outcome measurement as separate research.

**Implications for Completion Promises**

For learning systems, completion promises should reference design alignment:

- ❌ "Feature implemented" — no reference to learning science
- ❌ "Users learn better" — unmeasurable in a loop
- ✓ "Strict grading implemented per Leitner protocol" — design alignment
- ✓ "Socratic mode produces 85% questions in test dialogues" — behavioral verification

### Hypothesis-Driven Experimentation

Unlike implementation (where the goal is known), experimentation explores unknowns. The loop needs:

```markdown
# Experiment: Opening Question Variants

## Hypothesis
Opening questions that reference specific passages produce more text-grounded dialogues
than generic opening questions.

## Experimental Design
- **Control**: Generic opening ("What questions do you have about this chapter?")
- **Treatment**: Passage-specific ("On page 12, the author claims X. Do you agree?")
- **Measure**: Count of passage citations in subsequent dialogue turns
- **N**: 10 dialogues per condition across 3 different chapters

## Completion Criteria
Generate all 60 dialogues (10 × 2 conditions × 3 chapters).
Output: `<promise>DIALOGUES-GENERATED</promise>`

Note: Analysis and hypothesis evaluation are separate tasks.
```

### Generative Completion

For dialogue generation tasks, completion isn't "tests pass" but "artifacts produced":

```markdown
## Artifact Specification

Each generated dialogue must:
1. Be saved to `experiments/opening-questions/{condition}/{chapter}/{run_id}.jsonl`
2. Contain at least 8 exchange turns
3. Include metadata header with condition, chapter, timestamp
4. Not exceed 4000 tokens total (to control for length)

Completion check: Count files in output directory matches expected (60).
```

### Feature Development with Emergent Requirements

Unlike protocol implementation (where theory specifies behavior), feature development often discovers requirements through iteration:

```markdown
# Feature: Learner Difficulty Calibration

## Initial Specification
Track patterns across sessions to calibrate Socratic question difficulty.

## Known Requirements
- Store session outcomes (did learner struggle? succeed easily?)
- Compute difficulty metrics per topic/chapter
- Adjust question generation based on difficulty history

## Discovery Process
This feature requires iteration to understand what "difficulty" means operationally.
First iteration should:
1. Implement basic session outcome storage
2. Propose 2-3 candidate difficulty metrics
3. Document trade-offs for human review

Do NOT proceed to full implementation until difficulty metric is approved.
Output after first phase: `<promise>CALIBRATION-DESIGN-READY</promise>`
```

This is the escalation pattern in action—the loop does discovery work, then pauses for human decision before proceeding.

### Multi-Model Orchestration (Advanced Extension)

For advanced use cases like generating Claude/Gemini dialogues as learning material, additional considerations apply:

- **Turn-taking protocol**: Which model speaks when? How are turns signaled?
- **Context synchronization**: Each model needs the full dialogue history
- **Persona consistency**: Each model should maintain a distinct intellectual stance
- **Termination conditions**: When has the dialogue explored the topic sufficiently?

This is an advanced application that builds on the core principles but requires additional coordination machinery. Document as an extension rather than a core pattern.

---

## AFK vs HOTL: Considerations for Mixed-Mode Usage

### AFK (Away From Keyboard) Mode

Long-running overnight loops require:

**Robust Iteration Limits**: Set `--max-iterations` conservatively. Better to wake up to "ran out of iterations" than "spent $200 on API credits."

**Coarse-Grained Progress Markers**: Since you won't intervene, ensure each commit represents meaningful progress. The git log should tell a story you can follow in the morning.

**Generous Error Recovery**: The loop will encounter transient failures (network issues, rate limits). Documentation should specify retry behavior and fallback strategies.

**Clear Stopping Points**: Define what "done for now" looks like even if the full task isn't complete. The loop should commit progress and stop cleanly rather than thrashing.

### HOTL (Human On The Loop) Mode

Interactive sessions with monitoring allow:

**Finer-Grained Escalation**: You can intervene at decision points. Mark more situations as "pause for human input" since the human is present.

**Real-Time Adjustment**: If the loop is heading in a wrong direction, you can cancel and re-prompt. Design prompts to be adjustable mid-stream.

**Shorter Iteration Budgets**: With monitoring, you don't need large safety margins. Use what's needed.

**Exploratory Tasks**: HOTL mode suits tasks where the goal is discovery rather than known-endpoint implementation. You guide the exploration.

### Mixed-Mode Strategy

For projects requiring both modes:

1. **Use AFK for well-specified implementation tasks** with clear completion criteria
2. **Use HOTL for exploration and decision-making** where human judgment adds value
3. **Design handoff points** where AFK work produces artifacts HOTL sessions can review and refine
4. **Document mode-specific prompts** so you can invoke the right style for each context

---

## Summary: The Ralph-Loop Documentation Checklist

When designing documentation for ralph-loop usage, verify:

### Specification Completeness
- [ ] Interface contracts are explicit
- [ ] Edge cases are enumerated
- [ ] Theoretical foundations are documented (for research software)
- [ ] Success criteria are measurable and verifiable

### Architectural Clarity
- [ ] Major decisions are captured in ADRs
- [ ] Decision boundaries are marked (what requires human input)
- [ ] Integration points with existing system are specified

### Verification Layers
- [ ] Functional tests defined
- [ ] Property tests defined (where applicable)
- [ ] Theory tests defined (for research software)
- [ ] Completion promise references appropriate verification level

### Progress Legibility
- [ ] Progress tracking mechanism specified (append-only files, structured state)
- [ ] Commit discipline expectations documented
- [ ] Partial progress is recoverable

### Task Scoping
- [ ] Task is decomposed to fit iteration budget
- [ ] PRD or equivalent structures larger initiatives
- [ ] Incremental completion markers defined

---

## Conclusion: Documentation as Executable Specification

Ralph-loop-appropriate documentation is documentation that could, in principle, be executed. Not in the sense of running code, but in the sense that an agent following the documentation literally would produce the intended outcome.

This is a higher standard than typical documentation. It requires:
- Precision where ambiguity would cause divergent interpretations
- Explicitness where implicit knowledge would cause confusion
- Measurability where subjective quality would cause endless iteration
- Modularity where monolithic tasks would exceed iteration budgets

The reward is documentation that serves both autonomous loops and human developers—clear enough to run overnight, clear enough to onboard a new team member, clear enough to revisit after months away from the codebase.

The Ralph Wiggum character never stops trying. Good documentation ensures that trying leads somewhere.
