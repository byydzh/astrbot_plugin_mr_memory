# Architecture

MR Memory separates raw evidence, distilled graph memory, private reconstruction,
and AstrBot's main LLM. The private memory agent uses its own configured provider;
the main LLM receives only a bounded evidence brief or calls one consultation tool.

## Non-negotiable group isolation

The memory tenant is the current group event's AstrBot
`unified_msg_origin` (UMO). It is derived inside the plugin and is never accepted
from an LLM tool argument or user-provided query parameter.

Every operation follows these rules:

1. Reject events without a platform ID, group ID, and UMO.
2. Derive the scope from the current `AstrMessageEvent`.
3. Select a physically separate SQLite file named by the scope's SHA-256.
4. Pass the scope into every storage read and write inside that file.
5. Filter every graph query by that scope.
6. Re-check both sides of joins that could otherwise connect two scopes.
7. Never expose an API that allows an LLM to choose another scope.

AstrBot main-agent isolation is therefore not part of MR Memory's trust boundary.
Even if the main agent mixes context, it cannot ask this plugin to read another
group's memory.

## Runtime boundaries

- Raw messages are immutable evidence and are stored per group scope.
- Distillation produces Cue--Tag--Episode, Person--Aspect--Semantic, and
  Topic--Episode units in the same scope.
- The private reconstruction loop has seven typed, scoped, read-only tools.
- The main LLM does not see those low-level tools by default.
- Empty graph scopes skip the private provider call entirely.

## Storage policy

Keep source messages, graph revisions, provenance, feedback, and administrator
decisions. Do not permanently keep hidden model reasoning, duplicated prompts,
or attachment blobs. Only distilled nodes should receive embeddings by default.

At the current deployment's observed traffic, the expected steady-state growth
is roughly 250--500 MB per year with structured traces. This estimate must be
revisited before enabling attachment storage or per-message embeddings.

## Feedback and revision TODO

Feedback-driven revision is a required architecture feature, not an optional
extension. It will use append-only state transitions:

```text
candidate -> provisional -> confirmed
                         -> disputed -> superseded / retracted
```

Planned requirements:

- preserve the original evidence and every revision;
- distinguish activation/salience from factual confidence;
- treat negative emotion as a strong review trigger, not sufficient proof alone;
- allow explicit self-correction or administrator correction to supersede memory;
- stage write proposals before committing them;
- provide an administrator review queue with confirm/edit/reject/defer actions;
- retain evidence IDs through reconstruction so feedback can target the memory
  actually used by the main LLM;
- use hysteresis so weak feedback flags a memory while stronger evidence is
  required to reverse a recently revised conclusion.

## Wake-up policy TODO

Use three complementary triggers:

- background consolidation after a message threshold or idle interval;
- a cheap deterministic gate before main-LLM requests;
- explicit consultation from the main LLM when the injected brief is insufficient.

The persistent component is the scheduler, queue, graph revision, and activation
state. LLM calls remain bounded and event-driven rather than continuously running.
