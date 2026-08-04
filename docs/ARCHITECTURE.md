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
- Candidate initialization embeds distilled units and queries inside the plugin
  with a local FastEmbed/ONNX model; it has no AstrBot provider or remote
  inference boundary.
- The private reconstruction loop has seven typed, scoped, read-only tools.
- The main LLM does not see those low-level tools by default.
- Empty graph scopes skip the private provider call entirely.
- Historical experiments pass a strict `before_sent_at` cutoff through candidate
  search and every traversal tool. They still use an isolated database because a
  later topic or semantic revision could otherwise overwrite historical state.

## Reconstruction control

The LLM chooses graph paths, but it must not be the only component allowed to stop
the search. The first real-call ablation showed that the model could retrieve the
correct source event on its first tool call and still browse until it discarded the
answer. A deterministic host evidence gate has therefore been validated offline:

1. the event must come from the initial high-score episode candidates;
2. raw event context must contain source keys;
3. raw evidence and the query must have salient lexical overlap;
4. passing the gate means “synthesize now”, never “the claim is true”.

The gate remains a runtime TODO. It should be switchable so later masked experiments
can compare identical runs with and without host stopping.

## Storage policy

Keep source messages, graph revisions, provenance, feedback, and administrator
decisions. Do not permanently keep hidden model reasoning, duplicated prompts,
or attachment blobs. Only distilled nodes should receive embeddings by default.

Developer observability uses three privacy-minimized tables. Experiment records
store query hashes, status, and bounded metadata; usage records store token classes
and latency; reconstruction records store tool arguments, evidence source keys, and
result hashes. Runtime private-agent usage comes from AstrBot runner aggregate stats,
so every internal LLM turn is included even though hidden reasoning is not retained.

At the current deployment's observed traffic, the expected steady-state growth
is roughly 250--500 MB per year with structured traces. This estimate must be
revisited before enabling attachment storage or per-message embeddings.

## Feedback working graph and revision

Feedback-driven behavioral revision is implemented as an append-only evidence
path plus a bounded active view:

```text
request/action/response -> later feedback -> prospective hypothesis
                                      \-> signed credit for activated paths
```

The host, not the maintenance LLM, enforces proposal binding, time order, group
and sender scope, a configurable evidence threshold, and atomic writes. Generic
style preferences use `activation_mode=always`; task-conditioned behavior uses
`activation_mode=semantic` with evidence-derived lexical triggers and a bounded
private-agent semantic gate. Utility and factual confidence are deliberately
separate.

Implemented retention rules:

- preserve the original evidence and every revision;
- distinguish activation/salience from factual confidence;
- treat negative emotion as a strong review trigger, not sufficient proof alone;
- stage write proposals before one host-validated commit transaction;
- retain evidence IDs through reconstruction so feedback can target the memory
  actually used by the main LLM;
- decay utility, bound the active view, and make merge/unmerge reversible without
  deleting provenance.

Still planned: administrator confirm/edit/reject/defer controls, explicit
self-correction flows, and semantic-fact supersede/retract states. The current
loop learns prospective behavior but does not rewrite an old factual claim.

## Wake-up policy

Use three complementary triggers:

- a cheap deterministic activation gate before main-LLM requests (implemented);
- private semantic activation and bounded feedback maintenance on a main request
  (implemented, default off);
- explicit consultation from the main LLM when the injected brief is insufficient
  (implemented);
- background consolidation after a message threshold or idle interval (TODO).

The persistent component is the scheduler, queue, graph revision, and activation
state. LLM calls remain bounded and event-driven rather than continuously running.
