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

- Raw message revisions are the truth layer and are stored per group scope. Edits and
  recalls never leave their old derived graph active.
- Participants are keyed by host-derived platform account IDs inside one group.
  Nicknames are aliases, not entity keys; ambiguous aliases are never merged.
- Visible Bot responses, reply/mention relations, and bounded attachment descriptors
  enter the same truth layer as user messages.
- Distillation produces Cue--Tag--Episode, Participant--Aspect--Claim, and
  Topic--Episode units in the same scope.
- Feedback maintenance may additionally produce a plastic association graph for
  group-local meanings, symbols, behaviors, preferences, procedures, and traversal
  paths. Its node kinds cannot represent accounts or privileges.
- Candidate initialization embeds distilled units and queries inside the plugin
  with a local FastEmbed/ONNX model; it has no AstrBot provider or remote
  inference boundary.
- The private reconstruction loop has the paper's seven typed, scoped, read-only
  tools plus one read-only learned-association traversal tool.
- The main LLM does not see those low-level tools by default.
- Empty graph scopes skip the private provider call entirely.
- Historical experiments pass a strict `before_sent_at` cutoff through candidate
  search and every traversal tool. They still use an isolated database because a
  later topic or semantic revision could otherwise overwrite historical state.

## Reconstruction control

The private LLM is the semantic gate. Embedding distance initializes candidates but
does not decide relevance, and a request is not skipped merely because every vector
score is low. Once a group has graph memory, each main-LLM request gets one bounded
semantic tick; an empty graph still skips the provider entirely.

The first real-call ablation also showed that a model can retrieve the correct source
event and then browse until it discards the answer. A deterministic host evidence
gate remains available as an optional latency/cost optimization:

1. the event must come from the initial high-score episode candidates;
2. raw event context must contain source keys;
3. raw evidence and the query must have salient lexical overlap;
4. passing the gate means “synthesize now”, never “the claim is true”.

It is disabled by default and never acts as a semantic truth threshold. When enabled,
its decision and visited source keys enter the run ledger so masked experiments can
compare identical runs with and without early stopping.

## Storage policy

Keep source message revisions, graph revisions, provenance, feedback, and
administrator decisions. Do not permanently keep hidden model reasoning, duplicated
prompts, attachment blobs, signed URLs, or local file paths. The storage boundary
allowlists components and retains only attachment type/name/reference hash. Distilled
nodes, Participant aliases, and learned association documents may receive embeddings;
the scores remain candidate priors rather than write or truth authority.

Each construction batch is oldest-first and content-hash bound. Read-only overlap
messages provide continuity but cannot advance the checkpoint. Every target source
must either be cited by an episode/claim or appear in the persisted ignored-source
ledger. An interrupted `PROCESSING` lease is recovered as a bounded retry when the
database is reopened. Episode identity depends on its evidence set rather than LLM
wording.

Structured claims separate speaker from subject, keep multiple source rows and exact
spans, and use `ACTIVE`, `UNRESOLVED`, `QUARANTINED`, `CONFLICTED`, `SUPERSEDED`,
`RETRACTED`, or `STALE`. Revisions can target only active claims for the same subject.
Single-source identity or privilege claims are quarantined. See
[the identity model](IDENTITY_MODEL.md). Quarantined claims remain visible to the
administrator console and construction audit, but are excluded from automatic
runtime retrieval until independent evidence promotes them.

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
                                      \-> versioned plastic association mutation
```

The host, not the maintenance LLM, enforces proposal binding, time order, group
and sender scope, a configurable evidence threshold, and evidence-bound writes. A
plastic mutation is unavailable until the same feedback proposal reaches
`COMMITTED`; negative mutation is further restricted to plastic edges recorded in
the eligible response's activation trace. Generic
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
- reject blame assignment to an alternative path discovered only during later
  maintenance;
- decay utility, bound the active view, and make merge/unmerge reversible without
  deleting provenance.

The construction path now has explicit semantic-fact supersede/retract states. The
feedback loop can revise the separate plastic association graph but still cannot
rewrite deterministic account identity or directly overwrite a structured factual
claim; administrator confirm/edit/reject/defer controls remain planned.

## Wake-up policy

Use four complementary triggers:

- a bounded private semantic tick before every main-LLM request when graph memory
  exists (implemented);
- embedding and lexical candidates as priors, not final relevance gates (implemented);
- explicit consultation from the main LLM when the injected brief is insufficient
  (implemented);
- separate bounded construction and feedback workers after a message threshold,
  proposal, or maintenance interval (implemented; never run when capture is disabled).

The persistent component is the scheduler, maintenance jobs, graph revision, and
bounded operational state (focus, selected edge IDs, last decision, and evidence
keys). Hidden reasoning is never serialized. LLM calls remain bounded and
event-driven rather than continuously running.
Every group has two rolling 24-hour private-token ledgers. `online` covers live
construction, reconstruction, and feedback maintenance; `backfill` covers only
one-off imported-history construction. A checkpoint carries one immutable processing
class, live work has scheduling priority, and reaching either budget pauses only that
class without blocking the main response or consuming the other ledger.
