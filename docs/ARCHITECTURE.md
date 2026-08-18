# Architecture

MR Memory separates raw evidence, distilled graph memory, private reconstruction,
and AstrBot's main LLM. The private memory agent uses its own configured provider.
The main LLM receives only a bounded, host-validated surface packet compiled from
an `EvidenceCertificateV2`, or calls one consultation tool; it never receives the
private reconstruction tools directly.

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
- Provider-mediated reconstruction is split into a one-pass L2 Evidence Reader and
  a host-approved, bounded L3 `EccrOrchestrator`. The main LLM does not see either
  layer's low-level read actions.
- Empty graph scopes skip the private provider call entirely.
- Historical experiments and the production path bind candidate search, reply
  lookup, identity resolution, graph reads, and every L3 action to the same strict
  cutoff and message-row upper bound. Experiments still use an isolated database
  because a later topic or semantic revision could otherwise overwrite historical
  state.

## Layered reconstruction control

Embedding distance initializes candidates but does not decide relevance or truth.
The host owns scope, time, identity, source allowlists, cache validation, routing,
budgets, and all writes. The private model may interpret only the bounded evidence
the host provides; it cannot select a group, relax a cutoff, authorize a deeper
route, or turn an operational failure into semantic absence.

### L0 request snapshot

Every answer-time memory request first creates one immutable `RequestSnapshot`. It
contains the host-derived UMO and scope hash, strict `cutoff_at`, the transaction-order
`message_upper_bound`, current request and reply source keys, sender participant key,
query and context hashes, capture time, and two revision vectors.

The data revision vector covers message, deletion, identity, graph, relation, and
feedback heads. The inference revision vector covers the retriever, embedding model,
fusion policy, reader model and protocol, certificate schema, surface compiler, and
route policy. Every read belonging to the request must satisfy both the strict time
cutoff and the row upper bound. The current message and later same-second arrivals
therefore cannot enter the request, while earlier same-second rows remain visible.

Append-only messages that arrive after capture are excluded by those two bounds and
do not interrupt an in-flight L2/L3 call. A deletion, identity change, graph rewrite,
relation revision, or feedback revision that can alter evidence already visible in
the frozen window makes the request stale and fails closed.

### L0--L3 route

```text
L0 RequestSnapshot and deterministic host checks
        |
        v
L1a exact evidence-pack cache
        |
        v
L1b dependency-revalidated semantic-certificate cache
        |
        v
host-owned route policy
        |
        +-- L2 one-pass Evidence Reader
        +-- L3 bounded ECCR (compile / discriminate / audit discovery)
        |
        v
EvidenceCertificateV2 -> surface compiler -> main LLM -> shadow verifier
```

- **L0** performs snapshot capture, deterministic host checks, and host-only returns.
  An empty eligible memory scope does not call the Provider.
- **L1a** reuses only an exact evidence packet bound to scope, normalized query,
  context, reply target, row bound, data heads, and retrieval revisions. It is a
  retrieval cache, not a semantic answer cache; every hit is source-audited against
  the current snapshot.
- **L1b** reuses a semantic certificate only after rebuilding the exact packet,
  checking its hash, re-auditing cited sources and recorded dependencies, confirming
  the inference revisions, and rebinding the certificate envelope to the current
  snapshot. Similar wording alone never authorizes reuse.
- **L2** gives one bounded packet to the Evidence Reader for one semantic read. It
  may return `CERTIFIED`, `PARTIAL`, `SEMANTIC_NONE`, `SAFETY_ABSTAIN`, or
  `REQUEST_L3`. A protocol-only repair may correct one malformed envelope, but it
  does not create an open-ended semantic traversal.
- **L3** runs only when deterministic host policy allows it. The shared production
  `EccrOrchestrator` has explicit model-call, retrieval-round, and deadline bounds
  and performs compile, discriminate, and audit-discovery phases through
  host-validated read actions. Its temporary argument graph does not directly write
  long-term memory.

`REQUEST_L3` is advisory. Only the host may approve it, based on request kind,
identity ambiguity, high-risk attribution, multiple events, conflicting evidence,
revision questions, explicit deep recall, feedback audit, mode, budget, and runtime
state. If required depth is not authorized, the route returns a safety abstention
instead of silently making an L2 call. Semantic status and operational status remain
separate, so `RUNNING`, `BUDGET_BLOCKED`, `PROVIDER_UNAVAILABLE`, timeout, and failure
can never be stored or presented as `SEMANTIC_NONE`.

### Certificate, surface, and answer audit

L2 and L3 both terminate in `EvidenceCertificateV2`. The certificate is bound to the
request snapshot, packet hash, source and participant allowlists, and data/inference
revisions. Each evidence atom records subject and speaker attribution, evidence
stance, source keys and spans, provenance, importance, and confidence. The
certificate separately retains conflicts, unresolved conditions, open obligations,
`must_include`, `must_not_upgrade`, and an explicit stop reason.

Before injection, the surface compiler validates that contract and emits a bounded
JSON packet. It may remove optional atoms to fit the limit, but it fails closed if a
required anchor or upgrade guard cannot fit. The main LLM receives that packet as
untrusted reference data, not as instructions. After the answer, a shadow verifier
records missing anchors, lost attribution or uncertainty, and forbidden certainty
upgrades; it does not rewrite the certificate or promote model prose into memory
truth.

### Singleflight and invalidation

Concurrent work is coalesced only when the semantic certificate key, full snapshot
digest, and target route level all match. Only the singleflight producer performs
budget preflight and Provider work; waiters share its terminal result. A waiter
timeout does not cancel the shared task, and plugin unload drains then boundedly
cancels outstanding layered tasks.

L1 entries have explicit TTLs and cache status. Message/deletion/identity/graph/
relation/feedback revisions invalidate affected data or dependency records. Changes
to the retriever, embedding model, fusion policy, reader model/protocol, certificate
schema, surface compiler, or route policy change the relevant keys and prevent stale
reuse. Certificate dependencies allow targeted invalidation; cache cleanup and
interrupted reconstruction-job recovery remain bounded and auditable.

### Runtime modes

- `low_latency` automatically checks L0/L1 and starts or joins L2 on a miss. An
  ordinary chat waits only for the configured short `runtime_l2_wait_seconds`; when
  that waiter ends, the task may finish in the background for later exact reuse.
  Automatic L3 remains off unless `runtime_auto_deep_analysis` is enabled.
- `balanced` uses the same host route with a configured bounded L2 wait, normally
  chosen longer than for `low_latency`. The mode does not itself authorize L3;
  `runtime_auto_deep_analysis` still controls automatic escalation.
- `research` waits synchronously for routed semantic work and authorizes bounded L3
  when the host observes a qualifying risk or the reader requests it.
- `manual_only` disables automatic answer-time reconstruction. The optional
  consultation tool can still request an explicit, bounded deep recall.

Explicit memory queries and deep-recall requests use a bounded synchronous route even
outside `research`; ordinary chat may return while background L2 continues. A legacy
saved value of `every_request` maps to `research` so a hot reload does not silently
reduce the former synchronous/deep behavior.

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

Schema 16 adds request snapshots, evidence-pack caches, memory certificates,
certificate dependencies, invalidation events, and reconstruction-job lifecycle to
each physical group database. Startup recovers only bounded interrupted state, and
periodic cleanup expires stale snapshots, packs, certificates, dependencies, and
orphaned jobs without crossing group scope.

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

Automatic wake-up means “evaluate the layered host route”, not “call the Provider on
every request”. Use four complementary triggers:

- L0/L1 snapshot and cache checks before a main-LLM request when automatic wake-up is
  enabled; only an authorized cache miss reaches L2 or L3 (implemented);
- embedding and lexical candidates as bounded priors inside an exact evidence packet,
  not final relevance or truth gates (implemented);
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
