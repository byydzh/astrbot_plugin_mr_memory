# Architecture

MR Memory keeps the existing per-group truth store and rewrites the answer-time
path as a bounded evidence reconstruction service. While `/chat` is waiting, the
plugin performs exactly one call to its configured resident Evidence Reader. That
Reader jointly resolves textual references and memory meaning from a compact,
source-backed packet; it cannot call tools, repair its response, switch providers,
or request a deeper route. AstrBot's selected main model then receives the compiled
memory surface and answers normally.

## Non-negotiable group isolation

The tenant is the current group event's host-derived AstrBot
`unified_msg_origin` (UMO). The plugin never accepts a UMO from a user, model, or
tool argument. Every operation derives scope from the current event, selects one
physically separate SQLite database named by the scope hash, rechecks that scope on
every storage/graph operation, and freezes a cutoff plus message-row upper bound.

## Data plane retained

- Raw message revisions remain the truth layer. Edits, recalls, and self-erasure
  invalidate dependent graph state without deleting provenance.
- Participants use host-observed platform account IDs. Nicknames are aliases,
  never account keys; ambiguous aliases stay separate.
- Reply/mention relations, visible Bot outputs, and bounded attachment descriptors
  share the same truth boundary.
- Background distillation may build Cue--Tag--Episode,
  Participant--Aspect--Claim, Topic--Episode, semantic-memory, and plastic
  association units.
- Feedback maintenance may stage and commit source-backed behavioral hypotheses.
  It cannot rewrite account identity or promote model prose to truth.
- Local embeddings initialize candidates only. Distance is not relevance or truth.
- SQLite planner statistics are refreshed with `ANALYZE` when a group database is
  opened so populated visibility joins use indexed plans.

## Online serving plane

```text
AstrBot on_llm_request
  -> freeze one RequestSnapshot for this event
  -> retrieve aliases + trigram FTS5/bounded CJK bigrams + embeddings + graph
  -> expand history/activity only for actual identity candidates
  -> compile one globally bounded, source-deduplicated EvidenceAtomPack
  -> audit selected source revisions/content against the frozen snapshot
  -> one strict evidence-reader.compact-host-speaker semantic pass
  -> re-audit the same source fingerprints
  -> compile <= local_serving_max_chars memory surface
  -> append exactly once to the main-model request
  -> AstrBot main model answers normally
```

Each non-empty answer-time reconstruction starts one MR Memory provider call. Its
usage and phase timings are written to the ledger. Monetary cost remains
`UNKNOWN_NO_BILLING_EVIDENCE` unless the provider supplies billing evidence; the
main model's incremental cost is also unknown without an on/off measurement.

### Route rules

- Textual identity is never pre-decided by a string resolver. Structured current
  sender/mention/reply accounts are host anchors; nicknames, semantic subjects and
  historical observations remain competing candidates for the Reader.
- Participant activity is fetched only for activity-analysis requests. Incidental
  episode speakers do not trigger per-person history/activity expansion.
- Raw lexical recall always combines bounded trigram FTS/BM25 with a bounded CJK
  bigram substring branch so two-character entities are not structurally invisible.
  Both branches exclude the current request and obey cutoff plus row upper bounds;
  they only retrieve sources and never decide identity or meaning.
- Query-bound EvidenceAtomPack reuse is disabled: the current request changes the
  exact snapshot watermark, so the old cache normally added I/O without hits.
- `CHAT` can receive source-backed learned patterns whose deterministic cues match;
  opening an interaction trace alone never counts as activation.

### Deadline and readiness

- `_execute_local_memory_serving` owns one `local_serving_timeout_seconds`
  abnormal-hang guard for the request-local evidence pipeline. It defaults to 180
  seconds and is configurable from 1 to 600 seconds. It is not a response-latency
  gate: runtime readiness, per-group service opening, interaction tracing, experiment
  setup, snapshot, direct/full retrieval, materialization, compilation, source audit,
  and reconstruction-ledger durations are recorded separately so slow work remains
  observable without being converted into absent memory. No alternate source or
  background result is used. Only the final experiment status write is outside that
  guard so a terminal result can still be persisted after expiry.
- A timeout or local error is logged and injects nothing. AstrBot continues its own
  configured reply; MR Memory does not select another provider, data source, cache
  approximation, or substitute answer.
- A cancelled native SQLite or embedding call can finish in its executor thread;
  the request does not use its late result. Terminal failure persistence is awaited
  outside the retrieval hang guard and records the last completed stage.
- Production deployments that require semantic retrieval can explicitly set
  `embedding_preload_on_startup=true`. A request waits for the same startup task
  within the 180-second guard instead of treating an in-progress preload as missing
  memory. A completed preload failure remains an explicit error; preload is not
  enabled by default because its memory cost is deployment-specific.
- Repeated hooks for the same source/query join one local task. A different query
  cancels the obsolete task. Retained outcomes are bounded and cleaned after send.
- Full retrieval is not protected by a plugin-wide lock. SQLite keeps its own
  connection lock around each database operation, while local embedding uses a fair,
  cancellation-safe inference slot. Passage indexing releases that slot between
  configured batches so an online query is not trapped behind an entire maintenance
  batch.
- Alias observations are write-side materialized and snapshot-indexed. Candidate
  coverage is explicit; incomplete coverage forbids `UNIQUE_ALIAS`.
- Source validation reads cited rows in one bounded query and performs fail-closed
  pre/post comparison of `revision_no` and `content_sha256`. An edit during the
  Reader call cannot mix old text with new relation metadata.

## Envelope contract

`MR_MEMORY_EVIDENCE_ATOM_PACK_V1` contains a strict global source catalog plus
source-key views for identity candidates, semantic/feedback evidence, lexical hits,
episodes, history, activity and recent context. Raw payload appears once even when
several views reference the same message. `evidence-reader.compact-host-speaker` exposes
canonical identifiers as `sN`/`pN`; the model returns semantic fields only and the
host injects snapshot/revision/hash/validation fields before strict parsing.

The compiler applies these rules:

1. Reply anchors are mandatory. Every present, query-relevant stratum (person,
   semantic, lexical, episode, history, recent, and activity only for activity
   questions) first competes for one minimum-coverage source. A source shared by
   several still-uncovered strata wins by marginal coverage; remaining slots use
   the stable relevance order. Repeated aliases or lexical hits therefore cannot
   erase the episodic and recent layers.
2. The source cap is global and deterministic. Available/selected counts are
   recorded per stratum. Truncation disables
   `SEMANTIC_NONE`; incomplete identity candidate coverage also disables
   `UNIQUE_ALIAS`.
3. Derived views are pruned to selected sources. Conflicting immutable payloads for
   a selected source fail closed instead of choosing one copy.
4. The compiler derives two independent provenance relations from the selected
   compact pack. `participant_source_keys` is the identity/alias evidence
   allowlist: each candidate may cite only its explicit alias observations,
   participant-scoped messages or same-record sender sources, never a sibling
   candidate's or semantic subject's sources. `participant_speaker_source_keys`
   is stricter and comes only from a record carrying both `source_key` and
   `sender_participant_key`. A mention or third-party statement can therefore be
   evidence *about* a resolved subject without becoming that subject's direct
   speech. These maps constrain provenance; the Reader still performs joint
   natural-language identity resolution.
5. Temporal adjacency is not reply. Bot text is not independent human truth.
6. Reader output is one unfenced JSON object. Unknown aliases, host-owned fields,
   missing provenance and invalid state coupling are terminal protocol errors; the
   host does not silently normalize or repair them.

True source keys stay host-side for audit/feedback traces and become `s1`, `s2`, …
inside model-visible JSON. Source text is explicitly untrusted data, not instruction.

The source cap bounds distinct raw messages, not characters or upstream retrieval
work. Long selected messages can still produce a large Reader prompt, and episode,
graph and exact-vector candidate expansion occurs before pack compaction. The
current raw lexical path is bounded and batched, while complete retrieval remains
data-dependent; measurements must label the stage they cover rather than reporting
raw search latency as end-to-end retrieval latency.

## Feedback attribution

Creating a trace does not activate a hypothesis or edge. Only IDs retained in the
final injected envelope may be recorded as presented, and retrieval assigns no
fixed relevance score or positive credit.

After AstrBot reaches `after_message_sent`, the plugin records the exact compressed
brief and presented IDs that were injected. This hook is not a downstream delivery
acknowledgement; the message adapter is outside the plugin dependency boundary.
Later feedback can target the presented path without treating presentation as proof
of use or correctness.

## Background semantic work

The configured MR Memory provider serves bounded background construction/feedback
maintenance and the single online resident Reader. L3 ECCR, provider repair and
`mr_consult_subconscious` are not part of the online route. The consult tool is
deactivated and returns an explicit removal error if stale framework state invokes
it.

Background construction and the online Reader have separate thinking controls.
Background distillation may keep full reasoning enabled; the resident Reader
defaults to thinking disabled because its bounded evidence contract requires a
visible JSON result, and hidden reasoning must not consume the response budget or
answer latency. This changes only generation options for the same single provider
call; it is not a retry, alternate model or degraded route.

Background work uses per-group budgets and bounded queues, keeps provider failures
as terminal failures, cannot block/replace AstrBot's main reply, and never turns
operational failure into semantic absence. Provider, validation, budget, and
interrupted-checkpoint failures are not requeued by a timer or plugin reload.

## Storage and observability

Keep source/graph revisions, identity evidence, feedback proposals, administrator
decisions, and bounded traces. Do not persist hidden reasoning, signed URLs,
attachment blobs, credentials, or duplicated full prompts.

Local observations distinguish compilation, actual injection, after-send hook,
missing downstream delivery acknowledgement, zero query-time memory-provider usage, unknown main
model incremental cost, envelope size/source count/truncation, and local elapsed
time. `COMPLETED` describes only the local stage; it does not certify answer quality
or downstream delivery. Quality requires case output plus human review, and latency claims
must name their measurement boundary.
