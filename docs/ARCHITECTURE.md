# Architecture

MR Memory keeps the existing per-group truth store and rewrites the answer-time
path as a local evidence service. The plugin does not call a private reader,
alternate provider, repair model, or deep-research model while `/chat` is waiting.
AstrBot's already-selected main model is the only query-time model: it receives one
bounded, source-backed JSON envelope and decides how to use that evidence.

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
  -> deterministic current/reply identity resolution
  -> choose local route
       direct identity/activity question: bounded deterministic local query
       other CHAT: lexical + local embedding + bounded graph/raw expansion (<= 3000 chars)
       explicit memory question: same local retrieval with the larger configured budget
  -> audit every selected source against the frozen snapshot
  -> compile <= local_serving_max_chars JSON envelope
  -> append exactly once to the main-model request
  -> AstrBot main model answers normally
```

Answer-time MR Memory provider calls, provider tokens, and external API cost are
exactly zero. The envelope adds input to the main model, so its incremental Token,
latency, and billing impact remain `UNKNOWN_NOT_MEASURED` unless actual host usage
or a controlled on/off measurement exists.

### Route rules

- Ordinary chat uses the full local serving plane, so enabling the plugin can
  actually affect answers without adding a second query-time model. Its envelope
  remains capped at 3000 characters.
- Explicit identity and participant-activity questions use the smaller
  deterministic route when the target can be bound to snapshot evidence.
- Strong recall cues (`回忆`, `记得`, `之前`, `历史`, `谁说过`, `记忆`) select the
  larger explicit-memory envelope budget; they do not enable a remote reader.
- A quoted “这些人是谁” request resolves names from both the current query and the
  frozen reply text, while preserving multiple-account ambiguity.
- `CHAT` can receive source-backed learned patterns whose deterministic cues match;
  opening an interaction trace alone never counts as activation.

### Deadline and readiness

- `_execute_local_memory_serving` owns the single
  `local_serving_timeout_seconds` deadline. It bounds snapshot capture, retrieval,
  validation, and compilation; feedback-trace setup and terminal ledger persistence
  are outside that retrieval deadline.
- A timeout or local error is logged and injects nothing. AstrBot continues its own
  configured reply; MR Memory does not select another provider, data source, cache
  approximation, or substitute answer.
- A cancelled native SQLite or embedding call can finish in its executor thread;
  the request does not use its late result. Terminal failure persistence is awaited
  outside the retrieval deadline and records the last completed stage.
- Production deployments that require semantic retrieval can explicitly set
  `embedding_preload_on_startup=true`. A query fails clearly while that preload is
  in progress or after preload failure; preload is not enabled by default because
  its memory cost is deployment-specific.
- Repeated hooks for the same source/query join one local task. A different query
  cancels the obsolete task. Retained outcomes are bounded and cleaned after send.

## Envelope contract

`mr-local-serving.v1` contains current-event identity, bounded alias resolution,
source-backed claims/conflicts/unresolved items, selected graph connections,
applicable learned patterns, activity/reply context, short source aliases, raw
source records, truncation, and stage-specific cost accounting.

The compiler applies these rules:

1. A row with at least one retained source remains visible when other citations do
   not fit; it exposes total source count and `sources_truncated=true`.
2. A source absent from actual packet records never becomes a placeholder.
3. Conflicts and unresolved items receive capacity before extra ordinary claims.
   Tight profiles remove derived graph connections before source-backed brief rows.
4. Sources are allocated round-robin across evidence items/episodes.
5. Identity/ambiguity lists shrink with the profile and expose `*_truncated=true`.
6. Temporal adjacency is not reply. Bot text is not independent human truth.
   Anonymous speaker tokens or matching nicknames are not stable accounts.
7. Top-level `retrieval.truncated` is true whenever any item, text, alias, source,
   candidate, or profile was reduced.

True source keys stay host-side for audit/feedback traces and become `s1`, `s2`, …
inside model-visible JSON. Source text is explicitly untrusted data, not instruction.

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

The configured MR Memory provider remains available only to bounded background
construction and feedback maintenance. Legacy L2 certificates, L3 ECCR,
query-bound reconstruction, provider repair, and `mr_consult_subconscious` are not
part of the online route. The consult tool is deactivated and returns an explicit
removal error if stale framework state invokes it.

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
