# Changelog

## 0.17.1

- Restore the 0.16 answer-time inference structure as a deliberately weak,
  measurable baseline: host prefetch, one full-reasoning private-LLM semantic
  decision, and conditional escalation into the bounded graph-tool Agent loop.
  Embedding and host code no longer make the final semantic relevance decision.
- Retain 0.17's exact public-brief/source provenance, accepted-feedback graph
  materialization, clickable run details, end-to-end timing, and in-flight task
  drain during plugin hot reload.
- Correct the configuration and console language so `every_request`, its Token
  budget and observed latency describe the network LLM call that actually occurs.
- Add a resumable, provenance-bound and cutoff-audited research pilot that compares deterministic cache,
  the deliberately weak 0.16 one-pass gate, and a full read-only MR tool loop on
  identical candidates, with isolated SQLite clones, durable usage accounting,
  source-key gold audits, and a paper-style evidence report.

## 0.17.0

- Serve ordinary answer-time memory from the already distilled, locally embedded
  working graph; the main AstrBot LLM remains the relevance gate and automatic
  recall no longer waits for or bills a second network LLM. Manual consultation
  retains the independent provider's deep multi-step traversal.
- Persist each new public memory brief, its exact cited source keys, candidate
  ledger and interaction trace. The console's recent-call rows now open a focused
  provenance graph showing evidence, generated claims, feedback hypotheses and
  graph mutations; legacy rows explicitly disclose when only evidence can be
  reconstructed.
- Materialize every accepted new feedback behavior as an evidence-bound plastic
  cue/scenario-to-behavior path when the model omits its optional graph mutation.
  Model attribution and semantics remain authoritative; the host fallback only
  prevents an accepted decision from becoming an isolated row.
- Record true local end-to-end recall latency even when no provider usage event
  exists, and distinguish the new local working-memory path in runtime history.
- Drain active interaction and reconstruction tasks before a plugin hot reload
  closes their SQLite handles, so a reload cannot cut through an in-flight main
  reply or manual deep consultation.

## 0.16.3

- Normalize the legacy maintenance terminal state `COMPLETED` to `DONE` so a
  deduplicated feedback job can be scheduled again after hot reload.
- Read experiment statuses case-insensitively, keep in-flight calls distinct
  from failures, and report end-to-end wall latency even when a timed-out call
  has no provider usage record.
- Recover a terminal structured object from provider output or terminal
  reasoning; when strict validation still fails, make one protocol-only repair
  call while preserving full reasoning for the primary semantic decision.
- Reuse an active host-owned relation definition during ordinary edge upserts;
  only an explicit relation revision may create a new schema version.
- Show actual failure types and running calls in the runtime console instead of
  labeling every uppercase completed run as failed.
- Preserve validation details and model/host latency in recent-call diagnostics;
  cancelled housekeeping jobs no longer appear as live operational errors.
- Accept the runtime packet's nested proposal identity when deriving feedback
  evidence allowlists; explicitly reopen failed feedback batches while proposals
  remain, and release worker leases immediately on plugin hot reload.
- Replace serial attribution, per-item Harrier association lookup, and synthesis
  with one full-reasoning feedback decision. Host validation still bounds trace,
  account, evidence, edge, and group identity.

## 0.16.2

- Atomically validate distillation source snapshots, graph writes, coverage
  records, and processing checkpoints so edits or recalls cannot commit stale
  memory.
- Fail open instead of queueing concurrent automatic wakes for one group, and
  keep token-budget checks inside that serialization boundary.
- Derive reconstruction and feedback allowlists from the bounded JSON actually
  delivered to the private model; non-brief decisions cannot activate paths.
- Persist low-traffic distillation deadlines across hot reloads and record
  timestamp-only source corrections as first-class revisions.
- Use one shared version constant across AstrBot registration, runtime status,
  extractor metadata, and package metadata.

## 0.16.1

- Keep graph-neighbor inspection local so selecting a connected entry no longer
  discards an in-progress shortest-path start point.
- Report neighborhood truncation only when that neighborhood actually exceeds
  the canvas limit.
- Replace per-vector SQLite visibility probes with one bounded query per memory
  class, preventing semantic retrieval from starving live capture and graph reads.
- Cache validated group scope bindings in memory instead of reacquiring the
  shared SQLite lock for every incoming message.
- Retry one empty provider completion for reconstruction and feedback stages,
  with each attempt recorded separately in the token ledger.
- Recover only expired maintenance leases when another storage handle opens the
  same database, so read-oriented tooling cannot reclaim a live worker job.

## 0.16.0

- Turn the memory graph into a query-first analysis surface: search the full
  analyzable graph, focus a result, and expand its one-to-three-hop neighborhood.
- Add shortest-path inspection between any two visible memory entries and list
  every relation of the selected node with direct navigation to its neighbor.
- Add deterministic complex-network measures on the filtered undirected
  projection: density, average degree, weak components, giant-component ratio,
  clustering coefficient, reciprocity, sampled path length, degree distribution,
  and k-core decomposition.
- Add server-side node, relation, epistemic-state, degree, k-core, connected-only,
  and giant-component filters so metrics describe the same graph being queried.
- Replace the type-column layout with radial ego/path layouts and a deterministic
  force layout for global structure; label only the focus and structural hubs.

## 0.15.1

- Split online feedback into a compact semantic attribution gate and a second
  learning pass that runs only for attributable feedback.
- Bound feedback evidence packets, retrieve existing graph associations by local
  embedding only when learning is needed, and record both stages separately in
  the runtime ledger.
- Label the staged feedback path clearly in the management console.

## 0.15.0

- Replace automatic reconstruction's open-ended tool loop with one full-reasoning
  semantic decision over a host-prefetched, source-key-bounded evidence packet.
  The model may answer `brief`, return `none`, or explicitly escalate to the old
  deep traversal path; manual consultation still forces deep traversal.
- Microbatch up to six feedback proposals after a short debounce and decide them
  in one full-reasoning call. Host code gathers raw response, feedback, activated
  behavior hypotheses, and graph evidence before the call and strictly validates
  every returned evidence key and mutation.
- Give feedback maintenance its own rolling budget and reset control. Historical
  backfill remains an uncapped one-off ledger, while answer-time reconstruction
  and live construction keep their separate online budget.
- Retain attributable feedback below the activation threshold as `PROVISIONAL`
  instead of dropping it. Repeated matching evidence accumulates utility and may
  promote the behavior memory to `ACTIVE`; provisional memory never affects a
  response before promotion.
- Migrate legacy feedback jobs that were incorrectly blocked by the online budget
  into cancelled audit records, then create one clean batch job per group.
- Add precise wakeups for feedback retry and budget expiry instead of waiting for
  the daily construction sweeper.
- Add privacy-safe 24-hour latency, token, outcome and queue summaries plus a
  recent-call ledger for automatic reconstruction and feedback maintenance.
- Redesign the Plugin Page around operator questions: whether memory is working,
  why it ran, how long it took, how much it cost, what it changed, and what is
  waiting. Move the graph, account binding and raw evidence to focused secondary
  views and keep internal identifiers behind technical details.

## 0.14.0

- Remove the AngelEye detector/importer, import commands, web routes, UI, and tests
  from the runtime plugin. Historical ingestion is deployment tooling, not a
  reusable memory feature.
- Stop reclassifying pre-existing plugin data during schema upgrades. `BACKFILL`
  must now be explicit at ingestion, while later live adapter observations always
  win and retain `adapter_live` provenance.
- Treat external history construction as a finite uncapped backlog with a separate
  immutable audit ledger. Keep the rolling limit only for online construction,
  reconstruction, and feedback; add an audited online-budget reset that resumes
  only matching budget-wait jobs.
- Restrict startup recovery, threshold triggers, the periodic sweeper, and automatic
  retries to `LIVE` messages. `BACKFILL` can be selected explicitly through the
  authenticated manual API for a deployment-local one-shot driver, but the plugin
  never turns a finite import into a permanent automatic maintenance policy.
- Replace five-minute budget polling noise with persistent `BUDGET_WAIT` jobs whose
  next eligible time is calculated from the exact rolling-ledger expiry.
- Use reversible batch-local `mN` evidence IDs and `pN` participant IDs in
  construction prompts, omit duplicate plain-text components, and replace verbose
  per-message ignore reasons with a compact ignored-ID ledger. Canonical source and
  participant keys are restored before strict host validation and persistence.
- Record prompt protocol, prompt size, total tokens, and tokens per target for each
  successful construction run. The console now shows ingestion provenance,
  online/backfill progress and ledgers, recent failures, and online-budget reset.
- Keep a provider-portable construction default of 80 messages, while allowing
  long-context deployments to opt into larger batches. Production measurements
  at 320 and 500 targets reached 297 and 259 tokens per target respectively;
  this deployment uses 500 only for its finite history backlog.
- Remove fixed graph-unit quotas that made a 500-target batch silently classify
  416 ordinary messages as ignored. Output cardinality now scales with the target
  set, while strict evidence, identity, and host validation remain unchanged.
- Stop imposing the old 32k completion ceiling on DeepSeek V4 construction. The
  deployment can use its configured 384k limit, and streamed progress plus actual
  prompt/token/latency metadata are recorded without storing hidden reasoning.

## 0.13.0

- Split each group's private-LLM ledger into online and one-off history-backfill
  classes. Only an external migration tool may explicitly label data `BACKFILL`;
  plugin upgrades never reinterpret previously captured messages.
- Keep live construction ahead of historical backlog and never mix the two classes
  in one distillation checkpoint. A later idempotent history sync cannot demote an
  already live-captured message back into the history queue.
- Reserve one normal call of headroom before either budget can start more work, so
  a final ordinary batch does not cross the configured ceiling.
- Move feedback maintenance onto a dedicated worker and preserve proposal-specific
  dedupe keys in the periodic sweeper, preventing historical construction from
  starving feedback or generating duplicate jobs.
- Reject the generic word `可以` as a standalone cheap feedback signal while
  retaining explicit positive and corrective follow-ups for the private semantic
  gate.
- Bound every private tool result and the initial candidate set as valid JSON,
  reduce broad repeated-media evidence, and instruct the agent not to repeat
  identical broad calls. This removes the observed multi-turn context explosion.
- Pin sensitive OpenAI/httpcore SDK namespaces to INFO because their DEBUG request
  dump can contain private group evidence even when AstrBot's configured level is
  INFO.
- Expose per-scope online/backfill usage and pending counts in the console, and
  persist bounded exception details for future construction/reconstruction audits.

## 0.12.1

- Initialize tool visibility, maintenance workers, provider checks, and saved
  distillation queues on both cold startup and zero-restart plugin reload.
- Discover existing per-group databases after reload so pending historical
  construction resumes without waiting for a new message in each group.
- Distinguish the full OneBot group inventory from AngelEye's on-demand cache;
  the console now labels the number of materialized MR databases separately
  from the actual joined-group count and reports how many groups have no cached
  history source.
- Give background construction and feedback maintenance their own 300-second
  timeout while keeping the interactive reconstruction timeout at 45 seconds.
- Keep DeepSeek V4 construction thinking enabled by default and consume its stream
  so long reasoning calls are not mistaken for a dead non-streaming request. The
  setting remains explicit for diagnostics instead of silently downgrading the model.
- Bound construction to 40 source messages and a non-constraining 32768 output-token
  safety ceiling; concise graph-unit counts and the per-group daily ledger control
  cost while normal calls stop naturally.
- Work around AstrBot 4.27.1's dropped OpenAI-provider generation kwargs inside
  the plugin, without patching AstrBot core or changing the selected Provider.
- Retry terminal maintenance jobs only on explicit runtime bootstrap, so a
  fixed deployment recovers old failures without creating an infinite sweep loop.
- Requeue messages that exhausted their three construction attempts only at that
  same explicit reload boundary, including small groups with no ordinary pending
  messages left.
- Log only response lengths, token counts, and elapsed time for construction so
  provider-mode failures are diagnosable without exposing group-chat content.
- Retry deterministic JSON/evidence validation once with a bounded repair request;
  strict host validation remains unchanged and repeated blind maintenance retries
  are avoided.
- Deterministically discard isolated invalid optional graph units, re-run the same
  strict validator, and audit any newly uncovered raw evidence instead of failing
  an otherwise useful 40-message batch.
- Verify the exact 40-message production prompt with thinking enabled: the model
  stopped naturally after 22,094 completion tokens and about 200 seconds; the host
  then rejected one unsafe participant binding without weakening identity rules.

## 0.12.0

- Make the local embedding backend and model first-class settings instead of
  hiding the selected model behind a boolean switch.
- Expose the actual runtime triggers: automatic wake on every main-LLM request
  versus tool-only consultation, message-count construction threshold, maximum
  construction delay, feedback attribution window, and the high-level consult
  bridge.
- Correct UMO guidance to use the platform *instance* ID rather than the adapter
  type. An empty scope list explicitly means all groups.
- Add a read-only AngelEye history detector and a background, idempotent,
  group-isolated import flow. The console shows source coverage, target counts,
  platform mapping, progress, malformed-row skips, and the LLM cost boundary
  before starting.
- Show the effective scope, wake, construction, feedback, and embedding policies
  in the console instead of requiring users to infer them from hidden defaults.

## 0.11.0

- Persist explicit `HYPOTHESIS`, `SUPPORTED`, `CONTESTED`, and `CONFIRMED`
  states plus an uncertainty note on learned association edges. A repeated upsert
  cannot silently overwrite the epistemic state; later evidence must use the
  auditable `revise_edge` operation.
- Let ordinary incremental construction create new group-local association
  hypotheses and revise existing ones, so semantic learning is not limited to
  feedback on a Bot response. Competing interpretations remain separate graph paths.
- Add a bounded per-group repeated-media index that stores only adapter reference
  hashes, counts, sender cardinality, and a tiny source-key reservoir. No media bytes,
  URLs, OCR, captions, or automatic vision calls are retained.
- Give the private agent a read-only repeated-media context tool and explicitly ban
  visual inference from opaque hashes. The console renders unresolved association
  paths as dashed lines and exposes their uncertainty in the inspector.
- Reduce the normal settings surface to seven operational choices. Low-level tuning
  remains schema-backed for compatibility but is hidden from the dashboard.

## 0.10.0

- Add a group-scoped plastic association graph for learned meanings, symbols,
  behaviors, preferences, procedures, and traversal paths while keeping account
  identity and raw provenance in the deterministic truth layer.
- Let the private LLM register and version relation types, and propose evidence-bound
  edge upserts, reinforcement, inhibition, retirement, relation revision, and
  reversible node merges under host validation.
- Treat local embedding distance only as a candidate-generation prior. Run a bounded
  semantic tick for every main-LLM request once the group has graph memory; the
  private LLM owns relevance and stopping, with the deterministic early-stop gate
  retained only as an optional optimization.
- Persist bounded subconscious operational state and maintenance jobs so restarts do
  not erase focus, active edge IDs, or queued work; hidden reasoning is never stored.
- Record plastic-edge activation in observable interaction traces and assign feedback
  credit only to paths that actually influenced the eligible response. A graph
  mutation is rejected until its feedback proposal is host-committed.
- Extend the authenticated console, dashboard metrics, runtime ledger, and regression
  suite for dynamic relation and plastic graph inspection.
- Add a real-history local-semantics A/B harness with self-contained SVG reports and
  complete per-call token accounting for the “好女孩” and “阿拉蕾” cases.

## 0.9.0

- Add truth-layer Participant identities keyed by group scope, platform, and account
  ID, with time-bounded alias history, exact ambiguity handling, structured mention
  and reply relations, and administrator-confirmed aliases without account merging.
- Capture visible Bot output as ordinary `BOT` evidence linked by `RESPONDS_TO`, and
  handle platform recalls, edits, revisions, derived-memory invalidation, and
  self-service account erasure with future-capture suppression.
- Replace recent-window construction with per-message checkpoints, oldest-first
  batches, overlap context, retry bounds, content-hash verification, stable
  evidence-set episode keys, interrupted-batch recovery, and an explicit
  cited-or-ignored coverage ledger.
- Replace free-form person memory with structured claims that separate speaker and
  subject, validate exact evidence spans, support multiple sources, mark epistemic
  state, quarantine high-risk or uncertain claims, and implement conflict,
  supersede, retract, stale, and revision states. Quarantined claims remain outside
  automatic retrieval until independent-source promotion.
- Compute authoritative episode time from source messages and rebuild topic summaries
  from their currently linked active episodes.
- Add owner-type vector quotas, a minimum similarity threshold, a cheap history-intent
  gate, and the validated host evidence stop gate to the runtime reconstruction loop.
- Require the private reconstruction agent to return machine-validated
  claim/source/conflict/unresolved JSON; every item cites visited evidence, and
  truncation occurs only at complete structural units.
- Move automatic construction and feedback maintenance to a bounded background queue;
  add a lexical/reply feedback shutter and a per-group rolling private-token budget.
- Split optional Sentence Transformers/Harrier dependencies from the default
  FastEmbed install.
- Extend the authenticated console with Participant nodes, identity metrics, alias
  history, ambiguity reporting, pending checkpoints, and administrator alias binding.
- Add AstrBot 4.27.1 runtime-contract CI and truth-v2 regression coverage.

## 0.8.0

- Add a persistent, inspectable interaction graph for main-agent requests, tool
  actions, visible responses, later feedback, and prospective hypotheses without
  storing hidden chain-of-thought.
- Add a private feedback-maintenance agent with evidence inspection, scoped
  hypothesis lookup, and one atomic host-validated mutation transaction.
- Add signed backward utility assignment while keeping factual evidence confidence
  independent from behavioral feedback.
- Add explicit `always` and `semantic` activation modes, a lexical fast gate, and
  private-agent semantic activation for paraphrases.
- Add configurable commit thresholds, strict group/sender/cutoff checks, bounded
  media-redacted evidence, proposal binding, and fail-open runtime hooks.
- Add decay, dormancy, bounded active views, and reversible hypothesis merges
  without deleting provenance.
- Extend the AstrBot Plugin Page graph with Action--Feedback--Prospective Hypothesis
  nodes and add a strict real-history feedback A/B harness with token accounting.

## 0.7.0

- Add strict historical cutoffs to graph search and all seven traversal tools.
- Add reverse-window masked-call construction and controlled A/B tooling for real
  retained group calls without future leakage.
- Add privacy-minimized experiment, token-usage, and reconstruction-step ledgers.
- Record full private-agent aggregate usage at runtime instead of counting only
  the final response, and expose recent per-group totals through `/mrmem usage`.
- Validate a deterministic host evidence gate that stops needless graph browsing
  once a high-score episode is verified against raw source messages.

## 0.6.0

- Add a plugin-owned Sentence Transformers backend alongside FastEmbed/ONNX.
- Preserve checkpoint dtype with `dtype=auto` and support query-only named
  prompts required by asymmetric models such as Harrier.
- Bound Sentence Transformers batch size and maximum sequence length for
  low-memory deployments.
- Add an opt-in startup preload probe for deployment resource verification.

## 0.5.0

- Add an AstrBot-native Plugin Page console under `pages/console`.
- Add authenticated Web APIs for scope overview, graph inspection, source-message
  search, episode evidence, and explicit distillation.
- Visualize Cue--Tag--Episode, Person--Aspect--Semantic, and Topic--Episode links.
- Persist and verify each physical SQLite database's group-scope identity.
- Keep console scope selection opaque and server-resolved; raw UMO values are never
  accepted as API routing inputs.

## 0.4.0

- Reproduce the paper's validated episode, cue/tag, semantic-memory, and topic
  construction path.
- Add a plugin-owned local FastEmbed/ONNX backend and a dependency-free hash
  backend for offline tests; no AstrBot Embedding Provider or remote embedding
  API is used.
- Initialize reconstruction from vector-matched cues, episodes, and topics.
- Add `/mrmem distill` for explicit per-group graph construction.
- Inject dynamic memory evidence as temporary user content on AstrBot 4.27.
- Add an end-to-end offline reproduction fixture and regression test.

## 0.3.0

- Add a plugin-owned provider, defaulting to
  `deepseek/deepseek-v4-flash`, for memory reconstruction.
- Run the seven graph traversal tools inside a bounded private tool loop.
- Hide low-level traversal tools from the main LLM by default.
- Add automatic pre-request memory briefs and one optional high-level
  `mr_consult_subconscious` bridge tool.
- Skip the provider call entirely when the current session has no graph units.
- Enforce group isolation with event-derived scopes, one SQLite file per group,
  scoped SQL joins, and cross-group regression tests.

## 0.2.0

- Replace the passive all-in-one LLM search tool with the seven typed memory
  traversal tools from MRAgent Table 4.
- Add Cue--Tag--Episode, Person--Aspect--Semantic, and Topic--Episode schema.
- Enable the traversal toolkit by default when the plugin is explicitly loaded;
  development isolation remains a deployment boundary.

## 0.1.0

- 建立默认关闭的 AstrBot 事件适配层。
- 建立 SQLite 真值层、FTS5 索引和图记忆预留表。
- 增加幂等 JSONL 离线回放工具。
- 增加群隔离、全文检索和回放测试。
