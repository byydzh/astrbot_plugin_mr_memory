const state = {
  overview: null,
  scopeId: "",
  graph: null,
  participants: [],
  selectedNodeId: "",
  graphQuery: "",
  graphFocusId: "",
  graphPathSource: "",
  graphPathTarget: "",
  runDetail: null,
  runDetailNodeId: "",
  activeSection: "runtime",
  loadedSections: new Set(),
  visibleTypes: new Set([
    "participant", "cue", "episode", "semantic", "topic",
    "plastic", "action", "feedback", "hypothesis",
  ]),
  transform: { x: 0, y: 0, scale: 1 },
  virtualSize: { width: 1120, height: 680 },
  isPanning: false,
  panStart: null,
};

const elements = {
  connectionDot: document.getElementById("connection-dot"),
  connectionLabel: document.getElementById("connection-label"),
  healthBanner: document.getElementById("health-banner"),
  healthTitle: document.getElementById("health-title"),
  healthCopy: document.getElementById("health-copy"),
  healthLabel: document.getElementById("health-label"),
  metricRecallLatency: document.getElementById("metric-recall-latency"),
  metricRecallCaption: document.getElementById("metric-recall-caption"),
  metricRecallTokens: document.getElementById("metric-recall-tokens"),
  metricFeedbackLatency: document.getElementById("metric-feedback-latency"),
  metricFeedbackCaption: document.getElementById("metric-feedback-caption"),
  metricFeedbackResults: document.getElementById("metric-feedback-results"),
  metricFeedbackQueue: document.getElementById("metric-feedback-queue"),
  metricFeedbackQueueCaption: document.getElementById("metric-feedback-queue-caption"),
  metricScopes: document.getElementById("metric-scopes"),
  metricMessages: document.getElementById("metric-messages"),
  metricEpisodes: document.getElementById("metric-episodes"),
  metricSemantics: document.getElementById("metric-semantics"),
  metricParticipants: document.getElementById("metric-participants"),
  metricPending: document.getElementById("metric-pending"),
  metricEmbeddings: document.getElementById("metric-embeddings"),
  metricStorage: document.getElementById("metric-storage"),
  policyScope: document.getElementById("policy-scope"),
  policyWake: document.getElementById("policy-wake"),
  policyDistill: document.getElementById("policy-distill"),
  policyFeedback: document.getElementById("policy-feedback"),
  policyEmbedding: document.getElementById("policy-embedding"),
  policyBudget: document.getElementById("policy-budget"),
  policyFeedbackBudget: document.getElementById("policy-feedback-budget"),
  scopeSelect: document.getElementById("scope-select"),
  scopeSummary: document.getElementById("scope-summary"),
  scopeMeta: document.getElementById("scope-meta"),
  scopeErrors: document.getElementById("scope-errors"),
  refreshBtn: document.getElementById("refresh-btn"),
  resetBudgetBtn: document.getElementById("reset-budget-btn"),
  resetFeedbackBudgetBtn: document.getElementById("reset-feedback-budget-btn"),
  distillBtn: document.getElementById("distill-btn"),
  graphLimit: document.getElementById("graph-limit"),
  graphDepth: document.getElementById("graph-depth"),
  graphSearchForm: document.getElementById("graph-search-form"),
  graphSearchInput: document.getElementById("graph-search-input"),
  graphSearchResults: document.getElementById("graph-search-results"),
  graphOverviewBtn: document.getElementById("graph-overview-btn"),
  graphStructureFilter: document.getElementById("graph-structure-filter"),
  graphDegreeFilter: document.getElementById("graph-degree-filter"),
  graphCoreFilter: document.getElementById("graph-core-filter"),
  graphEpistemicFilter: document.getElementById("graph-epistemic-filter"),
  graphRelationFilter: document.getElementById("graph-relation-filter"),
  graphPathStatus: document.getElementById("graph-path-status"),
  graphPathClear: document.getElementById("graph-path-clear"),
  graphViewKicker: document.getElementById("graph-view-kicker"),
  graphViewTitle: document.getElementById("graph-view-title"),
  graphMetricNodes: document.getElementById("graph-metric-nodes"),
  graphMetricEdges: document.getElementById("graph-metric-edges"),
  graphMetricDegree: document.getElementById("graph-metric-degree"),
  graphMetricDensity: document.getElementById("graph-metric-density"),
  graphMetricComponents: document.getElementById("graph-metric-components"),
  graphMetricGiant: document.getElementById("graph-metric-giant"),
  graphMetricClustering: document.getElementById("graph-metric-clustering"),
  graphMetricCore: document.getElementById("graph-metric-core"),
  fitGraphBtn: document.getElementById("fit-graph-btn"),
  graphStage: document.getElementById("graph-stage"),
  graphSvg: document.getElementById("memory-graph"),
  graphViewport: document.getElementById("graph-viewport"),
  graphEdges: document.getElementById("graph-edges"),
  graphEdgeLabels: document.getElementById("graph-edge-labels"),
  graphNodes: document.getElementById("graph-nodes"),
  graphEmpty: document.getElementById("graph-empty"),
  graphLoading: document.getElementById("graph-loading"),
  graphEmptyTitle: document.getElementById("graph-empty-title"),
  graphEmptyCopy: document.getElementById("graph-empty-copy"),
  graphCaption: document.getElementById("graph-caption"),
  inspectorKicker: document.getElementById("inspector-kicker"),
  inspectorHeading: document.getElementById("inspector-heading"),
  inspector: document.getElementById("inspector-content"),
  messageForm: document.getElementById("message-search-form"),
  messageQuery: document.getElementById("message-query"),
  messageSender: document.getElementById("message-sender"),
  messageBody: document.getElementById("message-table-body"),
  participantStatus: document.getElementById("participant-status"),
  participantBody: document.getElementById("participant-table-body"),
  runtimeTableBody: document.getElementById("runtime-table-body"),
  runDetailDialog: document.getElementById("run-detail-dialog"),
  runDetailClose: document.getElementById("run-detail-close"),
  runDetailTitle: document.getElementById("run-detail-title"),
  runDetailSubtitle: document.getElementById("run-detail-subtitle"),
  runDetailMetrics: document.getElementById("run-detail-metrics"),
  runDetailWarning: document.getElementById("run-detail-warning"),
  runDetailGraphCaption: document.getElementById("run-detail-graph-caption"),
  runDetailFit: document.getElementById("run-detail-fit"),
  runDetailGraphStage: document.getElementById("run-detail-graph-stage"),
  runDetailGraph: document.getElementById("run-detail-graph"),
  runDetailViewport: document.getElementById("run-detail-viewport"),
  runDetailEdges: document.getElementById("run-detail-edges"),
  runDetailNodes: document.getElementById("run-detail-nodes"),
  runDetailLoading: document.getElementById("run-detail-loading"),
  runDetailEmpty: document.getElementById("run-detail-empty"),
  runDetailEmptyTitle: document.getElementById("run-detail-empty-title"),
  runDetailEmptyCopy: document.getElementById("run-detail-empty-copy"),
  runDetailMemoryState: document.getElementById("run-detail-memory-state"),
  runDetailIdentityBasis: document.getElementById("run-detail-identity-basis"),
  runDetailPayloadBasis: document.getElementById("run-detail-payload-basis"),
  runDetailInspectorTitle: document.getElementById("run-detail-inspector-title"),
  runDetailInspectorContent: document.getElementById("run-detail-inspector-content"),
  runDetailLedgerSummary: document.getElementById("run-detail-ledger-summary"),
  runDetailLedgerJson: document.getElementById("run-detail-ledger-json"),
  runDetailResultJson: document.getElementById("run-detail-result-json"),
  queuePending: document.getElementById("queue-pending"),
  queueProvisional: document.getElementById("queue-provisional"),
  queueBudgetWait: document.getElementById("queue-budget-wait"),
  queueOldest: document.getElementById("queue-oldest"),
  queueExplanation: document.getElementById("queue-explanation"),
  aliasBindForm: document.getElementById("alias-bind-form"),
  aliasAccountId: document.getElementById("alias-account-id"),
  aliasValue: document.getElementById("alias-value"),
  distillDialog: document.getElementById("distill-dialog"),
  distillForm: document.getElementById("distill-form"),
  distillLimit: document.getElementById("distill-limit"),
  distillCancel: document.getElementById("distill-cancel"),
  distillConfirm: document.getElementById("distill-confirm"),
  toastRegion: document.getElementById("toast-region"),
};

const SVG_NS = "http://www.w3.org/2000/svg";
const typeNames = {
  participant: "账户主体",
  cue: "检索线索",
  episode: "情节记忆",
  semantic: "人物与事实",
  topic: "主题",
  action: "回答与工具",
  feedback: "后续反馈",
  hypothesis: "行为记忆",
  plastic: "群内语义",
};

function svgElement(name, attributes = {}) {
  const node = document.createElementNS(SVG_NS, name);
  Object.entries(attributes).forEach(([key, value]) => {
    node.setAttribute(key, String(value));
  });
  return node;
}

function textElement(name, text, className = "") {
  const node = document.createElement(name);
  if (className) node.className = className;
  node.textContent = text;
  return node;
}

function formatNumber(value) {
  return new Intl.NumberFormat("zh-CN").format(Number(value || 0));
}

function formatDuration(value) {
  const milliseconds = Number(value || 0);
  if (!milliseconds) return "—";
  if (milliseconds < 1000) return `${Math.round(milliseconds)} 毫秒`;
  const seconds = milliseconds / 1000;
  return `${seconds >= 10 ? seconds.toFixed(1) : seconds.toFixed(2)} 秒`;
}

function formatAge(value) {
  const seconds = Math.max(0, Number(value || 0));
  if (!seconds) return "—";
  if (seconds < 60) return `${Math.round(seconds)} 秒`;
  if (seconds < 3600) return `${Math.round(seconds / 60)} 分钟`;
  return `${(seconds / 3600).toFixed(seconds < 10800 ? 1 : 0)} 小时`;
}

function formatWindow(secondsValue) {
  const seconds = Math.max(0, Number(secondsValue || 0));
  if (seconds < 3600) return `${Math.max(1, Math.round(seconds / 60))} 分钟`;
  const hours = seconds / 3600;
  return `${Number.isInteger(hours) ? hours : hours.toFixed(1)} 小时`;
}

function formatBytes(value) {
  let bytes = Number(value || 0);
  const units = ["B", "KB", "MB", "GB"];
  let index = 0;
  while (bytes >= 1024 && index < units.length - 1) {
    bytes /= 1024;
    index += 1;
  }
  const digits = index === 0 ? 0 : bytes >= 100 ? 0 : 1;
  return `${bytes.toFixed(digits)} ${units[index]}`;
}

function formatTime(value) {
  if (value === null || value === undefined || value === "") return "—";
  let date;
  if (typeof value === "number" || /^\d+$/.test(String(value))) {
    const numeric = Number(value);
    date = new Date(numeric > 1e12 ? numeric : numeric * 1000);
  } else {
    const normalized = String(value).includes("T")
      ? String(value)
      : `${String(value).replace(" ", "T")}Z`;
    date = new Date(normalized);
  }
  if (Number.isNaN(date.getTime())) return String(value);
  return new Intl.DateTimeFormat("zh-CN", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  }).format(date);
}

function truncate(value, length = 18) {
  const text = String(value || "");
  return text.length > length ? `${text.slice(0, length - 1)}…` : text;
}

function compactModelName(value) {
  const text = String(value || "");
  return text.includes("/") ? text.split("/").at(-1) : text || "未配置";
}

function showToast(message, kind = "success") {
  const toast = textElement("div", String(message), `toast ${kind}`);
  elements.toastRegion.appendChild(toast);
  window.setTimeout(() => toast.remove(), 4200);
}

function setConnection(status, label) {
  elements.connectionDot.className = `status-dot is-${status}`;
  elements.connectionLabel.textContent = label;
}

async function apiGet(endpoint, params = {}) {
  return window.AstrBotPluginPage.apiGet(endpoint, params);
}

async function apiPost(endpoint, body = {}) {
  return window.AstrBotPluginPage.apiPost(endpoint, body);
}

async function waitForPluginBridge(timeoutMs = 5000) {
  const startedAt = performance.now();
  while (!window.AstrBotPluginPage) {
    if (performance.now() - startedAt >= timeoutMs) {
      throw new Error("AstrBot 插件页面桥接加载超时");
    }
    await new Promise((resolve) => window.setTimeout(resolve, 25));
  }
  return window.AstrBotPluginPage;
}

function selectedScope() {
  return (state.overview?.scopes || []).find(
    (scope) => scope.storage_id === state.scopeId,
  );
}

function renderOverview() {
  const overview = state.overview;
  const totals = overview?.totals || {};
  const runtime = overview?.runtime || {};
  elements.metricScopes.textContent = formatNumber(totals.scopes);
  elements.metricMessages.textContent = formatNumber(totals.messages);
  elements.metricEpisodes.textContent = formatNumber(totals.episodes);
  elements.metricSemantics.textContent = formatNumber(totals.semantic_memories);
  elements.metricParticipants.textContent = formatNumber(totals.participants);
  elements.metricPending.textContent = formatNumber(totals.pending_distillation);
  elements.metricEmbeddings.textContent = formatNumber(totals.embeddings);
  elements.metricStorage.textContent = formatBytes(totals.database_bytes);
  const allowedUmos = runtime.allowed_umos || [];
  elements.policyScope.textContent = allowedUmos.length
    ? `${formatNumber(allowedUmos.length)} 个指定群聊`
    : "全部群";
  elements.policyWake.textContent = !runtime.subconscious_enabled
    ? "关闭"
    : runtime.runtime_wake_mode === "manual_only"
      ? "仅在主模型主动咨询时运行"
      : "每次回答由独立记忆模型判断；证据不足时继续图遍历";
  elements.policyDistill.textContent = runtime.auto_distillation_enabled
    ? `${formatNumber(runtime.auto_distillation_min_pending)} 条立即整理，最迟 ${Math.max(0.5, Number(runtime.maintenance_interval_seconds || 0) / 60)} 分钟一次`
    : "仅在管理员手动操作时整理";
  elements.policyFeedback.textContent = runtime.feedback_learning_enabled
    ? `${formatNumber(runtime.feedback_debounce_seconds)} 秒合并，最多 ${formatNumber(runtime.feedback_batch_size)} 条/次，可关联 ${formatWindow(runtime.feedback_window_seconds)}内的回答`
    : "关闭";
  elements.policyEmbedding.textContent = runtime.embedding_enabled
    ? `${compactModelName(runtime.embedding_model_name)} · ${runtime.embedding_model_loaded ? "正在使用" : "首次检索时加载"}`
    : "关闭";
  elements.policyBudget.textContent =
    runtime.private_daily_token_budget
      ? `${formatNumber(runtime.private_daily_token_budget)} Token / 群 / 24 小时`
      : "不限额";
  elements.policyFeedbackBudget.textContent =
    runtime.feedback_daily_token_budget
      ? `${formatNumber(runtime.feedback_daily_token_budget)} Token / 群 / 24 小时`
      : "不限额";

  const runtimeParts = [];
  runtimeParts.push(runtime.capture_enabled ? "正在记录群消息" : "未记录群消息");
  runtimeParts.push(runtime.feedback_learning_enabled ? "反馈学习已启用" : "反馈学习未启用");
  if (runtime.embedding_enabled) {
    if (!runtime.embedding_dependency_ready) {
      runtimeParts.push("本地语义检索不可用");
    } else if (runtime.embedding_model_loaded) {
      runtimeParts.push("本地语义检索已就绪");
    } else {
      runtimeParts.push("本地语义检索尚未首次使用");
    }
  }
  setConnection("online", runtimeParts.join(" · "));
}

function renderRuntimeHealth(scope) {
  const health = scope?.runtime_health || {};
  const reconstruction = health.reconstruction || {};
  const feedback = health.feedback || {};
  const queue = health.feedback_queue || {};
  const proposalStatus = queue.proposal_status || {};
  const hypothesisStatus = queue.hypothesis_status || {};
  const jobStatus = queue.job_status || {};
  const recallCalls = Number(reconstruction.calls || 0);
  const feedbackCalls = Number(feedback.calls || 0);
  const pending = Number(proposalStatus.pending || 0);
  const provisional = Number(hypothesisStatus.provisional || 0);
  const budgetWait = Number(jobStatus.budget_wait || 0);
  const recallFailures = Number(reconstruction.failed || 0);
  const feedbackFailures = Number(feedback.failed || 0);
  const effectiveFeedback = Number(feedback.committed || 0) + Number(feedback.provisional || 0);
  const decidedFeedback = effectiveFeedback
    + Number(feedback.ignored || 0)
    + Number(feedback.rejected || 0);

  elements.metricRecallLatency.textContent = recallCalls
    ? formatDuration(reconstruction.p50_elapsed_ms)
    : "暂无调用";
  elements.metricRecallTokens.textContent = recallCalls
    ? formatNumber(Math.round(Number(reconstruction.avg_tokens || 0)))
    : "暂无调用";
  elements.metricFeedbackLatency.textContent = feedbackCalls
    ? formatDuration(feedback.p50_elapsed_ms)
    : "暂无调用";
  elements.metricFeedbackResults.textContent = decidedFeedback
    ? `${formatNumber(effectiveFeedback)} / ${formatNumber(decidedFeedback)}`
    : "暂无结果";
  elements.metricFeedbackQueue.textContent = formatNumber(pending);
  elements.metricFeedbackQueueCaption.textContent = pending
    ? `最久已等待 ${formatAge(queue.oldest_pending_age_seconds)}`
    : "当前没有等待分析的反馈";

  elements.queuePending.textContent = formatNumber(pending);
  elements.queueProvisional.textContent = formatNumber(provisional);
  elements.queueBudgetWait.textContent = formatNumber(budgetWait);
  elements.queueOldest.textContent = formatAge(queue.oldest_pending_age_seconds);
  elements.queueExplanation.textContent = budgetWait
    ? "反馈正在等待独立额度恢复；不会被回答前回忆或消息整理挤占。"
    : provisional
      ? "待验证记忆不会直接影响回答；后续同向证据足够时才会启用。"
      : "普通聊天会被忽略；可归因但证据较弱的反馈会先保留为待验证。";

  const runtime = state.overview?.runtime || {};
  const issues = [];
  if (runtime.embedding_enabled && !runtime.embedding_dependency_ready) {
    issues.push("本地语义检索依赖不可用");
  }
  if (recallCalls && Number(reconstruction.p50_elapsed_ms || 0) > 3000) {
    issues.push(`回答前回忆 p50 为 ${formatDuration(reconstruction.p50_elapsed_ms)}`);
  }
  if (feedbackCalls && Number(feedback.p50_elapsed_ms || 0) > 20000) {
    issues.push(`反馈分析 p50 为 ${formatDuration(feedback.p50_elapsed_ms)}`);
  }
  if (budgetWait) issues.push(`${formatNumber(budgetWait)} 个反馈任务等待额度`);
  if (recallFailures) {
    const timeouts = Number(reconstruction.timeouts || 0);
    issues.push(
      `${formatNumber(recallFailures)} 次回答前回忆失败`
      + (timeouts ? `（其中 ${formatNumber(timeouts)} 次超时）` : ""),
    );
  }
  if (feedbackFailures) {
    const timeouts = Number(feedback.timeouts || 0);
    issues.push(
      `${formatNumber(feedbackFailures)} 次反馈分析失败`
      + (timeouts ? `（其中 ${formatNumber(timeouts)} 次超时）` : ""),
    );
  }

  if (issues.length) {
    elements.healthBanner.className = "health-banner is-warning";
    elements.healthLabel.textContent = "需要关注";
    elements.healthTitle.textContent = "记忆正在运行，但实时体验没有达到目标";
    elements.healthCopy.textContent = issues.join("；");
  } else if (!recallCalls && !feedbackCalls) {
    elements.healthBanner.className = "health-banner is-idle";
    elements.healthLabel.textContent = "等待数据";
    elements.healthTitle.textContent = "这个群还没有新的实时调用样本";
    elements.healthCopy.textContent = "出现回答前回忆或后续反馈后，这里会显示真实时延和消耗。";
  } else {
    elements.healthBanner.className = "health-banner is-healthy";
    elements.healthLabel.textContent = "运行正常";
    elements.healthTitle.textContent = "实时记忆链路处于目标范围内";
    const samples = [];
    if (recallCalls) samples.push(`回答前回忆 p95 ${formatDuration(reconstruction.p95_elapsed_ms)}`);
    if (feedbackCalls) samples.push(`反馈分析 p95 ${formatDuration(feedback.p95_elapsed_ms)}`);
    elements.healthCopy.textContent = `${samples.join("；")}。`;
  }

  renderRuntimeCalls(health.recent || []);
}

function renderRuntimeCalls(calls) {
  elements.runtimeTableBody.replaceChildren();
  if (!calls.length) {
    const row = document.createElement("tr");
    const cell = textElement("td", "最近 24 小时还没有实时调用。", "table-empty");
    cell.colSpan = 7;
    row.append(cell);
    elements.runtimeTableBody.append(row);
    return;
  }
  const phaseNames = { reconstruction: "回答前回忆", feedback: "反馈分析" };
  const outcomeNames = {
    useful: "提供了记忆",
    none: "没有相关记忆",
    committed: "已启用",
    provisional: "待验证",
    ignored: "普通聊天",
    rejected: "未采用",
    failed: "失败",
    running: "进行中",
    completed: "已完成",
  };
  const pathNames = {
    fast: "一次判断",
    deep_escalation: "判断后深挖",
    deep_forced: "主动深挖",
    one_pass_batch: "合并判断",
    one_pass_feedback_learning: "一次判断并学习",
    semantic_gate_then_learn: "先判断，必要时学习",
    materialized_local: "本地工作记忆",
    legacy: "旧版多轮",
  };
  calls.forEach((call) => {
    let outcome = String(call.outcome || "");
    if (outcome === "failed") {
      if (call.error_type === "TimeoutError") outcome = "超时";
      else if (call.error_type === "ValueError") outcome = "校验失败";
    }
    const outcomes = outcome
      .split(",")
      .filter(Boolean)
      .map((value) => outcomeNames[value] || value)
      .join("、");
    const row = document.createElement("tr");
    row.className = "runtime-call-row";
    row.tabIndex = 0;
    row.setAttribute("role", "button");
    row.setAttribute("aria-label", `查看 ${phaseNames[call.phase] || call.phase} 调用详情`);
    const outcomeCell = textElement("td", outcomes || "—");
    if (call.error_detail) outcomeCell.title = String(call.error_detail);
    const elapsedCell = textElement("td", formatDuration(call.elapsed_ms));
    elapsedCell.title = [
      `端到端 ${formatDuration(call.elapsed_ms)}`,
      `模型 ${formatDuration(call.llm_elapsed_ms)}`,
      call.first_chunk_ms ? `首段 ${formatDuration(call.first_chunk_ms)}` : "",
    ].filter(Boolean).join(" · ");
    const detailCell = document.createElement("td");
    const detailButton = textElement("button", "查看", "runtime-detail-button");
    detailButton.type = "button";
    detailButton.addEventListener("click", (event) => {
      event.stopPropagation();
      openRunDetail(call.run_id);
    });
    detailCell.append(detailButton);
    row.append(
      textElement("td", formatTime(call.started_at)),
      textElement("td", phaseNames[call.phase] || call.phase),
      outcomeCell,
      elapsedCell,
      textElement("td", formatNumber(call.tokens)),
      textElement("td", pathNames[call.path] || call.path || "—"),
      detailCell,
    );
    row.addEventListener("click", () => openRunDetail(call.run_id));
    row.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        openRunDetail(call.run_id);
      }
    });
    elements.runtimeTableBody.append(row);
  });
}

const memoryAccessNames = {
  read: "读取",
  write: "写入",
  upsert: "写入或更新",
  modify: "修改",
  context: "结构端点",
};

const memoryAccessOrder = ["read", "write", "upsert", "modify", "context"];

const memoryEffectStateNames = {
  RECORDED: "已记录",
  PARTIAL: "部分记录",
  NOT_APPLICABLE: "不适用",
  NO_ACTIVATION: "未激活记忆",
  INCOMPLETE_CAPTURE: "记录不完整",
  UNAVAILABLE_LEGACY: "旧记录不可还原",
};

function normalizedMemoryAccess(item) {
  const raw = Array.isArray(item?.access) ? item.access : [];
  const values = new Set(
    raw.map((value) => String(value || "").trim().toLowerCase())
      .filter((value) => Object.hasOwn(memoryAccessNames, value)),
  );
  return memoryAccessOrder.filter((value) => values.has(value));
}

function memoryAccessText(item) {
  const values = normalizedMemoryAccess(item);
  return values.length ? values.map((value) => memoryAccessNames[value]).join("、") : "未标记";
}

function memoryEffects(detail = state.runDetail) {
  const value = detail?.memory_effects;
  return value && typeof value === "object" ? value : {};
}

function memoryEffectCounts(effects, nodes, edges) {
  const provided = effects?.counts && typeof effects.counts === "object" ? effects.counts : {};
  const items = [...nodes, ...edges];
  const computed = Object.fromEntries(
    memoryAccessOrder.map((access) => [
      access,
      items.filter((item) => normalizedMemoryAccess(item).includes(access)).length,
    ]),
  );
  return Object.fromEntries(memoryAccessOrder.map((access) => {
    if (Object.hasOwn(provided, access) && provided[access] === null) return [access, null];
    const value = Number(provided[access]);
    return [access, Number.isFinite(value) && value >= 0 ? value : computed[access]];
  }));
}

function memoryEffectsAreTruncated(effects) {
  return effects?.truncated === true || effects?.counts?.truncated === true;
}

function formatMemoryEffectCount(value, effects) {
  if (value === null) return "未记录";
  if (memoryEffectsAreTruncated(effects)) {
    return Number(value) > 0 ? `至少 ${formatNumber(value)}` : "未知（已截断）";
  }
  return formatNumber(value);
}

function memoryEntityCount(effects, key, visibleCount) {
  const stateName = String(effects?.state || "").toUpperCase();
  const counts = effects?.counts && typeof effects.counts === "object" ? effects.counts : {};
  const totalKeys = [`total_${key}`, `${key}_total`];
  for (const totalKey of totalKeys) {
    const candidates = [effects?.[totalKey], counts?.[totalKey]];
    for (const candidate of candidates) {
      if (candidate === null) return { value: null, exactTotal: false };
      const parsed = Number(candidate);
      if (Number.isFinite(parsed) && parsed >= 0) return { value: parsed, exactTotal: true };
    }
  }
  if (Object.hasOwn(counts, key)) {
    if (counts[key] === null) return { value: null, exactTotal: false };
    const parsed = Number(counts[key]);
    return Number.isFinite(parsed) && parsed >= 0
      ? { value: parsed, exactTotal: !memoryEffectsAreTruncated(effects) }
      : { value: null, exactTotal: false };
  }
  if (["INCOMPLETE_CAPTURE", "UNAVAILABLE_LEGACY"].includes(stateName)) {
    return { value: null, exactTotal: false };
  }
  const collection = effects?.[key];
  if (!Array.isArray(collection)) return { value: null, exactTotal: false };
  return { value: visibleCount, exactTotal: !memoryEffectsAreTruncated(effects) };
}

function formatMemoryEntityCount(metric, effects) {
  if (metric.value === null) return "未记录";
  if (metric.exactTotal) return formatNumber(metric.value);
  if (memoryEffectsAreTruncated(effects)) {
    return metric.value > 0 ? `至少 ${formatNumber(metric.value)}` : "未知";
  }
  return formatNumber(metric.value);
}

function memoryEffectsMetricValue(effects, nodes, edges) {
  const stateName = String(effects?.state || "").toUpperCase();
  const nodeCount = memoryEntityCount(effects, "nodes", nodes.length);
  const edgeCount = memoryEntityCount(effects, "edges", edges.length);
  if (
    ["INCOMPLETE_CAPTURE", "UNAVAILABLE_LEGACY"].includes(stateName)
    && nodeCount.value === null
    && edgeCount.value === null
  ) return "未记录";
  return `${formatMemoryEntityCount(nodeCount, effects)} / ${formatMemoryEntityCount(edgeCount, effects)}`;
}

function memoryIdentityBasisText(effects) {
  if (effects?.identity_exact === true) return "账本确认身份";
  if (effects?.identity_exact === false) return "身份记录不完整";
  return "身份确认未记录";
}

function memoryPayloadBasisText(effects) {
  const value = effects?.payload_as_of;
  if (value === null || value === undefined || value === "") return "载荷解析时间未记录";
  if (typeof value === "object") {
    const mode = String(value.mode || value.source || "").toUpperCase();
    const at = value.at || value.timestamp;
    if (mode.includes("CURRENT")) return at ? `当前状态解析 · ${formatTime(at)}` : "当前状态解析";
    return at ? `状态解析截至 ${formatTime(at)}` : "载荷解析口径已记录";
  }
  const normalized = String(value).toUpperCase();
  if (normalized.includes("CURRENT")) return "当前状态解析";
  return `状态解析截至 ${formatTime(value)}`;
}

function renderMemoryEffectsBasis(effects) {
  const stateName = String(effects?.state || "").toUpperCase();
  elements.runDetailMemoryState.textContent = memoryEffectStateNames[stateName] || "记录状态未记录";
  elements.runDetailMemoryState.dataset.state = stateName || "UNKNOWN";
  elements.runDetailIdentityBasis.textContent = memoryIdentityBasisText(effects);
  elements.runDetailPayloadBasis.textContent = memoryPayloadBasisText(effects);
}

function memoryEffectsEmptyState(effects, run) {
  const reason = String(effects?.empty_reason || "").trim();
  const stateName = String(effects?.state || "").toUpperCase();
  const stateTitles = {
    PARTIAL: "仅能定位部分记忆操作",
    NO_ACTIVATION: "本次没有激活持久记忆",
    NOT_APPLICABLE: "本次不产生记忆读写",
    INCOMPLETE_CAPTURE: "本次记忆操作记录不完整",
    UNAVAILABLE_LEGACY: "旧记录未保存可定位的记忆条目",
  };
  const known = {
    no_activation: [
      "本次没有激活持久记忆",
      "模型可能只使用了当前对话或原始证据；展开调用处理账本可查看依据。",
    ],
    no_changes: [
      "本次没有写入或修改记忆",
      "反馈未形成已提交的持久记忆变更；展开调用处理账本可查看判定。",
    ],
    legacy_identity_missing: [
      "旧记录无法定位具体记忆条目",
      "当时未保存稳定的节点或连接 ID；不会用当前记忆图替代历史结果。",
    ],
    failed: [
      "调用未完成",
      "没有可确认的持久记忆读取、写入或修改记录。",
    ],
  };
  if (known[reason]) return { title: known[reason][0], copy: known[reason][1] };
  if (reason) {
    return {
      title: run?.status === "failed"
        ? "调用未完成"
        : stateTitles[stateName] || "本次没有可显示的持久记忆读写",
      copy: reason,
    };
  }
  return {
    title: stateTitles[stateName] || "本次没有可显示的持久记忆读写",
    copy: "该调用尚未保存可定位的持久记忆操作；展开调用处理账本可查看过程。",
  };
}

function layoutRunMemoryGraph(nodes, edges) {
  const width = 1040;
  const height = Math.max(540, 420 + Math.ceil(Math.max(0, nodes.length - 12) / 12) * 120);
  const center = { x: width / 2, y: height / 2 };
  const positions = new Map();
  const velocity = new Map();
  if (nodes.length === 1) {
    positions.set(nodes[0].id, center);
    return { width, height, positions };
  }
  const radius = Math.min(width, height) * Math.min(0.38, 0.2 + nodes.length * 0.012);
  [...nodes]
    .sort((a, b) => String(a.id).localeCompare(String(b.id)))
    .forEach((node, index) => {
      const angle = (Math.PI * 2 * index) / Math.max(1, nodes.length) - Math.PI / 2;
      positions.set(node.id, {
        x: center.x + Math.cos(angle) * radius,
        y: center.y + Math.sin(angle) * radius,
      });
      velocity.set(node.id, { x: 0, y: 0 });
    });
  const pairs = edges
    .map((edge) => [String(edge.source), String(edge.target)])
    .filter(([source, target]) => positions.has(source) && positions.has(target));
  const iterations = nodes.length > 80 ? 30 : 48;
  for (let iteration = 0; iteration < iterations; iteration += 1) {
    for (let left = 0; left < nodes.length; left += 1) {
      for (let right = left + 1; right < nodes.length; right += 1) {
        const a = nodes[left];
        const b = nodes[right];
        const pa = positions.get(a.id);
        const pb = positions.get(b.id);
        const va = velocity.get(a.id);
        const vb = velocity.get(b.id);
        let dx = pa.x - pb.x;
        let dy = pa.y - pb.y;
        const distanceSquared = Math.max(225, dx * dx + dy * dy);
        const distance = Math.sqrt(distanceSquared);
        const force = Math.min(2.1, 8200 / distanceSquared);
        dx /= distance;
        dy /= distance;
        va.x += dx * force;
        va.y += dy * force;
        vb.x -= dx * force;
        vb.y -= dy * force;
      }
    }
    pairs.forEach(([source, target]) => {
      const sourcePosition = positions.get(source);
      const targetPosition = positions.get(target);
      const sourceVelocity = velocity.get(source);
      const targetVelocity = velocity.get(target);
      const dx = targetPosition.x - sourcePosition.x;
      const dy = targetPosition.y - sourcePosition.y;
      const distance = Math.max(1, Math.sqrt(dx * dx + dy * dy));
      const force = (distance - 190) * 0.0032;
      sourceVelocity.x += (dx / distance) * force;
      sourceVelocity.y += (dy / distance) * force;
      targetVelocity.x -= (dx / distance) * force;
      targetVelocity.y -= (dy / distance) * force;
    });
    nodes.forEach((node) => {
      const position = positions.get(node.id);
      const nodeVelocity = velocity.get(node.id);
      nodeVelocity.x += (center.x - position.x) * 0.001;
      nodeVelocity.y += (center.y - position.y) * 0.001;
      nodeVelocity.x *= 0.8;
      nodeVelocity.y *= 0.8;
      position.x = Math.max(72, Math.min(width - 72, position.x + nodeVelocity.x));
      position.y = Math.max(72, Math.min(height - 72, position.y + nodeVelocity.y));
    });
  }
  return { width, height, positions };
}

function fitRunDetailMemoryGraph() {
  elements.runDetailGraphStage.scrollTo({ top: 0, left: 0, behavior: "smooth" });
}

function memoryNodeLabelLines(node) {
  const label = String(node?.label || node?.title || node?.id || "未命名记忆").trim();
  if (label.length <= 18) return [label];
  return [`${label.slice(0, 18)}…`];
}

function runDetailMemoryItems() {
  const effects = memoryEffects();
  const seen = new Set();
  const nodes = (Array.isArray(effects.nodes) ? effects.nodes : [])
    .filter((item) => item && typeof item === "object" && String(item.id || "").trim())
    .map((item) => ({ ...item, id: String(item.id) }))
    .filter((item) => {
      if (seen.has(item.id)) return false;
      seen.add(item.id);
      return true;
    });
  const nodeIds = new Set(nodes.map((node) => node.id));
  const edges = (Array.isArray(effects.edges) ? effects.edges : [])
    .filter((item) => item && typeof item === "object")
    .map((item) => ({
      ...item,
      source: String(item.source || ""),
      target: String(item.target || ""),
    }))
    .filter((item) => nodeIds.has(item.source) && nodeIds.has(item.target));
  return { effects, nodes, edges };
}

function renderRunDetailMemoryEffects() {
  elements.runDetailEdges.replaceChildren();
  elements.runDetailNodes.replaceChildren();
  const { effects, nodes, edges } = runDetailMemoryItems();
  elements.runDetailEmpty.classList.toggle("hidden", nodes.length > 0);
  if (!nodes.length) {
    const empty = memoryEffectsEmptyState(effects, state.runDetail?.run || {});
    elements.runDetailEmptyTitle.textContent = empty.title;
    elements.runDetailEmptyCopy.textContent = empty.copy;
    elements.runDetailGraph.style.height = "100%";
    elements.runDetailGraphCaption.textContent = empty.title;
    elements.runDetailInspectorTitle.textContent = "没有可定位的记忆条目";
    elements.runDetailInspectorContent.replaceChildren(
      textElement("p", empty.copy, "panel-note"),
    );
    return;
  }

  const layout = layoutRunMemoryGraph(nodes, edges);
  elements.runDetailGraph.setAttribute("viewBox", `0 0 ${layout.width} ${layout.height}`);
  elements.runDetailGraph.style.height = `${layout.height}px`;

  edges.forEach((edge, index) => {
    const source = layout.positions.get(edge.source);
    const target = layout.positions.get(edge.target);
    if (!source || !target) return;
    const access = normalizedMemoryAccess(edge);
    const pathId = `run-detail-memory-edge-${index}`;
    const relation = String(edge.relation || edge.type || "关联");
    let d;
    if (edge.source === edge.target) {
      d = `M ${source.x + 19} ${source.y - 12} C ${source.x + 82} ${source.y - 82}, ${source.x - 82} ${source.y - 82}, ${source.x - 19} ${source.y - 12}`;
    } else {
      const dx = target.x - source.x;
      const dy = target.y - source.y;
      const distance = Math.max(1, Math.sqrt(dx * dx + dy * dy));
      const startX = source.x + (dx / distance) * 28;
      const startY = source.y + (dy / distance) * 28;
      const endX = target.x - (dx / distance) * 31;
      const endY = target.y - (dy / distance) * 31;
      const bend = ((hashString(`${edge.source}:${edge.target}:${relation}`) % 3) - 1) * 18;
      const controlX = (startX + endX) / 2 - (dy / distance) * bend;
      const controlY = (startY + endY) / 2 + (dx / distance) * bend;
      d = `M ${startX} ${startY} Q ${controlX} ${controlY}, ${endX} ${endY}`;
    }
    const path = svgElement("path", {
      id: pathId,
      d,
      class: "run-detail-memory-edge",
      "data-edge-index": index,
      "data-source": edge.source,
      "data-target": edge.target,
      "data-access": access.join(" "),
      "marker-end": "url(#run-detail-arrow)",
      tabindex: "0",
      role: "button",
      "aria-label": `${relation}；本次操作：${memoryAccessText(edge)}`,
    });
    const edgeTitle = svgElement("title");
    edgeTitle.textContent = `${relation} · ${memoryAccessText(edge)}`;
    path.append(edgeTitle);
    path.addEventListener("click", () => selectRunDetailMemoryEdge(index, edge));
    path.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        selectRunDetailMemoryEdge(index, edge);
      }
    });
    elements.runDetailEdges.append(path);
    if (edges.length <= 40 && relation) {
      const label = svgElement("text", {
        class: "run-detail-memory-edge-label",
        "data-edge-index": index,
        "data-source": edge.source,
        "data-target": edge.target,
        "data-access": access.join(" "),
      });
      const textPath = svgElement("textPath", { href: `#${pathId}`, startOffset: "50%" });
      const edgeLabel = `${relation} · ${memoryAccessText(edge)}`;
      textPath.textContent = edgeLabel.length > 26 ? `${edgeLabel.slice(0, 26)}…` : edgeLabel;
      label.append(textPath);
      elements.runDetailEdges.append(label);
    }
  });

  nodes.forEach((node) => {
    const position = layout.positions.get(node.id);
    if (!position) return;
    const access = normalizedMemoryAccess(node);
    const group = svgElement("g", {
      class: "run-detail-memory-node",
      transform: `translate(${position.x} ${position.y})`,
      tabindex: "0",
      role: "button",
      "data-node-id": node.id,
      "data-type": node.type || "semantic",
      "data-access": access.join(" "),
      "aria-label": `${typeNames[node.type] || node.type || "记忆节点"}：${node.label || node.id}；本次操作：${memoryAccessText(node)}`,
    });
    group.append(svgElement("circle", {
      r: 29,
      class: "run-detail-memory-access-ring",
    }));
    group.append(svgElement("circle", {
      r: 21,
      class: "run-detail-memory-node-core",
    }));
    const operation = svgElement("text", {
      y: -36,
      class: "run-detail-memory-access-label",
      "text-anchor": "middle",
    });
    operation.textContent = memoryAccessText(node);
    group.append(operation);
    const label = svgElement("text", {
      y: 43,
      class: "run-detail-memory-node-label",
      "text-anchor": "middle",
    });
    const lines = memoryNodeLabelLines(node);
    lines.forEach((line, index) => {
      const span = svgElement("tspan", { x: 0, dy: index ? 14 : 0 });
      span.textContent = line;
      label.append(span);
    });
    group.append(label);
    group.addEventListener("click", () => selectRunDetailMemoryNode(node.id));
    group.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        selectRunDetailMemoryNode(node.id);
      }
    });
    elements.runDetailNodes.append(group);
  });

  const counts = memoryEffectCounts(effects, nodes, edges);
  const nodeCount = memoryEntityCount(effects, "nodes", nodes.length);
  const edgeCount = memoryEntityCount(effects, "edges", edges.length);
  const effectState = memoryEffectStateNames[String(effects.state || "").toUpperCase()]
    || "记录状态未记录";
  elements.runDetailGraphCaption.textContent = [
    `记忆节点 ${formatMemoryEntityCount(nodeCount, effects)}`,
    `连接 ${formatMemoryEntityCount(edgeCount, effects)}`,
    `读取 ${formatMemoryEffectCount(counts.read, effects)}`,
    `写入 ${formatMemoryEffectCount(counts.write, effects)}`,
    `写入或更新 ${formatMemoryEffectCount(counts.upsert, effects)}`,
    `修改 ${formatMemoryEffectCount(counts.modify, effects)}`,
    `结构端点 ${formatMemoryEffectCount(counts.context, effects)}`,
    effectState,
  ].join(" · ");
  const firstActive = nodes.find((node) => !normalizedMemoryAccess(node).includes("context"));
  selectRunDetailMemoryNode(firstActive?.id || nodes[0].id);
}

function selectRunDetailMemoryNode(nodeId) {
  const { nodes, edges } = runDetailMemoryItems();
  const selectedId = String(nodeId);
  const node = nodes.find((item) => item.id === selectedId);
  if (!node) return;
  state.runDetailNodeId = selectedId;
  const connected = new Set([selectedId]);
  edges.forEach((edge) => {
    if (edge.source === selectedId || edge.target === selectedId) {
      connected.add(edge.source);
      connected.add(edge.target);
    }
  });
  elements.runDetailNodes.querySelectorAll(".run-detail-memory-node").forEach((item) => {
    const id = item.getAttribute("data-node-id");
    item.classList.toggle("is-selected", id === selectedId);
    item.classList.toggle("is-muted", !connected.has(id));
  });
  elements.runDetailEdges.querySelectorAll(".run-detail-memory-edge, .run-detail-memory-edge-label").forEach((item) => {
    const active = item.getAttribute("data-source") === selectedId
      || item.getAttribute("data-target") === selectedId;
    item.classList.toggle("is-active", active);
    item.classList.toggle("is-muted", !active);
  });
  renderRunDetailMemoryInspector(node);
}

function selectRunDetailMemoryEdge(edgeIndex, edge) {
  const sourceId = String(edge.source);
  const targetId = String(edge.target);
  state.runDetailNodeId = "";
  elements.runDetailNodes.querySelectorAll(".run-detail-memory-node").forEach((item) => {
    const id = item.getAttribute("data-node-id");
    const endpoint = id === sourceId || id === targetId;
    item.classList.remove("is-selected");
    item.classList.toggle("is-muted", !endpoint);
  });
  elements.runDetailEdges.querySelectorAll(".run-detail-memory-edge, .run-detail-memory-edge-label").forEach((item) => {
    const active = item.getAttribute("data-edge-index") === String(edgeIndex);
    item.classList.toggle("is-active", active);
    item.classList.toggle("is-muted", !active);
  });
  renderRunDetailMemoryEdgeInspector(edge);
}

function renderRunDetailMemoryInspector(node) {
  elements.runDetailInspectorTitle.textContent = node.label || node.id || "记忆条目";
  elements.runDetailInspectorContent.replaceChildren();
  const badges = document.createElement("div");
  badges.className = "run-detail-memory-badges";
  badges.append(textElement("span", typeNames[node.type] || node.type || "记忆节点", "node-type-badge"));
  normalizedMemoryAccess(node).forEach((access) => {
    const chip = textElement("span", memoryAccessNames[access], "memory-access-chip");
    chip.dataset.access = access;
    badges.append(chip);
  });
  elements.runDetailInspectorContent.append(badges);
  appendDetailList(elements.runDetailInspectorContent, [
    ["本次操作", memoryAccessText(node)],
    ["节点类型", typeNames[node.type] || node.type],
    ["状态来源", node.state_source],
    ["状态", node.status],
    ["说明", node.detail],
    ["节点 ID", node.id],
  ]);
  const pre = document.createElement("pre");
  pre.className = "run-detail-node-json";
  pre.textContent = JSON.stringify(node, null, 2);
  elements.runDetailInspectorContent.append(pre);
}

function renderRunDetailMemoryEdgeInspector(edge) {
  const relation = String(edge.relation || edge.type || "记忆连接");
  elements.runDetailInspectorTitle.textContent = relation;
  elements.runDetailInspectorContent.replaceChildren();
  const badges = document.createElement("div");
  badges.className = "run-detail-memory-badges";
  badges.append(textElement("span", "记忆连接", "node-type-badge"));
  normalizedMemoryAccess(edge).forEach((access) => {
    const chip = textElement("span", memoryAccessNames[access], "memory-access-chip");
    chip.dataset.access = access;
    badges.append(chip);
  });
  elements.runDetailInspectorContent.append(badges);
  appendDetailList(elements.runDetailInspectorContent, [
    ["关系", relation],
    ["本次操作", memoryAccessText(edge)],
    ["陈述", edge.statement],
    ["认知状态", edge.epistemic_state],
    ["不确定性", edge.uncertainty],
    ["效用", edge.utility],
    ["状态来源", edge.state_source],
    ["起点", edge.source],
    ["终点", edge.target],
  ]);
  const pre = document.createElement("pre");
  pre.className = "run-detail-node-json";
  pre.textContent = JSON.stringify(edge, null, 2);
  elements.runDetailInspectorContent.append(pre);
}

function renderRunDetailLedger(detail) {
  const ledger = detail?.graph && typeof detail.graph === "object" ? detail.graph : {};
  const nodes = Array.isArray(ledger.nodes) ? ledger.nodes : [];
  const edges = Array.isArray(ledger.edges) ? ledger.edges : [];
  elements.runDetailLedgerSummary.textContent = nodes.length || edges.length
    ? `${formatNumber(nodes.length)} 个处理节点 · ${formatNumber(edges.length)} 条处理关系`
    : "没有已保存的处理流程";
  elements.runDetailLedgerJson.textContent = JSON.stringify(ledger, null, 2);
}

function renderRunDetail(detail) {
  const run = detail.run || {};
  const result = run.result || {};
  const metadata = run.metadata || {};
  const usage = Array.isArray(detail.usage) ? detail.usage : [];
  const tokens = usage.reduce((sum, item) => sum + Number(item.total || 0), 0);
  const modelMs = usage.reduce((sum, item) => sum + Number(item.elapsed_ms || 0), 0);
  const effects = memoryEffects(detail);
  const memoryNodes = Array.isArray(effects.nodes) ? effects.nodes : [];
  const memoryEdges = Array.isArray(effects.edges) ? effects.edges : [];
  const title = run.experiment_type === "runtime_feedback_maintenance"
    ? "这次反馈实际改了哪些记忆"
    : "这次回答实际激活了哪些记忆";
  elements.runDetailTitle.textContent = title;
  elements.runDetailSubtitle.textContent = `${run.run_id || ""} · ${formatTime(run.started_at)} · ${result.path || metadata.path || "未记录路径"}`;
  renderMemoryEffectsBasis(effects);
  elements.runDetailMetrics.replaceChildren();
  [
    ["状态", run.status || "—"],
    ["模型耗时", formatDuration(modelMs)],
    ["Token", formatNumber(tokens)],
    ["记忆节点 / 连接", memoryEffectsMetricValue(effects, memoryNodes, memoryEdges)],
  ].forEach(([label, value]) => {
    const card = document.createElement("article");
    card.append(textElement("span", label), textElement("strong", value));
    elements.runDetailMetrics.append(card);
  });
  const warnings = [
    ...(Array.isArray(effects.warnings) ? effects.warnings : []),
    ...(Array.isArray(detail.graph?.warnings) ? detail.graph.warnings : []),
  ];
  const missingRefs = Array.isArray(effects.missing_refs) ? effects.missing_refs : [];
  if (missingRefs.length) warnings.push(`${missingRefs.length} 个记忆引用无法定位，主画布未显示这些条目。`);
  const uniqueWarnings = [...new Set(warnings.map((warning) => String(warning || "").trim()).filter(Boolean))];
  elements.runDetailWarning.replaceChildren();
  elements.runDetailWarning.classList.toggle("hidden", !uniqueWarnings.length);
  uniqueWarnings.forEach((warning) => elements.runDetailWarning.append(textElement("p", warning)));
  renderRunDetailLedger(detail);
  elements.runDetailResultJson.textContent = JSON.stringify(result, null, 2);
  renderRunDetailMemoryEffects();
}

async function openRunDetail(runId) {
  if (!state.scopeId || !runId) return;
  state.runDetail = null;
  state.runDetailNodeId = "";
  elements.runDetailLoading.classList.remove("hidden");
  elements.runDetailEmpty.classList.add("hidden");
  elements.runDetailEdges.replaceChildren();
  elements.runDetailNodes.replaceChildren();
  elements.runDetailMetrics.replaceChildren();
  elements.runDetailWarning.classList.add("hidden");
  elements.runDetailMemoryState.textContent = "记录状态未读取";
  elements.runDetailMemoryState.dataset.state = "UNKNOWN";
  elements.runDetailIdentityBasis.textContent = "身份确认未记录";
  elements.runDetailPayloadBasis.textContent = "载荷解析时间未记录";
  elements.runDetailEmptyTitle.textContent = "本次没有可显示的持久记忆读写";
  elements.runDetailEmptyCopy.textContent = "展开调用处理账本可查看证据与处理过程。";
  elements.runDetailLedgerSummary.textContent = "正在读取调用处理账本";
  elements.runDetailLedgerJson.textContent = "正在读取…";
  elements.runDetailResultJson.textContent = "正在读取…";
  if (!elements.runDetailDialog.open) elements.runDetailDialog.showModal();
  try {
    const detail = await apiGet(
      `scopes/${state.scopeId}/runs/${encodeURIComponent(String(runId))}`,
    );
    if (detail.scope_id !== state.scopeId) return;
    state.runDetail = detail;
    renderRunDetail(detail);
  } catch (error) {
    elements.runDetailEmpty.classList.remove("hidden");
    elements.runDetailEmptyTitle.textContent = "调用详情读取失败";
    elements.runDetailEmptyCopy.textContent = error?.message || "无法读取本次记忆操作。";
    elements.runDetailLedgerJson.textContent = error?.message || "调用详情读取失败";
    elements.runDetailResultJson.textContent = error?.message || "调用详情读取失败";
    showToast(error?.message || "调用详情读取失败", "error");
  } finally {
    elements.runDetailLoading.classList.add("hidden");
  }
}

function populateScopes(previousScopeId = "") {
  const scopes = state.overview?.scopes || [];
  elements.scopeSelect.replaceChildren();
  if (!scopes.length) {
    const option = new Option("尚无群聊记忆数据", "");
    elements.scopeSelect.append(option);
    elements.scopeSelect.disabled = true;
    elements.distillBtn.disabled = true;
    elements.resetBudgetBtn.disabled = true;
    elements.resetFeedbackBudgetBtn.disabled = true;
    state.scopeId = "";
    state.participants = [];
    renderParticipants([]);
    elements.scopeSummary.textContent = "启用群聊记忆并收到消息后，这里会出现群聊。";
    elements.scopeMeta.textContent = "尚无群聊记忆库。";
    elements.scopeErrors.classList.add("hidden");
    showEmptyGraph();
    renderRuntimeHealth(null);
    return;
  }

  for (const scope of scopes) {
    const group = scope.group_id || "未知群";
    const platform = scope.platform_id || "unknown";
    const option = new Option(
      `${group} · ${platform} · ${formatNumber(scope.messages)} 条消息`,
      scope.storage_id,
    );
    elements.scopeSelect.append(option);
  }
  elements.scopeSelect.disabled = false;
  const available = scopes.some((scope) => scope.storage_id === previousScopeId);
  state.scopeId = available ? previousScopeId : scopes[0].storage_id;
  elements.scopeSelect.value = state.scopeId;
  elements.distillBtn.disabled = false;
  elements.resetBudgetBtn.disabled = false;
  elements.resetFeedbackBudgetBtn.disabled = false;
  state.loadedSections.clear();
  renderScopeMeta();
}

function renderScopeMeta() {
  const scope = selectedScope();
  if (!scope) {
    elements.scopeSummary.textContent = "尚未选择群聊";
    elements.scopeMeta.textContent = "尚未选择群范围";
    elements.scopeErrors.classList.add("hidden");
    renderRuntimeHealth(null);
    return;
  }
  const runtime = state.overview?.runtime || {};
  const usage = scope.token_usage_24h || {};
  const rawLedger = scope.token_ledger_24h || {};
  const pendingByClass = scope.pending_distillation_by_class || {};
  elements.scopeSummary.textContent = [
    `群 ${scope.group_id || "未知"}`,
    `${formatNumber(scope.messages)} 条消息`,
    `${formatNumber(scope.episodes)} 段情节记忆`,
    `最近消息 ${formatTime(scope.last_message_at)}`,
  ].join(" · ");
  elements.scopeMeta.textContent = [
    `UMO ${scope.umo}`,
    `${formatNumber(pendingByClass.live)} 条新消息待整理`,
    `深挖与整理 ${formatNumber(usage.online)}/${formatNumber(runtime.private_daily_token_budget)} Token`,
    `反馈 ${formatNumber(usage.feedback)}/${formatNumber(runtime.feedback_daily_token_budget)} Token`,
    Number(rawLedger.online || 0) !== Number(usage.online || 0)
      ? `回忆原始账本 ${formatNumber(rawLedger.online)} Token（已重置额度）`
      : "回忆额度未人工重置",
    Number(rawLedger.feedback || 0) !== Number(usage.feedback || 0)
      ? `反馈原始账本 ${formatNumber(rawLedger.feedback)} Token（已重置额度）`
      : "反馈额度未人工重置",
    `${formatNumber(scope.open_semantic_hypotheses)} 条群内语义仍有疑问`,
    `状态版本 ${formatNumber(scope.subconscious_revision)}`,
  ].join("  ·  ");
  const recentErrors = [
    ...(scope.recent_processing_errors || []),
    ...(scope.recent_maintenance_errors || []),
  ];
  if (recentErrors.length) {
    const latest = recentErrors[0];
    elements.scopeErrors.textContent = `最近异常：${latest.last_error || "未知错误"}（${formatTime(latest.updated_at)}）`;
    elements.scopeErrors.classList.remove("hidden");
  } else {
    elements.scopeErrors.classList.add("hidden");
  }
  renderRuntimeHealth(scope);
}

async function refreshOverview({ preserveSelection = true, loadScope = true } = {}) {
  const previousScopeId = preserveSelection ? state.scopeId : "";
  elements.refreshBtn.disabled = true;
  setConnection("loading", "正在同步运行状态");
  try {
    state.overview = await apiGet("overview");
    renderOverview();
    populateScopes(previousScopeId);
    if (loadScope && state.scopeId) {
      await loadSelectedScope();
    }
  } catch (error) {
    setConnection("error", "控制台连接失败");
    showToast(error?.message || "无法读取 MR Memory 状态", "error");
  } finally {
    elements.refreshBtn.disabled = false;
  }
}

function setGraphLoading(loading) {
  elements.graphLoading.classList.toggle("hidden", !loading);
}

function showEmptyGraph() {
  state.graph = null;
  state.selectedNodeId = "";
  elements.graphEdges.replaceChildren();
  elements.graphEdgeLabels.replaceChildren();
  elements.graphNodes.replaceChildren();
  elements.graphEmpty.classList.remove("hidden");
  elements.graphCaption.textContent = "当前群范围还没有可显示的图记忆。";
  renderInspectorPlaceholder();
}

async function loadSelectedScope() {
  if (!state.scopeId) {
    showEmptyGraph();
    return;
  }
  renderScopeMeta();
  if (state.activeSection === "runtime") return;
  if (state.loadedSections.has(state.activeSection)) return;
  if (state.activeSection === "graph") setGraphLoading(true);
  try {
    if (state.activeSection === "graph") await loadGraph();
    else if (state.activeSection === "identity") await loadParticipants();
    else if (state.activeSection === "evidence") await loadMessages();
    state.loadedSections.add(state.activeSection);
  } catch (error) {
    showToast(error?.message || "当前区域加载失败", "error");
  } finally {
    if (state.activeSection === "graph") setGraphLoading(false);
  }
}

async function activateSection(section) {
  state.activeSection = section;
  document.querySelectorAll("[data-section]").forEach((tab) => {
    const active = tab.dataset.section === section;
    tab.classList.toggle("is-active", active);
    tab.setAttribute("aria-selected", active ? "true" : "false");
  });
  document.querySelectorAll("[data-section-view]").forEach((view) => {
    view.classList.toggle("hidden", view.dataset.sectionView !== section);
  });
  await loadSelectedScope();
  if (section === "graph" && state.graph) window.requestAnimationFrame(fitGraph);
}

function graphRequestParams() {
  const params = {
    limit: Number(elements.graphLimit.value || 100),
    depth: Number(elements.graphDepth.value || 2),
    types: [...state.visibleTypes].sort().join(","),
    min_degree: Number(elements.graphDegreeFilter.value || 0),
    min_core: Number(elements.graphCoreFilter.value || 0),
    structure: elements.graphStructureFilter.value || "all",
  };
  if (state.graphQuery) params.query = state.graphQuery;
  if (state.graphFocusId) params.focus = state.graphFocusId;
  if (elements.graphEpistemicFilter.value) {
    params.epistemic = elements.graphEpistemicFilter.value;
  }
  if (elements.graphRelationFilter.value) {
    params.relation = elements.graphRelationFilter.value;
  }
  if (state.graphPathSource && state.graphPathTarget) {
    params.path_source = state.graphPathSource;
    params.path_target = state.graphPathTarget;
  }
  return params;
}

async function loadGraph() {
  const graph = await apiGet(
    `scopes/${state.scopeId}/graph`,
    graphRequestParams(),
  );
  if (state.scopeId !== graph?.scope?.storage_id) return;
  state.graph = graph;
  state.graphFocusId = graph.focus_node_id || state.graphFocusId;
  state.selectedNodeId = "";
  renderGraphMetrics();
  renderSearchResults();
  renderRelationOptions();
  updatePathStatus();
  renderGraph();
  const preferredNode = graph.mode === "path"
    ? graph.path?.target
    : graph.focus_node_id;
  if (preferredNode && (graph.nodes || []).some((node) => node.id === preferredNode)) {
    selectNode(preferredNode);
  } else {
    renderInspectorPlaceholder();
  }
}

function renderGraphMetrics() {
  const metrics = state.graph?.metrics || {};
  elements.graphMetricNodes.textContent = formatNumber(metrics.node_count);
  elements.graphMetricEdges.textContent = formatNumber(metrics.edge_count);
  elements.graphMetricDegree.textContent = Number(metrics.average_degree || 0).toFixed(2);
  elements.graphMetricDensity.textContent = `${(Number(metrics.density || 0) * 100).toFixed(2)}%`;
  elements.graphMetricComponents.textContent = formatNumber(metrics.connected_components);
  elements.graphMetricGiant.textContent = `最大分量 ${(Number(metrics.giant_component_ratio || 0) * 100).toFixed(1)}%`;
  elements.graphMetricClustering.textContent = Number(metrics.average_clustering || 0).toFixed(3);
  elements.graphMetricCore.textContent = formatNumber(metrics.max_core);
}

function renderSearchResults() {
  const matches = state.graph?.matches || [];
  elements.graphSearchResults.replaceChildren();
  elements.graphSearchResults.classList.toggle("hidden", !state.graphQuery);
  if (!state.graphQuery) return;
  if (!matches.length) {
    elements.graphSearchResults.append(
      textElement("p", `没有找到“${state.graphQuery}”。可以换一个称呼、证据片段或更短的关键词。`, "panel-note"),
    );
    return;
  }
  matches.slice(0, 9).forEach((match) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "graph-search-result";
    button.classList.toggle("is-active", match.id === state.graph?.focus_node_id);
    button.append(
      textElement("span", match.label || match.id),
      textElement("small", `${typeNames[match.type] || match.type} · 度 ${match.degree} · k-core ${match.core}`),
      textElement("small", truncate(match.detail || "暂无摘要", 76)),
    );
    button.addEventListener("click", () => focusGraphNode(match.id));
    elements.graphSearchResults.append(button);
  });
}

function renderRelationOptions() {
  const current = elements.graphRelationFilter.value;
  const options = [{ relation: "", count: null }, ...(state.graph?.relation_counts || [])];
  if (current && !options.some((item) => item.relation === current)) {
    options.splice(1, 0, { relation: current, count: null });
  }
  elements.graphRelationFilter.replaceChildren();
  options.forEach((item) => {
    const option = document.createElement("option");
    option.value = item.relation;
    option.textContent = item.relation
      ? `${item.relation}${item.count === null ? "" : ` (${formatNumber(item.count)})`}`
      : "全部关系";
    option.selected = item.relation === current;
    elements.graphRelationFilter.append(option);
  });
}

function updatePathStatus() {
  elements.graphPathClear.disabled = !state.graphPathSource && !state.graphPathTarget;
  const nodeMap = new Map((state.graph?.nodes || []).map((node) => [node.id, node]));
  const source = nodeMap.get(state.graphPathSource)?.label || state.graphPathSource;
  const target = nodeMap.get(state.graphPathTarget)?.label || state.graphPathTarget;
  if (!state.graphPathSource) {
    elements.graphPathStatus.textContent = "在右侧节点详情中选择起点和终点";
    return;
  }
  if (!state.graphPathTarget) {
    elements.graphPathStatus.textContent = `起点：${source}；再选择一个节点作为终点`;
    return;
  }
  const path = state.graph?.path || {};
  elements.graphPathStatus.textContent = path.found
    ? `${source} → ${target}：${formatNumber(path.length)} 跳`
    : `${source} 与 ${target} 在当前筛选网络中不连通`;
}

function hashString(value) {
  let hash = 2166136261;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

function calculateNeighborhoodLayout(nodes) {
  const width = 1180;
  const height = 720;
  const center = { x: width / 2, y: height / 2 };
  const positions = new Map();
  const focusId = state.graph?.focus_node_id;
  if (focusId) positions.set(focusId, center);
  const levels = new Map();
  nodes.forEach((node) => {
    if (node.id === focusId) return;
    const level = Math.max(1, Number(node.distance || 1));
    if (!levels.has(level)) levels.set(level, []);
    levels.get(level).push(node);
  });
  [...levels.entries()].sort((a, b) => a[0] - b[0]).forEach(([level, items]) => {
    items.sort((a, b) =>
      Number(b.degree || 0) - Number(a.degree || 0) ||
      String(a.type).localeCompare(String(b.type), "zh-CN") ||
      String(a.label).localeCompare(String(b.label), "zh-CN")
    );
    let cursor = 0;
    let ringIndex = 0;
    while (cursor < items.length) {
      const radius = 145 * level + ringIndex * 72;
      const capacity = Math.max(8, Math.floor((Math.PI * 2 * radius) / 76));
      const ringItems = items.slice(cursor, cursor + capacity);
      const phase = ((level + ringIndex) % 2) * (Math.PI / Math.max(1, ringItems.length));
      ringItems.forEach((node, index) => {
        const angle = phase + (Math.PI * 2 * index) / ringItems.length - Math.PI / 2;
        positions.set(node.id, {
          x: center.x + Math.cos(angle) * radius,
          y: center.y + Math.sin(angle) * radius,
        });
      });
      cursor += ringItems.length;
      ringIndex += 1;
    }
  });
  state.virtualSize = { width, height };
  return positions;
}

function calculatePathLayout(nodes) {
  const width = 1180;
  const height = 720;
  const positions = new Map();
  const ordered = [...nodes].sort(
    (a, b) => Number(a.path_index || 0) - Number(b.path_index || 0),
  );
  ordered.forEach((node, index) => {
    const ratio = ordered.length === 1 ? 0.5 : index / (ordered.length - 1);
    positions.set(node.id, {
      x: 120 + ratio * (width - 240),
      y: height / 2 + (index % 2 ? 30 : -30),
    });
  });
  state.virtualSize = { width, height };
  return positions;
}

function calculateOverviewLayout(nodes, edges) {
  const width = 1180;
  const height = 720;
  const center = { x: width / 2, y: height / 2 };
  const positions = new Map();
  const velocity = new Map();
  const componentIds = [...new Set(nodes.map((node) => Number(node.component_id || 0)))];
  const componentCenters = new Map();
  componentIds.forEach((componentId, index) => {
    if (componentId === 1) {
      componentCenters.set(componentId, center);
      return;
    }
    const angle = (Math.PI * 2 * (index - 1)) / Math.max(1, componentIds.length - 1);
    componentCenters.set(componentId, {
      x: center.x + Math.cos(angle) * 250,
      y: center.y + Math.sin(angle) * 215,
    });
  });
  nodes.forEach((node, index) => {
    const seed = hashString(node.id);
    const angle = ((seed % 10000) / 10000) * Math.PI * 2;
    const radius = 42 + Math.sqrt(index + 1) * 22;
    const componentCenter = componentCenters.get(Number(node.component_id || 0)) || center;
    positions.set(node.id, {
      x: componentCenter.x + Math.cos(angle) * radius,
      y: componentCenter.y + Math.sin(angle) * radius,
    });
    velocity.set(node.id, { x: 0, y: 0 });
  });
  const edgePairs = edges
    .map((edge) => [edge.source, edge.target])
    .filter(([source, target]) => positions.has(source) && positions.has(target));
  const iterations = nodes.length > 300 ? 34 : 54;
  for (let iteration = 0; iteration < iterations; iteration += 1) {
    for (let left = 0; left < nodes.length; left += 1) {
      const a = nodes[left];
      const pa = positions.get(a.id);
      const va = velocity.get(a.id);
      for (let right = left + 1; right < nodes.length; right += 1) {
        const b = nodes[right];
        const pb = positions.get(b.id);
        const vb = velocity.get(b.id);
        let dx = pa.x - pb.x;
        let dy = pa.y - pb.y;
        const distanceSquared = Math.max(100, dx * dx + dy * dy);
        const distance = Math.sqrt(distanceSquared);
        const force = Math.min(1.7, 6200 / distanceSquared);
        dx /= distance;
        dy /= distance;
        va.x += dx * force;
        va.y += dy * force;
        vb.x -= dx * force;
        vb.y -= dy * force;
      }
    }
    edgePairs.forEach(([source, target]) => {
      const sourcePosition = positions.get(source);
      const targetPosition = positions.get(target);
      const sourceVelocity = velocity.get(source);
      const targetVelocity = velocity.get(target);
      const dx = targetPosition.x - sourcePosition.x;
      const dy = targetPosition.y - sourcePosition.y;
      const distance = Math.max(1, Math.sqrt(dx * dx + dy * dy));
      const force = (distance - 92) * 0.0028;
      sourceVelocity.x += (dx / distance) * force;
      sourceVelocity.y += (dy / distance) * force;
      targetVelocity.x -= (dx / distance) * force;
      targetVelocity.y -= (dy / distance) * force;
    });
    nodes.forEach((node) => {
      const position = positions.get(node.id);
      const nodeVelocity = velocity.get(node.id);
      const componentCenter = componentCenters.get(Number(node.component_id || 0)) || center;
      nodeVelocity.x += (componentCenter.x - position.x) * 0.0018;
      nodeVelocity.y += (componentCenter.y - position.y) * 0.0018;
      nodeVelocity.x *= 0.82;
      nodeVelocity.y *= 0.82;
      position.x = Math.max(54, Math.min(width - 54, position.x + nodeVelocity.x));
      position.y = Math.max(54, Math.min(height - 54, position.y + nodeVelocity.y));
    });
  }
  state.virtualSize = { width, height };
  return positions;
}

function calculateLayout(nodes, edges) {
  if (state.graph?.mode === "neighborhood") return calculateNeighborhoodLayout(nodes);
  if (state.graph?.mode === "path") return calculatePathLayout(nodes);
  return calculateOverviewLayout(nodes, edges);
}

function renderGraph() {
  const graph = state.graph;
  const nodes = graph?.nodes || [];
  const visibleIds = new Set(nodes.map((node) => node.id));
  const edges = (graph?.edges || []).filter(
    (edge) => visibleIds.has(edge.source) && visibleIds.has(edge.target),
  );
  elements.graphEdges.replaceChildren();
  elements.graphEdgeLabels.replaceChildren();
  elements.graphNodes.replaceChildren();
  elements.graphEmpty.classList.toggle("hidden", nodes.length > 0);
  if (!nodes.length) {
    elements.graphEmptyTitle.textContent = state.graphQuery
      ? `没有找到“${state.graphQuery}”的连接`
      : "没有符合当前筛选的记忆连接";
    elements.graphEmptyCopy.textContent = state.graphQuery
      ? "换一个称呼、证据片段或更短的关键词。"
      : "放宽节点类型、最小度或 k-core 筛选后再试。";
    elements.graphCaption.textContent = "当前查询没有可绘制节点。";
    elements.graphViewKicker.textContent = state.graphQuery ? "检索结果" : "结构概览";
    elements.graphViewTitle.textContent = state.graphQuery ? "没有匹配条目" : "没有可显示结构";
    return;
  }

  const mode = graph.mode || "overview";
  if (mode === "neighborhood") {
    const focus = nodes.find((node) => node.id === graph.focus_node_id);
    elements.graphViewKicker.textContent = `${graph.depth} 跳邻域`;
    elements.graphViewTitle.textContent = focus?.label || "条目连接";
  } else if (mode === "path") {
    elements.graphViewKicker.textContent = "最短路径";
    elements.graphViewTitle.textContent = graph.path?.found
      ? `${formatNumber(graph.path.length)} 跳连接`
      : "当前筛选下不连通";
  } else {
    elements.graphViewKicker.textContent = "结构概览";
    elements.graphViewTitle.textContent = "这个群的关键连接";
  }

  const positions = calculateLayout(nodes, edges);
  const labelledIds = new Set(
    [...nodes]
      .sort((a, b) => Number(b.degree || 0) - Number(a.degree || 0))
      .slice(0, mode === "overview" ? 18 : 45)
      .map((node) => node.id),
  );
  if (graph.focus_node_id) labelledIds.add(graph.focus_node_id);
  edges.forEach((edge, index) => {
    const source = positions.get(edge.source);
    const target = positions.get(edge.target);
    if (!source || !target) return;
    const bend = ((hashString(`${edge.source}|${edge.target}|${index}`) % 7) - 3) * 3;
    const controlX = (source.x + target.x) / 2 - (target.y - source.y) * 0.018 + bend;
    const controlY = (source.y + target.y) / 2 + (target.x - source.x) * 0.018 + bend;
    const path = svgElement("path", {
      d: `M ${source.x} ${source.y} Q ${controlX} ${controlY} ${target.x} ${target.y}`,
      class: [
        "graph-edge",
        edge.type === "plastic_relation" &&
          ["HYPOTHESIS", "CONTESTED"].includes(edge.epistemic_state)
          ? "is-epistemically-open"
          : "",
      ].filter(Boolean).join(" "),
      "data-source": edge.source,
      "data-target": edge.target,
    });
    elements.graphEdges.append(path);

    if (edges.length <= 42) {
      const label = svgElement("text", {
        x: controlX,
        y: controlY - 5,
        class: "edge-label",
        "data-source": edge.source,
        "data-target": edge.target,
      });
      label.textContent = truncate(edge.relation, 13);
      elements.graphEdgeLabels.append(label);
    }
  });

  nodes.forEach((node) => {
    const position = positions.get(node.id);
    if (!position) return;
    const radius = 8 + Math.min(10, Math.log2(Number(node.degree || 0) + 1) * 2.25) +
      (node.id === graph.focus_node_id ? 3 : 0);
    const group = svgElement("g", {
      class: [
        "graph-node",
        labelledIds.has(node.id) ? "is-labeled" : "",
        node.path_index !== undefined ? "is-path" : "",
      ].filter(Boolean).join(" "),
      transform: `translate(${position.x} ${position.y})`,
      tabindex: "0",
      role: "button",
      "aria-label": `${typeNames[node.type] || node.type}: ${node.label}，度 ${node.degree || 0}，k-core ${node.core || 0}`,
      "data-node-id": node.id,
      "data-type": node.type,
    });
    group.append(svgElement("circle", { r: radius }));
    const label = svgElement("text", { y: radius + 16 });
    label.textContent = truncate(node.label, node.type === "cue" ? 16 : 20);
    group.append(label);
    group.addEventListener("click", (event) => {
      event.stopPropagation();
      selectNode(node.id);
    });
    group.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        selectNode(node.id);
      }
    });
    elements.graphNodes.append(group);
  });

  const metrics = graph.view_metrics || {};
  elements.graphCaption.textContent = [
    `画布 ${formatNumber(nodes.length)} 个节点 / ${formatNumber(edges.length)} 条关系`,
    `筛选全集 ${formatNumber(graph.metrics?.node_count)} 个节点`,
    graph.truncated ? `画布按上限显示 ${graph.limit} 个` : "画布未截断",
    `平均路径约 ${Number(graph.metrics?.average_path_length_estimate || 0).toFixed(2)} 跳`,
    metrics.connected_components > 1 ? `${formatNumber(metrics.connected_components)} 个画布分量` : "画布连通",
  ].join("  ·  ");
  window.requestAnimationFrame(fitGraph);
}

function applyTransform() {
  const { x, y, scale } = state.transform;
  elements.graphViewport.setAttribute("transform", `translate(${x} ${y}) scale(${scale})`);
}

function fitGraph() {
  const width = Math.max(320, elements.graphStage.clientWidth);
  const height = Math.max(320, elements.graphStage.clientHeight);
  const scale = Math.min(
    width / state.virtualSize.width,
    height / state.virtualSize.height,
    1.08,
  ) * 0.91;
  state.transform = {
    scale,
    x: (width - state.virtualSize.width * scale) / 2,
    y: (height - state.virtualSize.height * scale) / 2,
  };
  applyTransform();
}

async function focusGraphNode(nodeId) {
  state.graphFocusId = nodeId;
  state.graphPathSource = "";
  state.graphPathTarget = "";
  setGraphLoading(true);
  try {
    await loadGraph();
  } catch (error) {
    showToast(error?.message || "节点连接加载失败", "error");
  } finally {
    setGraphLoading(false);
  }
}

function selectNode(nodeId) {
  const graph = state.graph;
  const node = (graph?.nodes || []).find((item) => item.id === nodeId);
  if (!node) return;
  state.selectedNodeId = nodeId;
  const connected = new Set([nodeId]);
  (graph.edges || []).forEach((edge) => {
    if (edge.source === nodeId || edge.target === nodeId) {
      connected.add(edge.source);
      connected.add(edge.target);
    }
  });

  elements.graphNodes.querySelectorAll(".graph-node").forEach((element) => {
    const id = element.getAttribute("data-node-id");
    element.classList.toggle("is-selected", id === nodeId);
    element.classList.toggle("is-muted", !connected.has(id));
  });
  elements.graphEdges.querySelectorAll(".graph-edge").forEach((element) => {
    const active =
      element.getAttribute("data-source") === nodeId ||
      element.getAttribute("data-target") === nodeId;
    element.classList.toggle("is-active", active);
    element.classList.toggle("is-muted", !active);
  });
  elements.graphEdgeLabels.querySelectorAll(".edge-label").forEach((element) => {
    const active =
      element.getAttribute("data-source") === nodeId ||
      element.getAttribute("data-target") === nodeId;
    element.classList.toggle("is-muted", !active);
  });
  renderInspector(node);
}

function renderInspectorPlaceholder() {
  elements.inspectorKicker.textContent = "结构分析";
  elements.inspectorHeading.textContent = "关键节点与分布";
  elements.inspector.replaceChildren();
  const graph = state.graph;
  if (!graph) {
    const root = document.createElement("div");
    root.className = "inspector-placeholder";
    root.append(
      textElement("strong", "先选择一个群聊"),
      textElement("p", "读取结构后，这里会显示枢纽节点和度分布。"),
    );
    elements.inspector.append(root);
    return;
  }
  const root = document.createElement("div");
  root.className = "analysis-summary";
  root.append(
    textElement(
      "p",
      `指标基于当前筛选图的无向投影；最短路径与分量按弱连通计算。路径长度为 ${graph.metrics?.path_sample_size || 0} 个起点的估计值。`,
      "panel-note",
    ),
  );
  const metrics = graph.metrics || {};
  const networkSummary = document.createElement("section");
  networkSummary.className = "analysis-block";
  networkSummary.append(textElement("h3", "网络摘要"));
  appendDetailList(networkSummary, [
    ["无向关系", formatNumber(metrics.unique_edge_count)],
    ["孤立节点", formatNumber(metrics.isolated_nodes)],
    [
      "最大分量",
      `${formatNumber(metrics.giant_component_size)}（${(
        Number(metrics.giant_component_ratio || 0) * 100
      ).toFixed(1)}%）`,
    ],
    ["平均最短路径（估计）", Number(metrics.average_path_length_estimate || 0).toFixed(2)],
    ["直径（估计）", formatNumber(metrics.diameter_estimate)],
    ["有向互惠率", `${(Number(metrics.reciprocity || 0) * 100).toFixed(2)}%`],
  ]);
  root.append(networkSummary);
  const hubs = document.createElement("section");
  hubs.className = "analysis-block";
  hubs.append(textElement("h3", "连接度最高的条目"));
  const hubList = document.createElement("div");
  hubList.className = "hub-list";
  (graph.top_nodes || []).slice(0, 8).forEach((node) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "hub-button";
    button.append(
      textElement("strong", node.label || node.id),
      textElement("span", `度 ${node.degree}`),
      textElement("small", `${typeNames[node.type] || node.type} · k-core ${node.core} · 分量 C${node.component_id}`),
    );
    button.addEventListener("click", () => focusGraphNode(node.id));
    hubList.append(button);
  });
  if (!hubList.children.length) {
    hubList.append(textElement("p", "当前筛选下没有节点。", "panel-note"));
  }
  hubs.append(hubList);
  root.append(hubs);

  const distribution = document.createElement("section");
  distribution.className = "analysis-block";
  distribution.append(textElement("h3", "度分布"));
  const table = document.createElement("table");
  table.className = "degree-table";
  const head = document.createElement("thead");
  const headRow = document.createElement("tr");
  headRow.append(textElement("th", "度"), textElement("th", "节点数"));
  head.append(headRow);
  const body = document.createElement("tbody");
  (graph.degree_histogram || []).forEach((bucket) => {
    const row = document.createElement("tr");
    row.append(textElement("td", bucket.label), textElement("td", formatNumber(bucket.count)));
    body.append(row);
  });
  table.append(head, body);
  distribution.append(table);
  root.append(distribution);
  elements.inspector.append(root);
}

function appendDetailList(container, rows) {
  const list = document.createElement("dl");
  list.className = "detail-list";
  rows.forEach(([label, value]) => {
    if (value === null || value === undefined || value === "") return;
    const row = document.createElement("div");
    row.className = "detail-row";
    const term = textElement("dt", label);
    const detail = textElement("dd", String(value));
    row.append(term, detail);
    list.append(row);
  });
  container.append(list);
}

function renderInspector(node) {
  elements.inspectorKicker.textContent = "条目连接";
  elements.inspectorHeading.textContent = "它与哪些记忆相连";
  elements.inspector.replaceChildren();
  const root = document.createElement("div");
  root.append(
    textElement("span", typeNames[node.type] || node.type, "node-type-badge"),
    textElement("h3", node.label || "未命名节点", "inspector-title"),
    textElement("p", node.detail || "暂无摘要。", "inspector-detail"),
  );
  const connected = (state.graph?.edges || []).filter(
    (edge) => edge.source === node.id || edge.target === node.id,
  );
  const actions = document.createElement("div");
  actions.className = "inspector-actions";
  const focusButton = textElement("button", "以此为中心展开连接", "button secondary");
  focusButton.type = "button";
  focusButton.addEventListener("click", () => focusGraphNode(node.id));
  const pathStartButton = textElement("button", "设为路径起点", "button ghost");
  pathStartButton.type = "button";
  pathStartButton.addEventListener("click", () => {
    state.graphPathSource = node.id;
    state.graphPathTarget = "";
    updatePathStatus();
    showToast(`已将“${node.label}”设为路径起点。`);
  });
  const pathTargetButton = textElement("button", "设为路径终点", "button ghost");
  pathTargetButton.type = "button";
  pathTargetButton.disabled = !state.graphPathSource || state.graphPathSource === node.id;
  pathTargetButton.addEventListener("click", async () => {
    if (!state.graphPathSource || state.graphPathSource === node.id) return;
    state.graphPathTarget = node.id;
    setGraphLoading(true);
    try {
      await loadGraph();
    } catch (error) {
      showToast(error?.message || "最短路径计算失败", "error");
    } finally {
      setGraphLoading(false);
    }
  });
  actions.append(focusButton, pathStartButton, pathTargetButton);
  root.append(actions);
  appendDetailList(root, [
    ["节点 ID", node.id],
    ["画布内关系", connected.length],
    ["全集度", node.degree],
    ["入度 / 出度", `${node.in_degree || 0} / ${node.out_degree || 0}`],
    ["k-core", node.core],
    ["弱连通分量", node.component_id ? `C${node.component_id}（${formatNumber(node.component_size)} 个节点）` : ""],
    ["开始时间", node.started_at ? formatTime(node.started_at) : ""],
    ["结束时间", node.ended_at ? formatTime(node.ended_at) : ""],
    ["原始证据", node.source_count ? `${node.source_count} 条` : ""],
    ["置信度", node.confidence !== undefined ? `${Math.round(node.confidence * 100)}%` : ""],
    ["效用", node.utility !== undefined ? Number(node.utility).toFixed(3) : ""],
    ["作用域", node.scope_type ? `${node.scope_type}:${node.scope_key || ""}` : ""],
    ["激活模式", node.activation_mode || ""],
    ["触发线索", Array.isArray(node.trigger_cues) ? node.trigger_cues.join(" · ") : ""],
    ["状态", node.status || ""],
    ["学习时间", node.learned_at ? formatTime(node.learned_at) : ""],
    ["证据键", node.source_key || ""],
    ["内部主体键", node.canonical_key || ""],
    ["账户 ID", node.account_id || ""],
    ["账户类型", node.account_type || ""],
    ["可信状态", node.epistemic_status || ""],
  ]);
  const nodeMap = new Map(
    (state.graph?.nodes || []).map((item) => [item.id, item]),
  );
  if (connected.length) {
    const connections = document.createElement("section");
    connections.className = "analysis-block";
    connections.append(textElement("h3", `当前画布中的 ${formatNumber(connected.length)} 条连接`));
    const list = document.createElement("div");
    list.className = "connection-list";
    [...connected]
      .sort((a, b) => String(a.relation).localeCompare(String(b.relation), "zh-CN"))
      .forEach((edge) => {
        const outgoing = edge.source === node.id;
        const neighborId = outgoing ? edge.target : edge.source;
        const neighbor = nodeMap.get(neighborId);
        const button = document.createElement("button");
        button.type = "button";
        button.className = "connection-button";
        button.append(
          textElement("span", outgoing ? `— ${edge.relation} →` : `← ${edge.relation} —`),
          textElement("strong", neighbor?.label || neighborId),
          textElement("small", `${typeNames[neighbor?.type] || neighbor?.type || "节点"} · 度 ${neighbor?.degree || 0}`),
        );
        button.addEventListener("click", () => selectNode(neighborId));
        list.append(button);
      });
    connections.append(list);
    root.append(connections);
  }
  const plasticRelations = connected.filter(
    (edge) => edge.type === "plastic_relation",
  );
  if (plasticRelations.length) {
    const block = document.createElement("section");
    block.className = "evidence-block";
    block.append(textElement("h3", "群内语义与仍待确认的解释"));
    plasticRelations.forEach((edge) => {
      const source = nodeMap.get(edge.source);
      const target = nodeMap.get(edge.target);
      const card = document.createElement("article");
      card.className = "relation-card";
      card.append(
        textElement(
          "strong",
          `${source?.label || edge.source} —${edge.relation}→ ${target?.label || edge.target}`,
        ),
        textElement(
          "span",
          `${edge.epistemic_state || "HYPOTHESIS"} · 置信度 ${Math.round(Number(edge.confidence || 0) * 100)}%`,
          "relation-state",
        ),
      );
      if (edge.statement) {
        card.append(textElement("p", edge.statement, "inspector-detail"));
      }
      if (edge.uncertainty) {
        card.append(
          textElement("p", `未决：${edge.uncertainty}`, "relation-uncertainty"),
        );
      }
      block.append(card);
    });
    root.append(block);
  }
  if (node.source_text) {
    const block = document.createElement("section");
    block.className = "evidence-block";
    block.append(
      textElement("h3", "原始证据"),
      textElement("p", node.source_text, "inspector-detail"),
    );
    root.append(block);
  }
  elements.inspector.append(root);

  if (node.type === "episode" && node.entity_id) {
    loadEpisodeEvidence(node, root);
  }
}

async function loadEpisodeEvidence(node, root) {
  const requestedScopeId = state.scopeId;
  const block = document.createElement("section");
  block.className = "evidence-block";
  block.append(textElement("h3", "正在读取原始证据…"));
  root.append(block);
  try {
    const data = await apiGet(
      `scopes/${requestedScopeId}/episodes/${node.entity_id}`,
    );
    if (
      state.selectedNodeId !== node.id ||
      state.scopeId !== requestedScopeId ||
      data.scope_id !== requestedScopeId
    ) return;
    block.replaceChildren();
    block.append(textElement("h3", "检索线索"));
    const chips = document.createElement("div");
    chips.className = "keyword-list";
    (data.keywords || []).forEach((item) => {
      chips.append(textElement("span", `${item.cue} · ${item.tag}`, "keyword-chip"));
    });
    block.append(chips);
    block.append(textElement("h3", `原始消息 · ${(data.messages || []).length} 条`));
    (data.messages || []).forEach((message) => {
      const card = document.createElement("article");
      card.className = "evidence-message";
      const header = document.createElement("header");
      header.append(
        textElement("span", message.sender_name || message.sender_id || message.role),
        textElement("time", formatTime(message.sent_at)),
      );
      card.append(header, textElement("p", message.plain_text || "[非文本消息]"));
      block.append(card);
    });
  } catch (error) {
    block.replaceChildren(textElement("p", error?.message || "证据读取失败", "inspector-detail"));
  }
}

async function loadMessages() {
  if (!state.scopeId) return;
  const requestedScopeId = state.scopeId;
  elements.messageBody.replaceChildren();
  const loadingRow = document.createElement("tr");
  const loadingCell = textElement("td", "正在读取消息…", "table-empty");
  loadingCell.colSpan = 4;
  loadingRow.append(loadingCell);
  elements.messageBody.append(loadingRow);
  try {
    const data = await apiGet(`scopes/${state.scopeId}/messages`, {
      query: elements.messageQuery.value.trim(),
      sender: elements.messageSender.value.trim(),
      limit: 80,
    });
    if (
      state.scopeId !== requestedScopeId ||
      data.scope_id !== requestedScopeId
    ) return;
    renderMessages(data.messages || []);
  } catch (error) {
    renderMessageError(error?.message || "消息检索失败");
  }
}

function renderMessages(messages) {
  elements.messageBody.replaceChildren();
  if (!messages.length) {
    renderMessageError("没有匹配的原始消息。", false);
    return;
  }
  messages.forEach((message) => {
    const row = document.createElement("tr");
    row.append(
      textElement("td", formatTime(message.sent_at)),
      textElement("td", message.sender_name || message.sender_id || message.role),
      textElement("td", message.plain_text || "[非文本消息]"),
      textElement("td", message.source_key || ""),
    );
    elements.messageBody.append(row);
  });
}

function renderMessageError(message, isError = true) {
  elements.messageBody.replaceChildren();
  const row = document.createElement("tr");
  const cell = textElement("td", message, "table-empty");
  if (isError) cell.style.color = "var(--danger)";
  cell.colSpan = 4;
  row.append(cell);
  elements.messageBody.append(row);
}

async function loadParticipants() {
  if (!state.scopeId) {
    state.participants = [];
    renderParticipants([]);
    return;
  }
  const requestedScopeId = state.scopeId;
  const data = await apiGet(`scopes/${state.scopeId}/participants`, { limit: 500 });
  if (state.scopeId !== requestedScopeId || data.scope_id !== requestedScopeId) return;
  state.participants = data.participants || [];
  renderParticipants(state.participants);
}

function renderParticipants(participants) {
  elements.participantBody.replaceChildren();
  if (!participants.length) {
    const row = document.createElement("tr");
    const cell = textElement("td", "当前群范围还没有账户主体。", "table-empty");
    cell.colSpan = 6;
    row.append(cell);
    elements.participantBody.append(row);
    elements.participantStatus.textContent = "平台账户 ID 是不可变主键；同名账户不会自动合并。";
    return;
  }
  const aliasOwners = new Map();
  participants.forEach((participant) => {
    (participant.aliases || []).forEach((alias) => {
      const key = String(alias.normalized_alias || alias.alias || "").trim();
      if (!key) return;
      if (!aliasOwners.has(key)) aliasOwners.set(key, new Set());
      aliasOwners.get(key).add(String(participant.account_id));
    });
  });
  const ambiguousAliases = [...aliasOwners.values()].filter((owners) => owners.size > 1).length;
  elements.participantStatus.textContent = [
    `${formatNumber(participants.length)} 个账户主体`,
    `${formatNumber(ambiguousAliases)} 个重名别名保持歧义`,
    "平台账户 ID 永不因昵称相同而合并",
  ].join(" · ");
  participants.forEach((participant) => {
    const aliases = (participant.aliases || [])
      .map((alias) => `${alias.alias} (${formatTime(alias.last_seen_at)})`)
      .join(" · ");
    const row = document.createElement("tr");
    row.append(
      textElement("td", participant.account_id || "—"),
      textElement("td", participant.current_display_name || "—"),
      textElement("td", aliases || "—"),
      textElement("td", participant.account_type || "—"),
      textElement("td", formatTime(participant.last_seen_at)),
      textElement("td", participant.canonical_key || "—"),
    );
    elements.participantBody.append(row);
  });
}

function bindGraphInteractions() {
  elements.graphSvg.addEventListener("pointerdown", (event) => {
    if (event.button !== 0 || event.target.closest?.(".graph-node")) return;
    state.isPanning = true;
    state.panStart = {
      clientX: event.clientX,
      clientY: event.clientY,
      x: state.transform.x,
      y: state.transform.y,
    };
    elements.graphSvg.classList.add("is-panning");
    elements.graphSvg.setPointerCapture(event.pointerId);
  });
  elements.graphSvg.addEventListener("pointermove", (event) => {
    if (!state.isPanning || !state.panStart) return;
    state.transform.x = state.panStart.x + event.clientX - state.panStart.clientX;
    state.transform.y = state.panStart.y + event.clientY - state.panStart.clientY;
    applyTransform();
  });
  const endPan = (event) => {
    if (!state.isPanning) return;
    state.isPanning = false;
    state.panStart = null;
    elements.graphSvg.classList.remove("is-panning");
    if (elements.graphSvg.hasPointerCapture(event.pointerId)) {
      elements.graphSvg.releasePointerCapture(event.pointerId);
    }
  };
  elements.graphSvg.addEventListener("pointerup", endPan);
  elements.graphSvg.addEventListener("pointercancel", endPan);
  elements.graphSvg.addEventListener(
    "wheel",
    (event) => {
      event.preventDefault();
      const rect = elements.graphSvg.getBoundingClientRect();
      const pointerX = event.clientX - rect.left;
      const pointerY = event.clientY - rect.top;
      const oldScale = state.transform.scale;
      const factor = event.deltaY > 0 ? 0.9 : 1.1;
      const newScale = Math.max(0.18, Math.min(2.6, oldScale * factor));
      const worldX = (pointerX - state.transform.x) / oldScale;
      const worldY = (pointerY - state.transform.y) / oldScale;
      state.transform.scale = newScale;
      state.transform.x = pointerX - worldX * newScale;
      state.transform.y = pointerY - worldY * newScale;
      applyTransform();
    },
    { passive: false },
  );
  elements.graphSvg.addEventListener("click", (event) => {
    if (event.target.closest?.(".graph-node")) return;
    state.selectedNodeId = "";
    elements.graphNodes.querySelectorAll(".graph-node").forEach((element) => {
      element.classList.remove("is-selected", "is-muted");
    });
    elements.graphEdges.querySelectorAll(".graph-edge").forEach((element) => {
      element.classList.remove("is-active", "is-muted");
    });
    elements.graphEdgeLabels.querySelectorAll(".edge-label").forEach((element) => {
      element.classList.remove("is-muted");
    });
    renderInspectorPlaceholder();
  });
}

async function resetBudget(budgetClass) {
  if (!state.scopeId) return;
  const isFeedback = budgetClass === "feedback";
  const label = isFeedback ? "反馈学习" : "深挖与整理";
  if (!window.confirm(`重置当前群的${label} 24 小时额度？原始 Token 记录会继续保留。`)) return;
  const button = isFeedback ? elements.resetFeedbackBudgetBtn : elements.resetBudgetBtn;
  button.disabled = true;
  try {
    await apiPost(`scopes/${state.scopeId}/budget/reset`, {
      budget_class: budgetClass,
    });
    showToast(`${label}额度已重置，原始 Token 记录未删除。`);
    await refreshOverview({ preserveSelection: true, loadScope: false });
  } catch (error) {
    showToast(error?.message || `${label}额度重置失败`, "error");
  } finally {
    button.disabled = !state.scopeId;
  }
}

async function reloadGraphFromControls(errorMessage = "图分析加载失败") {
  if (!state.scopeId) return;
  setGraphLoading(true);
  try {
    await loadGraph();
    state.loadedSections.add("graph");
  } catch (error) {
    showToast(error?.message || errorMessage, "error");
  } finally {
    setGraphLoading(false);
  }
}

function bindEvents() {
  elements.refreshBtn.addEventListener("click", () => refreshOverview());
  elements.runDetailClose.addEventListener("click", () => elements.runDetailDialog.close());
  elements.runDetailFit.addEventListener("click", fitRunDetailMemoryGraph);
  elements.resetBudgetBtn.addEventListener("click", () => resetBudget("online"));
  elements.resetFeedbackBudgetBtn.addEventListener("click", () => resetBudget("feedback"));
  document.querySelectorAll("[data-section]").forEach((tab) => {
    tab.addEventListener("click", () => activateSection(tab.dataset.section));
  });
  elements.scopeSelect.addEventListener("change", async () => {
    state.scopeId = elements.scopeSelect.value;
    state.loadedSections.clear();
    state.graph = null;
    state.participants = [];
    state.graphQuery = "";
    state.graphFocusId = "";
    state.graphPathSource = "";
    state.graphPathTarget = "";
    elements.graphSearchInput.value = "";
    renderScopeMeta();
    await loadSelectedScope();
  });
  elements.graphSearchForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    state.graphQuery = elements.graphSearchInput.value.trim();
    state.graphFocusId = "";
    state.graphPathSource = "";
    state.graphPathTarget = "";
    await reloadGraphFromControls("记忆条目检索失败");
  });
  elements.graphOverviewBtn.addEventListener("click", async () => {
    state.graphQuery = "";
    state.graphFocusId = "";
    state.graphPathSource = "";
    state.graphPathTarget = "";
    elements.graphSearchInput.value = "";
    await reloadGraphFromControls();
  });
  elements.graphPathClear.addEventListener("click", async () => {
    state.graphPathSource = "";
    state.graphPathTarget = "";
    await reloadGraphFromControls();
  });
  [
    elements.graphLimit,
    elements.graphDepth,
    elements.graphStructureFilter,
    elements.graphDegreeFilter,
    elements.graphCoreFilter,
    elements.graphEpistemicFilter,
    elements.graphRelationFilter,
  ].forEach((control) => {
    control.addEventListener("change", () => reloadGraphFromControls());
  });
  elements.fitGraphBtn.addEventListener("click", fitGraph);
  document.querySelectorAll("[data-node-type]").forEach((checkbox) => {
    checkbox.addEventListener("change", async () => {
      if (checkbox.checked) state.visibleTypes.add(checkbox.dataset.nodeType);
      else state.visibleTypes.delete(checkbox.dataset.nodeType);
      state.selectedNodeId = "";
      await reloadGraphFromControls();
    });
  });
  elements.messageForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    await loadMessages();
  });
  elements.aliasBindForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!state.scopeId) return;
    const accountId = elements.aliasAccountId.value.trim();
    const alias = elements.aliasValue.value.trim();
    if (!accountId || !alias) return;
    const submit = elements.aliasBindForm.querySelector("button[type='submit']");
    submit.disabled = true;
    try {
      await apiPost(`scopes/${state.scopeId}/participants/bind_alias`, {
        account_id: accountId,
        alias,
      });
      elements.aliasValue.value = "";
      showToast(`已将别名“${alias}”绑定到账号 ${accountId}`);
      await loadParticipants();
      if (state.loadedSections.has("graph")) await loadGraph();
      await refreshOverview({ preserveSelection: true, loadScope: false });
    } catch (error) {
      showToast(error?.message || "别名绑定失败", "error");
    } finally {
      submit.disabled = false;
    }
  });
  elements.distillBtn.addEventListener("click", () => {
    elements.distillLimit.value = Math.min(
      500,
      Number(state.overview?.runtime?.distillation_max_messages || 80),
    );
    elements.distillDialog.showModal();
  });
  elements.distillCancel.addEventListener("click", (event) => {
    event.preventDefault();
    elements.distillDialog.close();
  });
  elements.distillForm.addEventListener("submit", (event) => {
    event.preventDefault();
    elements.distillConfirm.click();
  });
  elements.distillConfirm.addEventListener("click", async (event) => {
    event.preventDefault();
    if (!state.scopeId) return;
    const limit = Math.max(4, Math.min(500, Number(elements.distillLimit.value || 80)));
    elements.distillConfirm.disabled = true;
    elements.distillCancel.disabled = true;
    elements.distillConfirm.textContent = "正在整理…";
    try {
      const result = await apiPost(`scopes/${state.scopeId}/distill`, { limit });
      elements.distillDialog.close();
      showToast(
        `整理完成：新增 ${formatNumber(result.episodes)} 段情节记忆、${formatNumber(result.semantic_memories)} 条人物与事实，并更新 ${formatNumber(result.embedded_documents)} 个本地检索文档。`,
      );
      await refreshOverview({ preserveSelection: true, loadScope: true });
    } catch (error) {
      showToast(error?.message || "记忆整理失败", "error");
    } finally {
      elements.distillConfirm.disabled = false;
      elements.distillCancel.disabled = false;
      elements.distillConfirm.textContent = "开始整理";
    }
  });
  bindGraphInteractions();
}

async function initApp() {
  bindEvents();
  try {
    const bridge = await waitForPluginBridge();
    await bridge.ready();
    await refreshOverview({ preserveSelection: false, loadScope: true });
  } catch (error) {
    setConnection("error", "无法连接 AstrBot");
    showToast(error?.message || "控制台初始化失败", "error");
  }
}

initApp();
