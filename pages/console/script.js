const state = {
  overview: null,
  scopeId: "",
  graph: null,
  participants: [],
  selectedNodeId: "",
  visibleTypes: new Set([
    "participant", "cue", "episode", "semantic", "topic",
    "action", "feedback", "hypothesis",
  ]),
  transform: { x: 0, y: 0, scale: 1 },
  virtualSize: { width: 1120, height: 680 },
  isPanning: false,
  panStart: null,
};

const elements = {
  connectionDot: document.getElementById("connection-dot"),
  connectionLabel: document.getElementById("connection-label"),
  runtimeVersion: document.getElementById("runtime-version"),
  metricScopes: document.getElementById("metric-scopes"),
  metricMessages: document.getElementById("metric-messages"),
  metricEpisodes: document.getElementById("metric-episodes"),
  metricSemantics: document.getElementById("metric-semantics"),
  metricParticipants: document.getElementById("metric-participants"),
  metricPending: document.getElementById("metric-pending"),
  metricEmbeddings: document.getElementById("metric-embeddings"),
  metricStorage: document.getElementById("metric-storage"),
  scopeSelect: document.getElementById("scope-select"),
  scopeMeta: document.getElementById("scope-meta"),
  refreshBtn: document.getElementById("refresh-btn"),
  distillBtn: document.getElementById("distill-btn"),
  graphLimit: document.getElementById("graph-limit"),
  fitGraphBtn: document.getElementById("fit-graph-btn"),
  graphStage: document.getElementById("graph-stage"),
  graphSvg: document.getElementById("memory-graph"),
  graphViewport: document.getElementById("graph-viewport"),
  graphEdges: document.getElementById("graph-edges"),
  graphEdgeLabels: document.getElementById("graph-edge-labels"),
  graphNodes: document.getElementById("graph-nodes"),
  graphEmpty: document.getElementById("graph-empty"),
  graphLoading: document.getElementById("graph-loading"),
  graphCaption: document.getElementById("graph-caption"),
  inspector: document.getElementById("inspector-content"),
  messageForm: document.getElementById("message-search-form"),
  messageQuery: document.getElementById("message-query"),
  messageSender: document.getElementById("message-sender"),
  messageBody: document.getElementById("message-table-body"),
  participantStatus: document.getElementById("participant-status"),
  participantBody: document.getElementById("participant-table-body"),
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
  participant: "账户主体 PARTICIPANT",
  cue: "线索 CUE",
  episode: "情节 EPISODE",
  semantic: "语义 SEMANTIC",
  topic: "主题 TOPIC",
  action: "行动 ACTION",
  feedback: "反馈 FEEDBACK",
  hypothesis: "前瞻假设 HYPOTHESIS",
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
  elements.metricStorage.textContent = `数据库 ${formatBytes(totals.database_bytes)}`;
  elements.runtimeVersion.textContent = `MR Memory ${overview?.version || ""}`;

  const runtimeParts = ["已连接"];
  runtimeParts.push(runtime.capture_enabled ? "采集开启" : "采集关闭");
  runtimeParts.push(runtime.feedback_learning_enabled ? "反馈闭环开启" : "反馈闭环关闭");
  if (runtime.embedding_enabled) {
    if (!runtime.embedding_dependency_ready) {
      runtimeParts.push("FastEmbed 未安装");
    } else if (runtime.embedding_model_loaded) {
      runtimeParts.push("本地 Embedding 已加载");
    } else {
      runtimeParts.push("本地 Embedding 待首次加载");
    }
  }
  setConnection("online", runtimeParts.join(" · "));
}

function populateScopes(previousScopeId = "") {
  const scopes = state.overview?.scopes || [];
  elements.scopeSelect.replaceChildren();
  if (!scopes.length) {
    const option = new Option("尚无群聊记忆数据", "");
    elements.scopeSelect.append(option);
    elements.scopeSelect.disabled = true;
    elements.distillBtn.disabled = true;
    state.scopeId = "";
    state.participants = [];
    renderParticipants([]);
    elements.scopeMeta.textContent = "启用采集并收到群消息后，此处会出现独立群范围。";
    showEmptyGraph();
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
  renderScopeMeta();
}

function renderScopeMeta() {
  const scope = selectedScope();
  if (!scope) {
    elements.scopeMeta.textContent = "尚未选择群范围";
    return;
  }
  elements.scopeMeta.textContent = [
    `UMO ${scope.umo}`,
    `${formatNumber(scope.cues)} cues`,
    `${formatNumber(scope.participants)} participants`,
    `${formatNumber(scope.pending_distillation)} pending`,
    `${formatNumber(scope.topics)} topics`,
    `${formatNumber(scope.active_hypotheses)} active hypotheses`,
    `${formatNumber(scope.feedback_links)} feedback links`,
    `最近消息 ${formatTime(scope.last_message_at)}`,
  ].join("  ·  ");
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
  setGraphLoading(true);
  elements.distillBtn.disabled = true;
  try {
    await Promise.all([loadGraph(), loadMessages(), loadParticipants()]);
  } catch (error) {
    showToast(error?.message || "群范围数据加载失败", "error");
  } finally {
    setGraphLoading(false);
    elements.distillBtn.disabled = false;
  }
}

async function loadGraph() {
  const limit = Number(elements.graphLimit.value || 200);
  const graph = await apiGet(`scopes/${state.scopeId}/graph`, { limit });
  if (state.scopeId !== graph?.scope?.storage_id) return;
  state.graph = graph;
  state.selectedNodeId = "";
  renderGraph();
  renderInspectorPlaceholder();
}

function calculateLayout(nodes) {
  const grouped = {
    participant: [], cue: [], episode: [], semantic: [], topic: [],
    action: [], feedback: [], hypothesis: [],
  };
  nodes.forEach((node) => grouped[node.type]?.push(node));
  const maxPrimary = Math.max(
    grouped.participant.length,
    grouped.cue.length,
    grouped.episode.length,
    8,
  );
  const rightCount = grouped.semantic.length + grouped.topic.length;
  const feedbackCount = Math.max(
    grouped.action.length,
    grouped.feedback.length,
    grouped.hypothesis.length,
  );
  const height = Math.max(
    680,
    maxPrimary * 58 + 120,
    rightCount * 48 + 140,
    feedbackCount * 58 + 140,
  );
  const width = 1580;
  const positions = new Map();

  function distribute(items, x, minY, maxY, wobble = 0) {
    if (!items.length) return;
    const span = Math.max(0, maxY - minY);
    items.forEach((node, index) => {
      const ratio = items.length === 1 ? 0.5 : index / (items.length - 1);
      positions.set(node.id, {
        x: x + (wobble ? ((index % 3) - 1) * wobble : 0),
        y: minY + span * ratio,
      });
    });
  }

  distribute(grouped.participant, 90, 80, height - 80, 12);
  distribute(grouped.cue, 285, 80, height - 80, 16);
  distribute(grouped.episode, 525, 90, height - 90, 22);
  const topicEnd = grouped.topic.length ? Math.min(height * 0.4, 110 + grouped.topic.length * 62) : 90;
  distribute(grouped.topic, 800, 90, topicEnd, 12);
  distribute(
    grouped.semantic,
    800,
    grouped.topic.length ? Math.max(height * 0.48, topicEnd + 70) : 100,
    height - 90,
    18,
  );
  distribute(grouped.action, 1040, 90, height - 90, 14);
  distribute(grouped.feedback, 1260, 100, height - 100, 12);
  distribute(grouped.hypothesis, 1490, 90, height - 90, 16);

  state.virtualSize = { width, height };
  return positions;
}

function renderGraph() {
  const graph = state.graph;
  const nodes = (graph?.nodes || []).filter((node) => state.visibleTypes.has(node.type));
  const visibleIds = new Set(nodes.map((node) => node.id));
  const edges = (graph?.edges || []).filter(
    (edge) => visibleIds.has(edge.source) && visibleIds.has(edge.target),
  );
  elements.graphEdges.replaceChildren();
  elements.graphEdgeLabels.replaceChildren();
  elements.graphNodes.replaceChildren();
  elements.graphEmpty.classList.toggle("hidden", nodes.length > 0);
  if (!nodes.length) {
    elements.graphCaption.textContent = "没有符合当前类型筛选的节点。";
    return;
  }

  const positions = calculateLayout(nodes);
  edges.forEach((edge, index) => {
    const source = positions.get(edge.source);
    const target = positions.get(edge.target);
    if (!source || !target) return;
    const bend = ((index % 5) - 2) * 9;
    const controlX = (source.x + target.x) / 2;
    const controlY = (source.y + target.y) / 2 + bend;
    const path = svgElement("path", {
      d: `M ${source.x} ${source.y} Q ${controlX} ${controlY} ${target.x} ${target.y}`,
      class: "graph-edge",
      "data-source": edge.source,
      "data-target": edge.target,
    });
    elements.graphEdges.append(path);

    if (edges.length <= 90) {
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
    const radius = node.type === "episode"
      ? 17
      : node.type === "participant"
        ? 16
      : node.type === "cue"
        ? 11
        : node.type === "feedback"
          ? 16
          : 14;
    const group = svgElement("g", {
      class: "graph-node",
      transform: `translate(${position.x} ${position.y})`,
      tabindex: "0",
      role: "button",
      "aria-label": `${typeNames[node.type] || node.type}: ${node.label}`,
      "data-node-id": node.id,
      "data-type": node.type,
    });
    group.append(svgElement("circle", { r: radius }));
    const label = svgElement("text", { y: radius + 18 });
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

  elements.graphCaption.textContent = [
    `${formatNumber(nodes.length)} 个节点`,
    `${formatNumber(edges.length)} 条关系`,
    graph.truncated ? `仅显示最近 ${graph.limit} 个 Episode` : "完整范围",
    "拖动 / 滚轮缩放",
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
  elements.inspector.replaceChildren();
  const root = document.createElement("div");
  root.className = "inspector-placeholder";
  root.append(
    textElement("span", "⌁"),
    textElement("strong", "选择一个节点"),
    textElement("p", "Episode 可继续展开到关键词和原始聊天证据。"),
  );
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
  appendDetailList(root, [
    ["节点 ID", node.id],
    ["关联数量", connected.length],
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
    ["Source Key", node.source_key || ""],
    ["Participant Key", node.canonical_key || ""],
    ["账户 ID", node.account_id || ""],
    ["账户类型", node.account_type || ""],
    ["认知状态", node.epistemic_status || ""],
  ]);
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
    block.append(textElement("h3", "Cue / Tag"));
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

function bindEvents() {
  elements.refreshBtn.addEventListener("click", () => refreshOverview());
  elements.scopeSelect.addEventListener("change", async () => {
    state.scopeId = elements.scopeSelect.value;
    renderScopeMeta();
    await loadSelectedScope();
  });
  elements.graphLimit.addEventListener("change", async () => {
    setGraphLoading(true);
    try {
      await loadGraph();
    } catch (error) {
      showToast(error?.message || "图谱加载失败", "error");
    } finally {
      setGraphLoading(false);
    }
  });
  elements.fitGraphBtn.addEventListener("click", fitGraph);
  document.querySelectorAll("[data-node-type]").forEach((checkbox) => {
    checkbox.addEventListener("change", () => {
      if (checkbox.checked) state.visibleTypes.add(checkbox.dataset.nodeType);
      else state.visibleTypes.delete(checkbox.dataset.nodeType);
      state.selectedNodeId = "";
      renderGraph();
      renderInspectorPlaceholder();
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
      await Promise.all([loadParticipants(), loadGraph()]);
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
        `整理完成：${result.episodes} episodes，${result.semantic_memories} 条语义记忆，${result.embedded_documents} 个向量文档。`,
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
    if (!window.AstrBotPluginPage) {
      throw new Error("AstrBot Plugin Page 桥接未加载");
    }
    await window.AstrBotPluginPage.ready();
    await refreshOverview({ preserveSelection: false, loadScope: true });
  } catch (error) {
    setConnection("error", "无法连接 AstrBot");
    showToast(error?.message || "控制台初始化失败", "error");
  }
}

initApp();
