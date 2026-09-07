"use strict";
const $ = (id) => document.getElementById(id);
const state = { scope: "", tab: "runtime", overview: null, next: null, query: "", graphQuery: "" };
const kinds = { foreground: "回答前回忆", recall: "回答前回忆", reconstruction: "回答前回忆", background: "后台学习", consolidation: "后台学习", feedback: "反馈学习", episode: "共同经历", semantic: "人物与事实", association: "语义连接", topic: "主题" };
const statuses = { completed: "完成", partial: "部分完成", error: "失败", failed: "失败", timeout: "超时", running: "处理中", skipped: "未执行", empty: "没有相关内容", cancelled: "已取消" };
const number = (value) => value == null ? "未记录" : Number(value).toLocaleString("zh-CN");
const duration = (value) => value == null ? "未记录" : `${(Number(value) / 1000).toFixed(1)} 秒`;
function date(value) {
  if (!value) return "未记录";
  const normalized = typeof value === "string" && /^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d$/.test(value) ? value.replace(" ", "T") + "Z" : value;
  const d = new Date(typeof normalized === "number" ? normalized * 1000 : normalized);
  return Number.isNaN(d.getTime()) ? String(value) : d.toLocaleString("zh-CN", { timeZone: "Asia/Shanghai", hour12: false });
}
function el(tag, text, className) {
  const node = document.createElement(tag);
  if (text != null) node.textContent = text;
  if (className) node.className = className;
  return node;
}
function button(text, action, className) {
  const node = el("button", text, className);
  node.type = "button";
  node.addEventListener("click", () => attempt(action));
  return node;
}
function notice(message, error = false) {
  $("notice").textContent = message;
  $("notice").hidden = !message;
  $("notice").classList.toggle("error", error);
}
async function attempt(action) {
  try { return await action(); } catch (error) { notice(error.message || String(error), true); }
}
async function api(endpoint, params = {}, post = false) {
  const result = await window.AstrBotPluginPage[post ? "apiPost" : "apiGet"](endpoint, params);
  // The host bridge unwraps the standard success envelope.
  if (result?.status === "error") throw new Error(result.message || "请求未完成");
  return result?.status === "success" && "data" in result ? result.data : result;
}
function path(endpoint) { return `scopes/${encodeURIComponent(state.scope)}/${endpoint}`; }
function empty(parent, text) { parent.replaceChildren(el("p", text, "empty")); }
function tokens(usage) {
  if (!usage || !Object.keys(usage).length) return null;
  if (Array.isArray(usage)) {
    const values = usage.map(tokens).filter((v) => v != null);
    return values.length ? values.reduce((a, b) => a + b, 0) : null;
  }
  if (usage.total_tokens != null || usage.total != null) return usage.total_tokens ?? usage.total;
  if (usage.input_other != null || usage.input_cached != null) return Number(usage.input_other || 0) + Number(usage.input_cached || 0) + Number(usage.output || 0);
  const input = usage.input_tokens ?? usage.input ?? usage.prompt_tokens;
  const output = usage.output_tokens ?? usage.output ?? usage.completion_tokens;
  return input != null || output != null ? Number(input || 0) + Number(output || 0) : null;
}
function details(parent, title, value) {
  const node = el("details");
  node.append(el("summary", title), el("pre", typeof value === "string" ? value : JSON.stringify(value, null, 2)));
  parent.append(node);
}
function proseSection(parent, heading, text, missing = "没有保存这部分内容") {
  const section = el("section"); section.append(el("h3", heading), el("div", text || missing, "prose")); parent.append(section);
}
function openDetail(title) {
  $("detail-title").textContent = title;
  empty($("detail-content"), "正在读取…");
  if (!$("detail").open) $("detail").showModal();
}

async function refresh() {
  $("connection").textContent = "正在读取…";
  const overview = await api("overview"); state.overview = overview;
  const select = $("scope-select"); select.replaceChildren();
  for (const s of overview.scopes) {
    const option = el("option", `群 ${s.group_id} · ${s.platform_id}${s.enabled ? "" : "（未启用）"}`);
    option.value = s.id; select.append(option);
  }
  if (!overview.scopes.some((s) => s.id === state.scope)) state.scope = overview.scopes[0]?.id || "";
  select.value = state.scope; select.disabled = !state.scope; $("distill").disabled = !state.scope;
  $("connection").textContent = "已连接 AstrBot";
  if (overview.errors?.length) notice(overview.errors.join("\n"), true);
  if (!state.scope) {
    select.append(el("option", "还没有群聊记录"));
    $("scope-summary").textContent = "收到群消息后会建立记忆库";
    empty($("metrics"), "暂无记忆库"); return;
  }
  const scope = overview.scopes.find((s) => s.id === state.scope);
  $("scope-summary").textContent = scope.enabled ? "此群已启用 MR" : "此群保留历史记录，当前未启用";
  $("scope-meta").textContent = scope.umo;
  const r = overview.runtime;
  $("policies").replaceChildren(...[
    `群消息记录：${r.capture ? "启用" : "关闭"}`,
    `回答前回忆：${r.recall ? "启用" : "关闭"}`,
    `后台学习：${r.learning ? "启用" : "关闭"}`,
    `反馈学习：${r.feedback ? "启用" : "关闭"}`,
    `本地语义检索：${r.embedding_enabled === false ? "关闭" : r.embedding_loaded ? "模型已加载" : "模型尚未加载"}`,
    `回忆模型：${r.provider || "尚未配置"}`,
    ...(r.background_provider && r.background_provider !== r.provider ? [`后台模型：${r.background_provider}`] : []),
    `回忆异常超时：${r.memory_timeout_seconds} 秒`,
  ].map((text) => el("span", text, "policy")));
  await loadTab();
}

async function loadTab() {
  if (!state.scope) return;
  const scope = state.scope;
  if (state.tab === "runtime") {
    const [overview, runData] = await Promise.all([api(path("overview")), api(path("runs"))]);
    if (state.scope !== scope) return;
    const c = overview.counts;
    const rows = runData.runs || [];
    const latencies = rows.filter((r) => ["foreground", "recall", "reconstruction"].includes(r.kind) && r.elapsed_ms != null).map((r) => Number(r.elapsed_ms)).sort((a, b) => a - b);
    const mid = Math.floor(latencies.length / 2);
    const median = !latencies.length ? null : latencies.length % 2 ? latencies[mid] : (latencies[mid - 1] + latencies[mid]) / 2;
    const budget = state.overview.runtime.background_rolling24h_budget;
    const feedbackBudget = state.overview.runtime.feedback_rolling24h_budget;
    $("metrics").replaceChildren(...[
      ["回忆耗时中位数", duration(median), `最近记录中的 ${latencies.length} 次回答前回忆`],
      ["最近 24 小时后台用量", number(overview.background_tokens_rolling24h), `Token · 额度 ${budget > 0 ? number(budget) : "不限"}；反馈另用 ${number(overview.feedback_tokens_rolling24h)} / ${feedbackBudget > 0 ? number(feedbackBudget) : "不限"}`],
      ["原始消息", number(c.messages), `最近消息：${date(c.last_message_at)}`],
      ["待整理消息", number(c.pending), `上次整理：${date(overview.state.consolidated_at)}`],
      ["共同经历", number(c.episodes), "保留的情节记忆"],
      ["人物与事实", number(c.semantics), "当前有效的语义记忆"],
      ["语义连接", number(c.associations), "仍可用于搜索的关联"],
      ["检索向量", number(c.embeddings), "数据库中已保存的索引项"],
    ].map(([label, value, caption]) => { const card = el("article", null, "metric"); card.append(el("span", label), el("strong", value), el("small", caption)); return card; }));
    const body = $("runs"); body.replaceChildren();
    if (!rows.length) { const tr = el("tr"); const td = el("td", "尚无保存的调用详情。旧版本没有留下的过程不会补造。", "empty"); td.colSpan = 6; tr.append(td); body.append(tr); }
    for (const run of rows) {
      const tr = el("tr"); const question = el("td", null, "question");
      question.append(el("span", kinds[run.kind] || run.kind || "调用", "run-kind"), el("span", run.question || run.detail || "后台整理群聊"));
      const status = el("td"); status.append(el("span", statuses[run.status] || run.status || "未记录", `status ${run.status}`));
      const action = el("td"); action.append(button("查看", () => showRun(run.id)));
      tr.append(el("td", date(run.started_at), "time"), question, status, el("td", duration(run.elapsed_ms)), el("td", number(tokens(run.usage))), action); body.append(tr);
    }
  } else if (state.tab === "memory") { await searchMemory(); }
  else if (state.tab === "people") { await searchPeople(); }
  else { await searchMessages(); }
}

async function showRun(id) {
  openDetail("调用详情");
  const run = await api(path(`runs/${encodeURIComponent(id)}`));
  const parent = $("detail-content"); parent.replaceChildren();
  parent.append(el("p", `${date(run.started_at)} · ${kinds[run.kind] || run.kind} · ${statuses[run.status] || run.status} · ${duration(run.elapsed_ms)} · ${number(tokens(run.usage))} Token`, "meta"));
  proseSection(parent, "当时的问题", run.question, "后台学习，没有用户提问");
  proseSection(parent, "提供给主意识的语义背景", run.background, "没有保存背景，不能据此推测当时注入了什么");
  if (run.detail) proseSection(parent, "处理情况", typeof run.detail === "string" ? run.detail : JSON.stringify(run.detail, null, 2));
  if (run.items?.length) details(parent, "本次形成的记忆", run.items);
  const tools = run.tool_calls || [];
  const toolSection = el("section"); toolSection.append(el("h3", `搜索与阅读过程 · ${tools.length} 项记录`));
  if (!tools.length) toolSection.append(el("p", "本次没有保存工具调用过程。", "footnote"));
  for (const [i, tool] of tools.entries()) details(toolSection, `${i + 1}. ${tool.name || tool.tool || tool.function?.name || "工具调用"}`, tool);
  parent.append(toolSection);
  if (run.messages?.length) details(parent, "本次模型上下文（完整保存内容）", run.messages);
  details(parent, "原始用量记录", run.usage || "未记录");
}

async function searchMemory() {
  const scope = state.scope; state.graphQuery = $("memory-query").value.trim();
  const [data, graph] = await Promise.all([api(path("memories"), { query: state.graphQuery, kind: $("memory-kind").value }), api(path("graph"), { query: state.graphQuery })]);
  if (scope !== state.scope) return;
  const list = $("memory-list"); list.replaceChildren();
  if (!data.items.length) empty(list, "没有匹配的记忆，可换个关键词再查。");
  for (const item of data.items) {
    const card = button("", () => showMemory(item.kind, item.id), "memory-card");
    card.append(el("small", kinds[item.kind] || item.kind), el("h3", item.title || item.relation || "未命名记忆"), el("p", item.summary || item.statement), el("small", `${(item.source_ids || []).length} 条原始消息 · ${date(item.updated_at || item.created_at)}`)); list.append(card);
  }
  renderGraph(graph);
}
async function showMemory(kind, id) {
  openDetail(kinds[kind] || "记忆详情");
  const item = await api(path(`memory/${encodeURIComponent(kind)}/${encodeURIComponent(id)}`));
  const parent = $("detail-content"); parent.replaceChildren();
  proseSection(parent, item.title || item.relation || "记忆内容", item.summary || item.statement);
  if (item.source && item.target) parent.append(el("p", `${item.source} → ${item.relation} → ${item.target}`, "meta"));
  if (item.uncertainty) proseSection(parent, "尚不确定的部分", item.uncertainty);
  const sources = el("section"); sources.append(el("h3", "回到原始消息"));
  const actions = el("div", null, "source-actions");
  for (const messageId of item.source_ids || []) actions.append(button(`消息 ${messageId}`, () => showContext(messageId)));
  if (!actions.children.length) actions.append(el("p", "这条旧记忆未保留可打开的原文关联。", "footnote"));
  sources.append(actions); parent.append(sources);
  details(parent, "记忆原始字段", item);
}
function renderGraph(data) {
  const svg = $("graph"); svg.replaceChildren();
  const ns = "http://www.w3.org/2000/svg";
  const make = (tag, attrs) => { const n = document.createElementNS(ns, tag); for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, String(v)); return n; };
  $("graph-caption").textContent = `${data.nodes.length} 个节点 · ${data.edges.length} 条实际连接（每次最多 ${data.limit} 条）；点击节点查看邻接，点击下方关系查看原文。`;
  const positions = new Map();
  data.nodes.forEach((node, i) => { const a = i / Math.max(data.nodes.length, 1) * Math.PI * 2 - Math.PI / 2; const radius = i % 2 ? 145 : 195; positions.set(node.id, [340 + Math.cos(a) * radius * 1.35, 220 + Math.sin(a) * radius]); });
  for (const edge of data.edges) {
    const [x1, y1] = positions.get(edge.source_node_id); const [x2, y2] = positions.get(edge.target_node_id);
    const line = make("line", { x1, y1, x2, y2 }); const title = make("title", {}); title.textContent = edge.statement || edge.relation; line.append(title); svg.append(line);
  }
  for (const node of data.nodes) {
    const [x, y] = positions.get(node.id); const group = make("g", { class: "node", tabindex: 0, role: "button", "aria-label": `展开 ${node.label} 的相邻连接` });
    const title = make("title", {}); title.textContent = node.label;
    const text = make("text", { x, y: y + 21, "text-anchor": "middle" }); text.textContent = node.label.length > 13 ? node.label.slice(0, 13) + "…" : node.label;
    group.append(make("circle", { cx: x, cy: y, r: 7 }), text, title);
    const expand = () => attempt(async () => renderGraph(await api(path("graph"), { node_id: node.id })));
    group.addEventListener("click", expand); group.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); expand(); } }); svg.append(group);
  }
  const list = $("edge-list"); list.replaceChildren();
  if (!data.edges.length) { const text = make("text", { x: 340, y: 220, "text-anchor": "middle" }); text.textContent = "没有匹配的语义连接"; svg.append(text); }
  for (const edge of data.edges) list.append(button(`${edge.source} → ${edge.relation} → ${edge.target}`, () => showMemory("association", edge.id)));
}

async function searchPeople() {
  const scope = state.scope; const data = await api(path("participants"), { query: $("people-query").value.trim() });
  if (scope !== state.scope) return;
  const body = $("people"); body.replaceChildren();
  for (const person of data.participants) {
    const tr = el("tr"); const account = el("td"); account.append(button(person.account_id, () => { $("alias-account").value = person.account_id; $("alias-value").focus(); }, "account-btn"));
    tr.append(account, el("td", person.name), el("td", (person.aliases || []).join(" · ") || "未记录别名", "aliases"), el("td", person.membership === "platform_roster" ? `群成员列表 · ${date(person.roster_fetched_at)}` : "历史消息")); body.append(tr);
  }
  if (!data.participants.length) { const tr = el("tr"); const td = el("td", "没有匹配的账户", "empty"); td.colSpan = 4; tr.append(td); body.append(tr); }
}
function readableContent(content) {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) return content.map(readableContent).filter(Boolean).join("\n");
  if (!content || typeof content !== "object") return "";
  if (["plain", "text"].includes(String(content.type).toLowerCase())) return content.text || content.data?.text || "";
  return readableContent(content.content || content.nodes || content.messages || content.data?.content);
}
function appendMessages(parent, messages, contextButtons = true) {
  for (const message of messages) {
    const article = el("article", null, "message"); const header = el("header");
    header.append(el("strong", message.sender_name || message.sender_id || "未知发送者"), el("span", { USER: "群友", BOT: "实际发送", SYSTEM: "生成 / 工具活动" }[message.role] || message.role), el("span", date(message.sent_at)));
    if (contextButtons) header.append(button("查看前后文", () => showContext(message.id)));
    const rawText = message.plain_text || "";
    const text = !rawText || /^\[(node|nodes|forward)\]$/.test(rawText.trim()) ? readableContent(message.content) || rawText : rawText;
    article.append(header, el("div", text || "这条消息没有纯文本，请展开消息结构。", "message-text"));
    details(article, `消息 ${message.id} · 结构、引用与附件`, { content: message.content, relations: message.relations, attachments: message.attachments }); parent.append(article);
  }
}
async function searchMessages(older = false) {
  const scope = state.scope; const query = $("message-query").value.trim();
  const data = await api(path("messages"), { query, ...(older && state.next ? { before_id: state.next } : {}) });
  if (scope !== state.scope) return;
  if (!older) $("message-list").replaceChildren();
  appendMessages($("message-list"), data.messages);
  if (!$("message-list").children.length) empty($("message-list"), "没有匹配的原始消息。");
  state.next = data.next_before_id; $("older").hidden = !state.next;
}
async function showContext(id) {
  openDetail(`消息 ${id} 的前后文`);
  const data = await api(path(`context/${id}`));
  const parent = $("detail-content"); parent.replaceChildren();
  parent.append(el("p", "按时间先后显示，目标消息前后各最多 8 条。", "meta"));
  appendMessages(parent, data.messages, false);
  if (!data.messages.length) empty(parent, "找不到原文，可能已撤回或移除。");
}

$("close-detail").addEventListener("click", () => $("detail").close());
$("scope-select").addEventListener("change", () => { state.scope = $("scope-select").value; attempt(refresh); });
$("refresh").addEventListener("click", () => attempt(refresh));
document.querySelectorAll("[data-tab]").forEach((tab) => tab.addEventListener("click", () => {
  state.tab = tab.dataset.tab;
  document.querySelectorAll("[data-tab]").forEach((n) => n.removeAttribute("aria-current")); tab.setAttribute("aria-current", "page");
  document.querySelectorAll("[data-view]").forEach((view) => { view.hidden = view.dataset.view !== state.tab; }); attempt(loadTab);
}));
for (const [form, action] of [["memory-search", searchMemory], ["people-search", searchPeople], ["message-search", () => searchMessages()]]) $(form).addEventListener("submit", (event) => { event.preventDefault(); attempt(action); });
$("graph-reset").addEventListener("click", () => attempt(async () => renderGraph(await api(path("graph"), { query: state.graphQuery }))));
$("older").addEventListener("click", () => attempt(() => searchMessages(true)));
$("alias-form").addEventListener("submit", (event) => { event.preventDefault(); attempt(async () => {
  const data = await api(path("participants/bind_alias"), { account_id: $("alias-account").value.trim(), alias: $("alias-value").value.trim() }, true);
  notice(data.message); $("alias-value").value = ""; await searchPeople();
}); });
$("distill").addEventListener("click", () => attempt(async () => {
  const scope = state.scope; $("distill").disabled = true; $("distill").textContent = "正在整理…";
  notice("正在整理此群尚未学到的消息，本次仍按已配置的后台额度运行。");
  try {
    const result = await api(path("distill"), {}, true);
    notice(result?.reason || result?.detail || (result?.status === "completed" ? `整理完成，更新 ${result.written_count ?? "未记录数量的"} 项记忆。` : `本次整理状态：${statuses[result?.status] || result?.status || "未记录"}`), ["error", "failed"].includes(result?.status));
    if (scope === state.scope) await loadTab();
  } finally { $("distill").disabled = !state.scope; $("distill").textContent = "整理新消息"; }
}));
attempt(async () => {
  const deadline = performance.now() + 5000;
  while (!window.AstrBotPluginPage) {
    if (performance.now() > deadline) throw new Error("请从 AstrBot 的 MR 插件页面入口打开控制台。");
    await new Promise((resolve) => setTimeout(resolve, 25));
  }
  await refresh();
});
