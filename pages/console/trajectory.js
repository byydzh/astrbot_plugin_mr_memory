/* Recorded activity only: no model-generated summaries or inferred tool timing. */
(function (root) {
  "use strict";
  const phases = { input: "接收输入", readiness: "准备记忆", model: "模型调用", read: "搜索与阅读", write: "保存理解", index: "更新索引", inject: "提供背景", output: "形成输出", end: "本次结束" };
  const toolNames = { search_messages: "搜索原文", search_memories: "搜索记忆目录", semantic_search: "语义检索", context: "展开前后文", memory: "打开记忆", member: "查询成员", graph: "查看连接", activity: "活动统计", remember: "保存记忆" };
  const labelTitle = (title) => Object.entries(toolNames).reduce((value, [name, label]) => value.replaceAll(name, label), String(title || ""));
  const statuses = { running: "进行中", started: "进行中", completed: "完成", returned: "已返回", error: "失败", failed: "失败", cancelled: "已取消", partial: "部分完成", timeout: "超时", skipped: "未执行", unknown: "状态未记录", missing: "未保存返回" };
  const own = (object, key) => Object.prototype.hasOwnProperty.call(object || {}, key);
  const running = (status) => ["running", "started"].includes(status);
  const parse = (value) => { if (typeof value !== "string") return value; try { return JSON.parse(value); } catch { return value; } };
  function text(value) {
    if (typeof value === "string") return value;
    if (Array.isArray(value)) return value.map(text).filter(Boolean).join("\n");
    if (!value || typeof value !== "object") return "";
    return value.text || value.completion_text || text(value.content);
  }
  function normalize(run) {
    if (run.trace_version >= 1 || run.steps?.length) {
      const steps = [], paired = new Map();
      for (const [index, event] of [...(run.steps || [])].sort((a, b) => a.seq - b.seq).entries()) {
        const data = event.data || {}, phase = event.phase || "input";
        const toolId = data.tool_call_id;
        const explicitPair = data.operation_id != null || Boolean(toolId) || phase === "model" && data.turn != null;
        const pair = data.operation_id != null ? `${phase}:operation:${data.operation_id}` : toolId ? `tool:${data.turn ?? "unrecorded"}:${toolId}` : phase === "model" && data.turn != null ? `model:${data.turn}` : `${phase}:${event.title || ""}`;
        let step = paired.get(pair);
        if (!step || (!explicitPair && (!running(step.status) || running(event.status)))) {
          step = { id: `step:${event.seq ?? index}`, seq: event.seq ?? index, phase, title: event.title || phases[phase] || phase,
            at: event.at ?? null, status: event.status || "unknown", data: {}, events: [], toolId, legacy: false };
          steps.push(step); paired.set(pair, step);
        }
        step.status = event.status || step.status;
        step.data = { ...step.data, ...data };
        step.title = toolNames[step.data.name] || event.title || step.title;
        step.events.push(event);
        step.turn = step.data.turn;
        step.group = step.toolId && step.turn != null ? `turn:${step.turn}` : step.data.parallel_group || null;
        if (running(event.status)) step.startedAt = event.at;
        else step.endedAt = event.at;
      }
      return { steps, legacy: false };
    }
    const steps = [], calls = run.tool_calls || [], byId = new Map(), used = new Set();
    calls.forEach((call) => { const id = call.tool_call_id || call.id; if (id) { const entries = byId.get(String(id)) || []; entries.push(call); byId.set(String(id), entries); } });
    const pending = new Map(); let turn = 0;
    const add = (phase, title, data, extras = {}) => {
      const step = { id: `legacy:${steps.length}`, seq: steps.length, at: null, phase, title, status: "unknown", legacy: true, data, events: [], ...extras };
      steps.push(step); return step;
    };
    function toolStep(call, round, unplaced = false) {
      const id = call.id || call.tool_call_id, recorded = id ? (byId.get(String(id)) || []).find((record) => !used.has(record)) : null;
      if (recorded) used.add(recorded);
      const name = call.function?.name || call.name || call.tool || recorded?.name || "未记录工具名";
      const data = { name, arguments: parse(call.function?.arguments ?? call.arguments ?? recorded?.arguments),
        ...(recorded && own(recorded, "result") ? { result: recorded.result } : {}), elapsed_ms: recorded?.elapsed_ms ?? call.elapsed_ms };
      const step = add(name === "remember" ? "write" : "read", toolNames[name] || `调用 ${name}`, data,
        { toolId: id, turn: round, group: round != null ? `legacy-turn:${round}` : null, unplaced,
          status: recorded?.status || call.status || "unknown" });
      if (id) pending.set(`${round}:${id}`, step);
      return step;
    }
    for (const message of run.messages || []) {
      if (message.role === "user") add("input", steps.length ? "补充输入" : "当时的输入", { input: parse(text(message.content)) });
      else if (message.role === "assistant") {
        turn += 1;
        const requested = message.tool_calls || [];
        add("model", `第 ${turn} 轮模型输出`, { completion_text: text(message.content), requested_tools: requested.map((call) => call.function?.name || call.name) }, { turn });
        for (const call of requested) toolStep(call, turn);
      } else if (message.role === "tool") {
        let step = pending.get(`${turn}:${message.tool_call_id}`);
        if (!step) step = add("read", "未匹配到请求的工具返回", {}, { toolId: message.tool_call_id, unplaced: true });
        step.data.model_result = parse(message.content);
        if (!own(step.data, "result")) step.data.result = step.data.model_result;
        if (step.status === "unknown") step.status = "returned";
      }
    }
    for (const call of calls) if (!used.has(call)) {
      const step = toolStep(call, null, Boolean(run.messages?.length));
      if (own(call, "result")) step.data.result = call.result;
    }
    for (const step of steps) if (step.toolId && step.status === "unknown" && !own(step.data, "result")) step.status = "missing";
    if (!steps.length && run.question) add("input", "留下的问题", { text: run.question });
    return { steps, legacy: true };
  }
  const node = (tag, content, className) => {
    const result = document.createElement(tag); if (content != null) result.textContent = content;
    if (className) result.className = className; return result;
  };
  const action = (label, fn, className) => {
    const result = node("button", label, className); result.type = "button"; result.addEventListener("click", fn); return result;
  };
  const fieldNames = { current: "本条消息", recent: "近期对话", messages: "消息", working: "连续对话线索", input: "输入", readiness: "准备情况",
    question: "问题", background: "语义背景", text: "内容", completion_text: "模型普通输出", query: "检索问题", terms: "搜索词", arguments: "调用参数",
    sender_id: "发言账户", account_id: "账户", account_ids: "账户", related_account_id: "相关账户", name: "名称", title: "标题", summary: "记忆内容",
    kind: "类型", limit: "读取上限", before: "前文条数", after: "后文条数", message_id: "消息编号", start_at: "起始时间", end_at: "截止时间",
    source_ids: "原始消息", source_keys: "原文标识", source_speakers: "原文发言者", subject: "记忆主体", person: "人物", sources: "关联原文",
    content: "内容", plain_text: "消息正文", sender_name: "发言名称", sent_at: "发言时间", role: "角色", relation: "关系", source: "起点", target: "目标",
    statement: "连接内容", result: "实际结果", status: "状态", reason: "原因", detail: "处理情况", items: "记忆条目", written: "已写入记忆",
    input_other: "输入 Token（未命中缓存）", input_cached: "输入 Token（缓存）", output: "输出 Token", input_tokens: "输入 Token", output_tokens: "输出 Token", total_tokens: "总 Token",
    written_count: "写入数量", indexed: "已建立索引", pending: "等待索引", turn: "轮次", elapsed_ms: "耗时", now_unix: "请求时间", timezone: "时区",
    previous_question: "上次问题", previous_request_at: "上次请求时间", request_at: "请求时间", event: "事件", error: "错误", count: "数量" };
  function valueView(parent, value, helpers, depth = 0) {
    if (value == null) { parent.append(node("p", "未记录", "muted")); return; }
    if (typeof value !== "object") { parent.append(node("div", String(value), "trajectory-prose")); return; }
    if (Array.isArray(value)) {
      if (!value.length) { parent.append(node("p", "返回为空（0 项）", "muted")); return; }
      if (value.every((part) => part == null || typeof part !== "object")) { parent.append(node("div", value.join(" · "), "trajectory-prose")); return; }
      const list = node("div", null, "trajectory-results");
      value.forEach((part, index) => {
        const item = node("details", null, "trajectory-result");
        const title = part?.sender_name ? `${part.sender_name} · ${helpers.date(part.sent_at)}` : part?.title || part?.name || part?.relation || (part?.kind ? `${part.kind} ${part.id ?? ""}` : `第 ${index + 1} 项`);
        item.append(node("summary", title));
        const preview = text(part?.plain_text || part?.summary || part?.statement || part?.content || part?.memory?.summary);
        if (preview) item.querySelector("summary").append(node("span", preview, "result-preview"));
        const body = node("div", null, "result-body"); valueView(body, part, helpers, depth + 1); item.append(body); list.append(item);
      }); parent.append(list); return;
    }
    const dl = node("dl", null, "trajectory-fields");
    for (const [key, part] of Object.entries(value)) {
      if (key === "reasoning_content" || part == null || key === "source_keys" && value.source_ids) continue;
      const dt = node("dt", fieldNames[key] || key), dd = node("dd");
      if (["sent_at", "start_at", "end_at", "request_at", "previous_request_at", "now_unix"].includes(key)) dd.append(node("span", helpers.date(part)));
      else if (key === "source_ids" && Array.isArray(part)) {
        const links = node("div", null, "source-actions");
        for (const id of part) links.append(action(`消息 ${id}`, () => helpers.showContext(id)));
        dd.append(links); if (!part.length) dd.append(node("span", "未关联原文"));
      } else if (typeof part === "object" && depth > 0) {
        const extra = node("details"); extra.append(node("summary", Array.isArray(part) ? `${part.length} 项 · 展开阅读` : "展开内容"));
        const body = node("div"); valueView(body, part, helpers, depth + 1); extra.append(body); dd.append(extra);
      } else valueView(dd, part, helpers, depth + 1);
      dl.append(dt, dd);
    }
    parent.append(dl);
  }
  function section(parent, title, value, helpers) {
    const area = node("section", null, "trajectory-section"); area.append(node("h4", title)); valueView(area, value, helpers); parent.append(area);
  }
  class View {
    constructor(container, helpers) {
      this.container = container; this.helpers = helpers; this.selected = null; this.follow = true; this.lastDetail = "";
      this.root = node("div", null, "trajectory");
      this.meta = node("p", null, "meta"); this.intro = node("div", null, "trajectory-intro"); this.note = node("p", null, "trajectory-note");
      this.flow = node("nav", null, "trajectory-flow"); this.flow.setAttribute("aria-label", "已发生的状态路径");
      this.controls = node("div", null, "trajectory-controls");
      this.previous = action("← 上一步", () => this.move(-1)); this.next = action("下一步 →", () => this.move(1));
      this.position = node("span", null, "step-position");
      const followLabel = node("label", null, "trajectory-follow"); this.checkbox = document.createElement("input"); this.checkbox.type = "checkbox"; this.checkbox.checked = true;
      this.checkbox.addEventListener("change", () => { this.follow = this.checkbox.checked; if (this.follow && this.steps.length) this.select(this.steps.at(-1).id, true); });
      followLabel.append(this.checkbox, node("span", "跟随新步骤"));
      this.controls.append(this.previous, this.position, this.next, action("刷新记录", () => this.helpers.refresh()), followLabel);
      this.layout = node("div", null, "trajectory-layout"); this.timeline = node("nav", null, "trajectory-timeline"); this.timeline.setAttribute("aria-label", "运行步骤");
      this.detail = node("article", null, "trajectory-step-detail"); this.detail.setAttribute("aria-label", "选中步骤的内容");
      this.layout.append(this.timeline, this.detail); this.saved = node("div", null, "trajectory-saved");
      this.followUp = node("details", null, "trajectory-follow-up"); this.followUp.append(node("summary", "交给主意识之后"));
      this.followUpBody = node("div"); this.followUp.append(this.followUpBody);
      this.raw = node("details", null, "trajectory-raw"); this.raw.append(node("summary", "原始运行记录（JSON）")); this.rawText = node("pre"); this.raw.append(this.rawText);
      this.root.append(this.meta, this.intro, this.note, this.flow, this.controls, this.layout, this.saved, this.followUp, this.raw); container.replaceChildren(this.root);
    }
    update(run) {
      this.run = run; const normalized = normalize(run); this.steps = normalized.steps; this.legacy = normalized.legacy;
      const live = running(run.status), prior = this.selected;
      if (!this.selected) { this.follow = live; this.selected = live ? this.steps.at(-1)?.id : this.steps[0]?.id; }
      else if (this.follow) this.selected = this.steps.at(-1)?.id;
      if (!this.steps.some((step) => step.id === this.selected)) this.selected = this.steps[0]?.id;
      this.meta.textContent = `${this.helpers.date(run.started_at)} · ${this.helpers.kind(run.kind)} · ${statuses[run.status] || run.status || "状态未记录"} · ${this.helpers.duration(run.elapsed_ms)} · ${this.helpers.number(this.helpers.tokens(run.usage))} Token`;
      this.intro.replaceChildren();
      if (run.question) this.intro.append(node("small", "这次的问题"), node("h3", run.question));
      else {
        const input = this.steps.find((step) => step.phase === "input");
        const count = run.message_count ?? input?.data.message_count ?? input?.data.messages?.length ?? input?.data.input?.messages?.length;
        this.intro.append(node("h3", `${this.helpers.kind(run.kind)} · ${count == null ? "消息数量未记录" : `${count} 条群聊消息`}`),
          node("p", run.kind === "feedback" ? "回看机器人原回答与群友后续交流，学习其中的纠正和新的理解。" : "整理群聊经历，为后续互动保留有用的理解。", "muted"));
      }
      this.note.textContent = this.legacy ? "旧记录，步骤时间未记录。按已保存的模型轮次还原；同轮工具并列，返回按调用 ID 对应，无法推定实际完成顺序。" : "按实际发生顺序排列；同一工具的开始与返回合在一起，点开可查看每个事件。";
      this.checkbox.checked = this.follow; this.checkbox.disabled = !live;
      const top = this.timeline.scrollTop, active = this.timeline.contains(document.activeElement) ? document.activeElement.dataset.stepId : null;
      this.timeline.replaceChildren(); this.flow.replaceChildren();
      let lastFlow = null, lastGroup = null, groupNode = null;
      for (const [index, step] of this.steps.entries()) {
        const flowKey = step.group || `${step.phase}:${step.id}`;
        if (flowKey !== lastFlow) {
          if (lastFlow != null) this.flow.append(node("span", "→", "flow-arrow"));
          const siblings = step.group ? this.steps.filter((part) => part.group === step.group) : [step];
          const label = siblings.length > 1 ? `同轮工具 · ${siblings.length} 项` : phases[step.phase] || step.title;
          const chip = action(label, () => this.select(step.id), "flow-step");
          if (siblings.some((part) => part.id === this.selected)) chip.setAttribute("aria-current", "step");
          chip.dataset.status = siblings.some((part) => running(part.status)) ? "running" : siblings.some((part) => ["error", "failed"].includes(part.status)) ? "error" : step.status;
          this.flow.append(chip); lastFlow = flowKey;
        }
        if (step.group && step.group !== lastGroup) {
          groupNode = node("div", null, "trajectory-tool-group"); groupNode.append(node("small", `第 ${step.turn ?? "未记录"} 轮 · 同轮工具，按调用 ID 对应`)); this.timeline.append(groupNode);
        } else if (!step.group) groupNode = null;
        lastGroup = step.group;
        const item = action("", () => this.select(step.id), "trajectory-step"); item.dataset.stepId = step.id;
        if (step.id === this.selected) item.setAttribute("aria-current", "step");
        item.append(node("span", String(index + 1).padStart(2, "0"), "step-number"));
        const copy = node("span", null, "step-copy"); copy.append(node("strong", step.title), node("small", `${statuses[step.status] || step.status} · ${step.at ? this.helpers.date(step.at).split(" ").at(-1) : "时间未记录"}${step.unplaced ? " · 所在轮次未记录" : ""}`));
        if (step.toolId) copy.append(node("small", step.toolId, "tool-id")); item.append(copy); (groupNode || this.timeline).append(item);
      }
      this.timeline.scrollTop = top;
      if (active) [...this.timeline.querySelectorAll("button")].find((item) => item.dataset.stepId === active)?.focus({ preventScroll: true });
      this.renderSelected();
      if (this.follow && prior !== this.selected) this.timeline.querySelector('[aria-current="step"]')?.scrollIntoView({ block: "nearest", inline: "nearest" });
      const savedSignature = JSON.stringify([run.question, run.background, run.items, run.written, run.detail]);
      if (savedSignature !== this.savedSignature) {
        this.savedSignature = savedSignature; this.saved.replaceChildren();
        for (const [label, value] of [["当时的问题", run.question], ["留下的语义背景", run.background], ["保存的记忆内容", run.written?.length ? run.written : run.items], ["本次处理情况", run.detail]]) {
          if (value && (!Array.isArray(value) || value.length)) { const area = node("details"); area.append(node("summary", label)); const body = node("div"); valueView(body, value, this.helpers); area.append(body); this.saved.append(area); }
        }
      }
      const followUpSignature = JSON.stringify(run.response_events || []);
      if (followUpSignature !== this.followUpSignature) {
        this.followUpSignature = followUpSignature; this.followUpBody.replaceChildren(node("p", "以下是已保存的主意识活动，不计入上方 MR 的步骤、耗时和 Token。生成稿与实际发送分别显示。", "footnote"));
        if (run.response_events?.length) this.helpers.appendMessages(this.followUpBody, run.response_events, false);
        else this.followUpBody.append(node("p", "尚无已保存的后续活动", "empty"));
      }
      this.rawText.textContent = JSON.stringify(run, (key, value) => key === "reasoning_content" ? undefined : value, 2);
    }
    select(id, follow = false) { this.selected = id; this.follow = follow; this.update(this.run); }
    move(delta) { const index = this.steps.findIndex((step) => step.id === this.selected); const step = this.steps[index + delta]; if (step) this.select(step.id); }
    renderSelected() {
      const index = this.steps.findIndex((step) => step.id === this.selected), step = this.steps[index];
      this.previous.disabled = index <= 0; this.next.disabled = index < 0 || index >= this.steps.length - 1;
      this.position.textContent = this.steps.length ? `${index + 1} / ${this.steps.length}` : "尚无步骤";
      const signature = JSON.stringify(step);
      if (signature === this.lastDetail) return; this.lastDetail = signature; this.detail.replaceChildren();
      if (!step) { this.detail.append(node("p", running(this.run.status) ? "本次刚开始，等待第一条步骤记录…" : "此调用没有留下可还原的步骤，可展开下方已有内容。", "empty")); return; }
      this.detail.append(node("p", `${phases[step.phase] || step.phase} · ${statuses[step.status] || step.status}`, "eyebrow"), node("h3", step.title));
      const data = step.data;
      const timing = [step.at ? `记录于 ${this.helpers.date(step.at)}` : "步骤时间未记录", data.elapsed_ms != null ? `耗时 ${this.helpers.duration(data.elapsed_ms)}` : null, data.name ? `工具 ${data.name}` : null, step.toolId ? `调用 ${step.toolId}` : null].filter(Boolean).join(" · ");
      this.detail.append(node("p", timing, "meta"));
      if (step.phase === "write" && data.target) this.detail.append(node("p", { working_state: "保存连续对话的工作状态", long_term_memory: "写入长期记忆", processing_progress: "推进群聊整理进度", feedback_progress: "推进反馈学习进度" }[data.target] || `保存目标：${data.target}`, "trajectory-note"));
      const ordinary = data.completion_text ?? data.text;
      if (ordinary) section(this.detail, step.phase === "model" ? "模型普通输出" : "记录内容", ordinary, this.helpers);
      if (step.phase === "model" && !ordinary) this.detail.append(node("p", running(step.status) ? "正在等待模型返回。" : "这一轮没有保存普通文字输出。", "muted"));
      if (data.requested_tools?.length) section(this.detail, "这一轮请求的工具", data.requested_tools.map((name) => toolNames[name] || name), this.helpers);
      if (own(data, "arguments")) section(this.detail, "实际查询 / 参数", data.arguments, this.helpers);
      if (own(data, "result")) section(this.detail, step.phase === "write" ? "实际保存内容与来源" : "实际返回", data.result, this.helpers);
      if (own(data, "model_result") && JSON.stringify(data.model_result) !== JSON.stringify(data.result)) {
        const area = node("details"); area.append(node("summary", "模型实际收到的工具返回")); const body = node("div"); valueView(body, data.model_result, this.helpers); area.append(body); this.detail.append(area);
      }
      if (data.usage) section(this.detail, `本步用量 · ${this.helpers.number(this.helpers.tokens(data.usage))} Token`, data.usage, this.helpers);
      const rest = Object.fromEntries(Object.entries(data).filter(([key]) => !["text", "completion_text", "arguments", "result", "model_result", "usage", "requested_tools", "elapsed_ms", "tool_call_id", "name", "turn", "target", "operation_id", "parallel_group", "reasoning_content"].includes(key)));
      if (Object.keys(rest).length) section(this.detail, "本步记录", rest, this.helpers);
      if (["read", "write"].includes(step.phase) && !own(data, "result")) this.detail.append(node("p", running(step.status) ? "正在执行，尚未返回。" : "没有保存这一步的返回内容。", "muted"));
      if (step.events.length > 1) {
        const events = node("details"); events.append(node("summary", `本步的 ${step.events.length} 条事件`));
        for (const event of step.events) events.append(node("p", `${this.helpers.date(event.at)} · ${statuses[event.status] || event.status} · ${event.title || ""}`, "meta"));
        this.detail.append(events);
      }
    }
  }
  const api = { normalize, View, running, phases, labelTitle, renderValue: valueView };
  root.MRRunTrajectory = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof window === "undefined" ? globalThis : window);
