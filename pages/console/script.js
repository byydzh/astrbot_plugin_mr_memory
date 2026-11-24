"use strict";
const $ = (id) => document.getElementById(id);

const state = {
  scope: "",
  tab: "runtime",
  overview: null,
  scopeOverview: null,
  next: null,
  graphQuery: "",
  graphView: "backbone",
  graphGrowMode: true,
  regionFilter: "all",
  memoryReads: new Map(),
  memoryItems: [],
  memoryOffset: 0,
  memoryHasMore: false,
  selectedNodeKey: "",
  inspectedItem: null,
  graphTrail: [],
  runDetail: null,
  detailVersion: 0,
  pollTimer: null,
  pollBusy: false,
  resetBusy: false,
  workspaceSignature: "",
  manualMentions: [],
  manualDraftKey: "",
  manualRequestId: "",
  manualBusy: false,
  manualReceipt: null,
  manualPollTimer: null,
  mentionTimer: null,
};

/* =========================================================================
   DOMAIN MAPS: PLAN D REGIONS, KINDS, STATUSES
   ========================================================================= */
const REGIONS = {
  M: { code: "M", name: "M区 · 经历与跨期桥", short: "M 经历", desc: "情节经历、事件、语义概括与跨月桥" },
  S: { code: "S", name: "S区 · 熟人社会脑", short: "S 社会脑", desc: "人物理解、关系、情境认识与长期熟识" },
  K: { code: "K", name: "K区 · 规则与边界", short: "K 规则", desc: "知识条目、适用范围与边界否决约束" },
  E: { code: "E", name: "E区 · 原话依据", short: "E 原话", desc: "原始群聊发言与互动依据节点" },
};

const kinds = {
  foreground: "回答前回忆",
  recall: "回答前回忆",
  reconstruction: "回答前回忆",
  background: "后台学习",
  consolidation: "后台学习",
  feedback: "反馈学习",
  episode: "共同经历",
  event: "事件切片",
  semantic: "语义认识",
  bridge: "跨期桥",
  field: "主题场",
  understanding: "人物理解",
  observation: "观察笔记",
  relationship: "人际关系",
  situated_knowledge: "情境认识",
  familiarity: "熟识度",
  longitudinal_familiarity: "长期熟识",
  shared_social_state: "群内共识态",
  shared_social_state_verified: "已验证共识",
  atmosphere: "氛围感知",
  content: "知识条目",
  veto: "边界否决",
  scope: "适用范围",
  message: "原始消息",
  encounter: "对话回合",
  association: "语义连接",
  topic: "主题",
  pattern: "跨经历理解",
  node: "人物与概念",
  reflection: "延续思路",
};

const statuses = {
  completed: "完成",
  partial: "部分完成",
  error: "失败",
  failed: "失败",
  timeout: "超时",
  running: "处理中",
  queued: "已排队",
  pending: "待处理",
  processing: "整理中",
  processed: "已消费",
  skipped: "未执行",
  empty: "没有相关内容",
  cancelled: "已取消",
};

const reasonLabels = {
  waiting_window: "等待 0–6 时整理窗口或达到批次门槛",
  busy: "当前群正在整理中，记录已保留待处理",
  budget_exhausted: "24 小时后台额度已满，等待额度恢复",
  disabled: "当前群或后台整理已关闭",
};

const BIN_META = {
  far: { label: "跨月捷径 (far)", order: 0 },
  negative: { label: "边界 / 修订 (negative)", order: 1 },
  cross: { label: "跨区关联 (cross)", order: 2 },
  inside: { label: "区内锚点 (inside)", order: 3 },
  source_only: { label: "原文回指 (source_only)", order: 4 },
};

const number = (value) => (value == null ? "未记录" : Number(value).toLocaleString("zh-CN"));
const compactNum = (v) => {
  if (v == null) return "—";
  const n = Number(v);
  if (n >= 100000) return `${(n / 10000).toFixed(1)}w`;
  if (n >= 10000) return `${(n / 10000).toFixed(2)}w`;
  return n.toLocaleString("zh-CN");
};
const duration = (value) => (value == null ? "未记录" : `${(Number(value) / 1000).toFixed(1)} 秒`);

function date(value) {
  if (!value) return "未记录";
  const normalized =
    typeof value === "string" && /^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d$/.test(value)
      ? value.replace(" ", "T") + "Z"
      : value;
  const d = new Date(typeof normalized === "number" ? normalized * 1000 : normalized);
  return Number.isNaN(d.getTime())
    ? String(value)
    : d.toLocaleString("zh-CN", { timeZone: "Asia/Shanghai", hour12: false });
}

function shortDate(value) {
  const full = date(value);
  if (full === "未记录") return "";
  return full.replace(/^\d{4}\//, "").replace(/:\d\d$/, "");
}

function parseRegion(keyOrId, kind) {
  const s = String(keyOrId || "");
  if (s.startsWith("E|")) return "E";
  if (s.startsWith("M|")) return "M";
  if (s.startsWith("S|")) return "S";
  if (s.startsWith("K|")) return "K";
  const k = String(kind || "").toLowerCase();
  if (["message", "encounter"].includes(k)) return "E";
  if (
    [
      "understanding",
      "observation",
      "relationship",
      "situated_knowledge",
      "familiarity",
      "longitudinal_familiarity",
      "shared_social_state",
      "shared_social_state_verified",
      "atmosphere",
    ].includes(k)
  )
    return "S";
  if (["content", "veto", "scope"].includes(k)) return "K";
  return "M";
}

function deriveHeadline(text, fallback = "未命名节点", maxLen = 24) {
  const raw = String(text || "").replace(/\s+/g, " ").trim();
  if (!raw) return fallback;
  const firstBreak = raw.search(/[。！？；\n]/);
  let head = firstBreak > 3 && firstBreak <= maxLen + 8 ? raw.slice(0, firstBreak) : raw;
  if (head.length > maxLen) {
    const comma = head.slice(8, maxLen + 2).search(/[，、：:]/);
    head = comma >= 0 ? head.slice(0, 8 + comma) : head.slice(0, maxLen) + "…";
  }
  return head;
}

function deriveExcerpt(body, headline) {
  const raw = String(body || "").replace(/\s+/g, " ").trim();
  if (!raw) return "";
  const cleanHead = headline.replace(/…$/, "");
  if (cleanHead.length >= 6 && raw.startsWith(cleanHead)) {
    const rest = raw.slice(cleanHead.length).replace(/^[。！？；，、：:\s]+/, "");
    return rest || raw;
  }
  return raw;
}

function formatScopeLabel(s) {
  const gid = String(s.group_id || "");
  const isQQNum = /^\d{5,15}$/.test(gid);
  const shortId = isQQNum
    ? `QQ群 ${gid}`
    : gid.length > 14
    ? `官方群 ${gid.slice(0, 6)}…${gid.slice(-4)}`
    : `群 ${gid}`;
  return `${shortId} · ${s.platform_id}${s.enabled ? "" : "（未启用）"}`;
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
  const box = $("notice");
  box.textContent = message || "";
  box.hidden = !message;
  box.classList.toggle("error", Boolean(error));
}

async function attempt(action) {
  try {
    return await action();
  } catch (error) {
    notice(error?.message || String(error), true);
  }
}

async function api(endpoint, params = {}, post = false) {
  const result = await window.AstrBotPluginPage[post ? "apiPost" : "apiGet"](endpoint, params);
  if (result?.status === "error") throw new Error(result.message || "请求未完成");
  return result?.status === "success" && "data" in result ? result.data : result;
}

function path(endpoint, scope = state.scope) {
  return `scopes/${scope}/${endpoint}`;
}

function empty(parent, text) {
  parent.replaceChildren(el("p", text, "empty"));
}

function tokens(usage) {
  if (!usage || !Object.keys(usage).length) return null;
  if (Array.isArray(usage)) {
    const values = usage.map(tokens).filter((v) => v != null);
    return values.length ? values.reduce((a, b) => a + b, 0) : null;
  }
  if (usage.total_tokens != null || usage.total != null) return usage.total_tokens ?? usage.total;
  if (usage.input_other != null || usage.input_cached != null)
    return Number(usage.input_other || 0) + Number(usage.input_cached || 0) + Number(usage.output || 0);
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
  const section = el("section");
  section.append(el("h3", heading), el("div", text || missing, "prose"));
  parent.append(section);
}

function representationView(value) {
  if (value == null || typeof value !== "object")
    return el("span", value == null ? "null" : String(value));
  const list = el(Array.isArray(value) ? "ol" : "dl", null, "representation");
  for (const [key, part] of Object.entries(value)) {
    if (Array.isArray(value)) {
      const row = el("li");
      row.append(representationView(part));
      list.append(row);
    } else {
      const body = el("dd");
      body.append(representationView(part));
      list.append(el("dt", key), body);
    }
  }
  return list;
}

function beliefView(value) {
  if (value?.stance === "unconfirmed" && Object.keys(value).length === 1)
    return el("p", "尚未确认；目前保留为可继续理解的线索。");
  if (!value || typeof value !== "object" || Array.isArray(value)) return representationView(value);
  const labels = {
    stance: "当前看法",
    scope: "适用范围",
    reason: "理由",
    support: "支持",
    alternatives: "其他解释",
  };
  return representationView(
    Object.fromEntries(
      Object.entries(value).map(([key, part]) => [
        labels[key] || key,
        key === "stance" && part === "observed" ? "原始发言记录" : part,
      ])
    )
  );
}

/* =========================================================================
   THEME TOGGLE
   ========================================================================= */
let graphEngine = null;

function applyTheme(theme) {
  const next = theme === "light" ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  const toggle = $("theme-toggle");
  if (toggle) {
    toggle.textContent = next === "dark" ? "☀️" : "🌙";
    toggle.title = next === "dark" ? "切换到浅色主题" : "切换到深色主题";
  }
  try {
    window.localStorage?.setItem("mr_console_theme", next);
  } catch {
    /* ignore */
  }
  graphEngine?.draw();
}

(function initTheme() {
  let saved = "dark";
  try {
    saved = window.localStorage?.getItem("mr_console_theme") || "dark";
  } catch {
    saved = "dark";
  }
  applyTheme(saved);
})();

$("theme-toggle")?.addEventListener("click", () => {
  applyTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark");
});

/* =========================================================================
   INTERACTIVE HIDPI 4-REGION CANVAS GRAPH ENGINE
   ========================================================================= */
class SpatialGraphCanvas {
  constructor(canvas, stageEl, tooltipEl, callbacks) {
    this.canvas = canvas;
    this.stage = stageEl;
    this.tooltip = tooltipEl;
    this.cb = callbacks;
    this.ctx = canvas.getContext ? canvas.getContext("2d") : null;
    this.width = 900;
    this.height = 600;
    this.dpr = 1;

    this.nodes = [];
    this.edges = [];
    this.nodeMap = new Map();
    this.neighbors = new Map();
    this.centerKey = null;

    this.layoutMode = "zones"; // "zones" | "force"
    this.showAllLabels = false;
    this.visRegions = new Set(["M", "S", "K", "E"]);
    this.visBins = new Set(["far", "cross", "inside", "negative", "source_only"]);

    this.cam = { x: 0, y: 0, zoom: 1 };
    this.animTarget = null;
    this.rafId = null;

    this.hoverKey = null;
    this.selectedKey = null;
    this.dragNode = null;
    this.panStart = null;
    this.pointerMoved = false;

    this.bindEvents();
    this.resize();
  }

  palette() {
    const dark = document.documentElement.dataset.theme !== "light";
    return dark
      ? {
          bg: "#060911",
          grid: "rgba(148, 163, 184, 0.045)",
          zoneRing: "rgba(148, 163, 184, 0.14)",
          zoneLabel: "rgba(226, 232, 240, 0.65)",
          M: "#a78bfa",
          S: "#34d399",
          K: "#fb7185",
          E: "#38bdf8",
          edgeFar: "#f59e0b",
          edgeNeg: "#fb7185",
          edgeCross: "rgba(45, 212, 191, 0.64)",
          edgeInside: "rgba(148, 163, 184, 0.36)",
          edgeInactive: "rgba(100, 116, 139, 0.22)",
          pillBg: "rgba(9, 14, 26, 0.88)",
          pillBorder: "rgba(148, 163, 184, 0.28)",
          pillText: "#eef2fa",
          pillHiBorder: "#34d399",
          nodeStroke: "#060911",
          selectRing: "#ffffff",
        }
      : {
          bg: "#f5f8fc",
          grid: "rgba(15, 23, 42, 0.045)",
          zoneRing: "rgba(15, 23, 42, 0.12)",
          zoneLabel: "rgba(30, 41, 59, 0.72)",
          M: "#7c3aed",
          S: "#059669",
          K: "#e11d48",
          E: "#0284c7",
          edgeFar: "#d97706",
          edgeNeg: "#e11d48",
          edgeCross: "rgba(5, 150, 105, 0.48)",
          edgeInside: "rgba(100, 116, 139, 0.32)",
          edgeInactive: "rgba(148, 163, 184, 0.24)",
          pillBg: "rgba(255, 255, 255, 0.92)",
          pillBorder: "rgba(15, 23, 42, 0.16)",
          pillText: "#0f172a",
          pillHiBorder: "#059669",
          nodeStroke: "#ffffff",
          selectRing: "#0f172a",
        };
  }

  zoneCenters() {
    return {
      M: { x: -300, y: -165, r: 215, title: "M · 经历与跨期桥" },
      S: { x: 300, y: -165, r: 205, title: "S · 熟人社会脑" },
      K: { x: 295, y: 175, r: 190, title: "K · 规则与边界" },
      E: { x: -295, y: 175, r: 195, title: "E · 原话依据" },
    };
  }

  resize() {
    if (!this.stage || !this.canvas) return;
    const rect = this.stage.getBoundingClientRect?.() || { width: 900, height: 600 };
    const w = Math.max(320, Math.floor(rect.width || 900));
    const h = Math.max(320, Math.floor(rect.height || 600));
    this.dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.width = w;
    this.height = h;
    this.canvas.width = Math.floor(w * this.dpr);
    this.canvas.height = Math.floor(h * this.dpr);
    this.draw();
  }

  setData(data, centerKey = null) {
    this.centerKey = centerKey;
    const rawNodes = data?.nodes || [];
    const rawEdges = data?.edges || [];
    const deg = new Map();
    const nb = new Map();
    for (const n of rawNodes) {
      deg.set(n.key, 0);
      nb.set(n.key, new Set([n.key]));
    }
    for (const e of rawEdges) {
      if (deg.has(e.source_key)) deg.set(e.source_key, deg.get(e.source_key) + 1);
      if (deg.has(e.target_key)) deg.set(e.target_key, deg.get(e.target_key) + 1);
      nb.get(e.source_key)?.add(e.target_key);
      nb.get(e.target_key)?.add(e.source_key);
    }

    this.nodes = rawNodes.map((n) => {
      const region = parseRegion(n.id || n.key, n.kind);
      const d = deg.get(n.key) || 0;
      const isCenter = centerKey && (n.key === centerKey || n.id === centerKey);
      const r = isCenter ? 14 : Math.min(12.5, 5.2 + Math.sqrt(d) * 1.85);
      return {
        ...n,
        region,
        deg: d,
        r,
        shortLabel: deriveHeadline(n.label || n.title || n.id, n.id || n.key, 15),
        fullLabel: n.label || n.title || n.id || n.key,
        x: 0,
        y: 0,
        vx: 0,
        vy: 0,
      };
    });
    this.nodeMap = new Map(this.nodes.map((n) => [n.key, n]));
    this.neighbors = nb;
    this.edges = rawEdges
      .filter((e) => this.nodeMap.has(e.source_key) && this.nodeMap.has(e.target_key))
      .map((e, i) => ({
        ...e,
        bin: e.slot || (e.relation === "bridge" ? "far" : "inside"),
        active: e.active !== false,
        bendSign: i % 2 === 0 ? 1 : -1,
      }));

    this.computeLayout();
    this.fitToView(false);
    this.updateCounts();
    this.draw();
  }

  mergeData(data, anchorKey) {
    const rawNodes = data?.nodes || [];
    const rawEdges = data?.edges || [];
    const zones = this.zoneCenters();
    const anchor = this.nodeMap.get(anchorKey);
    const edgeSet = new Set(this.edges.map((e) => `${e.source_key}\u0000${e.target_key}\u0000${e.relation}`));
    let addedNodes = 0;

    for (const n of rawNodes) {
      if (this.nodeMap.has(n.key)) continue;
      const region = parseRegion(n.id || n.key, n.kind);
      const z = zones[region] || zones.M;
      const angle = (this.nodes.length + addedNodes) * 2.39996;
      const baseX = this.layoutMode === "zones" ? z.x : anchor ? anchor.x : 0;
      const baseY = this.layoutMode === "zones" ? z.y : anchor ? anchor.y : 0;
      const dist = this.layoutMode === "zones" ? 45 + (addedNodes % 7) * 20 : 85 + (addedNodes % 5) * 22;
      const entry = {
        ...n,
        region,
        deg: 1,
        r: 6.8,
        shortLabel: deriveHeadline(n.label || n.title || n.id, n.id || n.key, 15),
        fullLabel: n.label || n.title || n.id || n.key,
        x: baseX + Math.cos(angle) * dist,
        y: baseY + Math.sin(angle) * dist,
        vx: 0,
        vy: 0,
      };
      this.nodes.push(entry);
      this.nodeMap.set(entry.key, entry);
      this.neighbors.set(entry.key, new Set([entry.key]));
      addedNodes += 1;
    }

    let addedEdges = 0;
    for (const e of rawEdges) {
      if (!this.nodeMap.has(e.source_key) || !this.nodeMap.has(e.target_key)) continue;
      const ek = `${e.source_key}\u0000${e.target_key}\u0000${e.relation}`;
      if (edgeSet.has(ek)) continue;
      edgeSet.add(ek);
      this.edges.push({
        ...e,
        bin: e.slot || (e.relation === "bridge" ? "far" : "inside"),
        active: e.active !== false,
        bendSign: (this.edges.length + addedEdges) % 2 === 0 ? 1 : -1,
      });
      this.neighbors.get(e.source_key)?.add(e.target_key);
      this.neighbors.get(e.target_key)?.add(e.source_key);
      addedEdges += 1;
    }

    // Recompute degrees & radii smoothly
    const deg = new Map(this.nodes.map((n) => [n.key, 0]));
    for (const e of this.edges) {
      deg.set(e.source_key, (deg.get(e.source_key) || 0) + 1);
      deg.set(e.target_key, (deg.get(e.target_key) || 0) + 1);
    }
    for (const n of this.nodes) {
      n.deg = deg.get(n.key) || 0;
      n.r = n.key === this.centerKey ? 14 : Math.min(12.5, 5.2 + Math.sqrt(n.deg) * 1.85);
    }
    if (addedNodes > 0) this.relaxCollisions(26, false);
    this.updateCounts();
    this.draw();
    return { addedNodes, addedEdges };
  }

  updateCounts() {
    const counts = { M: 0, S: 0, K: 0, E: 0 };
    for (const n of this.nodes) counts[n.region] = (counts[n.region] || 0) + 1;
    for (const r of ["M", "S", "K", "E"]) {
      const elCnt = $(`cnt-${r.toLowerCase()}`);
      if (elCnt) elCnt.textContent = String(counts[r]);
    }
  }

  computeLayout() {
    const nodes = this.nodes;
    if (!nodes.length) return;
    const zones = this.zoneCenters();
    const golden = Math.PI * (3 - Math.sqrt(5));

    if (this.centerKey && this.nodeMap.has(this.centerKey) && this.layoutMode === "zones") {
      // Ego-centric star layout grouped by cognitive region quadrant
      const center = this.nodeMap.get(this.centerKey);
      center.x = 0;
      center.y = 0;
      const byReg = { M: [], S: [], K: [], E: [] };
      for (const n of nodes) {
        if (n.key === this.centerKey) continue;
        byReg[n.region].push(n);
      }
      const sectorAngles = {
        M: [-Math.PI * 0.92, -Math.PI * 0.56],
        S: [-Math.PI * 0.44, -Math.PI * 0.08],
        K: [Math.PI * 0.08, Math.PI * 0.44],
        E: [Math.PI * 0.56, Math.PI * 0.92],
      };
      for (const reg of ["M", "S", "K", "E"]) {
        const list = byReg[reg].sort((a, b) => b.deg - a.deg);
        const [a0, a1] = sectorAngles[reg];
        list.forEach((n, idx) => {
          const ring = Math.floor(idx / 7);
          const posInRing = idx % 7;
          const totalInRing = Math.min(7, list.length - ring * 7);
          const t = totalInRing <= 1 ? 0.5 : posInRing / (totalInRing - 1);
          const angle = a0 + (a1 - a0) * t;
          const dist = 165 + ring * 88 + (posInRing % 2) * 18;
          n.x = Math.cos(angle) * dist;
          n.y = Math.sin(angle) * dist;
        });
      }
      this.relaxCollisions(42, true);
      return;
    }

    if (this.layoutMode === "zones") {
      // 4-Region Spatial Zone Layout
      const byReg = { M: [], S: [], K: [], E: [] };
      for (const n of nodes) byReg[n.region].push(n);
      for (const reg of ["M", "S", "K", "E"]) {
        const list = byReg[reg].sort((a, b) => b.deg - a.deg);
        const z = zones[reg];
        list.forEach((n, idx) => {
          const rad = idx === 0 ? 0 : Math.sqrt((idx + 0.5) / Math.max(list.length, 1)) * (z.r * 0.78);
          const ang = idx * golden;
          n.x = z.x + Math.cos(ang) * rad;
          n.y = z.y + Math.sin(ang) * rad;
          n.vx = 0;
          n.vy = 0;
        });
      }
      // Zone-constrained force relaxation
      const iters = 95;
      for (let it = 0; it < iters; it++) {
        const cool = 1 - it / iters;
        const cap = 16 * cool + 1.0;
        // Pairwise repulsion & strict collision avoidance
        for (let i = 0; i < nodes.length; i++) {
          const a = nodes[i];
          for (let j = i + 1; j < nodes.length; j++) {
            const b = nodes[j];
            const dx = a.x - b.x || 0.01;
            const dy = a.y - b.y || 0.01;
            const dist = Math.hypot(dx, dy) || 0.01;
            const minGap = a.r + b.r + 24;
            let force = 0;
            if (dist < minGap) force = (minGap - dist) * 0.42;
            else if (a.region === b.region && dist < 140) force = 950 / (dist * dist);
            if (force > 0) {
              const fx = (dx / dist) * force;
              const fy = (dy / dist) * force;
              a.vx += fx;
              a.vy += fy;
              b.vx -= fx;
              b.vy -= fy;
            }
          }
        }
        // Soft edge pull (stronger intra-region, gentle cross-region alignment)
        for (const e of this.edges) {
          const a = this.nodeMap.get(e.source_key);
          const b = this.nodeMap.get(e.target_key);
          if (!a || !b || a === b) continue;
          const dx = b.x - a.x;
          const dy = b.y - a.y;
          const dist = Math.hypot(dx, dy) || 1;
          const k = a.region === b.region ? 0.032 : 0.008;
          const target = a.region === b.region ? 76 : 260;
          const f = (dist - target) * k;
          const fx = (dx / dist) * f;
          const fy = (dy / dist) * f;
          a.vx += fx;
          a.vy += fy;
          b.vx -= fx;
          b.vy -= fy;
        }
        // Pull back inside region territory
        for (const n of nodes) {
          const z = zones[n.region];
          n.vx += (z.x - n.x) * 0.045;
          n.vy += (z.y - n.y) * 0.045;
          const mag = Math.hypot(n.vx, n.vy) || 1;
          const step = Math.min(mag, cap);
          n.x += (n.vx / mag) * step;
          n.y += (n.vy / mag) * step;
          // Soft clamp inside region disc
          const rx = n.x - z.x;
          const ry = n.y - z.y;
          const rDist = Math.hypot(rx, ry);
          const maxR = z.r - 16;
          if (rDist > maxR) {
            n.x = z.x + (rx / rDist) * maxR;
            n.y = z.y + (ry / rDist) * maxR;
          }
          n.vx *= 0.3;
          n.vy *= 0.3;
        }
      }
    } else {
      // Free Force-Directed Layout
      const sorted = [...nodes].sort((a, b) => b.deg - a.deg);
      sorted.forEach((n, idx) => {
        const rad = idx === 0 ? 0 : Math.sqrt(idx) * 44;
        const ang = idx * golden;
        n.x = Math.cos(ang) * rad * 1.2;
        n.y = Math.sin(ang) * rad;
        n.vx = 0;
        n.vy = 0;
      });
      const iters = 110;
      const ideal = 92;
      for (let it = 0; it < iters; it++) {
        const cool = 1 - it / iters;
        const cap = 20 * cool + 1.2;
        for (let i = 0; i < nodes.length; i++) {
          const a = nodes[i];
          for (let j = i + 1; j < nodes.length; j++) {
            const b = nodes[j];
            const dx = a.x - b.x || 0.01;
            const dy = a.y - b.y || 0.01;
            const dist = Math.hypot(dx, dy) || 0.01;
            const rep = (ideal * ideal) / Math.max(dist * dist, 160);
            const fx = (dx / dist) * rep * 1.1;
            const fy = (dy / dist) * rep * 1.1;
            a.vx += fx;
            a.vy += fy;
            b.vx -= fx;
            b.vy -= fy;
          }
        }
        for (const e of this.edges) {
          const a = this.nodeMap.get(e.source_key);
          const b = this.nodeMap.get(e.target_key);
          if (!a || !b || a === b) continue;
          const dx = b.x - a.x;
          const dy = b.y - a.y;
          const dist = Math.hypot(dx, dy) || 1;
          const f = ((dist - ideal) / ideal) * 16;
          a.vx += (dx / dist) * f;
          a.vy += (dy / dist) * f;
          b.vx -= (dx / dist) * f;
          b.vy -= (dy / dist) * f;
        }
        for (const n of nodes) {
          n.vx += -n.x * 0.018;
          n.vy += -n.y * 0.022;
          const mag = Math.hypot(n.vx, n.vy) || 1;
          const step = Math.min(mag, cap);
          n.x += (n.vx / mag) * step;
          n.y += (n.vy / mag) * step;
          n.vx *= 0.35;
          n.vy *= 0.35;
        }
      }
      this.relaxCollisions(24, false);
    }
  }

  relaxCollisions(iters, lockCenter) {
    const nodes = this.nodes;
    for (let it = 0; it < iters; it++) {
      for (let i = 0; i < nodes.length; i++) {
        const a = nodes[i];
        for (let j = i + 1; j < nodes.length; j++) {
          const b = nodes[j];
          const dx = a.x - b.x || 0.01;
          const dy = a.y - b.y || 0.01;
          const dist = Math.hypot(dx, dy) || 0.01;
          const minGap = a.r + b.r + 22;
          if (dist < minGap) {
            const push = (minGap - dist) * 0.45;
            const ux = dx / dist;
            const uy = dy / dist;
            if (!(lockCenter && a.key === this.centerKey)) {
              a.x += ux * push;
              a.y += uy * push;
            }
            if (!(lockCenter && b.key === this.centerKey)) {
              b.x -= ux * push;
              b.y -= uy * push;
            }
          }
        }
      }
    }
  }

  worldToScreen(wx, wy) {
    return {
      x: (wx - this.cam.x) * this.cam.zoom + this.width / 2,
      y: (wy - this.cam.y) * this.cam.zoom + this.height / 2,
    };
  }

  screenToWorld(sx, sy) {
    return {
      x: (sx - this.width / 2) / this.cam.zoom + this.cam.x,
      y: (sy - this.height / 2) / this.cam.zoom + this.cam.y,
    };
  }

  fitToView(animate = true) {
    const vis = this.nodes.filter((n) => this.visRegions.has(n.region));
    if (!vis.length) {
      this.setCamera(0, 0, 1, animate);
      return;
    }
    let minX = Infinity,
      maxX = -Infinity,
      minY = Infinity,
      maxY = -Infinity;
    for (const n of vis) {
      minX = Math.min(minX, n.x - 55);
      maxX = Math.max(maxX, n.x + 55);
      minY = Math.min(minY, n.y - 55);
      maxY = Math.max(maxY, n.y + 55);
    }
    if (this.layoutMode === "zones" && !this.centerKey) {
      minX = Math.min(minX, -530);
      maxX = Math.max(maxX, 530);
      minY = Math.min(minY, -395);
      maxY = Math.max(maxY, 395);
    }
    const bw = Math.max(220, maxX - minX);
    const bh = Math.max(180, maxY - minY);
    const padTop = 96;
    const padBottom = 52;
    const padX = 56;
    const availW = Math.max(200, this.width - padX * 2);
    const availH = Math.max(180, this.height - padTop - padBottom);
    const zoom = Math.max(0.32, Math.min(1.45, Math.min(availW / bw, availH / bh)));
    const offsetY = (padTop - padBottom) / (2 * zoom);
    this.setCamera((minX + maxX) / 2, (minY + maxY) / 2 - offsetY, zoom, animate);
  }

  flyToNode(key) {
    const n = this.nodeMap.get(key);
    if (!n) return;
    this.selectedKey = key;
    if (!this.centerKey) {
      // Gently nudge camera while keeping the 4-region star map framed
      const nextX = this.cam.x * 0.65 + n.x * 0.35;
      const nextY = this.cam.y * 0.65 + n.y * 0.35;
      this.setCamera(nextX, nextY, Math.max(this.cam.zoom, 0.72), true);
    } else {
      this.setCamera(n.x, n.y, Math.max(this.cam.zoom, 1.2), true);
    }
  }

  setCamera(x, y, zoom, animate = true) {
    const clampedZ = Math.max(0.25, Math.min(4.0, zoom));
    if (!animate || typeof requestAnimationFrame !== "function") {
      this.cam = { x, y, zoom: clampedZ };
      this.draw();
      return;
    }
    const start = { ...this.cam };
    const t0 = performance.now();
    const dur = 240;
    if (this.rafId) cancelAnimationFrame(this.rafId);
    const step = (now) => {
      const p = Math.min(1, (now - t0) / dur);
      const ease = 1 - Math.pow(1 - p, 3);
      this.cam.x = start.x + (x - start.x) * ease;
      this.cam.y = start.y + (y - start.y) * ease;
      this.cam.zoom = start.zoom + (clampedZ - start.zoom) * ease;
      this.draw();
      if (p < 1) this.rafId = requestAnimationFrame(step);
      else this.rafId = null;
    };
    this.rafId = requestAnimationFrame(step);
  }

  hitTest(sx, sy) {
    const w = this.screenToWorld(sx, sy);
    let best = null;
    let bestDist = Infinity;
    const hitPad = 9 / this.cam.zoom;
    for (const n of this.nodes) {
      if (!this.visRegions.has(n.region)) continue;
      const d = Math.hypot(w.x - n.x, w.y - n.y);
      if (d <= n.r + hitPad && d < bestDist) {
        best = n;
        bestDist = d;
      }
    }
    return best;
  }

  updateTooltip(node, sx, sy) {
    if (!this.tooltip) return;
    if (!node) {
      this.tooltip.hidden = true;
      return;
    }
    const reg = REGIONS[node.region] || REGIONS.M;
    this.tooltip.replaceChildren();
    const head = el("div", null, "tt-head");
    head.append(
      el("span", reg.short, `reg-tag r-${node.region.toLowerCase()}`),
      el("span", kinds[node.kind] || node.kind, "kind-tag")
    );
    this.tooltip.append(
      head,
      el("div", node.id || node.key, "tt-addr"),
      el("div", node.fullLabel.slice(0, 110) + (node.fullLabel.length > 110 ? "…" : ""), "tt-body"),
      el("div", `${node.deg} 条邻边 · 单击检视 / 双击展开邻域`, "tt-meta")
    );
    const tx = Math.min(this.width - 325, Math.max(14, sx + 16));
    const ty = Math.min(this.height - 140, Math.max(14, sy + 16));
    this.tooltip.style.left = `${tx}px`;
    this.tooltip.style.top = `${ty}px`;
    this.tooltip.hidden = false;
  }

  bindEvents() {
    if (!this.canvas) return;
    const getXY = (e) => {
      const rect = this.canvas.getBoundingClientRect();
      return { sx: e.clientX - rect.left, sy: e.clientY - rect.top };
    };

    this.canvas.addEventListener("pointerdown", (e) => {
      const { sx, sy } = getXY(e);
      this.pointerMoved = false;
      const hit = this.hitTest(sx, sy);
      if (hit) {
        this.dragNode = hit;
      } else {
        this.panStart = { sx, sy, cx: this.cam.x, cy: this.cam.y };
        this.canvas.classList.add("dragging");
      }
    });

    this.canvas.addEventListener("pointermove", (e) => {
      const { sx, sy } = getXY(e);
      if (this.dragNode) {
        const w = this.screenToWorld(sx, sy);
        if (Math.hypot(w.x - this.dragNode.x, w.y - this.dragNode.y) > 2) this.pointerMoved = true;
        this.dragNode.x = w.x;
        this.dragNode.y = w.y;
        this.updateTooltip(null);
        this.draw();
        return;
      }
      if (this.panStart) {
        const dx = sx - this.panStart.sx;
        const dy = sy - this.panStart.sy;
        if (Math.hypot(dx, dy) > 3) this.pointerMoved = true;
        this.cam.x = this.panStart.cx - dx / this.cam.zoom;
        this.cam.y = this.panStart.cy - dy / this.cam.zoom;
        this.updateTooltip(null);
        this.draw();
        return;
      }
      const hit = this.hitTest(sx, sy);
      const nextKey = hit ? hit.key : null;
      this.canvas.classList.toggle("pointer", Boolean(hit));
      this.updateTooltip(hit, sx, sy);
      if (nextKey !== this.hoverKey) {
        this.hoverKey = nextKey;
        this.draw();
      }
    });

    const endDrag = (e) => {
      const { sx, sy } = getXY(e);
      const wasDragNode = this.dragNode;
      const moved = this.pointerMoved;
      this.dragNode = null;
      this.panStart = null;
      this.canvas.classList.remove("dragging");
      if (!moved) {
        const hit = wasDragNode || this.hitTest(sx, sy);
        if (hit) {
          this.selectedKey = hit.key;
          this.draw();
          this.cb.onSelect?.(hit);
        } else {
          this.selectedKey = null;
          this.draw();
        }
      }
    };

    this.canvas.addEventListener("pointerup", endDrag);
    this.canvas.addEventListener("pointerleave", () => {
      this.dragNode = null;
      this.panStart = null;
      this.hoverKey = null;
      this.canvas.classList.remove("dragging", "pointer");
      this.updateTooltip(null);
      this.draw();
    });

    this.canvas.addEventListener("dblclick", (e) => {
      const { sx, sy } = getXY(e);
      const hit = this.hitTest(sx, sy);
      if (hit) this.cb.onExpand?.(hit);
    });

    this.canvas.addEventListener(
      "wheel",
      (e) => {
        e.preventDefault();
        const { sx, sy } = getXY(e);
        const before = this.screenToWorld(sx, sy);
        const factor = e.deltaY < 0 ? 1.16 : 0.86;
        const nextZoom = Math.max(0.25, Math.min(4.0, this.cam.zoom * factor));
        this.cam.zoom = nextZoom;
        const after = this.screenToWorld(sx, sy);
        this.cam.x += before.x - after.x;
        this.cam.y += before.y - after.y;
        this.draw();
      },
      { passive: false }
    );

    if (typeof ResizeObserver !== "undefined") {
      new ResizeObserver(() => this.resize()).observe(this.stage);
    } else {
      window.addEventListener("resize", () => this.resize());
    }
  }

  draw() {
    const ctx = this.ctx;
    if (!ctx) return;
    const pal = this.palette();
    const W = this.width;
    const H = this.height;

    ctx.save();
    ctx.scale(this.dpr, this.dpr);

    // 1. Background + Subtle Spatial Grid
    ctx.fillStyle = pal.bg;
    ctx.fillRect(0, 0, W, H);
    const gridStep = Math.max(26, 48 * this.cam.zoom);
    const offX = ((W / 2 - this.cam.x * this.cam.zoom) % gridStep + gridStep) % gridStep;
    const offY = ((H / 2 - this.cam.y * this.cam.zoom) % gridStep + gridStep) % gridStep;
    ctx.strokeStyle = pal.grid;
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let x = offX; x < W; x += gridStep) {
      ctx.moveTo(x, 0);
      ctx.lineTo(x, H);
    }
    for (let y = offY; y < H; y += gridStep) {
      ctx.moveTo(0, y);
      ctx.lineTo(W, y);
    }
    ctx.stroke();

    // 2. 4-Region Spatial Zone Backdrops
    if (this.layoutMode === "zones" && !this.centerKey) {
      const zones = this.zoneCenters();
      for (const reg of ["M", "S", "K", "E"]) {
        if (!this.visRegions.has(reg)) continue;
        const z = zones[reg];
        const sc = this.worldToScreen(z.x, z.y);
        const sr = z.r * this.cam.zoom;
        const col = pal[reg];

        // Soft radial aura
        const grad = ctx.createRadialGradient(sc.x, sc.y, sr * 0.08, sc.x, sc.y, sr);
        grad.addColorStop(0, `${col}1c`);
        grad.addColorStop(1, `${col}00`);
        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.arc(sc.x, sc.y, sr, 0, Math.PI * 2);
        ctx.fill();

        // Dashed territory ring
        ctx.save();
        ctx.setLineDash([5, 6]);
        ctx.strokeStyle = `${col}38`;
        ctx.lineWidth = 1.15;
        ctx.beginPath();
        ctx.arc(sc.x, sc.y, sr, 0, Math.PI * 2);
        ctx.stroke();
        ctx.restore();

        // Zone label at top of ring
        ctx.font = `700 ${Math.max(10.5, Math.min(13, 12 * this.cam.zoom))}px Inter, "PingFang SC", sans-serif`;
        ctx.fillStyle = col;
        ctx.textAlign = "center";
        ctx.fillText(z.title, sc.x, sc.y - sr + 18);
      }
    } else if (this.centerKey && this.nodeMap.has(this.centerKey)) {
      const c = this.worldToScreen(0, 0);
      for (const ringR of [165, 255]) {
        ctx.save();
        ctx.setLineDash([4, 6]);
        ctx.strokeStyle = pal.zoneRing;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.arc(c.x, c.y, ringR * this.cam.zoom, 0, Math.PI * 2);
        ctx.stroke();
        ctx.restore();
      }
    }

    const focusKey = this.hoverKey || this.selectedKey;
    const focusNeighbors = focusKey ? this.neighbors.get(focusKey) || new Set([focusKey]) : null;

    // 3. Curved Semantic Edges
    const incidentLabels = [];
    for (const e of this.edges) {
      const a = this.nodeMap.get(e.source_key);
      const b = this.nodeMap.get(e.target_key);
      if (!a || !b) continue;
      if (!this.visRegions.has(a.region) || !this.visRegions.has(b.region)) continue;
      if (!this.visBins.has(e.bin) && !(e.bin === "source_only" && this.visBins.has("inside"))) continue;

      const isIncident = focusKey && (e.source_key === focusKey || e.target_key === focusKey);
      const dimmed = Boolean(focusKey && !isIncident);

      const p1 = this.worldToScreen(a.x, a.y);
      const p2 = this.worldToScreen(b.x, b.y);
      const dx = p2.x - p1.x;
      const dy = p2.y - p1.y;
      const dist = Math.hypot(dx, dy) || 1;
      const bend = e.bendSign * Math.min(34, dist * 0.14);
      const qx = (p1.x + p2.x) / 2 - (dy / dist) * bend;
      const qy = (p1.y + p2.y) / 2 + (dx / dist) * bend;

      ctx.save();
      let stroke = pal.edgeInside;
      let width = 1.1;
      if (!e.active) {
        stroke = pal.edgeInactive;
        ctx.setLineDash([3, 4]);
      } else if (e.bin === "far") {
        stroke = pal.edgeFar;
        width = 2.0;
      } else if (e.bin === "negative") {
        stroke = pal.edgeNeg;
        width = 1.6;
        ctx.setLineDash([5, 4]);
      } else if (e.bin === "cross") {
        stroke = pal.edgeCross;
        width = 1.3;
      }

      if (isIncident) {
        width = Math.max(width + 0.9, 2.2);
        ctx.globalAlpha = 0.95;
      } else if (dimmed) {
        ctx.globalAlpha = 0.08;
      } else {
        ctx.globalAlpha = e.bin === "far" ? 0.9 : 0.72;
      }

      ctx.strokeStyle = stroke;
      ctx.lineWidth = width;
      ctx.beginPath();
      ctx.moveTo(p1.x, p1.y);
      ctx.quadraticCurveTo(qx, qy, p2.x, p2.y);
      ctx.stroke();

      // Arrowhead & concise relation badge on incident edges
      if (isIncident && dist > 36) {
        const t = 0.78;
        const ax = (1 - t) * (1 - t) * p1.x + 2 * (1 - t) * t * qx + t * t * p2.x;
        const ay = (1 - t) * (1 - t) * p1.y + 2 * (1 - t) * t * qy + t * t * p2.y;
        const ang = Math.atan2(p2.y - qy, p2.x - qx);
        ctx.setLineDash([]);
        ctx.fillStyle = stroke;
        ctx.beginPath();
        ctx.moveTo(ax, ay);
        ctx.lineTo(ax - Math.cos(ang - 0.4) * 7, ay - Math.sin(ang - 0.4) * 7);
        ctx.lineTo(ax - Math.cos(ang + 0.4) * 7, ay - Math.sin(ang + 0.4) * 7);
        ctx.closePath();
        ctx.fill();

        if (incidentLabels.length < 8) {
          const mx = 0.25 * p1.x + 0.5 * qx + 0.25 * p2.x;
          const my = 0.25 * p1.y + 0.5 * qy + 0.25 * p2.y;
          const relShort = String(e.relation || "").length > 12 ? e.relation.slice(0, 12) + "…" : e.relation;
          incidentLabels.push({ x: mx, y: my, text: relShort, color: stroke });
        }
      }
      ctx.restore();
    }

    // 4. Nodes
    for (const n of this.nodes) {
      if (!this.visRegions.has(n.region)) continue;
      const p = this.worldToScreen(n.x, n.y);
      if (p.x < -40 || p.x > W + 40 || p.y < -40 || p.y > H + 40) continue;

      const isFocus = n.key === this.hoverKey || n.key === this.selectedKey;
      const isCenter = n.key === this.centerKey;
      const isNear = !focusNeighbors || focusNeighbors.has(n.key);
      const col = pal[n.region] || pal.M;
      const r = n.r * Math.max(0.75, Math.min(1.35, this.cam.zoom));

      ctx.save();
      ctx.globalAlpha = isNear ? 1 : 0.16;

      // Ambient halo for hubs / focus / center
      if (isFocus || isCenter || n.deg >= 4) {
        ctx.fillStyle = `${col}${isFocus ? "44" : "24"}`;
        ctx.beginPath();
        ctx.arc(p.x, p.y, r + (isFocus ? 8 : 5), 0, Math.PI * 2);
        ctx.fill();
      }

      // Main dot
      ctx.fillStyle = col;
      ctx.beginPath();
      ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
      ctx.fill();

      ctx.strokeStyle = isFocus ? pal.selectRing : pal.nodeStroke;
      ctx.lineWidth = isFocus ? 2.4 : 1.5;
      ctx.stroke();
      ctx.restore();
    }

    // 5. Collision-Aware Smart LOD Labels (ZERO overlap soup!)
    const placed = [];
    const collides = (box) =>
      placed.some(
        (b) => !(box.x2 + 4 < b.x1 || box.x1 - 4 > b.x2 || box.y2 + 3 < b.y1 || box.y1 - 3 > b.y2)
      );

    const drawPill = (x, y, text, borderCol, textCol, force = false) => {
      ctx.font = `600 11px Inter, "PingFang SC", sans-serif`;
      const tw = ctx.measureText(text).width;
      const pw = tw + 12;
      const ph = 20;
      const box = { x1: x - pw / 2, y1: y, x2: x + pw / 2, y2: y + ph };
      if (!force && collides(box)) return false;
      placed.push(box);

      ctx.save();
      ctx.fillStyle = pal.pillBg;
      ctx.strokeStyle = borderCol;
      ctx.lineWidth = force ? 1.4 : 1;
      const rad = 6;
      ctx.beginPath();
      ctx.roundRect ? ctx.roundRect(box.x1, box.y1, pw, ph, rad) : ctx.rect(box.x1, box.y1, pw, ph);
      ctx.fill();
      ctx.stroke();

      ctx.fillStyle = textCol;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(text, x, y + ph / 2 + 0.5);
      ctx.restore();
      return true;
    };

    // Priority 1: Focus & center nodes
    const priorityKeys = [this.hoverKey, this.selectedKey, this.centerKey].filter(Boolean);
    for (const k of priorityKeys) {
      const n = this.nodeMap.get(k);
      if (!n || !this.visRegions.has(n.region)) continue;
      const p = this.worldToScreen(n.x, n.y);
      drawPill(p.x, p.y + n.r * Math.min(1.3, this.cam.zoom) + 5, n.shortLabel, pal[n.region], pal.pillText, true);
    }

    // Priority 2: Incident relation badges (when focused)
    for (const rl of incidentLabels) {
      drawPill(rl.x, rl.y - 10, rl.text, rl.color, pal.pillText, false);
    }

    // Priority 3: 1-hop neighbors of focused node, or top-degree hubs
    const candidates = this.nodes
      .filter((n) => this.visRegions.has(n.region) && !priorityKeys.includes(n.key))
      .sort((a, b) => {
        const aNear = focusNeighbors?.has(a.key) ? 1 : 0;
        const bNear = focusNeighbors?.has(b.key) ? 1 : 0;
        if (aNear !== bNear) return bNear - aNear;
        return b.deg - a.deg;
      });

    const maxLabels = focusKey
      ? 24
      : this.showAllLabels || this.cam.zoom >= 1.75
      ? 72
      : 14;
    let drawn = 0;
    for (const n of candidates) {
      if (drawn >= maxLabels) break;
      if (focusKey && !focusNeighbors.has(n.key)) continue;
      const p = this.worldToScreen(n.x, n.y);
      if (p.x < 40 || p.x > W - 40 || p.y < 30 || p.y > H - 30) continue;
      if (
        drawPill(
          p.x,
          p.y + n.r * Math.max(0.8, Math.min(1.3, this.cam.zoom)) + 5,
          n.shortLabel,
          focusKey ? pal[n.region] : pal.pillBorder,
          pal.pillText,
          false
        )
      ) {
        drawn += 1;
      }
    }

    if (!this.nodes.length) {
      ctx.font = `600 13.5px Inter, "PingFang SC", sans-serif`;
      ctx.fillStyle = pal.zoneLabel;
      ctx.textAlign = "center";
      ctx.fillText("当前筛选条件下暂无语义连接，可清空关键词或切换脑区重试", W / 2, H / 2);
    }

    ctx.restore();
  }
}

/* =========================================================================
   NAVIGATION, DRAWER & LIVE POLLING
   ========================================================================= */
function switchTab(tabName) {
  if (tabName !== state.tab) notice("");
  state.tab = tabName;
  document.querySelectorAll("[data-tab]").forEach((n) => {
    if (n.dataset.tab === tabName) n.setAttribute("aria-current", "page");
    else n.removeAttribute("aria-current");
  });
  document.querySelectorAll("[data-view]").forEach((view) => {
    view.hidden = view.dataset.view !== tabName;
  });
  if (tabName === "memory") {
    setTimeout(() => graphEngine?.resize(), 20);
  }
  attempt(loadTab);
  schedulePoll();
}

function openDetail(title) {
  state.detailVersion += 1;
  state.runDetail = null;
  $("detail-title").textContent = title;
  empty($("detail-content"), "正在读取…");
  if (!$("detail").open) $("detail").showModal();
  schedulePoll();
  return { version: state.detailVersion, scope: state.scope };
}

function detailCurrent(ticket) {
  return ticket.version === state.detailVersion && ticket.scope === state.scope && $("detail").open;
}

function runHelpers() {
  return {
    date,
    duration,
    number,
    tokens,
    appendMessages,
    showContext: (id) => attempt(() => showContext(id)),
    refresh: () =>
      attempt(async () => {
        if (state.runDetail) await refreshRunDetail(state.runDetail);
      }),
    kind: (kind) => kinds[kind] || kind || "调用",
  };
}

function schedulePoll() {
  clearTimeout(state.pollTimer);
  state.pollTimer = null;
  if (document.visibilityState !== "visible" || !state.scope || state.pollBusy) return;
  if (
    state.tab === "runtime" ||
    (state.runDetail && MRRunTrajectory.running(state.runDetail.run?.status))
  ) {
    state.pollTimer = setTimeout(pollLive, 3000);
  }
}

async function pollLive() {
  state.pollTimer = null;
  if (state.pollBusy || document.visibilityState !== "visible" || !state.scope) return;
  state.pollBusy = true;
  try {
    const jobs = [];
    if (state.tab === "runtime") jobs.push(loadRuns());
    const detail = state.runDetail;
    if (detail && MRRunTrajectory.running(detail.run?.status)) jobs.push(refreshRunDetail(detail));
    const results = await Promise.allSettled(jobs);
    const failure = results.find((result) => result.status === "rejected");
    $("runs-live-note").textContent = failure
      ? "刷新暂未完成，稍后重试"
      : "最近 30 次 · 每 3 秒刷新 · 北京时间";
  } finally {
    state.pollBusy = false;
    schedulePoll();
  }
}

/* =========================================================================
   WORKSPACE 1: COGNITIVE OVERVIEW & RUNS
   ========================================================================= */
function renderNetworkHero(counts) {
  const net = counts?.network;
  const badge = $("engine-badge");
  const tilesEl = $("region-tiles");
  const barEl = $("region-bar");
  tilesEl.replaceChildren();
  barEl.replaceChildren();

  if (net && net.engine === "plan_d") {
    if (badge) badge.textContent = "PLAN D";
    $("hero-engine-label").textContent = "PLAN D · 四脑区认知记忆网络";
    $("hero-title").textContent = "全图四区实时拓扑分布";
    const reg = net.regions || {};
    const eCnt = Number(reg.E || 0);
    const mCnt = Number(reg.M || 0);
    const sCnt = Number(reg.S || 0);
    const kCnt = Number(reg.K || 0);
    const total = eCnt + mCnt + sCnt + kCnt;
    $("hero-total-nodes").textContent = number(total);
    $("hero-total-edges").textContent = number(net.edges);
    $("hero-watermark").textContent = net.seed_watermark != null ? `#${net.seed_watermark}` : "—";
    if ($("nav-badge-nodes")) $("nav-badge-nodes").textContent = compactNum(mCnt + sCnt + kCnt);

    const upperTotal = Math.max(1, mCnt + sCnt + kCnt);
    const segments = [
      ["r-e", Math.max(8, (eCnt / Math.max(1, total)) * 100), `E 原话 ${number(eCnt)}`],
      ["r-m", (mCnt / Math.max(1, total)) * 100, `M 经历与桥 ${number(mCnt)}`],
      ["r-s", (sCnt / Math.max(1, total)) * 100, `S 熟人社会脑 ${number(sCnt)}`],
      ["r-k", (kCnt / Math.max(1, total)) * 100, `K 规则与边界 ${number(kCnt)}`],
    ];
    for (const [cls, pct, tip] of segments) {
      const seg = el("div", null, `region-seg-bar ${cls}`);
      seg.style.width = `${Math.max(2, pct).toFixed(1)}%`;
      seg.title = tip;
      barEl.append(seg);
    }

    const tileDefs = [
      { key: "M", val: mCnt, sub: `上层占比 ${((mCnt / upperTotal) * 100).toFixed(0)}% · 点此探索星图` },
      { key: "S", val: sCnt, sub: `上层占比 ${((sCnt / upperTotal) * 100).toFixed(0)}% · 人物/关系/情境` },
      { key: "K", val: kCnt, sub: `上层占比 ${((kCnt / upperTotal) * 100).toFixed(0)}% · 知识/范围/边界` },
      { key: "E", val: eCnt, sub: `可溯源原始发言 · ${number(counts.messages)} 条群聊` },
    ];
    for (const t of tileDefs) {
      const meta = REGIONS[t.key];
      const tile = button(
        "",
        () => {
          if (t.key === "E") switchTab("messages");
          else {
            state.regionFilter = t.key;
            syncRegionSegButtons();
            switchTab("memory");
          }
        },
        `region-tile r-${t.key.toLowerCase()}`
      );
      const head = el("div", null, "rt-head");
      head.append(el("span", meta.name), el("span", "→"));
      tile.append(head, el("strong", number(t.val)), el("small", t.sub));
      tilesEl.append(tile);
    }
  } else {
    if (badge) badge.textContent = "LEGACY";
    $("hero-engine-label").textContent = "标准记忆存储模式";
    $("hero-title").textContent = "记忆条目概览";
    const total = (counts?.episodes || 0) + (counts?.semantics || 0) + (counts?.associations || 0);
    $("hero-total-nodes").textContent = number(total);
    $("hero-total-edges").textContent = number(counts?.associations || 0);
    $("hero-watermark").textContent = "—";
    for (const [cls, title, val, sub] of [
      ["r-m", "共同经历", counts?.episodes || 0, "保留的情节记忆"],
      ["r-s", "人物与事实", counts?.semantics || 0, "当前有效的语义记忆"],
      ["r-k", "语义连接", counts?.associations || 0, "仍可用于搜索的关联"],
      ["r-e", "原始消息", counts?.messages || 0, "完整归档群聊原文"],
    ]) {
      const tile = button("", () => switchTab(cls === "r-e" ? "messages" : "memory"), `region-tile ${cls}`);
      const head = el("div", null, "rt-head");
      head.append(el("span", title), el("span", "→"));
      tile.append(head, el("strong", number(val)), el("small", sub));
      tilesEl.append(tile);
    }
  }
}

function renderWorkspace(data) {
  if (!data.workspace) return;
  const signature = JSON.stringify([state.scope, data.workspace]);
  if (signature === state.workspaceSignature) return;
  state.workspaceSignature = signature;
  const parent = $("workspace");
  parent.replaceChildren();
  for (const selected of data.workspace.active || []) {
    const item = selected.memory;
    const card = button("", () => showMemoryDrawer(item.kind, item.id), "mem-item");
    const attention = item.attention;
    card.append(
      el("strong", item.title || item.kind, "mem-item-head"),
      el("p", item.content || item.summary, "mem-item-excerpt"),
      el("small", typeof attention === "string" ? attention : attention?.reason || "已选入持续注意", "mono-muted")
    );
    parent.append(card);
  }
  if (!parent.children.length)
    empty(parent, "目前没有选入持续注意的认识。记忆仍可主动搜索。");
}

async function loadRuns() {
  const scope = state.scope;
  const data = await api(path("runs"));
  if (scope !== state.scope || state.tab !== "runtime") return;
  renderRuns(data.runs || []);
  renderLearning(data);
  renderWorkspace(data);
}

function renderLearning(data) {
  const plan = $("learning-plan"),
    win = data.learning_window || state.overview?.runtime?.learning_window;
  if (!win) {
    plan.replaceChildren();
    return;
  }
  const allDay = !win.enabled || win.start === win.end;
  const heading = el("div", null, "learning-window");
  heading.append(el("strong", allDay ? "后台整理：全天可运行" : `夜间整理窗口：${win.start}–${win.end}`));
  heading.append(el("span", `服务当地时间 ${win.local_now} · ${win.timezone}`));
  heading.append(
    el(
      "small",
      win.open
        ? "当前处于整理时段；群消息记录与回答前回忆始终实时生效。"
        : `常规新任务排到 ${
            win.next_start_local || "下一工作时段"
          }；已开始的任务继续保存，消息采集、回答前回忆与手动触发照常运行。`
    )
  );
  plan.replaceChildren(heading);
  const labels = {
    deferred: "等待工作时段",
    budget_exhausted: "等待额度恢复",
    budget_wait: "等待可用额度",
    running: "正在整理",
    partial: "保存进度，待续接",
    completed: "本批完成",
    disabled: "已关闭",
    busy: "群内正在整理",
    skipped: "等待新材料",
    waiting: "等待更多交流",
    idle: "暂无待处理材料",
  };
  const task = (data.learning || []).find((item) => item.kind === "background");
  const runtime = data.learning_status?.background || {};
  const enabled = state.overview?.runtime?.learning;
  const row = el("div", null, "learning-row"),
    title = el("div", null, "learning-title");
  const status = !enabled
    ? "已关闭"
    : labels[runtime.status] || statuses[runtime.status] || (task ? "等待续接" : "等待新材料");
  title.append(el("strong", kinds.background), el("span", status, `status ${runtime.status || ""}`));
  if (task?.run_id) title.append(button("查看最近轨迹", () => showRun(task.run_id), "btn small"));
  row.append(title);
  if (task) {
    row.append(
      el(
        "p",
        `本批已理解 ${number(task.completed_count)} / ${number(
          task.material_count
        )} 条消息，已保存 ${number(task.memory_refs?.length || 0)} 项记忆。`
      )
    );
    const checkpoint =
      typeof task.checkpoint === "string" ? task.checkpoint : JSON.stringify(task.checkpoint);
    if (checkpoint) {
      const understanding = el("details", null, "learning-checkpoint");
      understanding.append(el("summary", "已形成的理解与断点说明"), el("p", checkpoint));
      row.append(understanding);
    }
  }
  if (runtime.reason) row.append(el("small", runtime.reason, "muted-note"));
  const pendingDrafts =
    (task?.pending_draft_count || 0) + (data.pending_learning_drafts?.background || 0);
  if (pendingDrafts)
    row.append(
      el(
        "p",
        `${number(pendingDrafts)} 项记忆草稿尚未保存，已保留供后续修复。`,
        "inset-banner"
      )
    );
  if (runtime.next_start_local && runtime.next_start !== win.next_start)
    row.append(el("small", `下次可继续：${runtime.next_start_local}`, "muted-note"));
  plan.append(row);
}

function renderRuns(rows) {
  const body = $("runs");
  body.replaceChildren();
  if ($("nav-badge-runs")) $("nav-badge-runs").textContent = rows.length ? String(rows.length) : "";
  if (!rows.length) {
    const tr = el("tr"),
      td = el("td", "尚无保存的调用详情。旧版本没有留下的过程不会补造。", "empty");
    td.colSpan = 6;
    tr.append(td);
    body.append(tr);
  }
  for (const run of rows) {
    const tr = el("tr"),
      question = el("td", null, "question");
    question.append(
      el("span", kinds[run.kind] || run.kind || "调用", "run-kind"),
      el(
        "span",
        run.question || (typeof run.detail === "string" ? run.detail : "") || "后台整理群聊"
      )
    );
    const status = el("td", null, "run-progress");
    status.append(
      el("span", statuses[run.status] || run.status || "未记录", `status ${run.status || ""}`)
    );
    status.append(
      el(
        "small",
        MRRunTrajectory.labelTitle(run.latest_title) ||
          (run.step_count ? `${run.step_count} 条步骤记录` : "旧记录 · 可查看已保存过程")
      )
    );
    const action = el("td", null, "col-r");
    action.append(
      button(
        MRRunTrajectory.running(run.status) ? "跟随运行" : "查看轨迹",
        () => showRun(run.id),
        MRRunTrajectory.running(run.status) ? "btn small primary" : "btn small"
      )
    );
    tr.append(
      el("td", shortDate(run.started_at), "time"),
      question,
      status,
      el("td", duration(run.elapsed_ms)),
      el("td", number(tokens(run.usage))),
      action
    );
    body.append(tr);
  }
}

function renderHistory(history) {
  if (history.status === "unsupported") {
    $("recover-history").disabled = true;
    $("history-status").textContent = "官方 QQ：在线消息持续记录";
    $("history-detail").textContent = history.last_error || "当前平台不支持历史消息回补。";
    return;
  }
  const active = history.status === "running";
  $("recover-history").disabled = !state.scope || !state.overview?.runtime.history_recovery || active;
  const progress = history.pending || history.last_result || {};
  const status = !state.overview?.runtime.history_recovery
    ? "已关闭"
    : statuses[history.status] || "等待核对";
  $("history-status").textContent = `${status} · 累计补入 ${number(
    history.total_inserted || 0
  )} 条遗漏消息${active ? `；本次已读取 ${progress.pages || 0} 页` : ""}`;
  const detailLines = [
    `最近核对：${date(history.checked_at || history.updated_at)}`,
    "重连时自动补查，平时每五分钟核对；只补入记忆库，不触发旧命令或重新回复。",
  ];
  if (history.error) detailLines.push(`本次未完成：${history.error}`);
  if (history.retry_at) detailLines.push(`下次重试：${date(history.retry_at)}`);
  if (history.unavailable)
    detailLines.push(
      `${date(history.unavailable.since)} 至 ${date(history.unavailable.until)} 平台未返回更早记录。`
    );
  $("history-detail").textContent = detailLines.join(" ");
}

async function showRun(id) {
  const ticket = openDetail("运行轨迹");
  const detail = { ...ticket, id, run: null, view: null };
  state.runDetail = detail;
  await refreshRunDetail(detail);
  schedulePoll();
}

async function refreshRunDetail(detail) {
  const run = await api(
    path(`runs/${detail.id}`, detail.scope)
  );
  if (state.runDetail !== detail || !detailCurrent(detail)) return;
  detail.run = run;
  $("detail-title").textContent = `${kinds[run.kind] || run.kind || "调用"} · 运行轨迹 #${detail.id}`;
  if (!detail.view) detail.view = new MRRunTrajectory.View($("detail-content"), runHelpers());
  detail.view.update(run);
}

/* =========================================================================
   WORKSPACE 2: SPATIAL MEMORY STAR MAP & DOCKED INSPECTOR
   ========================================================================= */
function syncRegionSegButtons() {
  document.querySelectorAll("[data-region-filter]").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.regionFilter === state.regionFilter);
  });
}

function renderGraphTrail() {
  const host = $("graph-trail");
  if (!host) return;
  host.replaceChildren();
  const baseLabel = state.graphQuery
    ? `检索「${state.graphQuery}」`
    : state.graphView === "backbone"
    ? "全网骨架"
    : "近期前沿";
  const regSuffix = state.regionFilter !== "all" ? ` · ${state.regionFilter}区` : "";
  const rootBtn = button(
    baseLabel + regSuffix,
    () => resetGraphToOverview(),
    `trail-btn${state.graphTrail.length === 0 ? " current" : ""}`
  );
  host.append(rootBtn);
  state.graphTrail.forEach((entry, idx) => {
    host.append(el("span", "›", "trail-sep"));
    const isLast = idx === state.graphTrail.length - 1;
    host.append(
      button(
        entry.label,
        () => {
          state.graphTrail = state.graphTrail.slice(0, idx + 1);
          return expandNeighborhood(entry.node, false, true);
        },
        `trail-btn${isLast ? " current" : ""}`
      )
    );
  });
}

function updateGraphCaption() {
  if (!graphEngine) return;
  const activeEdges = graphEngine.edges.filter((e) => e.active !== false).length;
  $("graph-caption").textContent = `${graphEngine.nodes.length} 节点 · ${graphEngine.edges.length} 条连接（${activeEdges} 条活跃）`;
}

function renderMemoryList() {
  const list = $("memory-list");
  list.replaceChildren();
  const filtered = state.memoryItems.filter((item) => {
    if (state.regionFilter === "all") return true;
    return parseRegion(item.id, item.kind) === state.regionFilter;
  });
  $("mem-list-count").textContent = `记忆条目 (${filtered.length}${state.memoryHasMore ? "+" : ""})`;
  if ($("mem-more-bar")) $("mem-more-bar").hidden = !state.memoryHasMore;
  if (!filtered.length) {
    empty(list, "当前脑区/条件下没有匹配的记忆，可切换「全部脑区」或换个关键词。");
    return;
  }
  for (const item of filtered) {
    const reg = parseRegion(item.id, item.kind);
    const body = item.summary || item.statement || item.title || "";
    const headline = deriveHeadline(body, item.title || item.relation || item.id, 24);
    const excerpt = deriveExcerpt(body, headline);
    const key = String(item.id).includes("|") ? String(item.id) : `${item.kind}:${item.id}`;
    const card = button(
      "",
      () => {
        state.selectedNodeKey = key;
        highlightListSelection();
        if (graphEngine?.nodeMap.has(key)) graphEngine.flyToNode(key);
        return inspectNode({ key, id: item.id, kind: item.kind, label: body });
      },
      `mem-item${state.selectedNodeKey === key ? " selected" : ""}`
    );
    card.dataset.nodeKey = key;

    const top = el("div", null, "mem-item-top");
    top.append(
      el("span", reg, `reg-tag r-${reg.toLowerCase()}`),
      el("span", kinds[item.kind] || item.kind, "kind-tag")
    );
    const revision = item.revision ?? item.revision_no;
    if (revision > 1) top.append(el("span", `r${revision}`, "rev-tag"));
    else if (revision === 1) top.append(el("span", "r1", "mono-muted"));

    const foot = el("div", null, "mem-item-foot");
    const srcCount = (item.source_ids || []).length;
    foot.append(
      el("span", srcCount ? `${srcCount} 条原话` : String(item.id).slice(0, 22)),
      el("span", item.time_origin === "historical_replay"
        ? (item.evidence_at != null ? `依据 ${shortDate(item.evidence_at)}` : "历史补建")
        : shortDate(item.updated_at || item.created_at))
    );

    card.append(top, el("div", headline, "mem-item-head"));
    if (excerpt && excerpt !== headline) card.append(el("div", excerpt, "mem-item-excerpt"));
    card.append(foot);
    list.append(card);
  }
}

function highlightListSelection() {
  const list = $("memory-list");
  if (!list) return;
  for (const child of list.children) {
    const hit = child.dataset?.nodeKey === state.selectedNodeKey;
    child.classList.toggle("selected", hit);
    if (hit) child.scrollIntoView?.({ block: "nearest" });
  }
}

function memoryFilters() {
  return {
    query: state.graphQuery,
    region: state.regionFilter,
    kind: $("memory-kind")?.value || "all",
  };
}

// The list, graph and inspector can refresh independently in the same group.
async function readMemoryPanel(panel, endpoint, params) {
  const scope = state.scope;
  const token = Symbol(panel);
  state.memoryReads.set(panel, token);
  const current = () => scope === state.scope && state.memoryReads.get(panel) === token;
  try {
    const data = await api(path(endpoint), params);
    return current() ? data : null;
  } catch (error) {
    if (current()) throw error;
    return null;
  }
}

async function loadMemoryPage(append = false) {
  const more = $("memory-more");
  if (append && (more.disabled || !state.memoryHasMore)) return;
  more.disabled = true;
  const offset = append ? state.memoryOffset : 0;
  let data;
  try {
    data = await readMemoryPanel("list", "memories", { ...memoryFilters(), offset, limit: 60 });
  } catch (error) {
    more.disabled = false;
    if (!append) empty($("memory-list"), error.message || "读取记忆失败");
    throw error;
  }
  if (!data) return;
  const incoming = data.items || [];
  const existing = append ? state.memoryItems : [];
  const seen = new Set(existing.map((x) => `${x.kind}:${x.id}`));
  state.memoryItems = existing.concat(incoming.filter((x) => !seen.has(`${x.kind}:${x.id}`)));
  state.memoryOffset = offset + incoming.length;
  state.memoryHasMore = Boolean(data.has_more) && incoming.length > 0;
  more.disabled = false;
  renderMemoryList();
}

function reloadMemory() {
  state.memoryHasMore = false;
  state.memoryOffset = 0;
  state.memoryItems = [];
  $("mem-more-bar").hidden = true;
  $("mem-list-count").textContent = "记忆条目";
  empty($("memory-list"), "正在读取记忆…");
  return Promise.all([loadMemoryPage(), resetGraphToOverview()]);
}

function searchMemory() {
  state.graphQuery = $("memory-query").value.trim();
  return reloadMemory();
}

function applyGraphData(graph, centerKey) {
  if (!graphEngine) {
    graphEngine = new SpatialGraphCanvas(
      $("graph-canvas"),
      $("canvas-stage"),
      $("graph-tooltip"),
      {
        onSelect: (node) => {
          state.selectedNodeKey = node.key;
          highlightListSelection();
          attempt(() => inspectNode(node));
        },
        onExpand: (node) => attempt(() => expandNeighborhood(node, true)),
      }
    );
  }
  graphEngine.setData(graph, centerKey);
  updateGraphCaption();
}

async function resetGraphToOverview() {
  const graph = await readMemoryPanel("graph", "graph", { ...memoryFilters(), view: state.graphView, limit: 260 });
  if (!graph) return;
  state.graphTrail = [];
  applyGraphData(graph, null);
  renderGraphTrail();
}

function expandNeighborhood(node, pushTrail = true, forceFocus = false) {
  state.selectedNodeKey = node.key || String(node.id);
  highlightListSelection();
  return Promise.all([loadNeighborhood(node, pushTrail, forceFocus), inspectNode(node)]);
}

async function loadNeighborhood(node, pushTrail, forceFocus) {
  const graph = await readMemoryPanel("graph", "graph", { kind: node.kind, id: node.id, limit: 120 });
  if (!graph) return;
  const centerKey = node.key || String(node.id);
  if (pushTrail) {
    const existingIdx = state.graphTrail.findIndex((t) => t.key === centerKey);
    if (existingIdx >= 0) state.graphTrail = state.graphTrail.slice(0, existingIdx + 1);
    else
      state.graphTrail.push({
        key: centerKey,
        label: deriveHeadline(node.label || node.fullLabel || node.id, String(node.id), 14),
        node,
      });
  }
  if (state.graphGrowMode && !forceFocus && graphEngine && graphEngine.nodes.length > 0) {
    const { addedNodes, addedEdges } = graphEngine.mergeData(graph, centerKey);
    graphEngine.flyToNode(centerKey);
    updateGraphCaption();
    if (addedNodes || addedEdges)
      notice(`原地生长：+${addedNodes} 个新节点，+${addedEdges} 条新连接`);
  } else {
    applyGraphData(graph, centerKey);
  }
  renderGraphTrail();
}

function memoryEndpoint(kind, id) {
  // AstrBot's parent bridge encodes each path segment before sending it over HTTP.
  return `memory/${kind}/${id}`;
}

async function inspectNode(node) {
  const pane = $("node-inspector");
  const badges = $("inspector-badges");
  const body = $("inspector-body");
  pane.hidden = false;
  const reg = parseRegion(node.id || node.key, node.kind);
  badges.replaceChildren(
    el("span", REGIONS[reg]?.short || reg, `reg-tag r-${reg.toLowerCase()}`),
    el("span", kinds[node.kind] || node.kind || "节点", "kind-tag")
  );
  empty(body, "正在加载节点详情…");
  setTimeout(() => graphEngine?.resize(), 20);

  state.inspectedItem = null;
  let item;
  try {
    item = await readMemoryPanel("inspector", memoryEndpoint(node.kind || "node", node.id));
  } catch (error) {
    empty(body, error.message || "读取节点详情失败");
    return;
  }
  if (!item) return;
  state.inspectedItem = item;
  renderInspectorContent(item, node);
}

function revisionEntry(h) {
  const legacy = Object.hasOwn(h, "snapshot");
  const before = legacy ? h.snapshot?.memory ?? h.snapshot?.record ?? h.snapshot : h.previous;
  const oldRevision = before?.revision_no ?? before?.revision ?? h.snapshot?.record?.revision;
  const created = !legacy && h.revision === 1 && before == null;
  const label = legacy
    ? [h.operation, "操作前快照", oldRevision != null ? `rev ${oldRevision}` : ""].filter(Boolean).join(" · ")
    : `rev ${h.revision} · ${created ? "创建" : "修订前快照"}`;
  const text = before == null
    ? (created ? "初始版本（首次创建）" : "未保存操作前快照")
    : typeof before === "string" ? before : before.body || before.summary || before.content || JSON.stringify(before);
  return { label, text, at: legacy ? h.created_at : h.at };
}

function renderInspectorContent(item, refNode) {
  const badges = $("inspector-badges");
  const body = $("inspector-body");
  const reg = parseRegion(item.id || refNode?.id, item.kind || refNode?.kind);
  badges.replaceChildren(
    el("span", REGIONS[reg]?.short || reg, `reg-tag r-${reg.toLowerCase()}`),
    el("span", kinds[item.kind] || item.kind || "记忆", "kind-tag")
  );
  const revision = item.revision ?? item.revision_no;
  if (revision != null) badges.append(el("span", `rev ${revision}`, "rev-tag"));

  body.replaceChildren();

  // Address bar + Expand neighborhood CTA
  const addrRow = el("div", null, "insp-addr-row");
  const addrStr = String(item.id ?? refNode?.id ?? "");
  addrRow.append(el("span", addrStr, "insp-addr"));
  if (addrStr && !addrStr.startsWith("graph~")) {
    addrRow.append(
      button(
        "✦ 以它为中心展开邻域",
        () =>
          expandNeighborhood(
            {
              key: addrStr.includes("|") ? addrStr : `${item.kind}:${addrStr}`,
              id: addrStr,
              kind: item.kind || refNode?.kind,
              label: item.summary || item.title || addrStr,
            },
            true
          ),
        "btn small primary"
      )
    );
  }
  body.append(addrRow);

  // Single clean prose body (ZERO title/summary duplication!)
  const fullText = item.summary || item.statement || item.title || "无正文内容";
  body.append(el("div", fullText, "insp-prose"));

  if (item.time_origin === "historical_replay") {
    const evidence = item.evidence_at == null ? "" : ` · 最近关联原话：${date(item.evidence_at)}`;
    const status = item.status ? ` · 状态 ${item.status}` : "";
    body.append(el("div", `历史补建 · 实际写入时间未记录${evidence}${status}`, "mono-muted"));
  } else if (item.updated_at || item.created_at) {
    body.append(
      el(
        "div",
        `更新时间：${date(item.updated_at || item.created_at)}${
          item.status ? ` · 状态 ${item.status}` : ""
        }`,
        "mono-muted"
      )
    );
  }

  if (Array.isArray(item.history) && item.history.length) {
    const sec = el("div", null, "insp-sec");
    sec.append(el("div", `版本演进轨迹 (${item.history.length} 次修订)`, "insp-sec-title"));
    for (const h of item.history) {
      const card = el("div", null, "rev-card");
      const head = el("div", null, "rev-card-head");
      const entry = revisionEntry(h);
      head.append(el("span", entry.label), el("span", h.time_origin === "historical_replay"
        ? "历史补建 · 时间未记录" : shortDate(entry.at)));
      card.append(head, el("div", entry.text, "rev-card-body"));
      sec.append(card);
    }
    body.append(sec);
  }

  // Legacy cognitive metadata if present
  if (item.uncertainty) proseSection(body, "尚不确定的部分", item.uncertainty);
  if (item.belief) {
    const sec = el("div", null, "insp-sec");
    sec.append(el("div", "当前把握与依据", "insp-sec-title"), beliefView(item.belief));
    body.append(sec);
  }
  if (item.attention != null) {
    const sec = el("div", null, "insp-sec");
    sec.append(el("div", "持续关注原因", "insp-sec-title"), representationView(item.attention));
    body.append(sec);
  }

  // Grouped 1-Hop Connections (compact & scannable, ZERO 200-char walls!)
  const links = item.navigation?.connections || [];
  if (links.length) {
    const sec = el("div", null, "insp-sec");
    sec.append(el("div", `相邻连接 (${links.length})`, "insp-sec-title"));

    const groups = new Map();
    for (const lk of links) {
      const bin = lk.slot || (lk.relation === "bridge" ? "far" : "inside");
      if (!groups.has(bin)) groups.set(bin, []);
      groups.get(bin).push(lk);
    }
    const sortedBins = [...groups.keys()].sort(
      (a, b) => (BIN_META[a]?.order ?? 9) - (BIN_META[b]?.order ?? 9)
    );

    for (const bin of sortedBins) {
      const list = groups.get(bin);
      const grp = el("div", null, "conn-group");
      grp.append(el("span", `${BIN_META[bin]?.label || bin} · ${list.length}`, "conn-group-label"));
      for (const lk of list.slice(0, 18)) {
        const outward =
          String(lk.source_ref?.id) === String(item.id) || lk.source_key === String(item.id);
        const targetRef = outward ? lk.target_ref : lk.source_ref;
        const targetTitle = (outward ? lk.target : lk.source) || targetRef?.title;
        if (!targetRef) continue;
        const tReg = parseRegion(targetRef.id, targetRef.kind);
        const relStr = String(lk.relation || "关联");
        const isLongRel = relStr.length > 16;

        const card = button(
          "",
          () => {
            const tKey = String(targetRef.id).includes("|")
              ? String(targetRef.id)
              : `${targetRef.kind}:${targetRef.id}`;
            if (graphEngine?.nodeMap.has(tKey)) graphEngine.flyToNode(tKey);
            return inspectNode({ key: tKey, id: targetRef.id, kind: targetRef.kind, label: targetTitle });
          },
          "conn-card"
        );
        const top = el("div", null, "conn-card-top");
        top.append(
          el("span", tReg, `reg-tag r-${tReg.toLowerCase()}`),
          el("span", outward ? "→" : "←", "mono-muted"),
          el(
            "span",
            isLongRel ? (bin === "far" ? "跨月桥说明" : "语义阐释") : relStr,
            `conn-rel ${bin}`
          ),
          el("span", kinds[targetRef.kind] || targetRef.kind, "mono-muted")
        );
        if (lk.active === false) top.append(el("span", "非活跃", "mono-muted"));
        card.append(top, el("div", deriveHeadline(targetTitle, targetRef.id, 34), "conn-target"));
        if (isLongRel) card.append(el("div", relStr, "conn-rel-long"));
        grp.append(card);
      }
      sec.append(grp);
    }
    body.append(sec);
  }

  // Grounded Source Messages
  const srcIds = item.source_ids || [];
  const srcSec = el("div", null, "insp-sec");
  srcSec.append(el("div", `原始群聊依据 (${srcIds.length})`, "insp-sec-title"));
  const srcActions = el("div", null, "source-actions");
  for (const mid of srcIds) {
    srcActions.append(button(`消息 #${mid}`, () => showContext(mid), "btn small"));
  }
  if (!srcIds.length) {
    srcActions.append(
      el(
        "span",
        links.length ? "可沿上方「相邻连接」跳至 E 区原话节点。" : "此节点无直接关联原话。",
        "muted-note"
      )
    );
  }
  srcSec.append(srcActions);
  body.append(srcSec);

  details(body, "原始节点字段 (JSON)", item);
}

async function showMemoryDrawer(kind, id) {
  const ticket = openDetail(kinds[kind] || "记忆详情");
  let item;
  try {
    item = await api(path(memoryEndpoint(kind, id)));
  } catch (error) {
    if (detailCurrent(ticket)) empty($("detail-content"), error.message || "读取记忆详情失败");
    return;
  }
  if (!detailCurrent(ticket)) return;
  const parent = $("detail-content");
  parent.replaceChildren();
  proseSection(parent, deriveHeadline(item.summary || item.title, "记忆详情", 36), item.summary || item.statement);
  const actions = el("div", null, "source-actions");
  for (const mid of item.source_ids || [])
    actions.append(button(`消息 #${mid}`, () => showContext(mid), "btn small"));
  if (actions.children.length) parent.append(actions);
  details(parent, "原始字段", item);
}

/* =========================================================================
   WORKSPACE 3: PEOPLE & PLATFORM MIGRATION
   ========================================================================= */
async function searchPeople() {
  const scope = state.scope;
  const data = await api(path("participants"), { query: $("people-query").value.trim() });
  if (scope !== state.scope) return;
  const body = $("people");
  body.replaceChildren();
  const pool = data.pool || { account_ids: [], capacity: 50, characters: 0, day: "", notice: "", prefix: "" };
  const pct = Math.min(100, Math.round((pool.account_ids.length / Math.max(1, pool.capacity)) * 100));
  if ($("member-pool-fill")) $("member-pool-fill").style.width = `${pct}%`;
  if ($("nav-badge-people")) $("nav-badge-people").textContent = String((data.participants || []).length);
  $("member-pool-summary").textContent = `活跃前缀 ${pool.account_ids.length} / ${pool.capacity} 人 · ${number(
    pool.characters
  )} 字符 · 每日刷新 ${pool.day}。${pool.notice || ""}`;
  $("member-prefix-preview").textContent = pool.prefix || "当前没有成员前缀";
  for (const person of data.participants || []) {
    const tr = el("tr");
    const account = el("td");
    account.append(
      button(
        person.account_id,
        () => {
          $("profile-account").value = person.account_id;
          $("profile-name").value = person.preferred_name || "";
          $("profile-aliases").value = (person.confirmed_aliases || []).join(", ");
          $("profile-avoided").value = (person.avoided_names || []).join(", ");
          $("profile-description").value = person.description || "";
          $("profile-name").focus();
        },
        "account-btn"
      )
    );
    if (pool.account_ids.includes(person.account_id))
      account.append(el("span", "前缀中", "prefix-tag"));
    tr.append(
      account,
      el("td", `${person.card || "无群名片"}\n${person.nickname || "未取得平台昵称"}`),
      el(
        "td",
        `历史：${(person.observed_names || []).join(" · ") || "无"}\n确认：${
          (person.confirmed_aliases || []).join(" · ") || "无"
        }`,
        "aliases"
      ),
      el(
        "td",
        `${person.preferred_name || "未设置"}\n不用：${
          (person.avoided_names || []).join(" · ") || "无"
        }`
      ),
      el(
        "td",
        `${
          person.edited_by?.startsWith("self:")
            ? "本人"
            : person.edited_by?.startsWith("admin")
            ? "管理员"
            : "平台同步"
        }${person.edited_at ? " · " + shortDate(person.edited_at) : ""}`
      )
    );
    body.append(tr);
  }
  if (!data.participants?.length) {
    const tr = el("tr");
    const td = el("td", "没有匹配的账户", "empty");
    td.colSpan = 5;
    tr.append(td);
    body.append(tr);
  }
}

let migrationPreview = null;
async function loadMigration(preview = false) {
  if (!state.scope) return;
  const scope = state.scope;
  const selected = preview ? $("migration-legacy").value : "";
  const data = await api(path("migration"), selected ? { legacy_scope: selected } : {});
  if (scope !== state.scope) return;
  migrationPreview = { ...data, scope, selected: data.legacy_scope || selected };
  const picker = $("migration-legacy");
  picker.replaceChildren(el("option", "选择旧群记忆库"));
  picker.firstChild.value = "";
  for (const group of data.choices || []) {
    const option = el("option", formatScopeLabel({ ...group, enabled: true }));
    option.value = group.id;
    picker.append(option);
  }
  picker.value = migrationPreview.selected;
  picker.disabled = data.bound || !data.official;
  $("migration-bind").disabled = !data.official || data.bound || !data.old_counts;
  $("migration-preview").disabled = !data.official;
  const matched = Object.values(data.matches || {}).filter((v) => v.legacy_id).length;
  $("migration-status").textContent = !data.official
    ? "请在顶部切换到新官方机器人已开始记录的群。"
    : `${data.message} 已观察 ${data.observed_people.length} 位成员，${matched} 位已找到对应。` +
      (data.old_counts
        ? ` 旧库有 ${number(data.old_counts.messages)} 条原文、${number(
            data.old_counts.episodes +
              data.old_counts.semantics +
              data.old_counts.associations +
              data.old_counts.patterns
          )} 项有效记忆。`
        : "");
  const botLinks = Object.entries(data.bot_matches || {}).filter(([, match]) => match.legacy_id);
  $("migration-bot-status").textContent = botLinks.length
    ? `机器人自身${data.bound ? "已自动接续" : "将在接续旧群时自动关联"}：` +
      botLinks.map(([account, match]) => `${account} → ${match.legacy_id}`).join("；")
    : "机器人自身随旧群接续自动关联，无需手动选择。";
  const body = $("migration-members");
  body.replaceChildren();
  for (const person of data.observed_people || []) {
    const match = data.matches[person.account_id];
    const row = el("tr"),
      select = el("select"),
      choice = el("option", "暂不关联（新成员 / 尚未确认）");
    choice.value = "";
    select.append(choice);
    for (const old of (data.old_people || []).filter((p) => p.account_type === "USER")) {
      const option = el("option", `${old.card || old.nickname || old.name} · ${old.account_id}`);
      option.value = old.account_id;
      select.append(option);
    }
    select.value = match?.legacy_id || "";
    select.disabled = !data.bound;
    const cell = el("td");
    cell.append(select);
    const actionCell = el("td", null, "col-r"),
      save = button(
        "保存关联",
        async () => {
          const result = await api(
            path("migration/member", scope),
            { account_id: person.account_id, legacy_id: select.value },
            true
          );
          if (state.scope === scope) {
            notice(result.message);
            await loadMigration();
          }
        },
        "btn small"
      );
    save.disabled = !data.bound;
    actionCell.append(save);
    row.append(
      el("td", `${person.card || person.nickname || person.name}\n${person.account_id}`),
      cell,
      el(
        "td",
        {
          manual: match?.legacy_id ? "管理员设置" : "管理员设置为不关联",
          unchanged_name: "唯一名称一致",
          self_reported: "群员自填",
          self_ignored: "本人已取消",
          ignored: "管理员设置为不关联",
        }[match?.method] || "待对照 / 名称有冲突"
      ),
      actionCell
    );
    body.append(row);
  }
}

/* =========================================================================
   WORKSPACE 4: MESSAGES & OPERATOR CORRECTION STUDIO
   ========================================================================= */
function uuidV4() {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    return (c === "x" ? r : (r & 0x3) | 0x8).toString(16);
  });
}

function stopManualPoll() {
  clearTimeout(state.manualPollTimer);
  state.manualPollTimer = null;
}

function resetManualComposer() {
  stopManualPoll();
  state.manualMentions = [];
  state.manualDraftKey = "";
  state.manualRequestId = "";
  state.manualReceipt = null;
  if ($("manual-text")) $("manual-text").value = "";
  if ($("manual-mention-query")) $("manual-mention-query").value = "";
  if ($("manual-mention-suggestions")) {
    $("manual-mention-suggestions").replaceChildren();
    $("manual-mention-suggestions").hidden = true;
  }
  renderMentionChips();
  setManualReply("");
  if ($("manual-receipt")) {
    $("manual-receipt").replaceChildren();
    $("manual-receipt").hidden = true;
  }
}

function renderMentionChips() {
  const host = $("manual-mention-chips");
  if (!host) return;
  host.replaceChildren();
  for (const item of state.manualMentions) {
    const chip = el("span", `${item.label} (${item.account_id})`, "mention-chip");
    const rm = el("button", "×");
    rm.type = "button";
    rm.setAttribute("aria-label", `移除 ${item.label}`);
    rm.addEventListener("click", () => {
      state.manualMentions = state.manualMentions.filter((m) => m.account_id !== item.account_id);
      renderMentionChips();
    });
    chip.append(rm);
    host.append(chip);
  }
}

function setManualReply(id) {
  const input = $("manual-reply-id");
  const clearBtn = $("manual-clear-reply");
  if (!input) return;
  input.value = id ? String(id) : "";
  if (clearBtn) clearBtn.hidden = !input.value;
}

function quoteMessageForCorrection(message) {
  setManualReply(message.id);
  const isRealUser =
    message.role === "USER" &&
    message.sender_id &&
    message.sender_id !== "console:operator";
  if (isRealUser && !state.manualMentions.some((m) => m.account_id === String(message.sender_id))) {
    state.manualMentions.push({
      account_id: String(message.sender_id),
      label: message.sender_name || String(message.sender_id),
    });
    renderMentionChips();
  }
  $("manual-text")?.focus();
}

async function queryManualMentions() {
  const box = $("manual-mention-suggestions");
  const query = $("manual-mention-query").value.trim();
  if (!state.scope || !query) {
    box.replaceChildren();
    box.hidden = true;
    return;
  }
  const scope = state.scope;
  const data = await api(path("participants"), { query });
  if (scope !== state.scope) return;
  box.replaceChildren();
  const candidates = (data.participants || []).filter(
    (p) => !state.manualMentions.some((m) => m.account_id === p.account_id)
  );
  if (!candidates.length) {
    box.append(el("p", "没有可添加的匹配群友", "muted-note"));
    box.hidden = false;
    return;
  }
  for (const person of candidates.slice(0, 12)) {
    const display = person.preferred_name || person.card || person.nickname || person.account_id;
    box.append(
      button(`${display} · ${person.account_id}`, () => {
        state.manualMentions.push({ account_id: person.account_id, label: display });
        renderMentionChips();
        $("manual-mention-query").value = "";
        box.replaceChildren();
        box.hidden = true;
      })
    );
  }
  box.hidden = false;
}

function extractMessageIdFromAddress(addr) {
  const m = String(addr || "").match(/^E\|message:(\d+)$/);
  return m ? Number(m[1]) : null;
}

function renderManualReceipt(receipt) {
  const box = $("manual-receipt");
  if (!box) return;
  box.replaceChildren();
  if (!receipt) {
    box.hidden = true;
    return;
  }
  box.hidden = false;
  const head = el("div", null, "receipt-head");
  const st = receipt.state || "queued";
  head.append(
    el("strong", "提交回执"),
    el("span", statuses[st] || st, `status ${st}`),
    el("code", (receipt.request_id || "").slice(0, 18))
  );
  if (receipt.run_id)
    head.append(button(`轨迹 #${receipt.run_id}`, () => showRun(receipt.run_id), "btn small"));
  if (receipt.request_id)
    head.append(
      button(
        "刷新",
        () => pollManualReceipt(state.scope, receipt.request_id, true),
        "btn small ghost"
      )
    );
  box.append(head);

  if (receipt.reason) box.append(el("p", reasonLabels[receipt.reason] || receipt.reason, "muted-note"));

  if (Array.isArray(receipt.messages) && receipt.messages.length) {
    const list = el("div", null, "source-actions");
    for (const msg of receipt.messages) {
      const mState = msg.state || "pending";
      list.append(el("span", `#${msg.message_id} · ${statuses[mState] || mState}`, `status ${mState}`));
      if (msg.message_id != null)
        list.append(button(`原文`, () => showContext(msg.message_id), "btn small ghost"));
    }
    box.append(list);
  }

  if (Array.isArray(receipt.runs) && receipt.runs.length) {
    for (const runItem of receipt.runs) {
      const changes = runItem.changes || [];
      const edges = runItem.edges || [];
      const diffWrap = el("details");
      diffWrap.open = changes.length > 0 || edges.length > 0;
      diffWrap.append(
        el(
          "summary",
          `运行 #${runItem.run_id} 改图差分 · ${changes.length} 节点 / ${edges.length} 边`
        )
      );
      if (!changes.length && !edges.length) {
        diffWrap.append(el("p", "模型已阅读本次材料，未写入新节点或边。", "muted-note"));
      } else {
        for (const ch of changes) {
          const block = el("div", null, "trajectory-section");
          block.append(
            el("h4", ch.address || "节点变更"),
            el("p", ch.after || ch.before || "已更新", "trajectory-prose")
          );
          if (Array.isArray(ch.basis_addresses) && ch.basis_addresses.length) {
            const links = el("div", null, "source-actions");
            for (const addr of ch.basis_addresses) {
              const mid = extractMessageIdFromAddress(addr);
              if (mid != null)
                links.append(button(`依据 #${mid}`, () => showContext(mid), "btn small ghost"));
              else links.append(el("code", String(addr)));
            }
            block.append(links);
          }
          diffWrap.append(block);
        }
        for (const edge of edges) {
          diffWrap.append(
            el(
              "p",
              `[${edge.action || "edge"}] ${edge.source} → ${edge.relation} → ${edge.target}`,
              "mono-muted"
            )
          );
        }
      }
      box.append(diffWrap);
    }
  }
}

async function pollManualReceipt(scope, requestId, manualTrigger = false) {
  stopManualPoll();
  if (!scope || scope !== state.scope || !requestId) return;
  try {
    const data = await api(
      path(`manual-messages/${requestId}`, scope)
    );
    if (scope !== state.scope) return;
    state.manualReceipt = { scope, request_id: requestId, data };
    renderManualReceipt(data);
    if (["queued", "running"].includes(data?.state) && document.visibilityState === "visible") {
      state.manualPollTimer = setTimeout(() => pollManualReceipt(scope, requestId), 3000);
    } else if (data?.state === "completed") {
      await searchMessages();
    }
  } catch (error) {
    if (manualTrigger) notice(error?.message || String(error), true);
  }
}

async function submitManualMessages(trigger) {
  if (!state.scope || state.manualBusy) return;
  const raw = $("manual-text").value.trim();
  if (!raw) {
    notice("请先输入需要补充或纠错的消息正文。", true);
    $("manual-text").focus();
    return;
  }
  const blocks = raw
    .split(/\n\s*\n/)
    .map((part) => part.trim())
    .filter(Boolean);
  const replyRaw = $("manual-reply-id").value.trim();
  const replyId = replyRaw ? Number(replyRaw) : null;
  if (replyRaw && (!Number.isInteger(replyId) || replyId < 1)) {
    notice("关联原话编号必须是正整数。", true);
    return;
  }
  const mentionedIds = state.manualMentions.map((m) => m.account_id);
  const messages = blocks.map((text, idx) => {
    const entry = { text };
    if (mentionedIds.length) entry.mentioned_account_ids = mentionedIds;
    if (replyId != null && idx === 0) entry.reply_to_message_id = replyId;
    return entry;
  });

  const draftKey = JSON.stringify({ scope: state.scope, trigger, messages });
  if (draftKey !== state.manualDraftKey || !state.manualRequestId) {
    state.manualDraftKey = draftKey;
    state.manualRequestId = uuidV4();
  }
  const requestId = state.manualRequestId;
  const scope = state.scope;

  state.manualBusy = true;
  $("manual-submit-window").disabled = true;
  $("manual-submit-now").disabled = true;
  try {
    const receipt = await api(
      path("manual-messages", scope),
      { request_id: requestId, trigger, messages },
      true
    );
    if (scope !== state.scope) return;
    state.manualReceipt = { scope, request_id: receipt.request_id || requestId, data: receipt };
    renderManualReceipt(receipt);
    notice(
      trigger === "now"
        ? "已写入管理员补充消息并触发整理，下方可查看消费状态与改图差分。"
        : "已写入管理员补充消息，将在下个整理时窗进入整理队列。"
    );
    $("manual-text").value = "";
    state.manualMentions = [];
    renderMentionChips();
    setManualReply("");
    state.manualDraftKey = "";
    state.manualRequestId = "";
    await searchMessages();
    if (["queued", "running"].includes(receipt?.state)) {
      stopManualPoll();
      state.manualPollTimer = setTimeout(
        () => pollManualReceipt(scope, receipt.request_id || requestId),
        2500
      );
    }
  } catch (error) {
    const msg = error?.message || String(error);
    const missingRoute = /404|not\s*found|未知|不支持|unsupported/i.test(msg);
    const box = $("manual-receipt");
    if (missingRoute && box) {
      box.hidden = false;
      box.replaceChildren(
        el("strong", "前端纠错投喂入口已就绪，等待后端接口挂载"),
        el("p", `服务端返回：${msg}`, "muted-note")
      );
      notice("当前运行中的插件后端尚未包含 manual-messages 接口。", true);
    } else {
      notice(msg, true);
    }
  } finally {
    state.manualBusy = false;
    $("manual-submit-window").disabled = !state.scope;
    $("manual-submit-now").disabled = !state.scope;
  }
}

function readableContent(content) {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) return content.map(readableContent).filter(Boolean).join("\n");
  if (!content || typeof content !== "object") return "";
  if (["plain", "text"].includes(String(content.type).toLowerCase()))
    return content.text || content.data?.text || "";
  return readableContent(content.content || content.nodes || content.messages || content.data?.content);
}

function appendMessages(parent, messages, contextButtons = true) {
  for (const message of messages || []) {
    const isOperator =
      message.sender_id === "console:operator" ||
      message.platform === "console" ||
      message.author_kind === "operator" ||
      message.source === "console_manual" ||
      message.role === "OPERATOR";
    const article = el("article", null, `message${isOperator ? " is-operator" : ""}`);
    const header = el("header");
    const botEvent = Array.isArray(message.content)
      ? message.content.find((part) => part.type === "bot_event")?.event
      : null;
    const eventName = {
      generated: "主意识生成稿",
      tool_call: "主意识工具请求",
      tool_result: "主意识工具返回",
      sent: "实际发送",
    }[botEvent];
    const roleLabel = isOperator
      ? "管理员补充"
      : eventName ||
        { USER: "群友", BOT: "实际发送", SYSTEM: "生成 / 工具活动" }[message.role] ||
        message.role ||
        "消息";
    const roleClass = isOperator
      ? "operator"
      : message.role === "USER"
      ? "user"
      : message.role === "BOT" || botEvent === "sent"
      ? "bot"
      : "";

    header.append(
      el("strong", message.sender_name || message.sender_id || (isOperator ? "管理员补充" : "未知发送者")),
      el("span", roleLabel, `msg-role-pill ${roleClass}`.trim()),
      el("span", shortDate(message.sent_at))
    );
    if (isOperator) header.append(el("span", "仅记忆库 · 不发群", "mono-muted"));

    if (contextButtons) {
      const actions = el("div", null, "msg-actions");
      actions.append(
        button("↩ 引用纠错", () => quoteMessageForCorrection(message), "btn small ghost"),
        button("前后文", () => showContext(message.id), "btn small")
      );
      header.append(actions);
    }

    const rawText = message.plain_text || "";
    const text =
      !rawText || /^\[(node|nodes|forward)\]$/.test(rawText.trim())
        ? readableContent(message.content) || rawText
        : rawText;
    article.append(header);
    const toolActivity = ["tool_call", "tool_result"].includes(botEvent)
      ? message.content.find((part) => part.type === botEvent)
      : null;
    if (toolActivity) {
      const content = el("div", null, "trajectory-section");
      MRRunTrajectory.renderValue(
        content,
        {
          name: toolActivity.name,
          arguments: toolActivity.arguments,
          ...(botEvent === "tool_result" ? { result: toolActivity.result } : {}),
        },
        runHelpers()
      );
      article.append(content);
    } else {
      article.append(el("div", text || "这条消息没有纯文本，请展开消息结构。", "message-text"));
    }
    details(article, `消息 #${message.id} · 结构、引用与附件`, {
      content: message.content,
      relations: message.relations,
      attachments: message.attachments,
    });
    parent.append(article);
  }
}

async function searchMessages(older = false) {
  const scope = state.scope;
  const query = $("message-query").value.trim();
  const data = await api(path("messages"), {
    query,
    ...(older && state.next ? { before_id: state.next } : {}),
  });
  if (scope !== state.scope) return;
  if (!older) $("message-list").replaceChildren();
  appendMessages($("message-list"), data.messages);
  if (!$("message-list").children.length) empty($("message-list"), "没有匹配的原始消息。");
  state.next = data.next_before_id;
  $("older").hidden = !state.next;
}

async function showContext(id) {
  const returnRun = state.runDetail?.id;
  const ticket = openDetail(`消息 #${id} 的前后文`);
  const data = await api(path(`context/${id}`));
  if (!detailCurrent(ticket)) return;
  const parent = $("detail-content");
  parent.replaceChildren();
  if (returnRun != null)
    parent.append(button("← 返回本次运行轨迹", () => showRun(returnRun), "btn small return-run"));
  parent.append(el("p", "按时间先后显示，目标消息前后各最多 8 条。", "meta"));
  appendMessages(parent, data.messages, false);
  if (!data.messages?.length) empty(parent, "找不到原文，可能已撤回或移除。");
}

/* =========================================================================
   REFRESH & TAB LOADER
   ========================================================================= */
function sortScopes(scopes) {
  const list = [...(scopes || [])];
  if (list.length <= 1) return list;
  const stats = list.map((s) => ({
    s, lastAt: Number(s.last_message_at) || 0, recent: Number(s.recent_message_count) || 0,
  }));
  const latestAt = Math.max(...stats.map((x) => x.lastAt), 0);
  const maxLogVol = Math.max(...stats.map((x) => Math.log10(x.recent + 1)), 1);
  const scored = stats.map((x) => {
    const ageHours =
      x.lastAt > 0 && latestAt > 0 ? Math.max(0, (latestAt - x.lastAt) / 3600) : null;
    const recency = ageHours == null ? 0 : 1 / (1 + ageHours / 48);
    const volume = Math.log10(x.recent + 1) / maxLogVol;
    const weight = 0.55 * recency + 0.45 * volume;
    return { ...x, weight };
  });
  scored.sort(
    (a, b) =>
      b.weight - a.weight ||
      b.lastAt - a.lastAt ||
      b.recent - a.recent ||
      a.s.umo.localeCompare(b.s.umo)
  );
  return scored.map((x) => x.s);
}

async function refresh() {
  const conn = $("connection");
  conn.textContent = "正在读取…";
  conn.classList.remove("ok");
  if (window.__MR_DEMO_MODE__ && $("demo-banner")) $("demo-banner").hidden = false;

  const overview = await api("overview");
  overview.scopes = sortScopes(overview.scopes || []);
  state.overview = overview;
  const select = $("scope-select");
  select.replaceChildren();
  for (const s of overview.scopes) {
    const option = el("option", formatScopeLabel(s));
    option.value = s.id;
    select.append(option);
  }
  if (!overview.scopes.some((s) => s.id === state.scope))
    state.scope = overview.scopes[0]?.id || "";
  select.value = state.scope;
  select.disabled = !state.scope;
  $("distill").disabled = !state.scope || !overview.runtime.background_enabled;
  $("experimental-workspace").hidden = !overview.runtime.advanced;
  $("reset-conversation").disabled = !state.scope || state.resetBusy;
  if ($("manual-submit-window")) $("manual-submit-window").disabled = !state.scope;
  if ($("manual-submit-now")) $("manual-submit-now").disabled = !state.scope;
  conn.textContent = window.__MR_DEMO_MODE__ ? "本地演示数据源" : "已连接 AstrBot";
  conn.classList.add("ok");
  if (overview.errors?.length) notice(overview.errors.join("\n"), true);
  if (!state.scope) {
    select.append(el("option", "还没有群聊记录"));
    $("scope-summary").textContent = "收到群消息后会建立记忆库";
    $("context-repair-target").textContent = "请先选择群聊";
    empty($("metrics"), "暂无记忆库");
    return;
  }
  const scope = overview.scopes.find((s) => s.id === state.scope);
  $("context-repair-target").textContent = `操作对象：${formatScopeLabel(scope)}`;
  $("scope-summary").textContent = scope.enabled ? "已启用 MR" : "历史归档（未启用）";
  $("scope-dot")?.classList.toggle("on", Boolean(scope.enabled));
  $("scope-meta").textContent = scope.umo;
  $("scope-meta").title = scope.umo;

  const r = overview.runtime;
  const policyDefs = [
    [`消息采集：${r.capture ? "开" : "关"}`, r.capture ? "on" : ""],
    [`回答回忆：${r.recall ? "开" : "关"}`, r.recall ? "on" : ""],
    [`自动建图：${r.learning ? "开" : "关"}`, r.learning ? "on" : ""],
    [`前缀上限：${r.member_prefix_capacity} 人`, "on"],
    [
      `语义向量：${
        r.embedding_enabled === false ? "关" : r.embedding_loaded ? "已就绪" : "待加载"
      }`,
      r.embedding_loaded ? "on" : r.embedding_enabled === false ? "" : "warn",
    ],
    [`回忆模型：${r.provider || "未配置"}`, r.provider ? "on" : "warn"],
    ...(r.background_provider && r.background_provider !== r.provider
      ? [[`后台模型：${r.background_provider}`, "on"]]
      : []),
    [`超时阈值：${r.memory_timeout_seconds}s`, ""],
  ];
  $("policies").replaceChildren(
    ...policyDefs.map(([text, cls]) => el("span", text, `policy-pill ${cls}`.trim()))
  );
  await loadTab();
  schedulePoll();
}

async function loadTab() {
  if (!state.scope) return;
  const scope = state.scope;
  if (state.tab === "runtime") {
    const [overview, runData] = await Promise.all([api(path("overview")), api(path("runs"))]);
    if (state.scope !== scope) return;
    state.scopeOverview = overview;
    const c = overview.counts;
    const rows = runData.runs || [];
    renderNetworkHero(c);

    const latencies = rows
      .filter((r) => ["foreground", "recall", "reconstruction"].includes(r.kind) && r.elapsed_ms != null)
      .map((r) => Number(r.elapsed_ms))
      .sort((a, b) => a - b);
    const mid = Math.floor(latencies.length / 2);
    const median = !latencies.length
      ? null
      : latencies.length % 2
      ? latencies[mid]
      : (latencies[mid - 1] + latencies[mid]) / 2;
    const budget = state.overview.runtime.background_rolling24h_budget;

    if ($("nav-badge-pending"))
      $("nav-badge-pending").textContent = c.pending ? compactNum(c.pending) : "";

    $("metrics").replaceChildren(
      ...[
        ["回忆耗时中位数", duration(median), `最近 ${latencies.length} 次回答前回忆`],
        [
          "24h 后台 Token",
          number(overview.background_tokens_rolling24h),
          `额度 ${budget > 0 ? number(budget) : "不限"}`,
        ],
        [
          "待整理队列",
          number(c.pending),
          `上次整理：${shortDate(overview.state.consolidated_at) || "未记录"}`,
        ],
        [
          "活跃语义连接",
          number(c.associations),
          `向量索引 ${number(c.embeddings)} · 群员 ${number(c.participants)}`,
        ],
      ].map(([label, value, caption]) => {
        const card = el("article", null, "kpi-card");
        card.append(el("span", label), el("strong", value), el("small", caption));
        return card;
      })
    );
    renderRuns(rows);
    renderLearning(overview);
    renderHistory(overview.history || {});
    renderWorkspace(overview);
  } else if (state.tab === "memory") {
    await reloadMemory();
  } else if (state.tab === "people") {
    await searchPeople();
    if ($("platform-migration").open) await loadMigration();
  } else {
    await searchMessages();
  }
}

/* =========================================================================
   EVENT WIRING
   ========================================================================= */
$("close-detail").addEventListener("click", () => $("detail").close());
$("detail").addEventListener("close", () => {
  state.detailVersion += 1;
  state.runDetail = null;
  schedulePoll();
});

$("inspector-close")?.addEventListener("click", () => {
  state.memoryReads.delete("inspector");
  $("node-inspector").hidden = true;
  setTimeout(() => graphEngine?.resize(), 20);
});

$("scope-select").addEventListener("change", () => {
  state.memoryReads.clear();
  state.memoryItems = [];
  state.memoryHasMore = false;
  state.memoryOffset = 0;
  renderMemoryList();
  if (graphEngine) applyGraphData({ nodes: [], edges: [] }, null);
  state.detailVersion += 1;
  state.runDetail = null;
  state.selectedNodeKey = "";
  state.graphTrail = [];
  if ($("node-inspector")) $("node-inspector").hidden = true;
  if ($("detail").open) $("detail").close();
  state.scope = $("scope-select").value;
  $("profile-form").reset();
  $("context-repair-result").hidden = true;
  clearTimeout(state.pollTimer);
  resetManualComposer();
  attempt(refresh);
});

document.addEventListener("visibilitychange", () => {
  schedulePoll();
  if (document.visibilityState !== "visible") stopManualPoll();
  else if (
    state.manualReceipt &&
    state.manualReceipt.scope === state.scope &&
    ["queued", "running"].includes(state.manualReceipt.data?.state)
  ) {
    pollManualReceipt(state.scope, state.manualReceipt.request_id);
  }
});

window.addEventListener("pagehide", () => {
  clearTimeout(state.pollTimer);
  stopManualPoll();
  state.detailVersion += 1;
  state.runDetail = null;
});

$("refresh").addEventListener("click", () => attempt(refresh));

document.querySelectorAll("[data-tab]").forEach((tab) =>
  tab.addEventListener("click", () => switchTab(tab.dataset.tab))
);

/* Memory Star Map HUD Controls */
document.querySelectorAll("[data-region-filter]").forEach((btn) =>
  btn.addEventListener("click", () => {
    state.regionFilter = btn.dataset.regionFilter;
    syncRegionSegButtons();
    attempt(reloadMemory);
  })
);

function syncGraphViewButtons() {
  $("graph-view-backbone")?.classList.toggle("active", state.graphView === "backbone");
  $("graph-view-recent")?.classList.toggle("active", state.graphView === "recent");
}
$("graph-view-backbone")?.addEventListener("click", () => {
  state.graphView = "backbone";
  syncGraphViewButtons();
  attempt(resetGraphToOverview);
});
$("graph-view-recent")?.addEventListener("click", () => {
  state.graphView = "recent";
  syncGraphViewButtons();
  attempt(resetGraphToOverview);
});
$("graph-grow-toggle")?.addEventListener("click", () => {
  state.graphGrowMode = !state.graphGrowMode;
  $("graph-grow-toggle").textContent = state.graphGrowMode ? "原地生长" : "单点聚焦";
  $("graph-grow-toggle").classList.toggle("active", state.graphGrowMode);
});
$("memory-more")?.addEventListener("click", () => attempt(() => loadMemoryPage(true)));

document.querySelectorAll("[data-vis-region]").forEach((btn) =>
  btn.addEventListener("click", () => {
    const r = btn.dataset.visRegion;
    if (!graphEngine) return;
    if (graphEngine.visRegions.has(r)) {
      if (graphEngine.visRegions.size > 1) graphEngine.visRegions.delete(r);
    } else {
      graphEngine.visRegions.add(r);
    }
    btn.classList.toggle("on", graphEngine.visRegions.has(r));
    graphEngine.draw();
  })
);

document.querySelectorAll("[data-vis-bin]").forEach((btn) =>
  btn.addEventListener("click", () => {
    const b = btn.dataset.visBin;
    if (!graphEngine) return;
    if (graphEngine.visBins.has(b)) graphEngine.visBins.delete(b);
    else graphEngine.visBins.add(b);
    btn.classList.toggle("on", graphEngine.visBins.has(b));
    graphEngine.draw();
  })
);

$("graph-layout-toggle")?.addEventListener("click", () => {
  if (!graphEngine) return;
  graphEngine.layoutMode = graphEngine.layoutMode === "zones" ? "force" : "zones";
  $("graph-layout-toggle").textContent = graphEngine.layoutMode === "zones" ? "四区星图" : "自由力导";
  $("graph-layout-toggle").classList.toggle("active", graphEngine.layoutMode === "force");
  graphEngine.computeLayout();
  graphEngine.fitToView(true);
});

$("graph-label-toggle")?.addEventListener("click", () => {
  if (!graphEngine) return;
  graphEngine.showAllLabels = !graphEngine.showAllLabels;
  $("graph-label-toggle").textContent = graphEngine.showAllLabels ? "展开标签" : "智能标签";
  $("graph-label-toggle").classList.toggle("active", graphEngine.showAllLabels);
  graphEngine.draw();
});

$("graph-zoom-in")?.addEventListener("click", () => {
  if (graphEngine) graphEngine.setCamera(graphEngine.cam.x, graphEngine.cam.y, graphEngine.cam.zoom * 1.25, true);
});
$("graph-zoom-out")?.addEventListener("click", () => {
  if (graphEngine) graphEngine.setCamera(graphEngine.cam.x, graphEngine.cam.y, graphEngine.cam.zoom * 0.8, true);
});
$("graph-fit")?.addEventListener("click", () => graphEngine?.fitToView(true));
$("graph-reset")?.addEventListener("click", () => attempt(resetGraphToOverview));

for (const [form, action] of [
  ["memory-search", searchMemory],
  ["people-search", searchPeople],
  ["message-search", () => searchMessages()],
]) {
  $(form).addEventListener("submit", (event) => {
    event.preventDefault();
    attempt(action);
  });
}

$("memory-kind")?.addEventListener("change", () => attempt(reloadMemory));
$("older").addEventListener("click", () => attempt(() => searchMessages(true)));

$("recover-history").addEventListener("click", () =>
  attempt(async () => {
    $("recover-history").disabled = true;
    try {
      const result = await api(path("history/recover"), {}, true);
      notice(result.message);
      await loadTab();
    } catch (error) {
      $("recover-history").disabled = !state.scope || !state.overview?.runtime.history_recovery;
      throw error;
    }
  })
);

$("profile-form").addEventListener("submit", (event) => {
  event.preventDefault();
  attempt(async () => {
    if (!$("profile-account").value) throw new Error("请先选择群员");
    const scope = state.scope;
    const list = (id) =>
      $(id)
        .value.replaceAll("，", ",")
        .split(",")
        .map((x) => x.trim())
        .filter(Boolean);
    const data = await api(
      path("participants/profile"),
      {
        account_id: $("profile-account").value,
        changes: {
          preferred_name: $("profile-name").value,
          aliases: list("profile-aliases"),
          avoided_names: list("profile-avoided"),
          description: $("profile-description").value,
        },
      },
      true
    );
    if (scope === state.scope) {
      notice(data.message);
      await searchPeople();
    }
  });
});

$("distill").addEventListener("click", () =>
  attempt(async () => {
    const scope = state.scope;
    $("distill").disabled = true;
    $("distill").textContent = "正在整理…";
    notice("正在整理此群尚未学到的消息，本次仍按已配置的后台额度运行。");
    try {
      const result = await api(path("distill"), {}, true);
      notice(
        result?.reason ||
          result?.detail ||
          (result?.status === "completed"
            ? `整理完成，更新 ${result.written_count ?? "未记录数量的"} 项记忆。`
            : `本次整理状态：${statuses[result?.status] || result?.status || "未记录"}`),
        ["error", "failed"].includes(result?.status)
      );
      if (scope === state.scope) await loadTab();
    } finally {
      $("distill").disabled = !state.scope || !state.overview?.runtime.background_enabled;
      $("distill").textContent = "整理新消息";
    }
  })
);

$("reset-conversation").addEventListener("click", async () => {
  if (!state.scope || state.resetBusy) return;
  const scope = state.scope;
  const group = state.overview.scopes.find((item) => item.id === scope);
  const label = formatScopeLabel(group);
  const control = $("reset-conversation"),
    result = $("context-repair-result");
  state.resetBusy = true;
  control.disabled = true;
  control.textContent = "正在开启新会话…";
  result.hidden = false;
  result.classList.remove("error-text");
  result.textContent = `正在处理 ${label}…`;
  try {
    const data = await api(path("conversation/reset", scope), {}, true);
    if (!["cleared", "already_empty", "no_conversation"].includes(data?.status))
      throw new Error(data?.message || "未收到会话清理结果，请刷新后查看。");
    const message = `${label}：${data.message}`;
    if (scope === state.scope) {
      result.textContent = message;
      result.hidden = false;
    } else notice(message);
  } catch (error) {
    const message = `${label}：${error.message || String(error)}`;
    if (scope === state.scope) {
      result.textContent = message;
      result.classList.add("error-text");
      result.hidden = false;
    } else notice(message, true);
  } finally {
    state.resetBusy = false;
    control.disabled = !state.scope;
    control.textContent = "清理主会话";
  }
});

$("platform-migration").addEventListener("toggle", () => {
  if ($("platform-migration").open) attempt(() => loadMigration());
});
$("migration-preview").addEventListener("click", () => attempt(() => loadMigration(true)));
$("migration-legacy").addEventListener("change", () => {
  $("migration-bind").disabled = true;
  migrationPreview = null;
});
$("migration-bind").addEventListener("click", () =>
  attempt(async () => {
    if (!migrationPreview || migrationPreview.scope !== state.scope || !migrationPreview.selected)
      throw new Error("请先预览当前两个群的对照结果");
    $("migration-bind").disabled = true;
    const result = await api(path("migration/bind"), { legacy_scope: migrationPreview.selected }, true);
    notice(result.message);
    await loadMigration();
  })
);

/* Manual correction composer events */
$("manual-mention-query")?.addEventListener("input", () => {
  clearTimeout(state.mentionTimer);
  state.mentionTimer = setTimeout(() => attempt(queryManualMentions), 160);
});
$("manual-reply-id")?.addEventListener("input", () => {
  $("manual-clear-reply").hidden = !$("manual-reply-id").value.trim();
});
$("manual-clear-reply")?.addEventListener("click", () => setManualReply(""));
$("manual-form")?.addEventListener("submit", (e) => {
  e.preventDefault();
  attempt(() => submitManualMessages("next_window"));
});
$("manual-submit-window")?.addEventListener("click", () =>
  attempt(() => submitManualMessages("next_window"))
);
$("manual-submit-now")?.addEventListener("click", () =>
  attempt(() => submitManualMessages("now"))
);

/* =========================================================================
   BOOTSTRAP BRIDGE
   ========================================================================= */
attempt(async () => {
  const deadline = performance.now() + 5000;
  while (!window.AstrBotPluginPage) {
    if (performance.now() > deadline)
      throw new Error("请从 AstrBot 的 MR 插件页面入口打开控制台。");
    await new Promise((resolve) => setTimeout(resolve, 25));
  }
  await refresh();
});
