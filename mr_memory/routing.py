from __future__ import annotations

from dataclasses import dataclass


ROUTE_POLICY_REVISION = "host-route-policy.v1"

POLICY_MODES = {"LOW_LATENCY", "BALANCED", "RESEARCH"}
REQUEST_KINDS = {"CHAT", "MEMORY_QUERY", "DEEP_RECALL", "FEEDBACK_AUDIT"}
CACHE_STATES = {"HIT", "MISS", "STALE"}
SEMANTIC_STATUSES = {
    "UNKNOWN",
    "CERTIFIED",
    "PARTIAL",
    "SEMANTIC_NONE",
    "SAFETY_ABSTAIN",
    "REQUEST_L3",
}
OPERATIONAL_STATUSES = {
    "READY",
    "RUNNING",
    "BUDGET_BLOCKED",
    "PROVIDER_UNAVAILABLE",
    "FAILED",
}


@dataclass(frozen=True, slots=True)
class RouteFeatures:
    """Host-observed inputs to one memory routing decision.

    Semantic results and operational state deliberately occupy different
    fields.  A provider timeout, a budget refusal, or an in-flight request can
    therefore never be re-labelled as ``SEMANTIC_NONE``.
    """

    request_kind: str = "CHAT"
    deterministic_sufficient: bool = False
    identity_only: bool = False
    identity_ambiguous: bool = False
    high_risk: bool = False
    multi_event: bool = False
    conflicting_evidence: bool = False
    revision_question: bool = False
    explicit_deep: bool = False
    feedback_audit: bool = False
    l1a_cache_state: str = "MISS"
    l1b_cache_state: str = "MISS"
    l1b_semantic_status: str = "UNKNOWN"
    live_semantic_status: str = "UNKNOWN"
    operational_status: str = "READY"
    singleflight_running: bool = False

    def __post_init__(self) -> None:
        request_kind = str(self.request_kind).strip().upper()
        l1a_state = str(self.l1a_cache_state).strip().upper()
        l1b_state = str(self.l1b_cache_state).strip().upper()
        l1b_status = str(self.l1b_semantic_status).strip().upper()
        live_status = str(self.live_semantic_status).strip().upper()
        operational = str(self.operational_status).strip().upper()
        if request_kind not in REQUEST_KINDS:
            raise ValueError("request_kind is unsupported")
        if l1a_state not in CACHE_STATES or l1b_state not in CACHE_STATES:
            raise ValueError("cache state is unsupported")
        if l1b_status not in SEMANTIC_STATUSES:
            raise ValueError("l1b_semantic_status is unsupported")
        if live_status not in SEMANTIC_STATUSES:
            raise ValueError("live_semantic_status is unsupported")
        if operational not in OPERATIONAL_STATUSES:
            raise ValueError("operational_status is unsupported")
        object.__setattr__(self, "request_kind", request_kind)
        object.__setattr__(self, "l1a_cache_state", l1a_state)
        object.__setattr__(self, "l1b_cache_state", l1b_state)
        object.__setattr__(self, "l1b_semantic_status", l1b_status)
        object.__setattr__(self, "live_semantic_status", live_status)
        object.__setattr__(self, "operational_status", operational)


@dataclass(frozen=True, slots=True)
class RouteDecision:
    level: str
    action: str
    execution: str
    deadline_ms: int
    semantic_status: str
    operational_status: str
    cache_layer: str
    join_singleflight: bool
    allow_provider_call: bool
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.level not in {"L0", "L1", "L2", "L3"}:
            raise ValueError("route level is unsupported")
        if self.execution not in {"RETURN", "SYNC", "ASYNC", "BLOCKED"}:
            raise ValueError("route execution is unsupported")
        if int(self.deadline_ms) < 0:
            raise ValueError("deadline_ms must be non-negative")
        if self.semantic_status not in SEMANTIC_STATUSES:
            raise ValueError("route semantic_status is unsupported")
        if self.operational_status not in OPERATIONAL_STATUSES:
            raise ValueError("route operational_status is unsupported")

    def as_dict(self) -> dict[str, object]:
        return {
            "level": self.level,
            "action": self.action,
            "execution": self.execution,
            "deadline_ms": self.deadline_ms,
            "semantic_status": self.semantic_status,
            "operational_status": self.operational_status,
            "cache_layer": self.cache_layer,
            "join_singleflight": self.join_singleflight,
            "allow_provider_call": self.allow_provider_call,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True, slots=True)
class RoutePolicy:
    """Deterministic host policy; the reader may request L3 but cannot route it."""

    mode: str = "BALANCED"
    allow_l3: bool = True
    l2_deadline_ms: int = 1500
    l3_deadline_ms: int = 15000

    def __post_init__(self) -> None:
        mode = str(self.mode).strip().upper()
        if mode not in POLICY_MODES:
            raise ValueError("route policy mode is unsupported")
        if int(self.l2_deadline_ms) <= 0 or int(self.l3_deadline_ms) <= 0:
            raise ValueError("route deadlines must be positive")
        object.__setattr__(self, "mode", mode)

    @property
    def revision(self) -> str:
        return (
            f"{ROUTE_POLICY_REVISION}:{self.mode.casefold()}:"
            f"l3={int(self.allow_l3)}:{self.l2_deadline_ms}:{self.l3_deadline_ms}"
        )

    def _return(
        self,
        *,
        level: str,
        action: str,
        semantic_status: str,
        cache_layer: str,
        reasons: tuple[str, ...],
    ) -> RouteDecision:
        return RouteDecision(
            level=level,
            action=action,
            execution="RETURN",
            deadline_ms=0,
            semantic_status=semantic_status,
            operational_status="READY",
            cache_layer=cache_layer,
            join_singleflight=False,
            allow_provider_call=False,
            reasons=reasons,
        )

    def _blocked(
        self,
        *,
        level: str,
        operational_status: str,
        reasons: tuple[str, ...],
    ) -> RouteDecision:
        action = {
            "BUDGET_BLOCKED": "BLOCK_BUDGET",
            "PROVIDER_UNAVAILABLE": "BLOCK_PROVIDER",
            "FAILED": "REPORT_FAILURE",
        }.get(operational_status, "REPORT_OPERATIONAL_STATE")
        return RouteDecision(
            level=level,
            action=action,
            execution="BLOCKED",
            deadline_ms=0,
            semantic_status="UNKNOWN",
            operational_status=operational_status,
            cache_layer="NONE",
            join_singleflight=False,
            allow_provider_call=False,
            reasons=reasons,
        )

    def _execution(self, features: RouteFeatures, level: str) -> str:
        if features.request_kind in {"MEMORY_QUERY", "DEEP_RECALL"}:
            return "SYNC"
        if self.mode == "RESEARCH":
            return "SYNC"
        if level == "L3" and features.high_risk:
            return "SYNC"
        return "ASYNC"

    def _run(
        self,
        features: RouteFeatures,
        *,
        level: str,
        reasons: tuple[str, ...],
    ) -> RouteDecision:
        execution = self._execution(features, level)
        join = bool(features.singleflight_running)
        deadline = self.l3_deadline_ms if level == "L3" else self.l2_deadline_ms
        return RouteDecision(
            level=level,
            action=("JOIN_" if join else "START_") + level,
            execution=execution,
            deadline_ms=deadline if execution == "SYNC" else 0,
            semantic_status="UNKNOWN",
            operational_status="RUNNING",
            cache_layer="L1A" if features.l1a_cache_state == "HIT" else "NONE",
            join_singleflight=join,
            allow_provider_call=not join,
            reasons=reasons,
        )

    def decide(self, features: RouteFeatures) -> RouteDecision:
        if features.deterministic_sufficient or features.identity_only:
            return self._return(
                level="L0",
                action="RETURN_DETERMINISTIC",
                semantic_status="CERTIFIED",
                cache_layer="NONE",
                reasons=("deterministic host evidence is sufficient",),
            )

        if features.l1b_cache_state == "HIT" and features.l1b_semantic_status in {
            "CERTIFIED",
            "PARTIAL",
            "SEMANTIC_NONE",
            "SAFETY_ABSTAIN",
        }:
            return self._return(
                level="L1",
                action="RETURN_CERTIFICATE",
                semantic_status=features.l1b_semantic_status,
                cache_layer="L1B",
                reasons=("valid semantic-certificate cache hit",),
            )

        l3_requested = features.live_semantic_status == "REQUEST_L3"
        l3_required = any(
            (
                features.explicit_deep,
                features.feedback_audit,
                features.request_kind in {"DEEP_RECALL", "FEEDBACK_AUDIT"},
                features.high_risk,
                features.multi_event,
                features.conflicting_evidence,
                features.revision_question,
                l3_requested,
            )
        )

        if features.identity_ambiguous and not l3_required:
            return self._return(
                level="L1",
                action="RETURN_SAFETY_ABSTAIN",
                semantic_status="SAFETY_ABSTAIN",
                cache_layer="NONE",
                reasons=("identity binding is ambiguous",),
            )

        target = "L3" if l3_required and self.allow_l3 else "L2"
        if l3_required and not self.allow_l3:
            return self._return(
                level="L1",
                action="RETURN_SAFETY_ABSTAIN",
                semantic_status="SAFETY_ABSTAIN",
                cache_layer="NONE",
                reasons=("host policy denied the requested L3 route",),
            )

        if features.operational_status in {
            "BUDGET_BLOCKED",
            "PROVIDER_UNAVAILABLE",
            "FAILED",
        }:
            return self._blocked(
                level=target,
                operational_status=features.operational_status,
                reasons=(
                    "operational failure is not semantic absence",
                    f"target={target}",
                ),
            )

        reasons: list[str] = []
        if l3_requested:
            reasons.append("reader requested L3; host policy approved")
        if features.identity_ambiguous:
            reasons.append("identity ambiguity requires deeper evidence closure")
        if features.high_risk:
            reasons.append("high-risk attribution")
        if features.multi_event:
            reasons.append("multi-event synthesis")
        if features.conflicting_evidence:
            reasons.append("conflicting evidence")
        if features.revision_question:
            reasons.append("belief revision question")
        if features.explicit_deep or features.feedback_audit:
            reasons.append("explicit deep analysis")
        if not reasons:
            reasons.append("semantic certificate cache miss")
        return self._run(features, level=target, reasons=tuple(reasons))
