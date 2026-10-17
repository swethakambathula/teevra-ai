"""TEEVRA AI API Gateway - FastAPI backend.

Exposes the full REST contract, SSE event stream, WebSocket dashboard updates,
and coordinates the ADK agents over real A2A HTTP. Local mode uses the in-memory
operational state store (Firestore equivalent) and simulated Pub/Sub bus.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends, Query, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import httpx

from teevra_shared_schemas import Severity, HeatStatus
from teevra_simulation import (
    MACHINES, JOBS, WORKERS, TOTAL_SLOTS, FROZEN_WINDOW_SLOTS, DEMO_START,
    slot_to_ist, ist_to_slot, HEAT_POLICY, WORKSTATIONS, tariff_rate_at_ist,
    get_worker,
)
from teevra_simulation.state import get_state, reset_state
from teevra_scheduler_engine import CPSatSchedulingEngine
from teevra_a2a import A2ARemoteClient

app = FastAPI(title="TEEVRA AI API Gateway", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ENGINE = CPSatSchedulingEngine(total_slots=TOTAL_SLOTS)
RECOVERY_URL = "http://127.0.0.1:8072"
HEAT_URL = "http://127.0.0.1:8071"

# --------------------------------------------------------------------------- #
# Auth (local demo: bearer role header; Firebase in cloud mode)
# --------------------------------------------------------------------------- #
def current_role(authorization: Optional[str] = Header(None)) -> str:
    if not authorization:
        return "PLANNER"  # local demo default
    parts = authorization.split()
    if len(parts) == 2 and parts[0] == "Bearer":
        token = parts[1]
        if token.startswith("role:"):
            return token[5:]
        if token == "demo-supervisor":
            return "SUPERVISOR"
        if token == "demo-planner":
            return "PLANNER"
        if token == "demo-worker":
            return "WORKER"
    return "PLANNER"


def require_planner(role: str = Depends(current_role)):
    if role not in ("PLANNER", "SUPERVISOR", "ADMIN"):
        raise HTTPException(status_code=403, detail="planner role required")
    return role


# --------------------------------------------------------------------------- #
# SSE subscribers
# --------------------------------------------------------------------------- #
class Hub:
    def __init__(self):
        self.subscribers: list[asyncio.Queue] = []

    async def subscribe(self):
        q: asyncio.Queue = asyncio.Queue()
        self.subscribers.append(q)
        return q

    def unsubscribe(self, q):
        if q in self.subscribers:
            self.subscribers.remove(q)

    async def publish(self, event: dict):
        data = json.dumps(event, default=str)
        dead = []
        for q in self.subscribers:
            try:
                q.put_nowait(data)
            except Exception:
                dead.append(q)
        for q in dead:
            self.unsubscribe(q)


HUB = Hub()


# --------------------------------------------------------------------------- #
# Health
# --------------------------------------------------------------------------- #
@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "teevra-api-gateway", "mode": "local"}


@app.get("/api/readiness")
async def readiness():
    return {"ready": True}


# --------------------------------------------------------------------------- #
# Demo control
# --------------------------------------------------------------------------- #
@app.post("/api/demo/reset")
async def demo_reset(role: str = Depends(require_planner)):
    reset_state()
    state = get_state()
    # seed baseline schedule
    base = ENGINE.build_baseline_schedule(
        {"machines": MACHINES, "jobs": JOBS, "now_slot": 0}
    )
    state.schedules[base.version_id] = base
    state.current_version_id = base.version_id
    for m in MACHINES:
        state.machine_status[m.machine_id] = m.status.value
    await HUB.publish({
        "type": "demo.reset",
        "payload": {"version_id": base.version_id, "entries": len(base.entries)},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    return {"status": "reset", "version_id": base.version_id, "entries": len(base.entries)}


@app.post("/api/demo/seed")
async def demo_seed(role: str = Depends(require_planner)):
    return await demo_reset(role)


# --------------------------------------------------------------------------- #
# Machine fault events (Pub/Sub adapter)
# --------------------------------------------------------------------------- #
@app.post("/api/events/machine-fault")
async def machine_fault(payload: dict, role: str = Depends(current_role)):
    state = get_state()
    event_id = payload.get("event_id") or f"fault-{uuid.uuid4().hex[:8]}"
    machine_id = payload["machine_id"]
    ts = payload.get("timestamp") or datetime.now(timezone.utc).isoformat()
    dedup_key = f"MACHINE_BREAKDOWN:{machine_id}:{ts}"
    if dedup_key in state.event_dedup:
        return {"status": "duplicate", "event_id": event_id}
    state.event_dedup.add(dedup_key)

    now_slot = ist_to_slot(datetime.fromisoformat(ts))
    repair = int(payload.get("estimated_repair_minutes", 180))
    fault_end = now_slot + max(1, repair // 15)

    # mark machine down
    state.machine_status[machine_id] = "DOWN"

    event = {
        "event_id": event_id,
        "machine_id": machine_id,
        "event_type": "MACHINE_BREAKDOWN",
        "timestamp": ts,
        "estimated_repair_minutes": repair,
        "severity": payload.get("severity", "HIGH"),
        "source": payload.get("source", "OPERATOR"),
        "now_slot": now_slot,
        "fault_end_slot": fault_end,
    }
    state.fault_events.append(event)

    # Feature 3: Auto-trigger breakdown recovery solver immediately on fault event
    # (Pub/Sub message -> solver builds rescue plan with 2-hour frozen window)
    current = state.schedules.get(state.current_version_id) if state.current_version_id else None
    recovery_plan = None
    if current is not None:
        heat_blocked = [w for w, risk in state.workstation_risk.items() if risk == "STOP_WORK"]
        recovery_plan = ENGINE.recover_after_breakdown(
            current_version=current,
            fault_machine=machine_id,
            fault_start_slot=now_slot,
            fault_end_slot=fault_end,
            now_slot=now_slot,
            heat_blocked_stations=heat_blocked,
            time_limit_seconds=30.0,
        )
        state.schedules[recovery_plan.schedule_version.version_id] = recovery_plan.schedule_version
        state.candidate_version_id = recovery_plan.schedule_version.version_id

    await HUB.publish({
        "type": "machine.fault",
        "payload": event,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    result = {"status": "accepted", "event_id": event_id, "fault_end_slot": fault_end}
    if recovery_plan:
        r = recovery_plan.solver_result
        result["recovery"] = {
            "plan_id": recovery_plan.plan_id,
            "version_id": recovery_plan.schedule_version.version_id,
            "solver_status": r.solver_status.value,
            "is_feasible": r.is_feasible,
            "is_optimal": r.is_optimal,
            "jobs_unchanged": r.jobs_unchanged,
            "jobs_reassigned": r.jobs_reassigned,
            "jobs_delayed": r.jobs_delayed,
            "jobs_blocked": r.jobs_blocked,
            "expected_output_percent": r.expected_output_percent,
            "electricity_cost_before_inr": r.electricity_cost_before_inr,
            "electricity_cost_after_inr": r.electricity_cost_after_inr,
            "recovery_duration_ms": r.recovery_duration_ms,
            "human_approval_required": r.human_approval_required,
            "auto_commit_allowed": recovery_plan.auto_commit_allowed,
        }
        await HUB.publish({
            "type": "schedule.recovery",
            "payload": result["recovery"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
    return result


@app.post("/api/events/machine-restored")
async def machine_restored(payload: dict, role: str = Depends(current_role)):
    state = get_state()
    machine_id = payload["machine_id"]
    state.machine_status[machine_id] = "AVAILABLE"
    await HUB.publish({
        "type": "machine.restored",
        "payload": payload,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    return {"status": "ok"}


# --------------------------------------------------------------------------- #
# Schedule
# --------------------------------------------------------------------------- #
@app.get("/api/schedule/current")
async def schedule_current(role: str = Depends(current_role)):
    state = get_state()
    vid = state.current_version_id
    v = state.schedules.get(vid) if vid else None
    if v is None:
        return {"status": "no_schedule"}
    return {
        "status": "ok",
        "version_id": v.version_id,
        "is_committed": v.is_committed,
        "frozen_window": {
            "start_slot": v.frozen_window.start_slot,
            "end_slot": v.frozen_window.end_slot,
        },
        "entries": [e.model_dump() for e in v.entries],
        "machines": {m.machine_id: state.machine_status.get(m.machine_id, m.status.value) for m in MACHINES},
    }


@app.get("/api/schedule/history")
async def schedule_history(role: str = Depends(current_role)):
    state = get_state()
    return {
        "versions": [
            {"version_id": v.version_id, "created_at": v.created_at.isoformat(), "committed": v.is_committed}
            for v in state.schedules.values()
        ]
    }


@app.get("/api/schedule/diff")
async def schedule_diff(previous: str = Query(...), new: str = Query(...), role: str = Depends(current_role)):
    state = get_state()
    prev = state.schedules.get(previous)
    newv = state.schedules.get(new)
    if not prev or not newv:
        raise HTTPException(status_code=404, detail="version not found")
    ev = ENGINE.compare_schedules(prev, newv)
    return {"previous_version": previous, "new_version": new, "change_evidence": [e.model_dump() for e in ev]}


@app.post("/api/schedule/recover")
async def schedule_recover(payload: dict, role: str = Depends(require_planner)):
    state = get_state()
    current = state.schedules.get(state.current_version_id) if state.current_version_id else None
    if current is None:
        raise HTTPException(status_code=400, detail="no current schedule")
    fault_machine = payload.get("fault_machine", "CNC-A")
    start = int(payload.get("fault_start_slot", 20))
    end = int(payload.get("fault_end_slot", 31))
    now_slot = int(payload.get("now_slot", start))
    heat_blocked = [w for w, risk in state.workstation_risk.items() if risk == "STOP_WORK"]
    plan = ENGINE.recover_after_breakdown(
        current_version=current,
        fault_machine=fault_machine,
        fault_start_slot=start,
        fault_end_slot=end,
        now_slot=now_slot,
        heat_blocked_stations=heat_blocked,
        time_limit_seconds=30.0,
    )
    r = plan.solver_result
    result = r.model_dump()
    result["is_feasible"] = r.is_feasible
    result["is_optimal"] = r.is_optimal
    state.schedules[plan.schedule_version.version_id] = plan.schedule_version
    state.candidate_version_id = plan.schedule_version.version_id
    await HUB.publish({
        "type": "schedule.recovery",
        "payload": result,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    return {"status": "ok", "plan": result, "version_id": plan.schedule_version.version_id}


@app.post("/api/schedule/approve")
async def schedule_approve(payload: dict, role: str = Depends(require_planner)):
    state = get_state()
    vid = payload.get("version_id") or state.candidate_version_id
    if not vid or vid not in state.schedules:
        raise HTTPException(status_code=404, detail="no candidate version")
    v = state.schedules[vid]
    v.is_committed = True
    v.commit_correlation_id = str(uuid.uuid4())
    state.current_version_id = vid
    await HUB.publish({"type": "schedule.approved", "payload": {"version_id": vid}})
    return {"status": "approved", "version_id": vid}


@app.post("/api/schedule/reject")
async def schedule_reject(payload: dict, role: str = Depends(require_planner)):
    state = get_state()
    vid = payload.get("version_id") or state.candidate_version_id
    if vid in state.schedules:
        del state.schedules[vid]
    if state.candidate_version_id == vid:
        state.candidate_version_id = None
    await HUB.publish({"type": "schedule.rejected", "payload": {"version_id": vid}})
    return {"status": "rejected", "version_id": vid}


@app.post("/api/schedule/rollback")
async def schedule_rollback(role: str = Depends(require_planner)):
    state = get_state()
    committed = [v for v in state.schedules.values() if v.is_committed]
    if not committed:
        raise HTTPException(status_code=404, detail="nothing to rollback")
    v = sorted(committed, key=lambda x: x.created_at or "")[-1]
    state.current_version_id = v.version_id
    await HUB.publish({"type": "schedule.rollback", "payload": {"version_id": v.version_id}})
    return {"status": "rolled_back", "version_id": v.version_id}


# --------------------------------------------------------------------------- #
# Heat Guardian
# --------------------------------------------------------------------------- #
@app.get("/api/heat/workstations")
async def heat_workstations(role: str = Depends(current_role)):
    state = get_state()
    results = []
    for ws in WORKSTATIONS:
        wid = ws["workstation_id"]
        risk = state.workstation_risk.get(wid, "UNKNOWN")
        results.append({
            "workstation_id": wid,
            "name": ws["name"],
            "machine_id": ws.get("machine_id"),
            "risk_level": risk,
            "base_wbgt_c": ws["base_wbgt_c"],
        })
    return {"status": "ok", "workstations": results}


@app.get("/api/heat/measurements")
async def heat_measurements(role: str = Depends(current_role)):
    state = get_state()
    return {"status": "ok", "readings": state.heat_readings[-50:], "count": len(state.heat_readings)}


@app.post("/api/events/heat-reading")
async def heat_reading(payload: dict, role: str = Depends(current_role)):
    state = get_state()
    reading = {
        "workstation_id": payload.get("workstation_id", "WS-F1"),
        "temperature_c": payload.get("temperature_c", 35.0),
        "relative_humidity": payload.get("relative_humidity", 70.0),
        "natural_wet_bulb_c": payload.get("natural_wet_bulb_c", 30.0),
        "globe_temperature_c": payload.get("globe_temperature_c", 34.0),
        "sensor_fresh_seconds": payload.get("sensor_fresh_seconds", 10),
        "sensor_quality": payload.get("sensor_quality", "GOOD"),
        "recorded_at": payload.get("timestamp") or datetime.now(timezone.utc).isoformat(),
    }
    state.heat_readings.append(reading)
    # evaluate risk
    nwb = reading.get("natural_wet_bulb_c"); globe = reading.get("globe_temperature_c")
    wbgt = round(0.7 * nwb + 0.3 * globe, 2) if nwb is not None and globe is not None else None
    quality = reading.get("sensor_quality", "GOOD"); fresh = reading.get("sensor_fresh_seconds")
    risk = "UNKNOWN"
    if wbgt is not None and quality != "FAULTY" and (fresh is None or fresh <= 300):
        if wbgt >= HEAT_POLICY["stop_work_min_wbgt_c"]:
            risk = "STOP_WORK"
        elif wbgt >= HEAT_POLICY["high_risk_max_wbgt_c"]:
            risk = "HIGH_RISK"
        elif wbgt >= HEAT_POLICY["caution_max_wbgt_c"]:
            risk = "CAUTION"
        else:
            risk = "NORMAL"
    wid = reading["workstation_id"]
    state.workstation_risk[wid] = risk
    await HUB.publish({
        "type": "heat.reading",
        "payload": {"workstation_id": wid, "wbgt_c": wbgt, "risk": risk},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

    # Feature 4: Heat STOP_WORK -> automatically request recovery plan
    # (heat guardian triggers rescheduler to move workers off hot stations)
    recovery_triggered = None
    if risk == "STOP_WORK":
        current = state.schedules.get(state.current_version_id) if state.current_version_id else None
        if current is not None:
            heat_blocked = [w for w, r in state.workstation_risk.items() if r == "STOP_WORK"]
            now_slot = ist_to_slot(datetime.now(timezone.utc).astimezone(ZoneInfo("Asia/Kolkata")))
            rec_plan = ENGINE.recover_after_heat_event(
                current_version=current,
                blocked_stations=[wid],
                now_slot=now_slot,
                time_limit_seconds=30.0,
            )
            state.schedules[rec_plan.schedule_version.version_id] = rec_plan.schedule_version
            state.candidate_version_id = rec_plan.schedule_version.version_id
            rr = rec_plan.solver_result
            recovery_triggered = {
                "plan_id": rec_plan.plan_id,
                "version_id": rec_plan.schedule_version.version_id,
                "solver_status": rr.solver_status.value,
                "is_feasible": rr.is_feasible,
                "jobs_reassigned": rr.jobs_reassigned,
                "jobs_delayed": rr.jobs_delayed,
                "recovery_duration_ms": rr.recovery_duration_ms,
            }
            await HUB.publish({
                "type": "heat.recovery_triggered",
                "payload": recovery_triggered,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

    return {"status": "ok", "workstation_id": wid, "wbgt_c": wbgt, "risk_level": risk, "recovery_triggered": recovery_triggered}


# --------------------------------------------------------------------------- #
# Worker check-ins
# --------------------------------------------------------------------------- #
@app.post("/api/worker/checkin")
async def worker_checkin(payload: dict, role: str = Depends(current_role)):
    state = get_state()
    from teevra_shared_schemas.heat import CheckIn, CheckInResponse
    checkin = CheckIn(
        checkin_id=f"ci-{uuid.uuid4().hex[:8]}",
        worker_id=payload.get("worker_id", "WRK-01"),
        workstation_id=payload.get("workstation_id", "WS-CNC-A"),
        language=payload.get("language", "en"),
        response=CheckInResponse(payload.get("response", "OK")),
        voice_consent=bool(payload.get("voice_consent", False)),
        responded_at=datetime.now(timezone.utc).isoformat(),
        reminder_sent_at=datetime.now(timezone.utc).isoformat(),
    )
    state.checkins.append(checkin)
    if checkin.response == CheckInResponse.SOS:
        from teevra_shared_schemas.heat import SupervisorAlert
        alert = SupervisorAlert(
            alert_id=f"alert-{uuid.uuid4().hex[:8]}",
            alert_type="WORKER_SOS",
            severity="CRITICAL",
            workstation_id=checkin.workstation_id,
            worker_id=checkin.worker_id,
            message=f"SOS check-in from worker {checkin.worker_id}",
        )
        state.alerts[alert.alert_id] = alert
        await HUB.publish({"type": "worker.sos", "payload": alert.model_dump(), "timestamp": datetime.now(timezone.utc).isoformat()})
        return {"status": "ok", "alert_id": alert.alert_id, "escalated": True}
    await HUB.publish({"type": "worker.checkin", "payload": checkin.model_dump(), "timestamp": datetime.now(timezone.utc).isoformat()})
    return {"status": "ok", "checkin_id": checkin.checkin_id, "response": checkin.response.value}


# --------------------------------------------------------------------------- #
# Feature 4: Worker check-in initiation + Telugu voice delivery
# --------------------------------------------------------------------------- #

TELUGU_MESSAGES = {
    "prompt": "మీకు ఎలా ఉంది? నీళ్లు తాగారా?",
    "ok": "సరిగ్గా / OK",
    "too_hot": "చాలా వేడి / Too Hot",
    "sos": "సహాయం కావాలి / SOS",
    "consent": "గోసపి ఉపయోగించాలనుకుంటున్నారా? GPS ట్రాకింగ్ లేదు.",
}

ENGLISH_MESSAGES = {
    "prompt": "How are you? Have you had water?",
    "ok": "OK",
    "too_hot": "Too Hot",
    "sos": "SOS Need Help",
    "consent": "Consent to voice processing? No GPS tracking.",
}


def _voice_messages(language: str) -> dict:
    return TELUGU_MESSAGES if language == "te" else ENGLISH_MESSAGES


@app.post("/api/worker/checkin/initiate")
async def checkin_initiate(payload: dict, role: str = Depends(current_role)):
    """Feature 4: Initiate scheduled voice check-ins for workers on shift.

    Creates CheckIn records (MISSED by default) with Telugu/English voice
    prompts. Workers reply via /api/worker/checkin. SOS triggers a CRITICAL
    supervisor alert immediately.
    """
    state = get_state()
    from teevra_shared_schemas.heat import CheckIn, CheckInResponse

    language = payload.get("language", "te")
    msgs = _voice_messages(language)
    voice_consent = bool(payload.get("voice_consent", False))
    worker_id = payload.get("worker_id")
    workstation_id = payload.get("workstation_id")

    targets = []
    if worker_id:
        try:
            targets.append(get_worker(worker_id))
        except KeyError:
            raise HTTPException(status_code=404, detail=f"unknown worker_id: {worker_id}")
    else:
        # Initiate for workers on the current shift (demo: all floor workers)
        now_ist = datetime.now(timezone.utc).astimezone(ZoneInfo("Asia/Kolkata"))
        current_shift = "A" if 6 <= now_ist.hour < 14 else "B"
        targets = [
            w for w in WORKERS
            if w["role"] in ("WORKER", "OPERATOR") and w["shift"] == current_shift
        ][:10]  # demo cap

    checkins = []
    for w in targets:
        wid = w["worker_id"]
        ws_id = workstation_id or w.get("workstation", "WS-CNC-A")
        ci = CheckIn(
            checkin_id=f"ci-{uuid.uuid4().hex[:8]}",
            worker_id=wid,
            workstation_id=ws_id,
            language=language,
            response=CheckInResponse.MISSED,
            voice_consent=voice_consent,
            responded_at=None,
            reminder_sent_at=datetime.now(timezone.utc).isoformat(),
        )
        state.checkins.append(ci)
        checkins.append({
            "checkin_id": ci.checkin_id,
            "worker_id": wid,
            "workstation_id": ws_id,
            "language": language,
            "voice_prompt": msgs["prompt"],
            "voice_url": f"/api/worker/checkin/voice/{ci.checkin_id}",
            "options": [msgs["ok"], msgs["too_hot"], msgs["sos"]],
            "consent": msgs["consent"],
            "voice_consent_required": voice_consent,
        })

    await HUB.publish({
        "type": "checkin.initiated",
        "payload": {"checkins": checkins, "count": len(checkins), "language": language},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    return {"status": "ok", "checkins": checkins, "count": len(checkins)}


@app.get("/api/worker/checkin/voice/{checkin_id}")
async def checkin_voice(checkin_id: str, role: str = Depends(current_role)):
    """Feature 4: Return Telugu/English voice prompt for a check-in.

    Local demo mode returns a prerecorded audio hint URL plus the prompt text.
    Live mode (gemini-3.8-live, us-central1) requires voice consent first.
    """
    state = get_state()
    target = None
    for c in state.checkins:
        if c.checkin_id == checkin_id:
            target = c
            break
    if target is None:
        raise HTTPException(status_code=404, detail=f"unknown checkin_id: {checkin_id}")
    msgs = _voice_messages(target.language)
    return {
        "checkin_id": checkin_id,
        "worker_id": target.worker_id,
        "language": target.language,
        "voice_prompt": msgs["prompt"],
        "voice_url": f"/static/voice/{target.language}/{checkin_id}.wav",
        "voice_consent": target.voice_consent,
        "consent_required": True,
        "consent_text": msgs["consent"],
        "options": [msgs["ok"], msgs["too_hot"], msgs["sos"]],
        "simulated": True,
        "disclaimer": (
            "Voice audio (if live) is processed in us-central1. "
            "Tap-only mode available without voice."
        ),
    }


@app.get("/api/worker/checkin/status")
async def checkin_status(worker_id: Optional[str] = None, role: str = Depends(current_role)):
    state = get_state()
    items = [c for c in state.checkins if not worker_id or c.worker_id == worker_id]
    return {"status": "ok", "checkins": [c.model_dump() for c in items], "count": len(items)}


# --------------------------------------------------------------------------- #
# Heat Stress Guardian: Three-Tier Worker Check-In + Machinery Restriction
# --------------------------------------------------------------------------- #

def _workers_on_machine(state, machine_id: str) -> list[str]:
    """Return worker IDs currently assigned to a machine in the active schedule."""
    vid = state.current_version_id
    version = state.schedules.get(vid) if vid else None
    if not version:
        return []
    workers = set()
    for e in version.entries:
        if e.machine_id == machine_id and e.worker_id:
            workers.add(e.worker_id)
    return list(workers)


def _is_machinery_disabled(state, worker_id: str) -> tuple[bool, Optional[datetime]]:
    """Check if a worker is currently restricted from machinery."""
    until = state.worker_machinery_disabled_until.get(worker_id)
    if until is None:
        return False, None
    if datetime.now(timezone.utc) > until:
        del state.worker_machinery_disabled_until[worker_id]
        return False, None
    return True, until


@app.post("/api/worker/heat-checkin")
async def worker_heat_checkin(payload: dict, role: str = Depends(current_role)):
    """Three-tier heat stress check-in for workers assigned to a machine/workstation.

    Tier 1 - OK: worker is fine, no restriction.
    Tier 2 - TOO_HOT (caution): worker must rest 15 min; machinery disabled for 15 min.
    Tier 3 - SOS: CRITICAL supervisor alert raised immediately.
    """
    state = get_state()
    from teevra_shared_schemas.heat import CheckIn, CheckInResponse, SupervisorAlert

    worker_id = payload.get("worker_id")
    workstation_id = payload.get("workstation_id", "WS-CNC-A")
    response = payload.get("response", "OK")
    language = payload.get("language", "en")
    voice_consent = bool(payload.get("voice_consent", False))

    if response not in {r.value for r in CheckInResponse}:
        raise HTTPException(status_code=400, detail=f"invalid response: {response}")

    # Record the check-in
    checkin = CheckIn(
        checkin_id=f"ci-{uuid.uuid4().hex[:8]}",
        worker_id=worker_id,
        workstation_id=workstation_id,
        language=language,
        response=CheckInResponse(response),
        voice_consent=voice_consent,
        responded_at=datetime.now(timezone.utc).isoformat(),
        reminder_sent_at=datetime.now(timezone.utc).isoformat(),
    )
    state.checkins.append(checkin)

    result = {
        "status": "ok",
        "checkin_id": checkin.checkin_id,
        "worker_id": worker_id,
        "workstation_id": workstation_id,
        "response": response,
        "tier": response,
        "machinery_disabled": False,
        "disabled_until": None,
        "alert_created": False,
    }

    if response == CheckInResponse.OK:
        state.worker_heat_status[worker_id] = HeatStatus.NORMAL
        result["guidance"] = "You are fine. Continue working."

    elif response == CheckInResponse.TOO_HOT:
        # Tier 2: Rest 15 min, disable machinery for 15 minutes
        disabled_until = datetime.now(timezone.utc) + timedelta(minutes=15)
        state.worker_machinery_disabled_until[worker_id] = disabled_until
        state.worker_heat_status[worker_id] = HeatStatus.HIGH_RISK
        result["machinery_disabled"] = True
        result["disabled_until"] = disabled_until.isoformat()
        result["guidance"] = "Too hot. Take rest for 15 minutes, then continue. You cannot use machinery for 15 minutes."

        # Create supervisor alert (MEDIUM severity)
        alert = SupervisorAlert(
            alert_id=f"alert-{uuid.uuid4().hex[:8]}",
            alert_type="WORKER_HEAT_CAUTION",
            severity="MEDIUM",
            workstation_id=workstation_id,
            worker_id=worker_id,
            message=f"Worker {worker_id} reported TOO_HOT at {workstation_id}. Machinery restricted for 15 minutes.",
            correlation_id=payload.get("correlation_id"),
        )
        state.alerts[alert.alert_id] = alert
        result["alert_created"] = True
        result["alert_id"] = alert.alert_id

        await HUB.publish({
            "type": "worker.heat_caution",
            "payload": result,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    elif response == CheckInResponse.SOS:
        # Tier 3: CRITICAL supervisor alert
        state.worker_heat_status[worker_id] = HeatStatus.STOP_WORK
        alert = SupervisorAlert(
            alert_id=f"alert-{uuid.uuid4().hex[:8]}",
            alert_type="WORKER_SOS",
            severity="CRITICAL",
            workstation_id=workstation_id,
            worker_id=worker_id,
            message=f"SOS from worker {worker_id} at {workstation_id}. Immediate assistance required.",
            correlation_id=payload.get("correlation_id"),
        )
        state.alerts[alert.alert_id] = alert
        result["machinery_disabled"] = True
        result["disabled_until"] = None
        result["guidance"] = "SOS. Stay where you are. Help is being dispatched."
        result["alert_created"] = True
        result["alert_id"] = alert.alert_id

        await HUB.publish({
            "type": "worker.sos",
            "payload": alert.model_dump(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    await HUB.publish({
        "type": "worker.heat_checkin",
        "payload": result,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    return result


@app.get("/api/worker/heat-status/{worker_id}")
async def worker_heat_status(worker_id: str, role: str = Depends(current_role)):
    """Return a worker's current heat status and machinery restriction state."""
    state = get_state()
    disabled, until = _is_machinery_disabled(state, worker_id)
    risk = state.worker_heat_status.get(worker_id, HeatStatus.NORMAL)
    return {
        "status": "ok",
        "worker_id": worker_id,
        "heat_status": risk.value,
        "machinery_disabled": disabled,
        "disabled_until": until.isoformat() if until else None,
    }


# --------------------------------------------------------------------------- #
# Alerts
# --------------------------------------------------------------------------- #
@app.get("/api/alerts")
async def alerts(role: str = Depends(current_role)):
    state = get_state()
    return {"status": "ok", "alerts": [a.model_dump() for a in state.alerts.values()], "count": len(state.alerts)}


@app.post("/api/alerts/{alert_id}/acknowledge")
async def ack_alert(alert_id: str, role: str = Depends(current_role)):
    state = get_state()
    a = state.alerts.get(alert_id)
    if not a:
        raise HTTPException(status_code=404, detail="alert not found")
    a.status = "ACKNOWLEDGED"
    a.acknowledged_at = datetime.now(timezone.utc).isoformat()
    a.acknowledged_by = role
    return {"status": "acknowledged", "alert_id": alert_id}


@app.post("/api/alerts/{alert_id}/resolve")
async def resolve_alert(alert_id: str, role: str = Depends(current_role)):
    state = get_state()
    a = state.alerts.get(alert_id)
    if not a:
        raise HTTPException(status_code=404, detail="alert not found")
    a.status = "RESOLVED"
    a.resolved_at = datetime.now(timezone.utc).isoformat()
    return {"status": "resolved", "alert_id": alert_id}


# --------------------------------------------------------------------------- #
# Agent observability traces
# --------------------------------------------------------------------------- #
@app.get("/api/agents/traces")
async def agent_traces(role: str = Depends(current_role)):
    state = get_state()
    return {"status": "ok", "traces": state.audit[-100:], "count": len(state.audit)}


# --------------------------------------------------------------------------- #
# SSE event stream
# --------------------------------------------------------------------------- #
@app.get("/api/events/stream")
async def events_stream(role: str = Depends(current_role)):
    q = await HUB.subscribe()

    async def event_generator():
        try:
            while True:
                data = await q.get()
                yield f"data: {data}\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            HUB.unsubscribe(q)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# --------------------------------------------------------------------------- #
# WebSocket dashboard updates
# --------------------------------------------------------------------------- #
@app.websocket("/api/events/ws")
async def events_ws(websocket: WebSocket):
    await websocket.accept()
    q = await HUB.subscribe()
    try:
        while True:
            data = await q.get()
            await websocket.send_text(data)
    except WebSocketDisconnect:
        pass
    finally:
        HUB.unsubscribe(q)


# --------------------------------------------------------------------------- #
# A2A agent calls (orchestrator -> recovery / heat guardian)
# --------------------------------------------------------------------------- #
@app.post("/api/a2a/recovery")
async def a2a_recovery(payload: dict, role: str = Depends(require_planner)):
    async with A2ARemoteClient(RECOVERY_URL, timeout_seconds=60.0) as client:
        try:
            card = await client.get_agent_card()
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"recovery agent unreachable: {exc}")
        result = await client.send_task(json.dumps(payload), correlation_id=str(uuid.uuid4()))
    return {"status": "ok", "agent_card": card.get("name"), "result": result}


@app.post("/api/a2a/heat")
async def a2a_heat(payload: dict, role: str = Depends(current_role)):
    async with A2ARemoteClient(HEAT_URL, timeout_seconds=30.0) as client:
        try:
            await client.get_agent_card()
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"heat guardian unreachable: {exc}")
        result = await client.send_task(json.dumps(payload), correlation_id=str(uuid.uuid4()))
    return {"status": "ok", "result": result}


# seed baseline on startup if empty
@app.on_event("startup")
async def _startup():
    state = get_state()
    if not state.schedules:
        base = ENGINE.build_baseline_schedule({"machines": MACHINES, "jobs": JOBS, "now_slot": 0})
        state.schedules[base.version_id] = base
        state.current_version_id = base.version_id


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
