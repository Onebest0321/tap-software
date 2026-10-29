"""
dataset_tools.py — inventaire et contrôle qualité du dataset brut
=================================================================

Travaille sur l'arborescence écrite par app.py :
    data/PXX/<clé>_finger_TAPn.csv | <clé>_imu_TAPn.csv | <clé>_markers.csv
    data/PXX/videos/…mp4
    data/anomalie/PXX/…           (essais refusés / déplacés)

avec  <clé> = Pxx_Sxx_Gesture_xx_repxx_YYYYMMDD_HHMMSS

Fonctions :
  scan_overview   matrice participants × gestes (complet / manquant / doublon / cassé)
  cell_detail     détail des enregistrements d'une case (+ marqueurs, taille, nettoyé ?)
  move_repetitions  déplace des répétitions (CSV + marqueurs + vidéo + _Clean) vers anomalie
  list_anomalies / restore_repetition
  rep_data        séries d'une répétition (brute ou nettoyée) pour les graphiques
"""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import threading
import time
from datetime import datetime
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError, ParserError

FILE_RE = re.compile(
    r"^(?P<key>(?P<participant>P\d+)_S(?P<session>\d+)_(?P<gesture>Gesture_\d+)_rep(?P<rep>\d+)"
    r"_(?P<stamp>\d{8}_\d{6}))_(?P<kind>finger_[A-Za-z0-9]+|imu_[A-Za-z0-9]+|markers)\.csv$"
)
PARTICIPANT_RE = re.compile(r"^P\d+$")
KEY_RE = re.compile(r"^P\d+_S\d+_Gesture_\d+_rep\d+_\d{8}_\d{6}$")

# Flux écrits par app.py à chaque répétition : un fichier par flux du registre d'appareils (finger_<slot>,
# imu_<slot>) + le fichier de marqueurs. Les slots (TAP1, TAP2, TAP3…) et leur type peuvent changer d'une
# configuration à l'autre : il n'y a donc plus de liste figée. Les flux ATTENDUS d'une répétition sont ceux
# que présente la majorité des répétitions du même participant (_expected_streams) ; une répétition à qui
# il en manque un est « cassée ».
FINGER_HEADER = ["Timestamp_LSL", "Timestamp_HW", "Type"] + [
    f"{f}_{a}" for f in ("Thumb", "Index", "Middle", "Ring", "Pinky") for a in "XYZ"]
IMU_HEADER = ["Timestamp_LSL", "Timestamp_HW", "Type", "Accel_X", "Accel_Y", "Accel_Z",
              "Gyro_X", "Gyro_Y", "Gyro_Z"]

# Seuils de l'audit d'intégrité (documentés dans DATASET.md ; modifiables ici).
AUDIT = {
    "min_duration_s": 0.5,      # geste plus court → suspect
    "max_duration_s": 60.0,     # geste plus long → suspect
    "coverage_warn": 0.90,      # échantillons reçus / attendus (débit nominal × durée)
    "coverage_error": 0.50,
    "counter_tol_abs": 5,       # tolérance sur les compteurs f= / imu= du marqueur STOP
    "counter_tol_rel": 0.02,
}
SEVERITY_RANK = {"ok": 0, "warning": 1, "error": 2}
FILE_CLASSES = ("markers", "finger", "imu", "cross")
CLASS_LABEL = {"markers": "Marqueurs", "finger": "Doigts", "imu": "IMU", "cross": "Cohérence marqueurs/capteurs"}
ANOMALY_DIR = "anomalie"
EXCLUDED_CLEAN_DIR = "_excluded"
VERSION_FILE = "version.json"        # marque un dossier de version nettoyée : <data_clean/jeu>/<nom>/version.json
DEFAULT_OWNER = "principal"          # dataset propriétaire d'une version dont version.json n'indique pas de "dataset"


def version_root(base: str, slug: str) -> str:
    """Racine d'une version nommée : un dossier de premier niveau du jeu, <data_clean/jeu>/<slug>/ (à côté du dataset `base`)."""
    return os.path.join(os.path.dirname(os.path.normpath(base)), slug) if slug else base


def owner_of(path: str) -> Optional[str]:
    """Si `path` est une version nommée (contient version.json), renvoie la racine du dataset nettoyé auquel elle appartient."""
    p = os.path.normpath(path)
    vf = os.path.join(p, VERSION_FILE)
    if not os.path.isfile(vf):
        return None
    try:
        with open(vf, encoding="utf-8") as f:
            ds = (json.load(f) or {}).get("dataset")
    except Exception:
        ds = None
    return os.path.join(os.path.dirname(p), ds or DEFAULT_OWNER)


def list_versions(base: str) -> List[tuple]:
    """[(slug, racine)] des versions nommées du dataset nettoyé `base`."""
    b = os.path.normpath(base)
    sd = os.path.dirname(b)
    out = []
    for n in (sorted(os.listdir(sd)) if os.path.isdir(sd) else []):
        r = os.path.join(sd, n)
        if os.path.normcase(r) != os.path.normcase(b) and os.path.isdir(r):
            o = owner_of(r)
            if o and os.path.normcase(o) == os.path.normcase(b):
                out.append((n, r))
    return out


def _base_clean_root(clean_root: str) -> str:
    """Racine du dataset nettoyé, que `clean_root` soit cette racine ou l'une de ses versions nommées.
    L'audit parle des données brutes : ses fichiers restent à la racine, communs à toutes les versions."""
    p = os.path.normpath(clean_root)
    return owner_of(p) or p


def _all_clean_roots(clean_root: str) -> List[str]:
    base = _base_clean_root(clean_root)
    return [base] + [r for _n, r in list_versions(base)]


# ─────────────────────────────────────────────────────────────────────────────
# SCAN
# ─────────────────────────────────────────────────────────────────────────────
def _slot_sort_key(slot: str):
    """TAP2 avant TAP10."""
    m = re.fullmatch(r"([A-Za-z]*)(\d+)", slot)
    return (m.group(1), int(m.group(2))) if m else (slot, 0)


def _stream_sort_key(stream: str):
    kind, _, slot = stream.partition("_")
    return (_slot_sort_key(slot), kind != "finger")


def _expected_streams(recs) -> tuple:
    """Flux capteurs attendus pour un ensemble de répétitions = l'ensemble de flux le plus courant
    (à égalité, le plus grand : mieux vaut signaler un fichier manquant que l'ignorer)."""
    from collections import Counter
    c = Counter(tuple(sorted((k for k in r["files"] if k != "markers"), key=_stream_sort_key)) for r in recs)
    c.pop((), None)
    if not c:
        return ()
    return max(c.items(), key=lambda kv: (kv[1], len(kv[0])))[0]


def _scan_folder(folder: str, expected: Optional[tuple] = None) -> Dict[str, dict]:
    """key → {meta, files:{kind: path}} pour un dossier à plat.
    `expected` : flux capteurs attendus ; par défaut déduits du dossier (_expected_streams)."""
    recs: Dict[str, dict] = {}
    if not os.path.isdir(folder):
        return recs
    with os.scandir(folder) as it:
        for e in it:
            if not e.is_file():
                continue
            m = FILE_RE.match(e.name)
            if not m:
                continue
            r = recs.setdefault(m["key"], {
                "key": m["key"], "participant": m["participant"], "session": int(m["session"]),
                "gesture": m["gesture"], "rep": int(m["rep"]), "stamp": m["stamp"], "files": {},
            })
            r["files"][m["kind"]] = e.path
    if expected is None:
        expected = _expected_streams(recs.values())
    for r in recs.values():
        r["expected"] = expected
        missing = [] if "markers" in r["files"] else ["markers"]
        missing += [k for k in expected if k not in r["files"]]
        if not expected and not any(k != "markers" for k in r["files"]):
            missing.append("capteur")                    # ni doigts ni IMU : inutilisable
        r["missing_files"] = missing
        r["complete"] = not missing
    return recs


def _participant_expected(data_root: str, participant: str) -> tuple:
    """Flux attendus d'un participant (dossier principal) : sert aussi pour juger ses essais déplacés."""
    recs = _scan_folder(os.path.join(data_root, participant))
    return next(iter(recs.values()))["expected"] if recs else ()

def _participants(root: str) -> List[str]:
    if not os.path.isdir(root):
        return []
    return sorted(d for d in os.listdir(root)
                  if PARTICIPANT_RE.match(d) and os.path.isdir(os.path.join(root, d)))


def _clean_file(clean_root: str, r: dict) -> str:
    return os.path.join(clean_root, r["gesture"], r["participant"], f"{r['key']}_Clean.csv")


def _cell_state(recs: List[dict], expected_reps: int) -> dict:
    """Qualifie une case (participant × geste)."""
    complete = [r for r in recs if r["complete"]]
    broken = [r for r in recs if not r["complete"]]
    by_rep: Dict[int, int] = {}
    for r in complete:
        by_rep[r["rep"]] = by_rep.get(r["rep"], 0) + 1
    expected = set(range(1, expected_reps + 1))
    present = set(by_rep)
    missing = sorted(expected - present)
    duplicates = sorted(k for k, n in by_rep.items() if n > 1)
    extra = sorted(present - expected)

    if not recs:
        state = "empty"
    elif broken:
        state = "broken"
    elif duplicates or extra:
        state = "dup"
    elif missing:
        state = "partial"
    else:
        state = "ok"
    return {"state": state, "n": len(recs), "missing": missing, "duplicates": duplicates,
            "extra": extra, "broken": len(broken)}


def scan_overview(data_root: str, clean_root: str, expected_gestures: List[str],
                  expected_reps: int = 5) -> dict:
    """Matrice de complétude + indicateurs globaux."""
    gest_tokens = [g.replace(" ", "_") for g in expected_gestures]
    participants = _participants(data_root)

    rows = []
    scanned: Dict[str, Dict[str, dict]] = {}
    totals = {"ok": 0, "partial": 0, "dup": 0, "broken": 0, "empty": 0}
    n_recordings = n_dup_recs = n_missing_reps = 0
    unknown_gestures = set()

    for p in participants:
        recs = _scan_folder(os.path.join(data_root, p))
        scanned[p] = recs
        by_gesture: Dict[str, List[dict]] = {}
        for r in recs.values():
            by_gesture.setdefault(r["gesture"], []).append(r)
        unknown_gestures |= set(by_gesture) - set(gest_tokens)

        cells = []
        p_ok = 0
        for g in gest_tokens:
            c = _cell_state(by_gesture.get(g, []), expected_reps)
            cells.append({"s": c["state"], "n": c["n"], "m": c["missing"], "d": c["duplicates"],
                          "x": c["extra"], "b": c["broken"]})
            totals[c["state"]] += 1
            n_recordings += c["n"]
            n_missing_reps += len(c["missing"]) if c["state"] != "empty" else expected_reps
            n_dup_recs += sum(1 for _ in c["duplicates"]) + len(c["extra"])
            p_ok += 1 if c["state"] == "ok" else 0

        n_clean = sum(1 for r in recs.values() if r["complete"] and os.path.isfile(_clean_file(clean_root, r)))
        anomalies = len(_scan_folder(os.path.join(data_root, ANOMALY_DIR, p)))
        rows.append({"participant": p, "cells": cells, "recordings": len(recs),
                     "gestures_ok": p_ok, "clean": n_clean, "anomalies": anomalies})

    amap = _issue_map({p: set(r) for p, r in scanned.items()}, clean_root)
    for row in rows:
        for c, g in zip(row["cells"], gest_tokens):
            a_ = amap.get((row["participant"], g))
            if a_:
                c["ae"], c["aw"] = a_["e"], a_["w"]

    expected_total = len(participants) * len(gest_tokens) * expected_reps
    return {
        "gestures": gest_tokens,
        "expected_reps": expected_reps,
        "participants": rows,
        "totals": {
            "participants": len(participants),
            "recordings": n_recordings,
            "expected_recordings": expected_total,
            "cells": totals,
            "missing_reps": n_missing_reps,
            "duplicate_or_extra": n_dup_recs,
            "unknown_gestures": sorted(unknown_gestures),
            "clean": sum(r["clean"] for r in rows),
            "anomalies": sum(r["anomalies"] for r in rows),
        },
        "audit": load_audit_summary(clean_root),
    }


# ─────────────────────────────────────────────────────────────────────────────
# DÉTAIL D'UNE CASE
# ─────────────────────────────────────────────────────────────────────────────
def _count_lines(path: str) -> int:
    try:
        with open(path, "rb") as f:
            return max(sum(1 for _ in f) - 1, 0)
    except OSError:
        return 0


def _read_markers(path: Optional[str]) -> List[dict]:
    if not path or not os.path.isfile(path):
        return []
    out = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                out.append({"t": float(row["Timestamp_LSL"]), "m": row["Marker"]})
            except (KeyError, ValueError):
                continue
    return out


def _window(markers: List[dict]):
    t0 = t1 = None
    for m in markers:
        if m["m"].startswith("START") and not m["m"].startswith("COUNTDOWN"):
            t0 = m["t"]
        elif m["m"].startswith("STOP"):
            t1 = m["t"]
    return t0, t1


def cell_detail(data_root: str, clean_root: str, participant: str, gesture: str,
                expected_reps: int = 5) -> dict:
    if not PARTICIPANT_RE.match(participant):
        raise ValueError("participant invalide")
    recs = sorted((r for r in _scan_folder(os.path.join(data_root, participant)).values()
                   if r["gesture"] == gesture), key=lambda r: (r["rep"], r["stamp"]))
    seen: Dict[int, int] = {}
    out = []
    for r in recs:
        seen[r["rep"]] = seen.get(r["rep"], 0) + 1
    for r in recs:
        files = r["files"]
        markers = _read_markers(files.get("markers"))
        t0, t1 = _window(markers)
        f_keys = sorted((k for k in files if k.startswith("finger")), key=_stream_sort_key)
        i_keys = sorted((k for k in files if k.startswith("imu")), key=_stream_sort_key)
        try:
            stamp_txt = datetime.strptime(r["stamp"], "%Y%m%d_%H%M%S").strftime("%d/%m %H:%M:%S")
        except ValueError:
            stamp_txt = r["stamp"]
        audit = audit_recording(files, r["expected"])
        out.append({
            "audit": audit,
            "missing_files": r["missing_files"],
            "key": r["key"], "rep": r["rep"], "session": r["session"],
            "time": stamp_txt, "stamp": r["stamp"],
            "complete": r["complete"], "duplicate": seen[r["rep"]] > 1 or r["rep"] > expected_reps,
            "finger_rows": sum(_count_lines(files[k]) for k in f_keys),
            "imu_rows": {k.split("_")[1]: _count_lines(files[k]) for k in i_keys},
            "duration_s": round(t1 - t0, 2) if (t0 is not None and t1 is not None) else None,
            "has_stop": t1 is not None,
            "clean": os.path.isfile(_clean_file(clean_root, r)),
            "files": sorted(k for k in files),
            "kb": round(sum(os.path.getsize(p) for p in files.values()) / 1024, 1),
        })
    c = _cell_state(recs, expected_reps)
    return {"participant": participant, "gesture": gesture, "state": c["state"],
            "missing": c["missing"], "recordings": out}


# ─────────────────────────────────────────────────────────────────────────────
# DÉPLACEMENT / RESTAURATION
# ─────────────────────────────────────────────────────────────────────────────
def _video_for(data_root: str, participant: str, markers_path: Optional[str]) -> List[str]:
    """Vidéos associées : le nom contient t2flaskStart<LSL START avec '-' pour '.'>."""
    markers = _read_markers(markers_path)
    t0, _ = _window(markers)
    if t0 is None:
        return []
    token = f"_t2flaskStart{t0:.6f}".replace(".", "-")
    found = []
    for folder in (os.path.join(data_root, participant, "videos"),):
        if os.path.isdir(folder):
            found += [os.path.join(folder, n) for n in os.listdir(folder) if token in n]
    return found


MOVE_LOG_HEADER = ["time", "action", "participant", "key", "reason", "problems"]


def _log_move(data_root: str, action: str, participant: str, key: str, reason: str, problems: str = ""):
    folder = os.path.join(data_root, ANOMALY_DIR)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "moves_log.csv")
    if os.path.isfile(path):                     # migration d'un ancien journal sans la colonne « problems »
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        if rows and rows[0] != MOVE_LOG_HEADER:
            rows = [MOVE_LOG_HEADER] + [r + [""] * (len(MOVE_LOG_HEADER) - len(r)) for r in rows[1:]]
            with open(path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerows(rows)
    new = not os.path.isfile(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(MOVE_LOG_HEADER)
        w.writerow([datetime.now().isoformat(timespec="seconds"), action, participant, key, reason, problems])


def _read_move_log(data_root: str) -> Dict[str, dict]:
    """key → dernière entrée « move » (raison + signature des problèmes)."""
    path = os.path.join(data_root, ANOMALY_DIR, "moves_log.csv")
    out: Dict[str, dict] = {}
    if not os.path.isfile(path):
        return out
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("action") == "move":
                out[row["key"]] = {"time": row.get("time", ""), "reason": row.get("reason", ""),
                                   "problems": row.get("problems", "")}
            elif row.get("action") == "restore":
                out.pop(row["key"], None)
    return out


def move_repetitions(data_root: str, clean_root: str, participant: str, keys: List[str],
                     reason: str = "") -> dict:
    if not PARTICIPANT_RE.match(participant):
        raise ValueError("participant invalide")
    src = os.path.join(data_root, participant)
    dst = os.path.join(data_root, ANOMALY_DIR, participant)
    dst_vid = dst
    recs = _scan_folder(src)
    moved, skipped, details = [], [], []
    for key in keys:
        if not KEY_RE.match(key) or key not in recs or not key.startswith(participant + "_"):
            skipped.append(key)
            continue
        r = recs[key]
        tags = problem_tags(audit_recording(r["files"], r["expected"]))   # diagnostic AVANT de déplacer
        os.makedirs(dst, exist_ok=True)
        for p in list(r["files"].values()):
            shutil.move(p, os.path.join(dst, os.path.basename(p)))
        for v in _video_for(data_root, participant, os.path.join(dst, f"{key}_markers.csv")):
            shutil.move(v, os.path.join(dst_vid, os.path.basename(v)))
        for croot in _all_clean_roots(clean_root):          # la répétition sort de TOUTES les versions propres
            cf = _clean_file(croot, r)
            if os.path.isfile(cf):
                ex = os.path.join(croot, EXCLUDED_CLEAN_DIR, r["gesture"], participant)
                os.makedirs(ex, exist_ok=True)
                shutil.move(cf, os.path.join(ex, os.path.basename(cf)))
        _log_move(data_root, "move", participant, key, reason, tags)
        moved.append(key)
        details.append({"key": key, "problems": tags})
    if moved:
        _audit_forget(clean_root, moved)   # garde l'audit à jour : retire les répétitions déplacées
    return {"moved": moved, "skipped": skipped, "details": details}


def list_anomalies(data_root: str, participant: Optional[str] = None) -> List[dict]:
    parts = [participant] if participant else _participants(os.path.join(data_root, ANOMALY_DIR))
    log = _read_move_log(data_root)
    out = []
    for p in parts:
        for r in _scan_folder(os.path.join(data_root, ANOMALY_DIR, p), _participant_expected(data_root, p)).values():
            meta = log.get(r["key"], {})
            out.append({"participant": p, "key": r["key"], "gesture": r["gesture"], "rep": r["rep"],
                        "stamp": r["stamp"], "complete": r["complete"], "missing_files": r["missing_files"],
                        "origin": "manuel" if meta else "retry/auto", "reason": meta.get("reason", ""),
                        "problems": meta.get("problems", "")})
    return sorted(out, key=lambda x: (x["participant"], x["gesture"], x["rep"], x["stamp"]))


def restore_repetition(data_root: str, participant: str, key: str) -> dict:
    if not PARTICIPANT_RE.match(participant) or not KEY_RE.match(key):
        raise ValueError("paramètres invalides")
    src = os.path.join(data_root, ANOMALY_DIR, participant)
    recs = _scan_folder(src)
    if key not in recs:
        raise FileNotFoundError(key)
    dst = os.path.join(data_root, participant)
    os.makedirs(dst, exist_ok=True)
    for p in recs[key]["files"].values():
        shutil.move(p, os.path.join(dst, os.path.basename(p)))
    _log_move(data_root, "restore", participant, key, "")
    return {"restored": key}


# ─────────────────────────────────────────────────────────────────────────────
# DONNÉES D'UNE RÉPÉTITION POUR LES GRAPHIQUES
# ─────────────────────────────────────────────────────────────────────────────
SLOT_COL_RE = re.compile(r"^([A-Za-z0-9]+)_(?:Thumb|Index|Middle|Ring|Pinky|Accel|Gyro)_")   # <slot>_<capteur>_<axe>
FINGER_COLS = [f"{f}_{a}" for f in ("Thumb", "Index", "Middle", "Ring", "Pinky") for a in "XYZ"]
IMU_COLS = ["Accel_X", "Accel_Y", "Accel_Z", "Gyro_X", "Gyro_Y", "Gyro_Z"]


def _decimate(n: int, max_points: int) -> slice:
    step = max(1, int(np.ceil(n / max_points)))
    return slice(0, n, step)


def _sync_stats(df: pd.DataFrame) -> Optional[dict]:
    """Même calcul que l'étape 1 du pipeline, pour l'affichage (qualité de la synchro)."""
    if "Timestamp_HW" not in df.columns or "Timestamp_LSL" not in df.columns or len(df) < 3:
        return None
    hw, lsl = df["Timestamp_HW"].values, df["Timestamp_LSL"].values
    a, b = np.polyfit(hw, lsl, 1)
    corr = a * hw + b
    span = lsl[-1] - lsl[0]   # cadence moyenne (la médiane des Δt est faussée par les paquets)
    native = (len(lsl) - 1) / span if span > 0 else None
    return {"slope_a": float(a), "residual_std_ms": float(np.std(lsl - corr) * 1000),
            "jitter_before_ms": float(np.std(np.diff(lsl)) * 1000),
            "jitter_after_ms": float(np.std(np.diff(corr)) * 1000),
            "native_hz": float(native) if native else None}


def rep_data(data_root: str, clean_root: str, participant: str, key: str,
             source: str = "raw", max_points: int = 2500) -> dict:
    if not PARTICIPANT_RE.match(participant) or not KEY_RE.match(key) or not key.startswith(participant + "_"):
        raise ValueError("paramètres invalides")
    recs = _scan_folder(os.path.join(data_root, participant))
    rec = recs.get(key)
    if rec is None:
        recs = _scan_folder(os.path.join(data_root, ANOMALY_DIR, participant))
        rec = recs.get(key)
    if rec is None:
        raise FileNotFoundError(key)

    markers = _read_markers(rec["files"].get("markers"))
    t0, t1 = _window(markers)
    out = {"key": key, "source": source, "markers": [], "sensors": {}, "sync": {}, "window": None}

    if source == "clean":
        path = _clean_file(clean_root, rec)
        if not os.path.isfile(path):
            raise FileNotFoundError("Pas de version nettoyée pour cette répétition")
        df = pd.read_csv(path)
        sl = _decimate(len(df), max_points)
        t = df["Timestamp_LSL"].values[sl]
        devices = sorted({m.group(1) for m in (SLOT_COL_RE.match(c) for c in df.columns) if m}, key=_slot_sort_key)
        for dev in devices:
            f_cols = [c for c in FINGER_COLS if f"{dev}_{c}" in df.columns]
            i_cols = [c for c in IMU_COLS if f"{dev}_{c}" in df.columns]
            if f_cols:
                out["sensors"][f"finger_{dev}"] = {"t": t.round(5).tolist(),
                    "cols": {c: df[f"{dev}_{c}"].values[sl].round(4).tolist() for c in f_cols}}
            if i_cols:
                out["sensors"][f"imu_{dev}"] = {"t": t.round(5).tolist(),
                    "cols": {c: df[f"{dev}_{c}"].values[sl].round(4).tolist() for c in i_cols}}
        out["window"] = [0.0, float(df["Timestamp_LSL"].iloc[-1])] if len(df) else None
        out["rows"] = int(len(df))
        if t0 is not None:
            out["markers"] = [{"t": m["t"] - t0, "m": m["m"]} for m in markers
                              if m["m"].split("|")[0] in ("START", "STOP")]
        return out

    for kind, path in sorted(rec["files"].items()):
        if kind == "markers":
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        origin = t0 if t0 is not None else float(df["Timestamp_LSL"].iloc[0])
        sl = _decimate(len(df), max_points)
        cols = FINGER_COLS if kind.startswith("finger") else IMU_COLS
        out["sensors"][kind] = {
            "t": (df["Timestamp_LSL"].values[sl] - origin).round(5).tolist(),
            "cols": {c: df[c].values[sl].round(4).tolist() for c in cols if c in df.columns},
        }
        out["sync"][kind] = _sync_stats(df)
        out.setdefault("raw_rows", {})[kind] = int(len(df))
    if t0 is not None and t1 is not None:
        out["window"] = [0.0, t1 - t0]
    out["markers"] = [{"t": m["t"] - (t0 or 0), "m": m["m"]} for m in markers
                      if m["m"].split("|")[0] in ("COUNTDOWN_START", "START", "STOP", "RETRY")]
    out["has_clean"] = os.path.isfile(_clean_file(clean_root, rec))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# STATISTIQUES (lues dans le rapport de prétraitement)
# ─────────────────────────────────────────────────────────────────────────────
STATS_FIELDS = ["key", "participant", "gesture", "repetition", "sensor", "rows_raw", "rows_kept",
                "native_hz", "slope_a", "residual_std_ms", "jitter_before_ms", "jitter_after_ms",
                "arrival_delay_mean_ms", "arrival_delay_p95_ms", "arrival_delay_max_ms", "dup_removed", "spread_rows", "nan_inner_pts", "duration_s", "nan_filled", "n_warnings"]


def stats_rows(clean_root: str) -> dict:
    """Lignes du rapport (répétition × flux) au format compact {cols, rows}."""
    path = os.path.join(clean_root, "preprocessing_report.csv")
    if not os.path.isfile(path):
        return {"available": False}
    df = pd.read_csv(path)
    if "key" not in df.columns:
        df["key"] = ""
    df["key"] = df["key"].fillna("")
    errors = int((df["status"] != "ok").sum()) if "status" in df.columns else 0
    ok = df[df["status"] == "ok"] if "status" in df.columns else df
    ok = ok[ok["sensor"].fillna("") != ""]
    cols = [c for c in STATS_FIELDS if c in ok.columns]
    sub = ok[cols].astype(object).where(ok[cols].notna(), None)
    return {"available": True, "cols": cols, "rows": sub.values.tolist(), "errors": errors,
            "generated": os.path.getmtime(path)}


# ─────────────────────────────────────────────────────────────────────────────
# AUDIT D'INTÉGRITÉ  — quel TYPE de fichier pose problème ?
# ─────────────────────────────────────────────────────────────────────────────
# Classes de fichiers :  markers · finger · imu · cross (cohérence marqueurs ↔ capteurs)
# Gravité :              error   → répétition inutilisable telle quelle
#                        warning → suspect, à regarder
#
#  markers   missing_file · empty_file · unreadable · bad_header · bad_timestamp
#            no_start · multi_start · no_stop · multi_stop · order · duration
#            no_countdown · retry_marker · no_counters
#  finger/imu  missing_file · empty · unreadable · bad_header · nan · time_backwards
#            no_data_in_window · low_coverage · gap
#  cross     count_mismatch (compteurs f=/imu= du STOP ≠ lignes CSV dans [START, STOP])
#
# Les trous et la couverture sont mesurés sur l'horloge MATÉRIELLE (régulière) ;
# les timestamps LSL arrivent par paquets et ne conviennent pas à ces tests.
STOP_COUNTERS_RE = re.compile(r"\|f=(\d+)\|imu=(\d+)")


def _cls(stream: str) -> str:
    return "markers" if stream == "markers" else stream.split("_")[0]


def _issue(stream: str, code: str, severity: str, msg: str) -> dict:
    return {"stream": stream, "cls": _cls(stream) if stream != "cross" else "cross",
            "code": code, "severity": severity, "msg": msg}


def _read_csv_safe(path: str, stream: str, issues: List[dict]) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        issues.append(_issue(stream, "empty_file", "error", "Fichier vide (0 octet ou sans en-tête)"))
    except (ParserError, UnicodeDecodeError, ValueError) as e:
        issues.append(_issue(stream, "unreadable", "error", f"Fichier illisible : {str(e)[:80]}"))
    except OSError as e:
        issues.append(_issue(stream, "unreadable", "error", f"Lecture impossible : {e}"))
    return None


def _audit_markers(path: Optional[str]):
    """→ (info, issues, parsed) ; parsed = {t_start, t_stop, f, imu} (valeurs None si absentes)."""
    S, iss = "markers", []
    info = {"present": bool(path and os.path.isfile(path)), "rows": 0, "status": "ok"}
    if not info["present"]:
        return info, [_issue(S, "missing_file", "error", "Fichier de marqueurs absent")], None
    df = _read_csv_safe(path, S, iss)
    if df is None:
        return info, iss, None
    info["rows"] = int(len(df))
    if not {"Timestamp_LSL", "Marker"} <= set(df.columns):
        return info, iss + [_issue(S, "bad_header", "error", f"En-tête inattendu : {list(df.columns)[:4]}")], None
    if df.empty:
        return info, iss + [_issue(S, "empty", "error", "Aucun marqueur écrit")], None
    ts = pd.to_numeric(df["Timestamp_LSL"], errors="coerce")
    if ts.isna().any():
        return info, iss + [_issue(S, "bad_timestamp", "error", f"{int(ts.isna().sum())} timestamp(s) non numérique(s)")], None

    names = df["Marker"].astype(str).str.split("|").str[0].values
    t = ts.values
    starts, stops = t[names == "START"], t[names == "STOP"]
    parsed = {"t_start": None, "t_stop": None, "f": None, "imu": None}
    if len(starts) == 0:
        iss.append(_issue(S, "no_start", "error", "Pas de marqueur START (essai non démarré ou fichier tronqué)"))
    elif len(starts) > 1:
        iss.append(_issue(S, "multi_start", "error", f"{len(starts)} marqueurs START (essai relancé dans le même fichier ?)"))
    else:
        parsed["t_start"] = float(starts[0])
    if len(stops) == 0:
        iss.append(_issue(S, "no_stop", "error", "Pas de marqueur STOP (arrêt non enregistré : crash, déconnexion ?)"))
    elif len(stops) > 1:
        iss.append(_issue(S, "multi_stop", "error", f"{len(stops)} marqueurs STOP"))
    else:
        parsed["t_stop"] = float(stops[0])
        m = STOP_COUNTERS_RE.search(str(df["Marker"].values[names == "STOP"][0]))
        if m:
            parsed["f"], parsed["imu"] = int(m.group(1)), int(m.group(2))
        else:
            iss.append(_issue(S, "no_counters", "warning", "STOP sans compteurs f=/imu= (vérification croisée impossible)"))

    t0, t1 = parsed["t_start"], parsed["t_stop"]
    if t0 is not None and t1 is not None:
        if t1 <= t0:
            iss.append(_issue(S, "order", "error", f"STOP ({t1:.3f}) avant ou égal à START ({t0:.3f})"))
            parsed["t_start"] = parsed["t_stop"] = None
        else:
            d = t1 - t0
            if d < AUDIT["min_duration_s"] or d > AUDIT["max_duration_s"]:
                iss.append(_issue(S, "duration", "warning", f"Durée suspecte : {d:.2f} s"))
    if "COUNTDOWN_START" not in names or "COUNTDOWN_END" not in names:
        iss.append(_issue(S, "no_countdown", "warning", "Marqueurs de compte à rebours absents"))
    if "RETRY" in names:
        iss.append(_issue(S, "retry_marker", "warning", "Marqueur RETRY dans un fichier conservé"))
    if (np.diff(t) < 0).any():
        iss.append(_issue(S, "time_backwards", "warning", "Marqueurs non ordonnés dans le temps"))
    return info, iss, parsed


def _audit_sensor(stream: str, path: Optional[str], window, required: bool = True):
    """→ (info, issues). `window` = (t_start, t_stop) en temps LSL, ou None.
    `required` : flux attendu pour cette répétition (absent / vide = erreur)."""
    iss: List[dict] = []
    info = {"present": bool(path and os.path.isfile(path)), "rows": 0, "rows_window": None,
            "rate_hz": None, "max_gap_ms": None, "coverage": None, "status": "ok", "optional": not required}
    if not info["present"]:
        if required:
            iss.append(_issue(stream, "missing_file", "error", "Fichier absent"))
        return info, iss
    df = _read_csv_safe(path, stream, iss)
    if df is None:
        return info, iss
    expected = FINGER_HEADER if stream.startswith("finger") else IMU_HEADER
    absent = [c for c in expected if c not in df.columns]
    if absent:
        iss.append(_issue(stream, "bad_header", "error", f"Colonnes manquantes : {', '.join(absent[:4])}"))
        return info, iss
    info["rows"] = n = int(len(df))
    if n == 0:
        if required:
            iss.append(_issue(stream, "empty", "error", "Aucune ligne de données (en-tête seul)"))
        return info, iss

    num = df.drop(columns=["Type"]).apply(pd.to_numeric, errors="coerce")
    bad = int(num.isna().sum().sum())
    if bad:
        iss.append(_issue(stream, "nan", "warning", f"{bad} valeur(s) manquante(s) ou non numérique(s)"))
    lsl, hw = num["Timestamp_LSL"].values, num["Timestamp_HW"].values
    if (np.diff(lsl) < 0).any():
        iss.append(_issue(stream, "time_backwards", "warning", f"{int((np.diff(lsl) < 0).sum())} retour(s) en arrière du timestamp LSL"))
    # (l'écart entre horloge matérielle et horloge LSL n'est plus signalé : la régression d'horloge du
    #  prétraitement le corrige, ce n'est pas une anomalie de la répétition)

    dh = np.diff(hw[~np.isnan(hw)])
    med = float(np.median(dh[dh > 0])) if (dh > 0).any() else 0.0
    nominal = 1.0 / med if med > 0 else None
    info["rate_hz"] = round(nominal, 1) if nominal else None

    if window and window[0] is not None and window[1] is not None:
        t0, t1 = window
        mask = (lsl >= t0) & (lsl <= t1)
        k = int(mask.sum())
        info["rows_window"] = k
        if k == 0:
            iss.append(_issue(stream, "no_data_in_window", "error", "Aucun échantillon entre START et STOP"))
        elif nominal:
            cov = k / ((t1 - t0) * nominal)
            info["coverage"] = round(cov, 3)
            if cov < AUDIT["coverage_error"]:
                iss.append(_issue(stream, "low_coverage", "error", f"Seulement {cov:.0%} des échantillons attendus ({k})"))
            elif cov < AUDIT["coverage_warn"]:
                iss.append(_issue(stream, "low_coverage", "warning", f"{cov:.0%} des échantillons attendus ({k})"))
            hwi = hw[mask]
            if len(hwi) > 1:
                gap = float(np.nanmax(np.diff(hwi)))
                info["max_gap_ms"] = round(gap * 1000, 1)
                # le plus grand trou reste mesuré (info["max_gap_ms"]) mais n'est plus signalé : ni erreur, ni avertissement
    return info, iss


def _worst(issues: List[dict]) -> str:
    return max((i["severity"] for i in issues), key=lambda s: SEVERITY_RANK[s], default="ok")


def audit_recording(files: Dict[str, str], expected: Optional[tuple] = None) -> dict:
    """Audite UNE répétition (dict kind → chemin). Ne lève jamais.
    `expected` : flux capteurs attendus (voir _scan_folder) ; sans lui, seuls les fichiers présents sont audités.

    Retour : {status, classes:{markers|finger|imu|cross → ok|warning|error},
              files:{flux → info}, issues:[{stream, cls, code, severity, msg}]}"""
    m_info, m_iss, parsed = _audit_markers(files.get("markers"))
    window = (parsed["t_start"], parsed["t_stop"]) if parsed else None
    issues = list(m_iss)
    infos = {"markers": m_info}

    present = {k for k in files if k != "markers"}
    expected = set(expected) if expected is not None else present
    for st in sorted(expected | present, key=_stream_sort_key):
        info, iss = _audit_sensor(st, files.get(st), window, st in expected)
        infos[st] = info
        issues += iss

    # cohérence : compteurs du STOP ↔ lignes réellement écrites dans [START, STOP]
    if parsed and parsed["f"] is not None and window and window[0] is not None:
        for cls, key in (("finger", "f"), ("imu", "imu")):
            broken = {"missing_file", "empty", "empty_file", "unreadable", "bad_header", "no_data_in_window"}
            if any(i["cls"] == cls and i["code"] in broken for i in issues):
                continue                      # déjà signalé : inutile d'ajouter une alerte de cohérence
            parts = [infos[s]["rows_window"] for s in infos if s.startswith(cls) and infos[s]["rows_window"] is not None]
            if not parts:
                continue
            got, want = sum(parts), parsed[key]
            tol = max(AUDIT["counter_tol_abs"], AUDIT["counter_tol_rel"] * want)
            if abs(got - want) > tol:
                issues.append(_issue("cross", "count_mismatch", "warning",
                                     f"{cls} : marqueur STOP annonce {want}, CSV contient {got} ({got - want:+d})"))

    for st, info in infos.items():
        info["status"] = _worst([i for i in issues if i["stream"] == st])
    classes = {c: _worst([i for i in issues if i["cls"] == c]) for c in FILE_CLASSES}
    return {"status": _worst(issues), "classes": classes, "files": infos, "issues": issues}


def problem_tags(audit: dict) -> str:
    """'markers:no_stop;imu:gap' — signature compacte pour le journal des déplacements."""
    return ";".join(f"{i['cls']}:{i['code']}" for i in audit["issues"])


# ── audit global (tous les participants), avec rapport CSV + résumé JSON ─────
class AuditCancelled(Exception):
    pass


ISSUE_FIELDS = ["key", "participant", "gesture", "repetition", "stream", "cls", "code", "severity", "message"]


def audit_all(data_root: str, clean_root: str, progress: Optional[Callable[[dict], None]] = None,
              cancel: Optional[threading.Event] = None) -> dict:
    clean_root = _base_clean_root(clean_root)
    plan = [(p, r) for p in _participants(data_root) for r in _scan_folder(os.path.join(data_root, p)).values()]
    rows: List[dict] = []
    n_err = n_warn = 0
    by_rec_cls: Dict[str, Dict[str, int]] = {c: {"error": 0, "warning": 0} for c in FILE_CLASSES}
    for i, (p, r) in enumerate(plan):
        if cancel is not None and cancel.is_set():
            raise AuditCancelled()
        a = audit_recording(r["files"], r["expected"])
        n_err += a["status"] == "error"
        n_warn += a["status"] == "warning"
        for c, st in a["classes"].items():
            if st in ("error", "warning"):
                by_rec_cls[c][st] += 1
        for iss in a["issues"]:
            rows.append({"key": r["key"], "participant": p, "gesture": r["gesture"],
                         "repetition": f"rep{r['rep']:02d}", "stream": iss["stream"], "cls": iss["cls"],
                         "code": iss["code"], "severity": iss["severity"], "message": iss["msg"]})
        if progress and (i % 10 == 0 or i == len(plan) - 1):
            progress({"done": i + 1, "total": len(plan), "current": r["key"]})

    os.makedirs(clean_root, exist_ok=True)
    df = pd.DataFrame(rows, columns=ISSUE_FIELDS)
    df.to_csv(os.path.join(clean_root, "integrity_issues.csv"), index=False)
    summary = _audit_summary(df, len(plan), n_err, n_warn, by_rec_cls)
    with open(os.path.join(clean_root, "integrity_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def _audit_summary(df: pd.DataFrame, audited: int, n_err: int, n_warn: int, by_cls: dict) -> dict:
    codes = []
    if not df.empty:
        g = df.groupby(["cls", "code", "severity"]).agg(
            recordings=("key", "nunique"), issues=("key", "size")).reset_index()
        codes = sorted(g.to_dict("records"), key=lambda x: (-SEVERITY_RANK[x["severity"]], -x["recordings"]))
    return {"audited": audited, "clean": audited - n_err - n_warn, "with_errors": int(n_err),
            "with_warnings_only": int(n_warn), "by_class": by_cls, "codes": codes,
            "thresholds": AUDIT, "generated": time.time()}


def load_audit_summary(clean_root: str) -> Optional[dict]:
    clean_root = _base_clean_root(clean_root)
    path = os.path.join(clean_root, "integrity_summary.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _audit_forget(clean_root: str, keys: List[str]) -> None:
    """Retire des résultats d'audit les répétitions déplacées (issues + compteurs), sans relancer l'audit."""
    keys = set(keys)
    clean_root = _base_clean_root(clean_root)
    p = os.path.join(clean_root, "integrity_issues.csv")
    summ = load_audit_summary(clean_root)
    if not keys or summ is None or not os.path.isfile(p):
        return
    df = pd.read_csv(p)
    df = df[~df["key"].isin(keys)]
    df.to_csv(p, index=False)
    by = {}
    for c in FILE_CLASSES:
        d = df[df["cls"] == c]
        err = set(d[d["severity"] == "error"]["key"])
        warn = set(d[d["severity"] == "warning"]["key"]) - err
        by[c] = {"error": len(err), "warning": len(warn)}
    err_all = set(df[df["severity"] == "error"]["key"])
    warn_all = set(df[df["severity"] == "warning"]["key"]) - err_all
    summary = _audit_summary(df, max(0, summ.get("audited", 0) - len(keys)), len(err_all), len(warn_all), by)
    with open(os.path.join(clean_root, "integrity_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def audit_issues(clean_root: str, cls: str = "", code: str = "", limit: int = 300, severity: str = "") -> List[dict]:
    clean_root = _base_clean_root(clean_root)
    path = os.path.join(clean_root, "integrity_issues.csv")
    if not os.path.isfile(path):
        return []
    df = pd.read_csv(path)
    if cls:
        df = df[df["cls"] == cls]
    if code:
        df = df[df["code"] == code]
    if severity:
        df = df[df["severity"] == severity]
    df = df.sort_values(["severity", "participant", "gesture", "repetition"], ascending=[True, True, True, True])
    return df.head(limit).fillna("").to_dict("records")


def _issue_map(data_root_keys: Dict[str, set], clean_root: str) -> Dict[tuple, dict]:
    """(participant, gesture) → {e: nb répétitions en erreur, w: nb avec avertissements seuls}.
    Ne compte que les clés encore présentes (un déplacement après l'audit les retire)."""
    clean_root = _base_clean_root(clean_root)
    path = os.path.join(clean_root, "integrity_issues.csv")
    if not os.path.isfile(path):
        return {}
    df = pd.read_csv(path)
    if df.empty:
        return {}
    out: Dict[tuple, dict] = {}
    for key, g in df.groupby("key"):
        row = g.iloc[0]
        if key not in data_root_keys.get(row["participant"], set()):
            continue
        sev = "e" if (g["severity"] == "error").any() else "w"
        cell = out.setdefault((row["participant"], row["gesture"]), {"e": 0, "w": 0})
        cell[sev] += 1
    return out
