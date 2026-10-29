"""
preprocessing.py — chaîne de prétraitement intégrée au logiciel de capture
==========================================================================

Transforme les enregistrements bruts écrits par app.py
    data/PXX/<prefix>_finger_<slot>.csv   (un par TapStrap2)
    data/PXX/<prefix>_imu_<slot>.csv      (un par appareil)
    data/PXX/<prefix>_markers.csv
(<slot> = TAP1, TAP2, TAP3… : n'importe quel appareil peut occuper n'importe quel slot)
en un enregistrement unique, aligné, prêt pour la reconnaissance :
    <out>/Gesture_XX/PXX/<prefix>_Clean.csv

Les algorithmes sont ceux de Celaningfinali_v3.py (aucun changement
numérique) ; ce qui change :
  * aucune interaction (input/print/os.startfile) → importable depuis Flask ;
  * chaque répétition renvoie des statistiques de qualité (pente de la
    régression d'horloge, résidu, jitter avant/après, fréquence native,
    doublons, NaN comblés) → rapport CSV/JSON exploitable directement ;
  * progression + annulation → utilisable comme tâche de fond ;
  * normalisation Z-score par split (étape 5) sans prompt.

Étapes : 1 régression d'horloge · 2 découpage START/STOP · 3 rééchantillonnage
sur grille commune · 4 fusion · 5 (optionnelle) normalisation par split.
"""

from __future__ import annotations

import csv
import glob
import json
import logging
import os
import re
import shutil
import threading
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import interpolate

log = logging.getLogger("preprocessing")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_FREQ_HZ = 190          # valeur retenue dans le mémoire (< 200/208 Hz natifs)
DEFAULT_METHOD = "linear"      # "linear" | "pchip" | "akima"
METHODS = ("linear", "pchip", "akima")
NON_DATA_COLUMNS = ("Timestamp_LSL", "Timestamp_LSL_raw", "Timestamp_HW", "Type")
NORM_EXCLUDED = ("Timestamp_LSL", "Participant", "Gesture", "Repetition")

# Ex : P01_S01_Gesture_01_rep01_20260520_152408
NAME_PATTERN = re.compile(
    r"^(?P<participant>P\d+)_S(?P<session>\d+)_(?P<gesture>Gesture_\d+)_rep(?P<rep>\d+)_"
)
PARTICIPANT_DIR = re.compile(r"^P\d+$")

REPORT_FIELDS = [
    "key", "participant", "gesture", "repetition", "status", "error",
    "freq_hz", "method", "duration_s", "grid_points", "nan_filled", "n_warnings",
    "sensor", "rows_raw", "rows_kept", "native_hz", "slope_a", "intercept_b",
    "residual_std_ms", "jitter_before_ms", "jitter_after_ms", "dup_removed",
    "arrival_delay_mean_ms", "arrival_delay_p95_ms", "arrival_delay_max_ms",
    "out_path",
]


class Cancelled(Exception):
    """Levée quand l'examinateur annule le traitement. `results` = répétitions déjà traitées (pour le rapport)."""
    def __init__(self, results=None):
        super().__init__()
        self.results = results or []


# ─────────────────────────────────────────────────────────────────────────────
# DÉCOUVERTE DES FICHIERS
# ─────────────────────────────────────────────────────────────────────────────
def _key_from_path(path: str) -> str:
    name = os.path.basename(path).replace(".csv", "")
    m = re.search(r"^(.*?)(?:_finger_|_imu_|_markers)", name)
    return m.group(1) if m else name


def list_participants(data_root: str) -> List[str]:
    """Dossiers P01, P02, … présents (les dossiers anomalie/test sont ignorés)."""
    if not os.path.isdir(data_root):
        return []
    return sorted(d for d in os.listdir(data_root)
                  if PARTICIPANT_DIR.match(d) and os.path.isdir(os.path.join(data_root, d)))


def list_repetitions(folder: str) -> List[str]:
    """Clés de répétition ayant des marqueurs et au moins un flux capteur (doigts ou IMU) dans `folder` (à plat).
    Une configuration sans TapStrap2 (IMU seuls) ou sans IMU est donc acceptée."""
    groups: Dict[str, Dict[str, list]] = {}
    for kind, pat in (("finger", "*_finger_*.csv"), ("imu", "*_imu_*.csv"),
                      ("markers", "*_markers.csv")):
        for p in glob.glob(os.path.join(folder, pat)):
            groups.setdefault(_key_from_path(p), {"finger": [], "imu": [], "markers": []})[kind].append(p)
    return sorted(k for k, g in groups.items() if (g["finger"] or g["imu"]) and g["markers"])


def parse_name(key: str) -> Optional[dict]:
    m = NAME_PATTERN.match(key)
    if not m:
        return None
    return {"participant": m.group("participant"), "gesture": m.group("gesture"),
            "repetition": f"rep{m.group('rep')}"}


def clean_path(out_root: str, key: str) -> Optional[str]:
    info = parse_name(key)
    if not info:
        return None
    return os.path.join(out_root, info["gesture"], info["participant"], f"{key}_Clean.csv")


def stream_sets(data_root: str, participants: List[str]) -> Dict[str, tuple]:
    """clé de répétition → flux capteurs présents (ex. ('finger_TAP1', 'imu_TAP1', 'imu_TAP2')), d'après les noms de fichiers."""
    out: Dict[str, set] = {}
    for p in participants:
        folder = os.path.join(data_root, p)
        for pat in ("*_finger_*.csv", "*_imu_*.csv"):
            for path in glob.glob(os.path.join(folder, pat)):
                out.setdefault(_key_from_path(path), set()).add(stream_name(os.path.basename(path).replace(".csv", "")))
    return {k: tuple(sorted(v)) for k, v in out.items()}


def stream_differences(sets: Dict[str, tuple]) -> Optional[dict]:
    """Compare les flux des répétitions. None si tout le monde a les mêmes ; sinon
    {reference, n_reference, odd:[{key, missing, extra}]} (référence = l'ensemble le plus fréquent).
    Simple information : ces répétitions sont quand même nettoyées."""
    from collections import Counter
    if len(set(sets.values())) <= 1:
        return None
    ref, n = Counter(sets.values()).most_common(1)[0]
    odd = [{"key": k, "missing": sorted(set(ref) - set(v)), "extra": sorted(set(v) - set(ref))}
           for k, v in sorted(sets.items()) if v != ref]
    return {"reference": list(ref), "n_reference": n, "odd": odd}


def format_stream_notice(diff: dict, limit: int = 6) -> str:
    """Texte de la notification (une répétition par ligne, tronqué)."""
    head = (f"⚠ {len(diff['odd'])} répétition(s) n'ont pas les mêmes flux que les {diff['n_reference']} autres "
            f"(référence : {', '.join(diff['reference'])}). Elles sont nettoyées quand même, mais leurs colonnes "
            f"diffèrent :")
    lines = []
    for o in diff["odd"][:limit]:
        det = ([f"manque {', '.join(o['missing'])}"] if o["missing"] else []) + ([f"en plus {', '.join(o['extra'])}"] if o["extra"] else [])
        lines.append(f"  · {o['key']} — {' ; '.join(det)}")
    if len(diff["odd"]) > limit:
        lines.append(f"  · … et {len(diff['odd']) - limit} autre(s)")
    return "\n".join([head] + lines)


def _load_repetition(folder: str, key: str):
    """Charge les flux d'une répétition. Un flux sans aucune ligne (en-tête seul) est ignoré et
    signalé en avertissement au lieu de faire échouer toute la répétition (un fichier vide peut venir
    d'une copie, d'un appareil déconnecté ou d'un outil tiers).
    → (sensors, markers, flux_ignorés)"""
    sensors: Dict[str, dict] = {}
    skipped: List[str] = []
    for kind, pat in (("finger", "*_finger_*.csv"), ("imu", "*_imu_*.csv")):
        for p in sorted(glob.glob(os.path.join(folder, pat))):
            if _key_from_path(p) == key:
                name = os.path.basename(p).replace(".csv", "")
                df = pd.read_csv(p)
                if df.empty:
                    skipped.append(stream_name(name))
                else:
                    sensors[name] = {"df": df, "type": kind}
    marker_paths = [p for p in glob.glob(os.path.join(folder, "*_markers.csv")) if _key_from_path(p) == key]
    if not sensors or not marker_paths:
        raise FileNotFoundError(f"Fichiers incomplets ou sans données pour {key}")
    return sensors, pd.read_csv(marker_paths[0]), skipped


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 1 — RÉGRESSION D'HORLOGE   LSL_corr = a·HW + b
# ─────────────────────────────────────────────────────────────────────────────
def spread_packet_timestamps(hw):
    """Les échantillons d'un même paquet BLE partagent le MÊME timestamp matériel. On les répartit
    régulièrement entre ce timestamp et celui du paquet suivant (dernier paquet : même pas que le précédent).
    Les sauts en arrière ne sont pas touchés."""
    hw = np.asarray(hw, dtype=float)
    n = len(hw)
    if n < 2:
        return hw.copy()
    starts = np.flatnonzero(np.r_[True, np.diff(hw) != 0])
    counts = np.diff(np.r_[starts, n])
    T = hw[starts]
    step = np.empty(len(starts))
    step[:-1] = T[1:] - T[:-1]
    step[-1] = step[-2] if len(starts) > 1 else 0.0
    step = np.where(step > 0, step, 0.0)
    pos = np.arange(n) - np.repeat(starts, counts)
    return np.repeat(T, counts) + np.repeat(step, counts) * pos / np.repeat(counts, counts)


def step1_clock_regression(sensors: Dict[str, dict], stats: Dict[str, dict],
                           fix_slope: bool = False) -> Dict[str, dict]:
    """fix_slope=False : droite LSL = a·HW + b ajustée (a et b estimés).
    fix_slope=True  : a = 1 imposé (les deux horloges tournent à la même vitesse), seul le décalage
                      b = moyenne(LSL − HW) est estimé (moindres carrés avec pente fixée)."""
    out = {}
    for name, data in sensors.items():
        df = data["df"].copy()
        s = {"rows_raw": int(len(df))}
        if "Timestamp_HW" not in df.columns or "Timestamp_LSL" not in df.columns or len(df) < 2:
            s["note"] = "pas de régression (colonnes manquantes ou < 2 lignes)"
            stats[name] = s
            out[name] = {**data, "df": df}
            continue

        hw = df["Timestamp_HW"].values
        lsl = df["Timestamp_LSL"].values
        if fix_slope:
            a, b = 1.0, float(np.mean(lsl - hw))
        else:
            a, b = np.polyfit(hw, lsl, 1)
        lsl_fit = a * hw + b                      # droite ajustée (statistiques de synchro, inchangées)
        lsl_corr = a * spread_packet_timestamps(hw) + b   # temps corrigé : timestamps répartis dans chaque paquet
        # retard d'arrivée LSL vs HW : écart à la droite, ramené à 0 pour l'arrivée la plus précoce
        delay_ms = (lsl - lsl_fit - np.min(lsl - lsl_fit)) * 1000

        s.update({
            "slope_a": float(a),
            "intercept_b": float(b),
            "residual_std_ms": float(np.std(lsl - lsl_fit) * 1000),
            "jitter_before_ms": float(np.std(np.diff(lsl)) * 1000),
            "jitter_after_ms": float(np.std(np.diff(lsl_corr)) * 1000),
            "arrival_delay_mean_ms": float(np.mean(delay_ms)),
            "arrival_delay_p95_ms": float(np.percentile(delay_ms, 95)),
            "arrival_delay_max_ms": float(np.max(delay_ms)),
        })
        df["Timestamp_LSL_raw"] = df["Timestamp_LSL"]
        df["Timestamp_LSL"] = lsl_corr
        stats[name] = s
        out[name] = {**data, "df": df}
    return out


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 2 — DÉCOUPAGE SUR LES MARKERS
# ─────────────────────────────────────────────────────────────────────────────
def extract_markers(df_markers: pd.DataFrame):
    t_start = t_stop = None
    for _, row in df_markers.iterrows():
        marker, ts = str(row["Marker"]), float(row["Timestamp_LSL"])
        if marker.startswith("START") and not marker.startswith("COUNTDOWN"):
            t_start = ts
        elif marker.startswith("STOP"):
            t_stop = ts
    if t_start is None or t_stop is None:
        raise ValueError(f"Markers START/STOP introuvables ({df_markers['Marker'].tolist()})")
    return t_start, t_stop


def step2_trim(sensors: Dict[str, dict], df_markers: pd.DataFrame, stats: Dict[str, dict]):
    """Découpe et grille sur l'horloge corrigée a·HW+b (Timestamp_LSL issu de l'étape 1).
    Si la régression n'a pas pu être faite (colonnes manquantes ou < 2 lignes), Timestamp_LSL reste le temps d'arrivée."""
    t_start, t_stop = extract_markers(df_markers)
    out = {}
    for name, data in sensors.items():
        df = data["df"].copy()
        t = df["Timestamp_LSL"]
        mask = (t >= t_start) & (t <= t_stop)
        if not mask.any():
            raise ValueError(f"{name} : aucune ligne dans [{t_start:.6f}, {t_stop:.6f}] (Timestamp_LSL corrigé)")
        df = df[mask].copy()
        df["Timestamp_LSL"] = t[mask].values - t_start   # t = 0 au START
        stats.setdefault(name, {})["rows_kept"] = int(len(df))
        out[name] = {**data, "df": df}
    return out, t_start, t_stop


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 3 — RÉÉCHANTILLONNAGE SUR GRILLE COMMUNE
# ─────────────────────────────────────────────────────────────────────────────
def _interpolate(t, values, grid, method):
    if method == "linear":
        return interpolate.interp1d(t, values, kind="linear", bounds_error=False, fill_value=np.nan)(grid)
    if method == "pchip":
        return interpolate.PchipInterpolator(t, values, extrapolate=False)(grid)
    if method == "akima":
        return interpolate.Akima1DInterpolator(t, values)(grid, extrapolate=False)
    raise ValueError(f"Méthode d'interpolation inconnue : {method}")


def step3_resample(sensors, t_start, t_stop, freq_hz, method, stats, warnings: List[str]):
    duration = t_stop - t_start
    dt = 1.0 / freq_hz
    n_pts = int(np.floor(duration / dt)) + 1
    grid = np.linspace(0, duration, n_pts)
    nan_total = 0
    out = {}

    for name, data in sensors.items():
        df = data["df"].copy()
        t = df["Timestamp_LSL"].values
        s = stats.setdefault(name, {})

        if len(t) > 1:
            # cadence MOYENNE (n-1)/durée : la médiane des Δt est faussée quand les mesures arrivent par paquets
            span = t[-1] - t[0]
            native = (len(t) - 1) / span if span > 0 else float("inf")
            s["native_hz"] = float(native)
            if freq_hz > native * 1.5:
                warnings.append(f"{name} : ~{native:.0f} Hz natif, suréchantillonné à {freq_hz} Hz")
            elif native > freq_hz * 1.5:
                warnings.append(f"{name} : ~{native:.0f} Hz natif, sous-échantillonné à {freq_hz} Hz")

        dup = 0
        if len(t) > 1 and np.any(np.diff(t) <= 0):
            keep = np.concatenate(([True], np.diff(t) > 0))
            dup = int((~keep).sum())
            t = t[keep]
            df = df.loc[keep].reset_index(drop=True)
            warnings.append(f"{name} : {dup} timestamps non croissants supprimés")
        s["dup_removed"] = dup

        df_out = pd.DataFrame({"Timestamp_LSL": grid})
        for col in [c for c in df.columns if c not in NON_DATA_COLUMNS]:
            df_out[col] = _interpolate(t, df[col].values.astype(float), grid, method)

        n_nan = int(df_out.isnull().sum().sum())
        if n_nan:
            df_out = df_out.ffill().bfill()
        nan_total += n_nan
        out[name] = {**data, "df": df_out}

    return out, grid, nan_total


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 4 — FUSION
# ─────────────────────────────────────────────────────────────────────────────
def stream_name(full_name: str) -> str:
    """'…_imu_TAP1' → 'imu_TAP1' (distingue doigts et IMU d'un même Tap)."""
    m = re.search(r"_((?:imu|finger)_.+)$", full_name)
    return m.group(1) if m else full_name


def short_sensor_name(full_name: str) -> str:
    m = re.search(r"_(?:imu|finger)_(.+)$", full_name)
    return m.group(1) if m else full_name


def step4_merge(sensors, grid, ident: dict) -> pd.DataFrame:
    """Tous les capteurs sont déjà sur `grid` : simple concaténation de colonnes
    (équivalent à la jointure sur Timestamp_LSL de la V3, sans le coût ni le
    risque d'une jointure sur flottants)."""
    parts = [pd.DataFrame({"Timestamp_LSL": grid})]
    for name, data in sensors.items():
        short = short_sensor_name(name)
        df = data["df"].drop(columns=["Timestamp_LSL"])
        parts.append(df.rename(columns={c: f"{short}_{c}" for c in df.columns}))
    df_final = pd.concat(parts, axis=1)
    df_final.insert(0, "Participant", ident["participant"])
    df_final.insert(1, "Gesture", ident["gesture"])
    df_final.insert(2, "Repetition", ident["repetition"])
    return df_final


# ─────────────────────────────────────────────────────────────────────────────
# UNE RÉPÉTITION DE BOUT EN BOUT
# ─────────────────────────────────────────────────────────────────────────────
def process_repetition(folder: str, key: str, out_root: str,
                       freq_hz: int = DEFAULT_FREQ_HZ, method: str = DEFAULT_METHOD,
                       overwrite: bool = False, keep_intermediate: bool = False,
                       fix_slope: bool = False) -> dict:
    """Traite une répétition. Ne lève jamais : renvoie un dict de résultat."""
    ident = parse_name(key)
    res = {"key": key, "status": "error", "error": "", "warnings": [], "sensors": {},
           "freq_hz": freq_hz, "method": method, "out_path": None,
           "participant": ident["participant"] if ident else "Unknown",
           "gesture": ident["gesture"] if ident else "Unknown",
           "repetition": ident["repetition"] if ident else "Unknown"}
    try:
        if not ident:
            raise ValueError(f"Nom inattendu (Pxx_Sxx_Gesture_xx_repxx_…) : {key}")

        target = clean_path(out_root, key)
        res["out_path"] = target
        if os.path.isfile(target) and not overwrite:
            res["status"] = "skipped"
            return res

        sensors, df_markers, skipped = _load_repetition(folder, key)
        if skipped:
            res["warnings"].append("flux vide(s) ignoré(s) : " + ", ".join(skipped))
        stats: Dict[str, dict] = {}

        s1 = step1_clock_regression(sensors, stats, fix_slope)
        s2, t_start, t_stop = step2_trim(s1, df_markers, stats)
        s3, grid, nan_filled = step3_resample(s2, t_start, t_stop, freq_hz, method, stats, res["warnings"])
        df_final = step4_merge(s3, grid, ident)

        os.makedirs(os.path.dirname(target), exist_ok=True)
        tmp = target + ".tmp"                    # écriture atomique : un arrêt brutal ne laisse jamais un _Clean.csv tronqué
        df_final.to_csv(tmp, index=False)
        os.replace(tmp, target)

        if keep_intermediate:
            inter = os.path.join(out_root, "_intermediate", key)
            for tag, group in (("step1", s1), ("step2", s2), ("step3", s3)):
                os.makedirs(os.path.join(inter, tag), exist_ok=True)
                for name, d in group.items():
                    d["df"].to_csv(os.path.join(inter, tag, f"{name}_{tag}.csv"), index=False)

        res.update({"status": "ok", "sensors": stats, "duration_s": float(t_stop - t_start),
                    "grid_points": int(len(grid)), "nan_filled": int(nan_filled)})
    except Cancelled:
        raise
    except Exception as exc:  # noqa: BLE001 — on remonte l'erreur dans le rapport
        res["error"] = f"{type(exc).__name__}: {exc}"
        log.warning("Échec %s : %s", key, res["error"])
    return res


# ─────────────────────────────────────────────────────────────────────────────
# LOT : PLUSIEURS PARTICIPANTS (progression + annulation)
# ─────────────────────────────────────────────────────────────────────────────
def count_pending(data_root: str, out_root: str, participants: List[str]) -> dict:
    """Pour l'UI : répétitions brutes et déjà nettoyées par participant."""
    info = {}
    for p in participants:
        reps = list_repetitions(os.path.join(data_root, p))
        done = sum(1 for k in reps if (clean_path(out_root, k) and os.path.isfile(clean_path(out_root, k))))
        info[p] = {"raw": len(reps), "clean": done}
    return info


def default_workers() -> int:
    return max(1, min((os.cpu_count() or 2) - 1, 8))


def run_batch(data_root: str, out_root: str, participants: List[str],
              freq_hz: int = DEFAULT_FREQ_HZ, method: str = DEFAULT_METHOD,
              overwrite: bool = False, keep_intermediate: bool = False,
              fix_slope: bool = False,
              progress: Optional[Callable[[dict], None]] = None,
              cancel: Optional[threading.Event] = None,
              workers: Optional[int] = None) -> List[dict]:
    """Traite toutes les répétitions des participants donnés.

    workers > 1 : répétitions traitées en parallèle (processus) — chaque répétition
    est indépendante, le gain est quasi linéaire. Repli automatique en séquentiel
    si le pool de processus n'est pas disponible."""
    if method not in METHODS:
        raise ValueError(f"method doit être dans {METHODS}")
    plan = [(p, k) for p in participants for k in list_repetitions(os.path.join(data_root, p))]
    workers = default_workers() if workers is None else max(1, int(workers))
    results: List[dict] = []
    done_keys = set()

    def emit(r):
        results.append(r)
        done_keys.add(r["key"])
        if progress:
            progress({"done": len(results), "total": len(plan), "current": r["key"], "last": r})

    if workers > 1 and len(plan) > 3:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from concurrent.futures.process import BrokenProcessPool
        ex = ProcessPoolExecutor(max_workers=workers)
        try:
            futs = [ex.submit(process_repetition, os.path.join(data_root, p), k, out_root,
                              freq_hz, method, overwrite, keep_intermediate, fix_slope) for p, k in plan]
            for fut in as_completed(futs):
                if cancel is not None and cancel.is_set():
                    raise Cancelled()
                emit(fut.result())
        except Cancelled:
            ex.shutdown(wait=True, cancel_futures=True)       # les répétitions en cours se terminent proprement
            for fu in futs:                                    # …et sont gardées pour le rapport
                if fu.done() and not fu.cancelled():
                    try:
                        r = fu.result()
                    except Exception:
                        continue
                    if r["key"] not in done_keys:
                        results.append(r); done_keys.add(r["key"])
            raise Cancelled(results)
        except BrokenProcessPool:
            log.warning("Pool de processus indisponible → repli séquentiel")
        finally:
            ex.shutdown(wait=True, cancel_futures=True)

    for p, k in plan:                       # séquentiel (ou reprise après repli)
        if k in done_keys:
            continue
        if cancel is not None and cancel.is_set():
            raise Cancelled(results)
        emit(process_repetition(os.path.join(data_root, p), k, out_root, freq_hz, method,
                                overwrite, keep_intermediate, fix_slope))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# RAPPORT QUALITÉ  (une ligne par répétition × capteur)
# ─────────────────────────────────────────────────────────────────────────────
def write_report(results: List[dict], out_root: str, merge_existing: bool = True) -> dict:
    """Écrit preprocessing_report.csv et preprocessing_summary.json à la racine
    de sortie. Fusionne avec un rapport précédent (traitements incrémentaux)."""
    os.makedirs(out_root, exist_ok=True)
    csv_path = os.path.join(out_root, "preprocessing_report.csv")

    rows = []
    for r in results:
        if r["status"] == "skipped":
            continue
        base = {f: r.get(f, "") for f in REPORT_FIELDS if f not in ("sensor",)}
        base["n_warnings"] = len(r["warnings"])
        if r["sensors"]:
            for sname, s in r["sensors"].items():
                row = dict(base, sensor=stream_name(sname))
                for f in ("rows_raw", "rows_kept", "native_hz", "slope_a", "intercept_b",
                          "residual_std_ms", "jitter_before_ms", "jitter_after_ms", "dup_removed",
                          "arrival_delay_mean_ms", "arrival_delay_p95_ms", "arrival_delay_max_ms"):
                    row[f] = s.get(f, "")
                rows.append(row)
        else:
            rows.append(dict(base, sensor=""))

    df_new = pd.DataFrame(rows, columns=REPORT_FIELDS)
    if merge_existing and os.path.isfile(csv_path) and not df_new.empty:
        old = pd.read_csv(csv_path)
        if "key" not in old.columns:
            old["key"] = ""
        old["key"] = old["key"].fillna("")
        new_keys = set(df_new["key"])
        old = old[~old["key"].isin(new_keys)]
        old = old.reindex(columns=REPORT_FIELDS)
        df_new = pd.concat([old, df_new], ignore_index=True)
    if not df_new.empty or not os.path.isfile(csv_path):
        df_new.to_csv(csv_path, index=False)

    summary = summarize(pd.read_csv(csv_path)) if os.path.isfile(csv_path) else {}
    with open(os.path.join(out_root, "preprocessing_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def summarize(df: pd.DataFrame) -> dict:
    """Agrégats pour le tableau de bord (et pour la section Évaluation du papier)."""
    if df.empty:
        return {"repetitions": 0}
    id_cols = ["key"] if "key" in df.columns and df["key"].fillna("").ne("").all() \
        else ["participant", "gesture", "repetition"]
    reps = df.drop_duplicates(id_cols)
    ok = reps[reps["status"] == "ok"]
    out = {
        "repetitions": int(len(reps)),
        "ok": int(len(ok)),
        "errors": int((reps["status"] == "error").sum()),
        "repetitions_with_warnings": int((ok["n_warnings"] > 0).sum()) if len(ok) else 0,
        "repetitions_with_nan_filled": int((ok["nan_filled"] > 0).sum()) if len(ok) else 0,
        "sensors": {},
    }
    d = df[df["status"] == "ok"]
    for sensor, g in d.groupby("sensor"):
        if not sensor:
            continue

        def q(col):
            v = pd.to_numeric(g[col], errors="coerce").dropna()
            return {"mean": float(v.mean()), "std": float(v.std(ddof=0)),
                    "p50": float(v.quantile(.5)), "p95": float(v.quantile(.95)),
                    "max": float(v.max())} if len(v) else None

        out["sensors"][sensor] = {
            "n": int(len(g)),
            "residual_std_ms": q("residual_std_ms"),
            "jitter_before_ms": q("jitter_before_ms"),
            "jitter_after_ms": q("jitter_after_ms"),
            "native_hz": q("native_hz"),
            "slope_a": q("slope_a"),
            "dup_removed_total": int(pd.to_numeric(g["dup_removed"], errors="coerce").fillna(0).sum()),
        }
    return out


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 5 — NORMALISATION Z-SCORE PAR SPLIT (stats = bucket 'train' uniquement)
# ─────────────────────────────────────────────────────────────────────────────
def _list_clean(out_root: str) -> List[str]:
    skip = ("normalized_", "/_excluded/", "/_intermediate/")
    return sorted(p for p in glob.glob(os.path.join(out_root, "**", "*_Clean.csv"), recursive=True)
                  if not any(t in p.replace("\\", "/") for t in skip))


def _zscore_stats(paths: List[str]) -> dict:
    s, s2, n = {}, {}, {}
    for p in paths:
        df = pd.read_csv(p)
        for c in df.columns:
            if c in NORM_EXCLUDED or not pd.api.types.is_numeric_dtype(df[c]):
                continue
            v = df[c].dropna().values.astype(float)
            if v.size:
                s[c] = s.get(c, 0.0) + v.sum()
                s2[c] = s2.get(c, 0.0) + np.square(v).sum()
                n[c] = n.get(c, 0) + v.size
    stats = {}
    for c, cnt in n.items():
        mean = s[c] / cnt
        stats[c] = {"mean": float(mean), "std": float(np.sqrt(max(s2[c] / cnt - mean ** 2, 0.0))), "n": int(cnt)}
    return stats


def normalize_by_split(out_root: str, split_files: List[str]) -> List[dict]:
    """Pour chaque split (OUT_UD.json, OUT_UI.json…) : stats sur 'train' seulement,
    appliquées à tous les _Clean.csv → normalized_<scenario>/…_Clean_Norm.csv."""
    cleans = _list_clean(out_root)
    if not cleans:
        raise FileNotFoundError("Aucun *_Clean.csv — lancer d'abord le prétraitement.")

    def key_of(path):
        m = NAME_PATTERN.match(os.path.basename(path))
        return (m.group("participant"), m.group("gesture"),
                int(m.group("session")) * 100 + int(m.group("rep"))) if m else None

    summaries = []
    for split_path in split_files:
        with open(split_path, "r", encoding="utf-8") as f:
            split = json.load(f)
        scenario = split.get("scenario", os.path.splitext(os.path.basename(split_path))[0])
        train_keys = {(e["user"], cls, e["id"]) for cls, b in split["classes"].items() for e in b.get("train", [])}
        train_paths = [p for p in cleans if key_of(p) in train_keys]
        stats = _zscore_stats(train_paths)

        scen_dir = os.path.join(out_root, f"normalized_{scenario}")
        os.makedirs(scen_dir, exist_ok=True)
        with open(os.path.join(scen_dir, f"normalization_stats_zscore_{scenario}.json"), "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        for p in cleans:
            df = pd.read_csv(p)
            for c, st in stats.items():
                if c in df.columns:
                    df[c] = (df[c] - st["mean"]) / st["std"] if st["std"] > 1e-12 else df[c] - st["mean"]
            rel = os.path.relpath(os.path.dirname(p), out_root)
            dest = os.path.join(scen_dir, rel)
            os.makedirs(dest, exist_ok=True)
            df.to_csv(os.path.join(dest, os.path.basename(p).replace("_Clean.csv", "_Clean_Norm.csv")), index=False)

        summaries.append({"scenario": scenario, "train_files": len(train_paths),
                          "referenced": len(train_keys), "files_written": len(cleans)})
    return summaries


# ─────────────────────────────────────────────────────────────────────────────
# Ligne de commande (même module, sans passer par Flask)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Prétraitement des enregistrements TapStrap")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="data_clean")
    ap.add_argument("--freq", type=int, default=DEFAULT_FREQ_HZ)
    ap.add_argument("--method", choices=METHODS, default=DEFAULT_METHOD)
    ap.add_argument("--participants", nargs="*", default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--keep-intermediate", action="store_true")
    ap.add_argument("--fix-slope", action="store_true", help="impose a = 1 (n'estime que le décalage b) au lieu d'ajuster la pente")
    ap.add_argument("--workers", type=int, default=None, help="processus en parallèle (défaut : auto)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    parts = a.participants or list_participants(a.data)
    res = run_batch(a.data, a.out, parts, a.freq, a.method, a.overwrite, a.keep_intermediate,
                    fix_slope=a.fix_slope, workers=a.workers, progress=lambda p: print(f"[{p['done']}/{p['total']}] {p['current']} → {p['last']['status']}"))
    print(json.dumps(write_report(res, a.out), indent=2, ensure_ascii=False))
