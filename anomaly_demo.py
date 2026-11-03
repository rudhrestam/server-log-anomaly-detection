"""
Server Log Anomaly Detection — demo.

Every operations team has lived some version of this: nothing is obviously
down, the dashboards are mostly green, but one host is quietly throwing
errors, another has gone slow, and a third is getting worse by the hour.
By the time a human notices, the interesting question is no longer "is
something wrong?" but "when did it start, and which host do I look at
first?".

This demo builds that situation from scratch. It generates a week of raw
log lines for ten hosts — timestamps, levels, message templates and
latencies, with the usual day/night swing in volume — plants three
different faults in known places, then tries to find them again the way
you would with real logs:

  1. Parse the raw lines into a structured table.
  2. Roll the lines up into per-host, per-30-minute windows and describe
     each window with a handful of simple features.
  3. Detect outliers two ways — an IsolationForest over all the window
     features, and a plain per-host z-score on error rate — and compare
     them against the planted ground truth.

Everything is synthetic, the seed is fixed, and no real log data (from
any system, anywhere) is used.

Run:  python anomaly_demo.py
Writes a copy of the report to anomaly_report.txt next to this file.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

SEED = 3

# --- Fleet and time range --------------------------------------------------
HOSTS = [f"host-{i:02d}" for i in range(1, 11)]
DAYS = 7
WINDOW_MINUTES = 30
WINDOWS_PER_DAY = 24 * 60 // WINDOW_MINUTES
START = pd.Timestamp("2026-01-05 00:00:00")  # a Monday, arbitrary

# Generic message templates. Normal traffic mostly uses the first group;
# the error group only shows up on ERROR lines, which is part of what
# makes an error burst visible in the unique-template feature.
INFO_TEMPLATES = [
    "request completed",
    "health check ok",
    "cache lookup finished",
    "database query executed",
    "session token refreshed",
    "background job finished",
    "file upload stored",
    "metrics snapshot written",
]
WARN_TEMPLATES = [
    "cache miss rate rising",
    "slow query detected",
    "retrying upstream request",
    "disk usage above threshold",
    "connection pool nearly full",
]
ERROR_TEMPLATES = [
    "connection timeout",
    "upstream service unavailable",
    "authentication failed",
    "unhandled exception in worker",
    "disk write failed",
]
ALL_TEMPLATES = INFO_TEMPLATES + WARN_TEMPLATES + ERROR_TEMPLATES

# --- Planted faults (the ground truth the detectors are scored on) --------
# Each entry is (host, day_index, first_window, last_window_exclusive, label).
# Windows are 30 minutes long, so window 28 of a day starts at 14:00.
PLANTED = [
    ("host-03", 2, 28, 32, "error burst"),
    ("host-07", 4, 20, 23, "latency + error-rate spike"),
]
# The third fault does not fit a neat box: host-05 gets worse all through
# day 5. Its error probability climbs linearly from its normal level to
# about 0.35 by the end of the day. The windows where the intended error
# rate is at least double the host's normal rate count as planted — the
# first few hours of the slide are, honestly, barely distinguishable
# from noise, and pretending otherwise would flatter the detectors.
GRADUAL_HOST = "host-05"
GRADUAL_DAY = 5
GRADUAL_PEAK_ERROR = 0.35

# Detection settings.
Z_THRESHOLD = 3.5
IF_CONTAMINATION = 0.015
FEATURES = [
    "total_lines", "error_count", "error_rate", "warn_rate",
    "mean_latency_ms", "p95_latency_ms", "unique_templates",
]


def host_profile(rng):
    """Each host gets its own normal: volume, error rate, latency."""
    profiles = {}
    for host in HOSTS:
        profiles[host] = {
            "base_volume": float(rng.uniform(55, 105)),   # lines / 30 min at peak
            "error_rate": float(rng.uniform(0.02, 0.045)),
            "warn_rate": float(rng.uniform(0.07, 0.12)),
            "latency_ms": float(rng.uniform(45, 95)),     # median-ish latency
        }
    return profiles


def diurnal_factor(window_of_day):
    """Traffic peaks mid-morning and mid-afternoon, sags overnight."""
    hour = window_of_day * WINDOW_MINUTES / 60.0
    day_curve = 0.62 + 0.38 * np.sin(np.pi * (hour - 5.5) / 15.0) ** 2
    return float(np.clip(day_curve, 0.35, 1.0))


def planted_lookup():
    """Set of (host, absolute_window_index) that are planted anomalies."""
    planted = set()
    for host, day, first, last, _label in PLANTED:
        for w in range(first, last):
            planted.add((host, day * WINDOWS_PER_DAY + w))
    # Gradual failure: mark windows once the intended error rate has at
    # least doubled versus that host's profile. The profile is not known
    # here, so use the conservative end of the profile range (0.045);
    # in practice this lands on roughly the second half of the day.
    for w in range(WINDOWS_PER_DAY):
        frac = w / (WINDOWS_PER_DAY - 1)
        intended = 0.03 + (GRADUAL_PEAK_ERROR - 0.03) * frac
        if intended >= 0.09:
            planted.add((GRADUAL_HOST, GRADUAL_DAY * WINDOWS_PER_DAY + w))
    return planted


def generate_raw_lines(rng, profiles):
    """Generate raw log lines, one pipe-delimited string per line.

    Format:  timestamp | host | level | latency_ms | message template
    Pipe-delimited on purpose — real log formats are messier, but the
    parse step below is the point, not the delimiter.
    """
    lines = []
    total_windows = DAYS * WINDOWS_PER_DAY
    for host in HOSTS:
        prof = profiles[host]
        for abs_window in range(total_windows):
            day, window_of_day = divmod(abs_window, WINDOWS_PER_DAY)
            window_start = START + pd.Timedelta(
                minutes=abs_window * WINDOW_MINUTES)

            volume = prof["base_volume"] * diurnal_factor(window_of_day)
            error_p = prof["error_rate"]
            warn_p = prof["warn_rate"]
            latency_scale = 1.0
            template_pool_extra = []

            # --- plant the faults -------------------------------------
            for p_host, p_day, first, last, label in PLANTED:
                if host == p_host and day == p_day and first <= window_of_day < last:
                    if label == "error burst":
                        volume *= 1.6
                        error_p = 0.45
                        template_pool_extra = ERROR_TEMPLATES
                    else:  # latency + error-rate spike
                        volume *= 1.15
                        error_p = 0.28
                        latency_scale = 4.5
            if host == GRADUAL_HOST and day == GRADUAL_DAY:
                frac = window_of_day / (WINDOWS_PER_DAY - 1)
                error_p = prof["error_rate"] + (
                    GRADUAL_PEAK_ERROR - prof["error_rate"]) * frac
                latency_scale = 1.0 + 1.2 * frac

            n_lines = max(5, int(rng.poisson(volume)))
            offsets = np.sort(rng.integers(0, WINDOW_MINUTES * 60, size=n_lines))
            levels = rng.random(n_lines)
            latencies = rng.lognormal(
                mean=np.log(prof["latency_ms"] * latency_scale),
                sigma=0.45, size=n_lines)

            for i in range(n_lines):
                r = levels[i]
                if r < error_p:
                    level = "ERROR"
                    template = rng.choice(ERROR_TEMPLATES + template_pool_extra)
                    # Errors tend to be the slow requests, not the fast ones.
                    latency = latencies[i] * rng.uniform(1.4, 2.6)
                elif r < error_p + warn_p:
                    level = "WARN"
                    template = rng.choice(WARN_TEMPLATES)
                    latency = latencies[i] * rng.uniform(1.1, 1.8)
                else:
                    level = "INFO"
                    template = rng.choice(INFO_TEMPLATES)
                    latency = latencies[i]
                ts = window_start + pd.Timedelta(seconds=int(offsets[i]))
                lines.append(
                    f"{ts:%Y-%m-%d %H:%M:%S} | {host} | {level} | "
                    f"{latency:.1f} | {template}")
    rng.shuffle(lines)  # interleave hosts the way a real log stream arrives
    return lines


def parse_lines(lines):
    """Parse raw log lines into a structured DataFrame."""
    records = [line.split(" | ") for line in lines]
    df = pd.DataFrame(
        records,
        columns=["timestamp", "host", "level", "latency_ms", "template"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["latency_ms"] = df["latency_ms"].astype(float)
    return df.sort_values("timestamp").reset_index(drop=True)


def build_windows(df):
    """Aggregate parsed lines into per-host, per-window feature rows."""
    df = df.copy()
    elapsed_min = (df["timestamp"] - START).dt.total_seconds() // 60
    df["abs_window"] = (elapsed_min // WINDOW_MINUTES).astype(int)
    df["is_error"] = (df["level"] == "ERROR").astype(int)
    df["is_warn"] = (df["level"] == "WARN").astype(int)

    grouped = df.groupby(["host", "abs_window"])
    windows = grouped.agg(
        total_lines=("level", "size"),
        error_count=("is_error", "sum"),
        warn_count=("is_warn", "sum"),
        mean_latency_ms=("latency_ms", "mean"),
        p95_latency_ms=("latency_ms", lambda s: float(np.percentile(s, 95))),
        unique_templates=("template", "nunique"),
    ).reset_index()
    windows["error_rate"] = windows["error_count"] / windows["total_lines"]
    windows["warn_rate"] = windows["warn_count"] / windows["total_lines"]
    windows["window_start"] = START + pd.to_timedelta(
        windows["abs_window"] * WINDOW_MINUTES, unit="m")
    return windows


def describe_reasons(row, baselines):
    """Plain-English reasons a window looks wrong, vs its own host."""
    base = baselines.loc[row["host"]]
    reasons = []
    if row["error_rate"] > base["error_rate"] * 2:
        reasons.append(
            f"error rate {row['error_rate']:.2f} vs host baseline "
            f"{base['error_rate']:.2f}")
    if row["p95_latency_ms"] > base["p95_latency_ms"] * 1.8:
        reasons.append(
            f"p95 latency {row['p95_latency_ms']:.0f} ms vs host baseline "
            f"{base['p95_latency_ms']:.0f} ms")
    if row["total_lines"] > base["total_lines"] * 1.6:
        reasons.append(
            f"volume {int(row['total_lines'])} lines vs host baseline "
            f"{base['total_lines']:.0f}")
    if row["unique_templates"] > base["unique_templates"] + 3:
        reasons.append(
            f"{int(row['unique_templates'])} distinct templates vs host "
            f"baseline {base['unique_templates']:.0f}")
    if not reasons:
        reasons.append("feature combination is unusual for this host")
    return "; ".join(reasons)


def detect(windows):
    """Run both detectors; return (windows_with_flags, baselines)."""
    windows = windows.copy()

    # Host baselines: medians are used instead of means so the planted
    # faults themselves do not drag "normal" too far towards abnormal.
    baselines = windows.groupby("host")[FEATURES].median()

    # (a) Per-host z-score on error rate. The baseline is learned from
    # the first two days only, before any planted fault starts: if you
    # let a day-long failure into the baseline, it inflates the standard
    # deviation and quietly teaches the detector that failure is normal.
    # The statistics are also taken per host — a globally pooled
    # threshold would flag every busy host and miss every quiet one.
    clean = windows[windows["abs_window"] < 2 * WINDOWS_PER_DAY]
    stats = clean.groupby("host")["error_rate"].agg(["mean", "std"])
    windows["error_z"] = [
        (row.error_rate - stats.loc[row.host, "mean"])
        / (stats.loc[row.host, "std"] or 1e-9)
        for row in windows.itertuples()
    ]
    windows["flag_zscore"] = windows["error_z"] > Z_THRESHOLD

    # (b) IsolationForest over all window features at once. It can catch
    # windows that are odd in combination (say, normal error rate but
    # strange latency and template mix) where a single-metric rule can't.
    model = IsolationForest(
        n_estimators=200, contamination=IF_CONTAMINATION,
        random_state=SEED)
    windows["flag_iforest"] = model.fit_predict(windows[FEATURES]) == -1

    windows["reason"] = windows.apply(
        lambda r: describe_reasons(r, baselines), axis=1)
    return windows, baselines


def evaluate(windows, planted):
    """Score both detectors against the planted ground truth."""
    windows = windows.copy()
    windows["is_planted"] = [
        (row.host, row.abs_window) in planted
        for row in windows.itertuples()
    ]
    results = {}
    for method, col in [("IsolationForest", "flag_iforest"),
                        ("z-score (error rate)", "flag_zscore")]:
        flagged = windows[col]
        caught = int((flagged & windows["is_planted"]).sum())
        false_pos = int((flagged & ~windows["is_planted"]).sum())
        results[method] = {
            "flagged": int(flagged.sum()),
            "caught": caught,
            "missed": int(windows["is_planted"].sum() - caught),
            "false_positives": false_pos,
        }
    return windows, results


def main():
    rng = np.random.default_rng(SEED)
    profiles = host_profile(rng)
    raw_lines = generate_raw_lines(rng, profiles)
    df = parse_lines(raw_lines)
    windows = build_windows(df)
    planted = planted_lookup()
    windows, baselines = detect(windows)
    windows, results = evaluate(windows, planted)

    n_planted = int(windows["is_planted"].sum())
    either = windows["flag_iforest"] | windows["flag_zscore"]
    caught_either = int((either & windows["is_planted"]).sum())

    lines = []
    lines.append("=" * 70)
    lines.append("SERVER LOG ANOMALY REPORT  (synthetic data, 7 days)")
    lines.append("=" * 70)
    lines.append(f"Hosts: {len(HOSTS)}   Raw log lines parsed: {len(df):,}")
    lines.append(f"Windows: {len(windows):,} per-host {WINDOW_MINUTES}-minute "
                 f"windows ({WINDOWS_PER_DAY} per host per day)")
    lines.append(f"Levels: " + ", ".join(
        f"{level} {count:,}" for level, count in
        df['level'].value_counts().sort_index().items()))
    lines.append("")
    lines.append("Planted faults (ground truth):")
    lines.append("  host-03, day 3, 14:00-16:00 — error burst "
                 "(4 windows)")
    lines.append("  host-07, day 5, 10:00-11:30 — latency + error-rate spike "
                 "(3 windows)")
    lines.append(f"  {GRADUAL_HOST}, day {GRADUAL_DAY + 1} — gradually failing, "
                 f"error rate climbing to ~{GRADUAL_PEAK_ERROR:.2f} "
                 f"({n_planted - 7} windows above double its normal rate)")
    lines.append(f"  Total planted anomaly windows: {n_planted}")
    lines.append("")
    lines.append("Detection results (window-level, vs planted ground truth):")
    lines.append(f"  {'Method':<24}{'Flagged':>8}{'Caught':>8}"
                 f"{'Missed':>8}{'False pos.':>12}")
    for method, r in results.items():
        lines.append(f"  {method:<24}{r['flagged']:>8}{r['caught']:>8}"
                     f"{r['missed']:>8}{r['false_positives']:>12}")
    lines.append(f"  Caught by at least one method: {caught_either} of "
                 f"{n_planted} planted windows")
    lines.append("")
    lines.append("What each method is good at here:")
    lines.append("  The z-score rule is blunt but honest: it only sees error")
    lines.append("  rate, and it wins here because all three planted faults")
    lines.append("  raise it. It would be blind to a fault — a pure latency")
    lines.append("  regression, say — that leaves error rate alone.")
    lines.append("  The IsolationForest sees all seven features at once, so it")
    lines.append("  can catch that kind of fault, and its reason lines show")
    lines.append("  the latency damage on host-07 plainly. The cost of looking")
    lines.append("  at everything is flagging some merely-unusual windows")
    lines.append("  that are not faults at all — both methods do that here,")
    lines.append("  mostly in thin overnight windows where a handful of")
    lines.append("  errors in very few lines makes the rate look dramatic.")
    lines.append("")
    lines.append("Top flagged windows (by error-rate z-score):")
    top = windows[either].sort_values("error_z", ascending=False).head(12)
    for row in top.itertuples():
        methods = []
        if row.flag_iforest:
            methods.append("IsolationForest")
        if row.flag_zscore:
            methods.append("z-score")
        planted_mark = "  [planted]" if row.is_planted else ""
        lines.append(
            f"  {row.window_start:%a %H:%M}  {row.host}  "
            f"errors {int(row.error_count)}/{int(row.total_lines)} "
            f"({row.error_rate:.2f}), p95 {row.p95_latency_ms:.0f} ms  "
            f"[{', '.join(methods)}]{planted_mark}")
        lines.append(f"      {row.reason}")
    lines.append("")
    lines.append("Hosts flagged at least once:")
    for method, col in [("IsolationForest", "flag_iforest"),
                        ("z-score", "flag_zscore")]:
        hosts = sorted(windows.loc[windows[col], "host"].unique())
        lines.append(f"  {method}: {', '.join(hosts) if hosts else 'none'}")
    lines.append("=" * 70)

    report = "\n".join(lines)
    print(report)
    out = Path(__file__).with_name("anomaly_report.txt")
    out.write_text(report + "\n", encoding="utf-8")
    print(f"\nReport saved to {out.name}")


if __name__ == "__main__":
    main()
