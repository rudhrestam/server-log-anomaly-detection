# Server Log Anomaly Detection (Demo)

Most outages don't start with a server falling over. They start with one host throwing a few more errors than usual, another getting quietly slower, a third getting worse by the hour while every dashboard still looks mostly green. The hard part is rarely collecting the logs — it's noticing the change early, on the right host, with enough evidence attached that the person on call believes the alert. This project builds that situation from scratch: a week of synthetic server logs, three different faults planted in known places, and two detectors scored honestly on whether they find them.

> This is a personal portfolio project exploring log anomaly detection. It is built on 100% synthetic data with original code, and does not represent work done for any employer. It is a generic demo, not a model of any real system.

## Why per-host baselines matter

Fixed thresholds fail in both directions. Set an alert at "error rate above 10%" and you'll miss a quiet host that normally runs at 2% and has tripled; set it at 5% and a naturally noisy host will page you every week until everyone starts ignoring it. Each host has its own normal — its own volume, its own error rate, its own latency — so each host has to be judged against itself. Both detectors in this demo do that, in different ways, and the comparison between them is the interesting part.

There's a second trap, and the demo is careful about it: what you learn "normal" from. If a day-long failure is part of your baseline data, it inflates the spread and teaches the detector that failure looks normal. The z-score detector here learns each host's baseline from the first two days only, before any planted fault starts.

## What the demo does

`anomaly_demo.py` generates raw log lines for 10 hosts over 7 days — each line a timestamp, host, level (INFO/WARN/ERROR), latency in ms, and a message picked from a small set of generic templates — with volume following a diurnal curve (busy mid-morning and mid-afternoon, thin overnight). Three faults are planted:

1. **An error burst** — host-03 throws errors at a ~45% rate for two hours on day 3, at 1.6× its normal volume.
2. **A latency + error-rate spike** — host-07 runs about 4.5× slower with a ~28% error rate for 90 minutes on day 5.
3. **A host gradually failing** — host-05's error rate climbs linearly all through day 6, ending near 0.35. The first hours of the slide are barely distinguishable from noise, so the ground truth only counts the windows where its intended error rate has at least doubled — 39 of the day's 48 windows.

The script then works the way a real pipeline would:

1. **Parse** the raw lines into a structured DataFrame.
2. **Aggregate** into per-host, per-30-minute windows with seven features: total lines, error count, error rate, warn rate, mean latency, p95 latency, and unique message templates.
3. **Detect, two ways:**
   - *Per-host z-score* on error rate (baseline from the first two days, flag at z > 3.5) — simple, and every alert explains itself.
   - *IsolationForest* over all seven window features at once — it can catch faults that no single metric shows, like a pure latency regression.
4. **Explain** each flagged window against its own host's baseline (e.g. "error rate 0.49 vs host baseline 0.02; p95 latency 1,265 ms vs host baseline 190 ms").
5. **Evaluate** both methods window-by-window against the planted ground truth: caught, missed, and false positives on normal windows.

## Example output

Actual output from a run (fixed seed 3, so you'll get the same numbers):

```
Hosts: 10   Raw log lines parsed: 190,559
Windows: 3,360 per-host 30-minute windows (48 per host per day)
Levels: ERROR 7,181, INFO 165,065, WARN 18,313

Planted faults (ground truth):
  host-03, day 3, 14:00-16:00 — error burst (4 windows)
  host-07, day 5, 10:00-11:30 — latency + error-rate spike (3 windows)
  host-05, day 6 — gradually failing, error rate climbing to ~0.35 (39 windows above double its normal rate)
  Total planted anomaly windows: 46

Detection results (window-level, vs planted ground truth):
  Method                   Flagged  Caught  Missed  False pos.
  IsolationForest               51      39       7          12
  z-score (error rate)          55      42       4          13
  Caught by at least one method: 42 of 46 planted windows

Top flagged windows (by error-rate z-score):
  Wed 14:30  host-03  errors 62/126 (0.49), p95 230 ms  [IsolationForest, z-score]  [planted]
      error rate 0.49 vs host baseline 0.02; volume 126 lines vs host baseline 69; 18 distinct templates vs host baseline 13
  Fri 10:30  host-07  errors 32/98 (0.33), p95 1265 ms  [IsolationForest, z-score]  [planted]
      error rate 0.33 vs host baseline 0.03; p95 latency 1265 ms vs host baseline 190 ms
  Sat 23:30  host-05  errors 21/49 (0.43), p95 474 ms  [IsolationForest, z-score]  [planted]
      error rate 0.43 vs host baseline 0.04; p95 latency 474 ms vs host baseline 142 ms; 17 distinct templates vs host baseline 13
```

Two honest readings of those numbers. First, the simple method wins here: the z-score catches 42 of 46 planted windows because all three planted faults happen to raise error rate — it would be blind to a fault that didn't, which is exactly the case for keeping the IsolationForest around. Second, both methods produce a dozen-ish false positives, mostly thin overnight windows where a handful of errors in very few lines makes the rate look dramatic. A production version needs a minimum-volume guard; the demo leaves the false positives visible rather than tuning them away, because they're the part of anomaly detection that actually pages people at 3 a.m.

## Run it

```bash
pip install -r requirements.txt
python anomaly_demo.py
```

Prints the anomaly report and saves a copy to `anomaly_report.txt`.

## Tech stack

Python · NumPy · pandas · scikit-learn (IsolationForest)

## Ideas for extension

- Add a minimum-volume rule (or a Bayesian-smoothed error rate) so thin overnight windows stop producing dramatic-looking rates from four errors in nineteen lines.
- Plant a pure latency regression that leaves error rate untouched, and watch the z-score miss it entirely while the IsolationForest catches it — the cleanest possible demonstration of why multi-feature detection earns its complexity.
- Replace the fixed two-day baseline with a rolling baseline that excludes flagged windows, so "normal" tracks slow drift (deployments, growth) without absorbing failures.
- Stream the windows instead of batching them: score each 30-minute window as it closes and measure detection delay — how many windows into the gradual failure each method takes to fire — since for the slow fault, *when* you catch it matters as much as *whether*.
