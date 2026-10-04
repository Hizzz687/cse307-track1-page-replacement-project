# 3-5 minute walkthrough script

## 0:00-0:30 - Goal

“My project is Track 1, learned page replacement. I compare FIFO, LRU, and Belady's
Optimal with a small decision-tree policy. The important stress test is a workload
distribution shift halfway through the trace.”

## 0:30-1:30 - Code

Open `src/page_replacement.py` and point out:

1. `simulate()` contains the common hit/miss loop.
2. FIFO removes the oldest page; LRU updates an order list on every hit.
3. Optimal searches for the next use of every resident page.
4. `train_learned_policy()` creates features and labels from Optimal.
5. The learned policy scores resident candidates using recency, frequency, and page ID.

## 1:30-2:15 - Workload

Open `run_experiment.py`. Explain that the first 1,200 requests mostly use pages 0-7
with periodic sequential scans. The final 1,200 requests use short bursts on pages
16-31 plus uniformly random requests. Seed 307 makes the result reproducible.

## 2:15-3:15 - Run and results

Run:

```powershell
python run_experiment.py
```

Show `results/metrics.csv` and `results/hit_ratio.svg`. Explain the before/after columns.
Belady's Optimal is the reference upper bound. Compare how the online policies respond
after the shift, especially whether FIFO loses its advantage or LRU adapts faster.

## 3:15-4:00 - Conclusion

“The experiment shows why no fixed heuristic is always best. A learned policy can combine
signals such as recency and frequency, but it inherits the limits of its training data.
The practical design is therefore a transparent classical baseline plus a measured,
lightweight adaptive layer.”

## Likely questions

- **Why include Optimal?** It is an offline upper-bound reference, not deployable online.
- **Why not use a large ML model?** The task asks for a lightweight learned component;
  an interpretable tree keeps the OS decision understandable.
- **What changes at the shift?** The previously useful locality/frequency signal becomes
  less reliable when references spread across new pages and bursts.
- **How is AI disclosed?** The README states that an AI coding assistant helped with
  implementation and documentation, while the experiment design and interpretation are
  part of the submitted work.
