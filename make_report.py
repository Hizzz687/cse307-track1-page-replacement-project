"""Generate the 2-3 page written report from the experiment CSV."""

from __future__ import annotations

import csv
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle, PageBreak

ROOT = Path(__file__).parent
REPORT = ROOT / "report"
CSV_PATH = ROOT / "results" / "metrics.csv"


def pct(value: str) -> str:
    return f"{float(value) * 100:.2f}%"


def main() -> None:
    if not CSV_PATH.exists():
        raise SystemExit("Run run_experiment.py before generating the report.")
    REPORT.mkdir(exist_ok=True)
    rows = list(csv.DictReader(CSV_PATH.open(encoding="utf-8")))
    by_key = {(row["policy"], row["segment"]): row for row in rows}
    output = REPORT / "Mim_Mahizur_Rahman_CSE307_Track1_Report.pdf"
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitleCenter", parent=styles["Title"], alignment=TA_CENTER, spaceAfter=12))
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=9, leading=11))
    doc = SimpleDocTemplate(str(output), pagesize=A4, rightMargin=42, leftMargin=42, topMargin=40, bottomMargin=40)
    story: list[object] = []
    story.append(Paragraph("Learning-Augmented Page Replacement", styles["TitleCenter"]))
    story.append(Paragraph("CSE-307: Operating Systems - Term Paper, Track 1", styles["Heading2"]))
    story.append(Paragraph("<b>Student:</b> Mim Mahizur Rahman &nbsp;&nbsp; <b>ID:</b> 202414402", styles["BodyText"]))
    story.append(Spacer(1, 10))
    story.append(Paragraph("1. Problem framing", styles["Heading2"]))
    story.append(Paragraph(
        "FIFO and LRU are useful online page-replacement policies, but each encodes a fixed "
        "assumption about locality. FIFO ignores recent use, while LRU assumes that recently "
        "used pages will remain useful. This paper studies how those assumptions behave when "
        "a workload changes from locality-heavy/sequential accesses to random and bursty accesses. "
        "Belady's Optimal policy is included as an offline lower-bound reference. The goal is not "
        "to replace classical algorithms completely, but to test whether a small learned layer can "
        "choose better eviction candidates using observable history features.", styles["BodyText"]))
    story.append(Paragraph("2. Implementation summary", styles["Heading2"]))
    story.append(Paragraph(
        "The simulator models a cache of eight frames and 32 virtual pages. A page hit updates "
        "the policy state without causing a fault. On a miss with a full cache, FIFO evicts the "
        "oldest resident page, LRU evicts the least recently used page, and Optimal evicts the "
        "resident page whose next use is farthest in the future. The learned policy uses a small "
        "decision tree. Its candidate features are recency (requests since last use), frequency "
        "in the observed prefix, and page identifier. During training, each candidate is labeled "
        "as a good victim when it is the victim selected by Optimal. During testing, the learned "
        "policy scores each resident candidate and evicts the candidate with the highest predicted "
        "victim probability.", styles["BodyText"]))
    story.append(Spacer(1, 8))
    story.append(Paragraph("3. Experimental setup", styles["Heading2"]))
    story.append(Paragraph(
        "The test trace contains 2,400 requests and uses deterministic random seed 307. "
        "Requests 0-1,199 are locality-heavy: most accesses target pages 0-5, with periodic "
        "sequential scans across pages 0-7. Requests 1,200-2,399 form the stress phase: 70% "
        "of events belong to short bursts on pages 16-31 and the remainder are uniformly random "
        "over all pages. This creates a clear distribution shift while keeping the experiment "
        "small enough to inspect. Every policy sees exactly the same trace and cache capacity. "
        "The primary metrics are page-fault count and hit ratio, calculated separately before "
        "and after the shift.", styles["BodyText"]))
    story.append(Spacer(1, 8))
    story.append(Image(str(ROOT / "results" / "hit_ratio.png"), width=6.6 * inch, height=3.8 * inch))
    story.append(Paragraph("Figure 1. Hit ratio in the two workload phases.", styles["Small"]))
    story.append(PageBreak())
    story.append(Paragraph("4. Results", styles["Heading2"]))
    table_data = [["Policy", "Before faults", "Before hit", "After faults", "After hit"]]
    for policy in ("fifo", "lru", "optimal", "learned"):
        before = by_key[(policy, "before_shift")]
        after = by_key[(policy, "after_shift")]
        table_data.append([policy.upper(), before["faults"], pct(before["hit_ratio"]), after["faults"], pct(after["hit_ratio"])])
    table = Table(table_data, colWidths=[1.2 * inch, 1.0 * inch, 1.0 * inch, 1.0 * inch, 1.0 * inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.HexColor("#e5e7eb")]),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 7),
    ]))
    story.append(table)
    story.append(Spacer(1, 12))
    story.append(Paragraph(
        "The numerical table should be read as a phase comparison rather than as a claim that "
        "the learned policy is universally optimal. Optimal is expected to lead because it can "
        "see the future. The more meaningful comparison is whether the learned policy remains "
        "competitive with FIFO and LRU after the trace changes. The distribution shift removes "
        "the reliability of the first-half locality signal: pages that were repeatedly useful "
        "become poor choices to retain once the second phase begins. LRU can adapt because each "
        "new reference changes recency, while FIFO is insensitive to references between evictions. "
        "The learned policy can use frequency and recency together, but it is still limited by "
        "training labels and by the fact that future behavior is not directly observable. "
        "Therefore, a learned layer is best viewed as an adaptive heuristic with a classical "
        "policy as a transparent baseline, not as a replacement for correctness guarantees.", styles["BodyText"]))
    story.append(Paragraph("5. Limitations and conclusion", styles["Heading2"]))
    story.append(Paragraph(
        "This is a synthetic experiment: real systems have dirty pages, write-back costs, TLB "
        "effects, multi-level caches, and non-stationary patterns more complex than this trace. "
        "The learned model is intentionally small and interpretable, and it is trained from "
        "Optimal's labels, which is not available online. Even with these limitations, the "
        "experiment demonstrates the central OS idea: policy quality depends on workload behavior. "
        "Classical algorithms provide strong, understandable baselines; a lightweight predictor "
        "can add adaptivity, but its value must be measured across the workload shift rather than "
        "only on a stationary trace.", styles["BodyText"]))
    story.append(Spacer(1, 10))
    story.append(Paragraph("Reproducibility: run `python run_experiment.py` followed by `python make_report.py`.", styles["Small"]))
    doc.build(story)
    print(output)


if __name__ == "__main__":
    main()
