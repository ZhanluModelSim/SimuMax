"""Overlap observability report generation."""

import json
import os
from datetime import datetime
from typing import Optional

from simumax.core.des_engine import OverlapSummary, ResourceType


class OverlapReport:

    @staticmethod
    def to_dict(summary: OverlapSummary) -> dict:
        report = {
            "global": {
                "total_compute_time_ms": round(summary.total_compute_time, 6),
                "total_comm_time_ms": round(summary.total_comm_time, 6),
                "overlapped_comm_time_ms": round(
                    summary.total_overlapped_comm_time, 6
                ),
                "exposed_comm_time_ms": round(
                    summary.total_exposed_comm_time, 6
                ),
                "overlap_ratio": f"{summary.overall_overlap_ratio:.1%}",
                "compute_utilization": f"{summary.compute_utilization:.1%}",
                "intra_link_utilization": (
                    f"{summary.intra_link_utilization:.1%}"
                ),
                "inter_link_utilization": (
                    f"{summary.inter_link_utilization:.1%}"
                ),
                "iteration_time_ms": round(summary.iteration_time, 6),
            },
            "per_module": {},
            "per_comm_type": {},
        }

        for path, stats in sorted(
            summary.per_module.items(),
            key=lambda x: (
                x[1].fwd_exposed_time + x[1].bwd_exposed_time
            ),
            reverse=True,
        ):
            report["per_module"][path] = {
                "fwd_compute_ms": round(stats.fwd_compute_time, 6),
                "fwd_comm_ms": round(stats.fwd_comm_time, 6),
                "fwd_overlapped_ms": round(stats.fwd_overlapped_time, 6),
                "fwd_exposed_ms": round(stats.fwd_exposed_time, 6),
                "fwd_overlap_ratio": f"{stats.fwd_overlap_ratio:.1%}",
                "bwd_compute_ms": round(stats.bwd_compute_time, 6),
                "bwd_comm_ms": round(stats.bwd_comm_time, 6),
                "bwd_overlapped_ms": round(stats.bwd_overlapped_time, 6),
                "bwd_exposed_ms": round(stats.bwd_exposed_time, 6),
                "bwd_overlap_ratio": f"{stats.bwd_overlap_ratio:.1%}",
            }

        for ct, stats in sorted(
            summary.per_comm_type.items(),
            key=lambda x: x[1].exposed_time,
            reverse=True,
        ):
            report["per_comm_type"][ct] = {
                "total_time_ms": round(stats.total_time, 6),
                "overlapped_time_ms": round(stats.overlapped_time, 6),
                "exposed_time_ms": round(stats.exposed_time, 6),
                "overlap_ratio": f"{stats.overlap_ratio:.1%}",
            }

        return report

    @staticmethod
    def generate(
        summary: OverlapSummary,
        output_dir: Optional[str] = None,
        filename: str = "overlap_report.json",
    ):
        if output_dir is None:
            ts_dir = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(os.getcwd(), "output", ts_dir)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)
        report = OverlapReport.to_dict(summary)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

    @staticmethod
    def print_summary(summary: OverlapSummary):
        print("=" * 64)
        print("  Overlap Summary")
        print("=" * 64)
        print(
            f"  Compute utilization:     "
            f"{summary.compute_utilization:.1%}"
        )
        print(
            f"  Intra-link utilization:  "
            f"{summary.intra_link_utilization:.1%}"
        )
        print(
            f"  Inter-link utilization:  "
            f"{summary.inter_link_utilization:.1%}"
        )
        print(
            f"  Overall overlap ratio:   "
            f"{summary.overall_overlap_ratio:.1%}"
        )
        print(
            f"  Total compute time:      "
            f"{summary.total_compute_time:.4f} ms"
        )
        print(
            f"  Total comm time:         "
            f"{summary.total_comm_time:.4f} ms"
        )
        print(
            f"  Overlapped comm time:    "
            f"{summary.total_overlapped_comm_time:.4f} ms"
        )
        print(
            f"  Exposed comm time:       "
            f"{summary.total_exposed_comm_time:.4f} ms"
        )
        if summary.iteration_time > 0:
            print(
                f"  Iteration time:          "
                f"{summary.iteration_time:.4f} ms"
            )
        print("-" * 64)

        sorted_modules = sorted(
            summary.per_module.items(),
            key=lambda x: (
                x[1].fwd_exposed_time + x[1].bwd_exposed_time
            ),
            reverse=True,
        )
        exposed_modules = [
            (p, s) for p, s in sorted_modules
            if s.fwd_exposed_time + s.bwd_exposed_time > 0
        ]
        if exposed_modules:
            print("  Top exposed modules:")
            for path, stats in exposed_modules[:10]:
                total_exp = stats.fwd_exposed_time + stats.bwd_exposed_time
                print(
                    f"    {path}: {total_exp:.4f} ms exposed "
                    f"(fwd {stats.fwd_overlap_ratio:.0%} / "
                    f"bwd {stats.bwd_overlap_ratio:.0%} overlapped)"
                )
        else:
            print("  No exposed communication detected.")

        if summary.per_comm_type:
            print("-" * 64)
            print("  Communication breakdown:")
            for ct, stats in sorted(
                summary.per_comm_type.items(),
                key=lambda x: x[1].exposed_time,
                reverse=True,
            ):
                print(
                    f"    {ct}: total={stats.total_time:.4f} ms, "
                    f"overlapped={stats.overlapped_time:.4f} ms, "
                    f"exposed={stats.exposed_time:.4f} ms "
                    f"({stats.overlap_ratio:.0%} overlapped)"
                )
        print("=" * 64)
