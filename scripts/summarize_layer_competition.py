#!/usr/bin/env python3
"""Print compact diagnostics: python scripts/summarize_layer_competition.py "$RUN"."""
import argparse
import json
from pathlib import Path


def summarize(root):
    report = json.loads((root / "report.json").read_text(encoding="utf-8"))
    lines = [f"RUN: {root.name}", f"SCHEMA: {report['schema']}",
             f"Valid original pools: {report['valid_original_pools']}/{len(report['examples'])}",
             f"Valid expanded pools: {report['valid_expanded_pools']}/{len(report['examples'])}"]
    for example in report["examples"]:
        lines.extend(["", f"{example['id']}: {example['question']}",
                      f"Anchor: {example['anchor']!r}", f"Candidates: {example['candidates']}",
                      f"Unusable/budget-omitted generations: {example['invalid_or_truncated_generations']}"])
        for pool_name in ("original_pool", "expanded_pool"):
            pool = example[pool_name]
            if not pool["valid"]:
                lines.append(f"  {pool_name}: unavailable ({pool['reason']})")
                continue
            lines.append(f"  {pool_name}: candidate indices {pool['candidate_indices']}")
            for metric, result in pool["metrics"].items():
                original = result["original"]
                specific = result["edited_detail_specificity"]
                lines.append(f"    {metric}: original late margin={original['late_target_margin']:.5g}, "
                             f"detail late margin={specific['late_target_margin']:.5g}, "
                             f"detail win fraction={specific['target_win_fraction']:.3f}, "
                             f"detail switches={specific['winner_switches']}")
                if metric == "mean_logprob":
                    controls = [f"{v['kind']}={v['original_minus_control']['late_target_margin']:.5g}"
                                for v in result["controls"]]
                    lines.append("    Paired-control late margins: " + ", ".join(controls))
    lines.append("\nMargins measure relative support; they are not correctness probabilities.")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    print(summarize(parser.parse_args().run))
