#!/usr/bin/env python3
"""Apply the frozen runs/opus55_round/protocol.json decision rules to dev_v2 results."""
import argparse
import csv
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--x1d2", type=Path, required=True)
    ap.add_argument("--a7m3", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    x = json.loads(args.x1d2.read_text())
    comps = [c for c in x["comparisons"] if "subset" not in c]
    def get(a, b, s, metric="model_psnr", src=comps):
        for c in src:
            if c["a"] == a and c["b"] == b and c["slice"] == s and c["metric"] == metric:
                return c
        return None
    slices = list(dict.fromkeys(c["slice"] for c in comps))
    floor = {s: abs(get("balanced_rep", "balanced", s)["mean_delta_db"]) for s in slices}
    def counts(a, b, s):
        c = get(a, b, s)
        lo, hi = c["ci95"]
        return {**c, "noise_floor_x2": 2*floor[s],
                "counts": abs(c["mean_delta_db"]) > 2*floor[s] and (lo > 0 or hi < 0)}
    arms = ["nm@estimated", "nm@oracle", "nm@half", "nm@double", "tails", "nm_tails@estimated",
            "nm_tails@oracle", "balanced_rep", "shadow", "control", "baseline"]
    table = []
    for s in slices:
        for a in arms:
            for ref in ("balanced", "control"):
                if a != ref and get(a, ref, s):
                    c = counts(a, ref, s)
                    table.append({"slice": s, "arm": a, "reference": ref, "mean_delta_db": c["mean_delta_db"],
                                  "sem": c["sem"], "ci95_lo": c["ci95"][0], "ci95_hi": c["ci95"][1],
                                  "sources": c["sources"], "sources_arm_worse": c["sources_a_worse"],
                                  "noise_floor_x2": c["noise_floor_x2"], "exceeds_floor_and_ci": c["counts"]})
    B, S = "native|1|low", "native|0.015625|high"
    rules = {
        "conditioning_bright": {"rule": "nm@estimated - control on native|1|low >= -0.10 dB",
                                "value": get("nm@estimated", "control", B)["mean_delta_db"]},
        "conditioning_shadow": {"rule": "nm@estimated - balanced on native|0.015625|high >= -0.30 dB",
                                "value": get("nm@estimated", "balanced", S)["mean_delta_db"]},
        "tails_slices": {"rule": "tails - balanced > floor on both tails slices",
                         "values": {s: counts("tails", "balanced", s) for s in ("native|0.0625|tails", "native|0.015625|tails")}},
        "tails_guards": {"rule": "tails - balanced >= -0.10 on native|1|low and native|0.015625|high",
                         "values": {s: get("tails", "balanced", s)["mean_delta_db"] for s in (B, S)}},
    }
    rules["conditioning_bright"]["pass"] = rules["conditioning_bright"]["value"] >= -.10
    rules["conditioning_shadow"]["pass"] = rules["conditioning_shadow"]["value"] >= -.30
    rules["tails_slices"]["pass"] = all(v["counts"] and v["mean_delta_db"] > 0 for v in rules["tails_slices"]["values"].values())
    rules["tails_guards"]["pass"] = all(v >= -.10 for v in rules["tails_guards"]["values"].values())
    out = {"noise_floor_abs_rep_minus_balanced_db": floor, "rules": rules,
           "estimator_median_std_ratio": x["estimator_median_std_ratio"]}
    if args.a7m3:
        a = json.loads(args.a7m3.read_text())
        ac = [c for c in a["comparisons"] if "subset" not in c and c["metric"] == "model_psnr"]
        out["a7m3_balanced_vs_equal_budget_control"] = {c["slice"]: {k: c[k] for k in ("mean_delta_db", "sem", "sources", "sources_a_worse")}
                                                        for c in ac if c["a"] == "balanced" and c["b"] == "control"}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/"decision.json").write_text(json.dumps(out, indent=1)+"\n")
    with open(args.output/"paired_comparisons.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
