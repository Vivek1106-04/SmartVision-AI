"""Score the app's repetition counter against the reference annotation.

Joins what the rule-based evaluators counted during the offline replay
(results/repcount_system.csv, produced by tools/bench/replay.mjs) against the
signal-derived reference (results/repcount_reference.csv) and reports the two
numbers the report needs: mean absolute error and exact-match rate.

The reference depends on how shallow a movement still counts as a repetition,
so the score is also recomputed at a looser and a stricter prominence setting.
If the verdict only holds at one setting it is not a verdict.

Outputs:
  results/repcount_accuracy.csv    - per-clip counted, reference and error
  results/repcount_sensitivity.csv - the same score at each prominence setting
"""

import argparse
import os

import pandas as pd


def score(merged: pd.DataFrame) -> pd.DataFrame:
    """Per-exercise and overall error between counted and reference reps."""
    def summarise(frame: pd.DataFrame, label: str) -> dict:
        error = frame["reps_counted"] - frame["reference_reps"]
        return {
            "exercise": label,
            "clips": len(frame),
            "reference_reps": int(frame["reference_reps"].sum()),
            "counted_reps": int(frame["reps_counted"].sum()),
            "mae": round(error.abs().mean(), 3),
            "mean_error": round(error.mean(), 3),
            "exact_match_rate": round((error == 0).mean(), 3),
            "within_one_rate": round((error.abs() <= 1).mean(), 3),
        }

    rows = [summarise(group, name) for name, group in merged.groupby("exercise")]
    rows.append(summarise(merged, "all"))
    return pd.DataFrame(rows)


def sensitivity(system: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Overall score recomputed against each prominence setting."""
    columns = [c for c in reference.columns if c.startswith("reference_reps_p")]
    rows = []
    for column in sorted(columns):
        merged = system.merge(reference[["video", column]], on="video")
        merged = merged.rename(columns={column: "reference_reps"})
        overall = score(merged).query("exercise == 'all'").iloc[0].to_dict()
        overall["prominence_deg"] = int(column.rsplit("p", 1)[1])
        rows.append(overall)
    frame = pd.DataFrame(rows)
    return frame[["prominence_deg"] + [c for c in frame.columns if c != "prominence_deg"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_dir", default="results")
    args = parser.parse_args()

    system = pd.read_csv(os.path.join(args.results_dir, "repcount_system.csv"))
    reference = pd.read_csv(os.path.join(args.results_dir, "repcount_reference.csv"))
    merged = system.merge(reference[["video", "reference_reps"]], on="video", how="inner")

    if len(merged) != len(system):
        missing = set(system["video"]) - set(merged["video"])
        raise SystemExit(f"No reference count for: {sorted(missing)}")

    summary = score(merged)
    per_clip = merged.assign(error=merged["reps_counted"] - merged["reference_reps"])

    out_path = os.path.join(args.results_dir, "repcount_accuracy.csv")
    per_clip.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")
    print()
    print(summary.to_string(index=False))

    spread = sensitivity(system, reference)
    spread_path = os.path.join(args.results_dir, "repcount_sensitivity.csv")
    spread.to_csv(spread_path, index=False)
    print(f"\nWrote {spread_path}")
    print()
    print(spread.drop(columns=["exercise"]).to_string(index=False))


if __name__ == "__main__":
    main()
