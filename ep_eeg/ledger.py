"""Append one row to results.tsv (untracked) from a summary.json."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
HDR = "commit\tstage\tarm\tkappa_val\tacc_val\tsettle_rate\tseeds\twall_s\tstatus\tdescription\n"
def append(summary_path, status, desc):
    s = json.loads(Path(summary_path).read_text()); f = ROOT / "results.tsv"
    if not f.exists(): f.write_text(HDR)
    row = [s["commit"], s["stage"], s["arm"], f'{s["kappa_val"]:.4f}', f'{s["acc_val"]:.4f}',
           f'{s["settle_rate"]:.4f}', str(len(s["seeds"])), f'{s["wall_s"]:.0f}', status, desc]
    with f.open("a") as h: h.write("\t".join(row) + "\n")
if __name__ == "__main__":
    append(sys.argv[1], sys.argv[2], sys.argv[3])
