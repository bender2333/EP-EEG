import sys
from ep_eeg.ledger import append
for line in open(sys.argv[1]):
    if line.strip():
        d, st, desc = line.rstrip("\n").split("|"); append(d + "/summary.json", st, desc)
