#!/usr/bin/env bash
cd /home/claude/ep-eeg
xargs -P 4 -L 1 bash runs/M14/grid/one.sh < runs/M14/grid/jobs.txt
echo QUEUE_DONE
