#!/bin/bash
# Run the commands of a jobs file one after another (one command per line; blank lines and # comments skipped).
# Each job's output goes to logs/<jobs file name>_<line number>.log; the queue keeps going if a job fails.
# The GPU is chosen by the caller, e.g.
#   CUDA_VISIBLE_DEVICES=0 nohup bash scripts/run_queue.sh jobs.txt > queue.log 2>&1 &
set -u
jobs=${1:?usage: run_queue.sh JOBS_FILE}
mkdir -p logs
tag=$(basename "$jobs"); tag=${tag%.*}
n=0
while IFS= read -r line || [ -n "$line" ]; do
  n=$((n + 1))
  case "$line" in ''|\#*) continue ;; esac
  echo "[$(date '+%F %T')] start $n: $line"
  bash -c "$line" > "logs/${tag}_$n.log" 2>&1 < /dev/null
  rc=$?
  if [ $rc -eq 0 ]; then
    echo "[$(date '+%F %T')] done  $n"
  else
    echo "[$(date '+%F %T')] FAILED $n (exit $rc, see logs/${tag}_$n.log)"
  fi
done < "$jobs"
echo "[$(date '+%F %T')] queue finished"
