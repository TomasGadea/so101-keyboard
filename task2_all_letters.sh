#!/usr/bin/env bash
set -u
set -o pipefail

# Run run_eval_2.sh for every letter and measure wall-clock time.
#
# Default order is keyboard order. You can override it by passing letters:
#   ./task2_all_letters.sh A B C
#
# Optional env vars:
#   STOP_ON_FAIL=1      stop immediately if one letter fails
#   PAUSE_SECONDS=0.5   sleep between letters
#   RUN_DIR=path        where logs/csv files are written

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
TASK2="$ROOT_DIR/run_eval_2.sh"

if [ ! -x "$TASK2" ]; then
  echo "Expected executable task script at: $TASK2" >&2
  exit 1
fi

if [ "$#" -gt 0 ]; then
  LETTERS=("$@")
else
  LETTERS=(Q W E R T Y U I O P A S D F G H J K L Z X C V B N M)
fi

STOP_ON_FAIL="${STOP_ON_FAIL:-0}"
PAUSE_SECONDS="${PAUSE_SECONDS:-0}"
RUN_DIR="${RUN_DIR:-$ROOT_DIR/new_approach_with_homography/timing/task2}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$RUN_DIR/task2_all_letters_${STAMP}.log"
CSV="$RUN_DIR/task2_all_letters_${STAMP}.csv"

mkdir -p "$RUN_DIR"
printf "letter,status,seconds\n" > "$CSV"

now_s() {
  python3 -c 'import time; print(f"{time.perf_counter():.6f}")'
}

elapsed_s() {
  python3 - "$1" "$2" <<'PY'
import sys
start = float(sys.argv[1])
end = float(sys.argv[2])
print(f"{end - start:.3f}")
PY
}

total_start="$(now_s)"
ok_count=0
fail_count=0

{
  echo "[task2-all] letters: ${LETTERS[*]}"
  echo "[task2-all] log: $LOG"
  echo "[task2-all] csv: $CSV"
  echo
} | tee -a "$LOG"

for letter in "${LETTERS[@]}"; do
  echo "===== task2 letter=$letter =====" | tee -a "$LOG"
  start="$(now_s)"

  set +e
  "$TASK2" "$letter" 2>&1 | tee -a "$LOG"
  status="${PIPESTATUS[0]}"
  set -e

  end="$(now_s)"
  seconds="$(elapsed_s "$start" "$end")"

  if [ "$status" -eq 0 ]; then
    result="ok"
    ok_count=$((ok_count + 1))
  else
    result="fail:$status"
    fail_count=$((fail_count + 1))
  fi

  printf "%s,%s,%s\n" "$letter" "$result" "$seconds" >> "$CSV"
  echo "[task2-all] letter=$letter status=$result seconds=${seconds}" | tee -a "$LOG"
  echo | tee -a "$LOG"

  if [ "$status" -ne 0 ] && [ "$STOP_ON_FAIL" = "1" ]; then
    echo "[task2-all] stopping after failure because STOP_ON_FAIL=1" | tee -a "$LOG"
    break
  fi

  if [ "$PAUSE_SECONDS" != "0" ]; then
    sleep "$PAUSE_SECONDS"
  fi
done

total_end="$(now_s)"
total_seconds="$(elapsed_s "$total_start" "$total_end")"

{
  echo "===== summary ====="
  echo "[task2-all] ok=$ok_count fail=$fail_count total_seconds=$total_seconds"
  echo "[task2-all] csv: $CSV"
  echo "[task2-all] log: $LOG"
} | tee -a "$LOG"

if [ "$fail_count" -gt 0 ]; then
  exit 1
fi
