#!/usr/bin/env bash
# MGnify 무한 재시도 — 한 번 실패해도 다시 시도
# Ctrl+C 두 번 (한 번 보내면 현재 run 만 죽임) 으로 멈춤

cd "$(dirname "$0")/../.."
LOG=data/raw/mgnify_v3/run_log.txt
mkdir -p data/raw/mgnify_v3
echo "Started $(date)" >> "$LOG"

while true; do
    echo "===== MGnify run starting $(date) =====" >> "$LOG"
    python data/scripts/collect_mgnify_extra.py \
        --max-extra 8000 --workers 4 --checkpoint-every 100 \
        2>&1 | tee -a "$LOG"
    EC=$?
    echo "===== MGnify run ended exit=$EC at $(date) =====" >> "$LOG"
    # 실패시 60초 대기 후 재시도 (API 진정 시간)
    sleep 60
done
