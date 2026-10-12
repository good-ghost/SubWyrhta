#!/usr/bin/env bash
# 녹화 정지 구간 검사 (공통 규칙: 5초 이하). 개발 컨테이너의 ffmpeg 를 쓴다.
# 사용: bash web/e2e/check-freeze.sh [영상 폴더, 기본 docs/e2e/movie]
set -euo pipefail

MOVIE_DIR=$(cd "${1:-$(dirname "$0")/../../docs/e2e/movie}" && pwd)
MAX_FREEZE_S=5
# 화면 차이 문턱 (0~1, SUB_ROBOT 과 같은 값). 1920x1080 에서 커서(28px)만 움직이는 변화는
# 0.0002 에도 못 미쳐 정지로 본다 (2026-10-08 측정). 기다리는 장면은 화면 요소를 바꿔야 한다
# (walkthrough.spec.ts 의 waitJob)
NOISE=0.0005

status=0
for movie in "$MOVIE_DIR"/*.webm; do
  name=$(basename "$movie")
  report=$(ffmpeg -hide_banner -nostats -i "$movie" -vf "freezedetect=n=$NOISE:d=$MAX_FREEZE_S" -map 0:v -f null - 2>&1 \
    | grep -o 'freeze_\(start\|duration\): [0-9.]*' || true)
  if [ -n "$report" ]; then
    echo "FREEZE $name"
    echo "$report" | paste - - | sed 's/^/  /'
    status=1
  else
    echo "ok     $name"
  fi
done
exit $status
