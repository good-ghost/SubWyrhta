#!/usr/bin/env bash
# E2E 미디어를 만든다 (web/e2e/fixtures, 저장소에 올리지 않는다). run.sh 가 없을 때 부른다.
# - speech.wav: espeak-ng 합성 음성 (영어 문장, 약 16초)
# - demo-talk.mp4: 단색 화면 + speech.wav
# - broken.mp4: 미디어가 아닌 파일 (실패 처리 장면용)
# espeak-ng 가 없으면 podman(개발 컨테이너 안에서는 distrobox-host-exec podman)의 debian 이미지로 만든다.
set -euo pipefail

FIXTURES="$(cd "$(dirname "$0")/.." && pwd)/fixtures"
TEXT='Welcome to the subtitle demo. This short recording was made with a speech synthesizer. The system extracts the audio, finds the speech, and writes subtitles. Each line should stay on screen long enough to read. Thank you for watching.'
IMAGE=docker.io/library/debian:bookworm-slim

mkdir -p "$FIXTURES"
if command -v espeak-ng >/dev/null 2>&1; then
  espeak-ng -v en-us -s 150 -w "$FIXTURES/speech.wav" "$TEXT"
else
  if command -v podman >/dev/null 2>&1; then
    PODMAN=(podman)
  elif command -v distrobox-host-exec >/dev/null 2>&1; then
    PODMAN=(distrobox-host-exec podman)
  else
    echo "espeak-ng 또는 podman 이 필요합니다" >&2
    exit 1
  fi
  "${PODMAN[@]}" run --rm -v "$FIXTURES:/out:Z" "$IMAGE" sh -c \
    "apt-get update -qq >/dev/null && apt-get install -y -qq espeak-ng >/dev/null 2>&1 && espeak-ng -v en-us -s 150 -w /out/speech.wav \"$TEXT\""
fi
ffmpeg -hide_banner -loglevel error -f lavfi -i color=c=0x00796b:s=640x360:r=10 -i "$FIXTURES/speech.wav" \
  -shortest -c:v libx264 -pix_fmt yuv420p -c:a aac -y "$FIXTURES/demo-talk.mp4"
printf 'this is not a media file\n' > "$FIXTURES/broken.mp4"
echo "E2E 미디어 생성: $FIXTURES"
