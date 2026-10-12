#!/usr/bin/env bash
# E2E 용 로컬 serve 서버. 실행마다 새 데이터 폴더·설정으로 띄운다 (관리 계정은 첫 장면이 화면에서 만든다).
# STT 는 실제로 처리한다: faster-whisper tiny, CPU int8 (모델 캐시는 실행 사이에 유지한다).
# Playwright webServer 가 부른다 (web/ 에서, 개발 컨테이너 안). 화면은 web/dist 를 쓰므로 먼저 빌드한다.
set -euo pipefail

WEB_DIR=$(cd "$(dirname "$0")/../.." && pwd)
REPO_DIR=$(dirname "$WEB_DIR")
ROOT="$WEB_DIR/test-results/e2e-server"
PORT="${E2E_PORT:-18950}"
MODEL_DIR="${E2E_MODEL_DIR:-$HOME/.cache/subwyrhta-e2e-models}"

rm -rf "$ROOT"
mkdir -p "$ROOT/data" "$MODEL_DIR"
# E2E 미디어는 저장소에 없다: 없으면 만든다 (espeak-ng 합성 음성)
if [ ! -f "$WEB_DIR/e2e/fixtures/speech.wav" ] || [ ! -f "$WEB_DIR/e2e/fixtures/demo-talk.mp4" ]; then
  bash "$WEB_DIR/e2e/server/make-fixtures.sh"
fi
# 관리 계정 admin 의 비밀번호 (테스트가 이 파일을 읽어 처음 설정·로그인한다)
token=$(head -c 24 /dev/urandom | od -An -tx1 | tr -d ' \n')
printf '%s' "$token" > "$ROOT/token"

# 처리 중 화면·실행 중 취소를 보일 긴 음성 (speech.wav 24번, 약 9분). 테스트가 small 모델로 처리한다
ffmpeg -hide_banner -loglevel error -stream_loop 23 -i "$WEB_DIR/e2e/fixtures/speech.wav" -ar 16000 -ac 1 \
  "$ROOT/speech-long.wav"

cat > "$ROOT/data/config.yaml" <<YAML
# E2E 용 설정 (web/e2e/server/run.sh 가 만든다)
stt:
  model: faster-whisper
  model_size: tiny
  model_dir: "$MODEL_DIR"
  device: cpu
  compute_type: int8
  batch_size: 8
  beam_size: 5
  temperature: 0.0
  source_lang: auto
audio:
  normalize: true
  separate_vocals: false
  denoise:
    enabled: false
    model: noisereduce
  vad:
    enabled: true
    pre_vad: true
segmentation:
  mode: advanced
output:
  format: srt
quality:
  enabled: true
  auto_fix: true
  export_report: false
web:
  max_upload_mb: 200
  delete_input_after_done: true
YAML

# 프로젝트의 .env(실제 키·토큰)를 읽지 않게 빈 env 파일을 준다
: > "$ROOT/e2e.env"

cd "$REPO_DIR"
# 작업 프로세스가 코어를 모두 쓰면 녹화 중인 브라우저 렌더링이 밀려 화면이 멈춰 보인다 (정지 구간 5초 초과).
# 로컬 운영 권장과 같이 스레드 수를 제한한다 (README "로컬 운영 시 부하")
export OMP_NUM_THREADS="${E2E_THREADS:-8}"
exec "$REPO_DIR/.venv/bin/python" -m src.main --env-file "$ROOT/e2e.env" serve --data "$ROOT/data" --host 127.0.0.1 --port "$PORT" \
  --static-dir "$WEB_DIR/dist"
