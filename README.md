# SubWyrhta

영상/음성 파일에서 고품질 자막(SRT/ASS/VTT/JSON)을 자동 생성하는 독립형 STT 도구.

번역 기능은 포함하지 않으며, 음성 인식과 자막 생성에 특화되어 있다.

## Features

- **다중 STT 엔진** — WhisperX, faster-whisper, whisper-turbo, Parakeet
- **4가지 세그멘테이션** — basic, advanced, extreme, llm-assisted
- **화자 분리** — pyannote 기반, 화자별 색상/위치 스타일 (ASS)
- **Pre-VAD** — STT 이전 발화 구간 분리로 정밀한 타임스탬프
- **오디오 향상** — 배경음 분리(demucs), 음량 정규화, 노이즈 억제
- **품질 관리** — CPS/줄길이/타임코드/빈 자막 검사 및 자동 교정
- **llm-assisted 분할** — 구독(ChatGPT·Claude CLI), 클라우드 API 키(Gemini·OpenRouter·NVIDIA NIM), 로컬(Ollama·OpenAI 호환), 분당 요청 수(RPM) 제한
- **배치 처리** — 디렉토리·glob·파일 목록 입력, `--workers` 병렬 처리(GPU 는 worker 하나), 실패 파일 건너뛰기
- **장면 감지** — 긴 공백(선택: 영상 컷)을 장면 경계로 보고 경계 자막 간격 보장
- **중단된 곳부터 재개** — 전처리·Pre-VAD·화자 분리 결과와 음성 인식 체크포인트(Pre-VAD 청크 단위)를 남겨 끝낸 단계를 건너뛴다 (`--resume`, 웹의 다시 시도)
- **웹 화면 (serve)** — 업로드, 작업 대기열(한 번에 하나), 완료 목록·미리 보기, 자막 검수(검수 메모를 보며 SRT 수정), 로그, 설정
- **Python API** — CLI 외에 코드에서 직접 호출 가능

## Requirements

- Python 3.12+
- ffmpeg (ffprobe 포함)
- Node.js·npm (웹 화면을 소스에서 다시 빌드할 때만. 빌드한 화면 `web/dist` 가 저장소에 들어 있다)

## Installation

```bash
# 저장소 클론
git clone https://github.com/good-ghost/SubWyrhta.git
cd SubWyrhta

# 가상환경 설정
python -m venv .venv
source .venv/bin/activate

# 의존성 설치 (코어·STT 엔진·웹·화자 분리·오디오 향상·LLM 전부)
pip install -r requirements.txt
# CPU 만 쓰는 서버: torch 는 CPU 휠로 (CUDA 휠보다 수 GB 작다)
#   pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# 개발 (pytest, ruff, mypy)
pip install -r requirements-dev.txt
```

## Quick Start

### 기본 자막 생성

```bash
python -m src.main transcribe video.mp4
```

`./output/video.en.srt` 파일이 생성된다.

### 출력 포맷 / 언어 지정

```bash
python -m src.main transcribe video.mp4 --output-format vtt --source-lang ko
```

### 화자 분리

```bash
export HF_TOKEN="hf_..."   # 또는 .env

python -m src.main diarize video.mp4 --num-speakers 2
```

토큰 발급과 모델 셋(`speaker-diarization-3.1`, `segmentation-3.0`, `speaker-diarization-community-1`)의 사용 조건 동의가 필요하다: [docs/OPERATIONS.md](docs/OPERATIONS.md) 2절.

화자별 색상이 적용된 ASS 자막(`./output/video.en.diarized.ass`)이 생성된다.

### 파일 정보 확인

```bash
python -m src.main info video.mp4
```

## CLI Commands

### transcribe

음성/영상 파일을 자막으로 변환한다.

```bash
python -m src.main transcribe <input> [options]
```

`<input>` 은 파일, 디렉토리, glob 패턴(`"videos/**/*.mp4"`, 따옴표로 감싼다) 중 하나다. 파일이 여럿이면 배치로 처리한다.

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--recursive` | 디렉토리·glob 을 하위 폴더까지 찾는다 | - |
| `--file-list` | 처리할 파일 목록 (한 줄에 하나, `#` 주석). `<input>` 없이도 쓴다 | - |
| `--workers` | 병렬 worker 수. GPU 는 worker 하나만 쓰고 나머지는 CPU | `1` |
| `--workers-device` | 모든 worker 의 디바이스를 직접 지정 | - |
| `--failed-list` | 실패한 파일 목록 저장 경로 (실패 파일은 건너뛰고 계속, 하나라도 실패하면 종료 코드 1) | - |
| `--output-dir` | 출력 디렉토리 | `./output` |
| `--output-format` | `srt` / `ass` / `vtt` / `json` | `srt` |
| `--output-name` | 출력 파일명 직접 지정 (파일 하나일 때만) | `{입력}.{언어}.{포맷}` |
| `--model` | STT 엔진 (`whisperx` / `faster-whisper` / `whisper-turbo` / `parakeet` / `distil-whisper`) | `whisperx` |
| `--model-size` | 모델 크기 | `large-v3` |
| `--model-dir` | 모델 캐시 경로 | `~/.cache/subwyrhta` |
| `--source-lang` | 음성 언어 (`auto`로 자동 감지) | `auto` |
| `--device` | `auto` / `cuda` / `cpu` / `mps` | `auto` |
| `--compute-type` | `auto` / `float16` / `float32` / `int8` | `auto` |
| `--batch-size` / `--beam-size` / `--temperature` | 배치 추론 크기, Beam search 크기, 샘플링 온도 | `16` / `5` / `0.0` |
| `--audio-track` | 오디오 트랙 번호 (0-based) | 기본 트랙 |
| `--start` / `--end` | 처리 시간 범위 (HH:MM:SS) | 전체 |
| `--split-by-chapters` | 챕터별 독립 자막 생성 | - |
| `--keep-audio` | 추출된 오디오 보존 | - |
| `--denoise` / `--separate-vocals` / `--normalize-audio` | 전처리 (각각 `--no-…`로 끈다) | 설정 파일 |
| `--vad` / `--pre-vad` | VAD, Pre-VAD (`--no-pre-vad`: 전체 추론 후 사후 재분할) | 설정 파일 |
| `--vad-export` | VAD 타임라인 JSON 저장 | - |
| `--segmentation-mode` / `--max-cps` | 세그멘테이션 모드, 최대 CPS | 설정 파일 |
| `--diarize` / `--num-speakers` / `--hf-token` | 화자 분리 | 설정 파일 |
| `--qc` / `--no-qc`, `--quality-report` | 품질 검사, 품질 리포트 JSON | 설정 파일 |
| `--export-words` | 단어 타임스탬프 JSON 함께 저장 | - |
| `--resume` / `--no-resume` | 체크포인트(`<입력>.checkpoint.json`)와 단계 결과(`<입력>.work/`)에서 재개 / 남은 것을 지우고 처음부터 | 남기기만 하고 쓰지 않음 |
| `--llm-provider` / `--llm-model` / `--llm-base-url` | llm-assisted 프로바이더 (`codex` / `claude-code` / `gemini` / `openrouter` / `nvidia-nim` / `ollama` / `local`), 모델, 로컬 서버 주소 | 설정 파일 |
| `--dry-run` | 실행 계획만 출력 | - |

`transcribe`·`diarize`·Python API·웹 화면은 같은 전체 파이프라인(`src/pipeline.py`)을 실행한다:
(화자 분리를 켜면 모델 접근 사전 확인) → 추출 → 전처리(demucs → 음량 정규화 → 노이즈 억제) → Pre-VAD → STT → 화자 분리 →
노이즈 필터(·구두점 복원) → 세그멘테이션(화자 분할 → 모드별 분할 → CPS 최적화 → 최소 표시 시간 → 텍스트 정규화) → 화자 이름 → 장면 경계 간격 → 품질 검사 → 출력.

### diarize

화자 분리 포함 자막을 생성한다 (기본 포맷 ASS, 파일명 `{입력}.{언어}.diarized.{포맷}`). transcribe 의 출력·구간 옵션(`--output-*`, `--keep-audio`, `--audio-track`, `--start`/`--end`), STT 옵션 일부(`--model`, `--model-size`, `--source-lang`, `--device`), LLM 옵션, `--dry-run` 에 아래 옵션을 더한다. 전처리·VAD·세그멘테이션 모드·품질 검사 등은 설정 파일 값을 쓴다.

```bash
python -m src.main diarize <input> [options]
```

| 옵션 | 설명 |
|------|------|
| `--num-speakers` | 화자 수 (`auto` 또는 정수) |
| `--hf-token` | HuggingFace 토큰 |
| `--speaker-map` | 화자 이름 매핑 (`SPEAKER_00:John,SPEAKER_01:Jane`) |
| `--speaker-profile` | 화자 프로파일 JSON 파일 |
| `--export-embeddings` | 화자 임베딩 내보내기 (`{입력}.embeddings.json`) |
| `--speaker-embeddings` | 이전 파일의 화자 임베딩. 같은 화자에게 이전과 같은 화자 ID 를 붙인다 (시리즈 영상의 화자 이름을 맞출 때) |
| `--no-speaker-split` | 화자 전환 분할 비활성화 |

### validate

기존 자막 파일의 품질을 검사한다.

```bash
python -m src.main validate subtitle.srt --auto-fix
```

| 옵션 | 설명 |
|------|------|
| `--auto-fix` | 자동 교정 후 저장 |
| `--output` | 교정 결과 저장 경로 (기본 `{원본}.fixed.{확장자}`) |
| `--max-chars` | 줄당 최대 글자 수 (기본 42) |
| `--max-cps` | 최대 CPS |
| `--min-duration` | 최소 표시 시간 (기본 1.0초) |

### enhance

오디오 품질 향상만 수행한다.

```bash
python -m src.main enhance audio.mp3 --denoise --separate-vocals
```

| 옵션 | 설명 |
|------|------|
| `--output-dir` | 출력 디렉토리 (기본 `./output`, 파일 `{입력}.enhanced.wav`) |
| `--denoise` | 노이즈 억제 |
| `--separate-vocals` | demucs 배경음 분리 |
| `--normalize` / `--no-normalize` | 음량 정규화 (기본 켬) |
| `--device` | demucs 디바이스 (`auto` / `cuda` / `cpu` / `mps`) |

### info / list-models

```bash
python -m src.main info video.mp4        # 파일 정보
python -m src.main list-models           # 엔진/모델/디바이스 목록
```

## Web UI (serve)

로컬 서버로 웹 화면을 띄운다. 브라우저에서 영상·음성을 올리면 작업 대기열에 들어가고, **한 번에 하나씩** 등록 순서대로 처리한다 (작업마다 자식 프로세스: 끝나면 모델 메모리가 모두 반환되고, 실행 중 작업을 취소할 수 있다).

```bash
pip install -r requirements.txt

python -m src.main serve --data ./data          # http://127.0.0.1:8950
python -m src.main --config ./config.yaml serve --data ./data --host 0.0.0.0 --port 8950
```

- 저장소에는 빌드한 화면(`web/dist`)이 들어 있어 Node.js 없이 바로 쓸 수 있다. 화면 소스(`web/src`)를 고쳐 다시 빌드하려면 `web/vendor/vue-smartview` 가 있어야 한다. 비공개 vue-smartview 의 부분 빌드라 공개 저장소에는 없다 (관리자는 `scripts/update-smartview.sh <VUE-SMARTVIEW 경로>` 로 만든다)
- 처음 접속하면 관리 계정을 만든다. 이후 그 계정으로 로그인한다 (세션 쿠키, 비밀번호는 `data/secrets.json`에 scrypt 해시, 권한 600)
- `--config`를 주지 않으면 `data/config.yaml`을 쓰고, 없으면 코드 기본값으로 만든다. 화면의 설정에서 저장하면 다음 작업부터 적용된다 (`web.session_hours`·서버 로그 수준은 서버 재시작 후)
- 데이터 폴더: `config.yaml`, `secrets.json`(관리 계정), `jobs.db`(작업 기록), `jobs/<id>/input`(업로드), `jobs/<id>/output`(자막), `jobs/<id>/checkpoint.json`(음성 인식 체크포인트), `jobs/<id>/work/`(단계 결과: 전처리 음성·Pre-VAD·화자 분리), `logs/subwyrhta.log`(JSON 로그), `uploads/`
- 새 작업에서 작업별 옵션(엔진, 모델 크기, 언어, 출력 포맷 여러 개, 세그멘테이션, 전처리, 화자 분리 등)을 고른다. 실패한 작업 모두 다시 시도, 끝난 작업 모두 지우기, 자막 미리 보기를 지원한다
- 완료 목록의 **자막 검수**는 SRT 자막 전체에 LLM 검수 리포트(`*.review.json`)의 메모를 붙여 표로 보인다 (시작, 끝, 자막, 검수 종류, 상태, 메모). 자막을 누르면 메모와 함께 수정 창이 열리고, 저장하면 SRT 에 바로 쓰고 그 자막을 검수 완료로 표시한다 (고치지 않고 저장해도 된다). 처음 고칠 때 원본을 `*.orig.srt` 로 남긴다. 같은 작업의 VTT·ASS·JSON 출력은 바꾸지 않는다
- 완료한 작업의 업로드 원본은 기본으로 지운다 (`web.delete_input_after_done`). 실패·취소한 작업은 남겨서 다시 시도할 수 있다 (끝낸 단계는 건너뛰고 음성 인식은 체크포인트에서 이어서)
- 오래 걸리는 작업을 맡길 때는 서버를 터미널(세션)과 떼어 띄운다 (`setsid nohup python -m src.main serve --data ./data > serve.log 2>&1 &`). 터미널이 닫혀 서버가 끝나도 다시 시도하면 끝낸 단계부터 이어서 한다
- 화자 분리 토큰은 화면에서 받지 않는다. 서버를 띄울 때 `HF_TOKEN` 환경 변수(`.env` 가능)로 준다
- 준비(`.env`, Hugging Face 모델 동의, LLM 프로바이더), 실행·재시작, 상태 확인, 작업 실패 시 조치는 [docs/OPERATIONS.md](docs/OPERATIONS.md) 를 따른다

### 로컬 운영 시 부하

| 항목 | 내용 |
|------|------|
| 동시 처리 | 1건. 대기 작업은 순서대로 기다린다 |
| CPU 스레드 | 작업 프로세스가 보이는 코어를 모두 쓴다. 다른 서비스와 함께 쓰면 `OMP_NUM_THREADS=8 python -m src.main serve …`처럼 제한한다 |
| 메모리 (추정, 실측 아님) | large-v3 CPU 기준 float32 약 9~11GB, int8 약 4~6GB. CPU 서버는 설정에서 `compute_type: int8`을 권장한다 |
| 모델 캐시 | `stt.model_dir`(또는 `SUBWYRHTA_MODEL_DIR`), `~/.cache/huggingface`, `~/.cache/torch` |
| 웹 서버 | torch 를 import 하지 않는다 (디바이스 확인은 시작 때 별도 프로세스로 한 번) |

## STT Engines

| 엔진 | 옵션 | 특징 |
|------|------|------|
| WhisperX | `--model whisperx` | 배치 추론 + forced alignment (기본값) |
| faster-whisper | `--model faster-whisper` | CTranslate2 고속 추론 |
| whisper-turbo | `--model whisper-turbo` | OpenAI turbo 모델 |
| Parakeet | `--model parakeet` | NVIDIA NeMo 영어 특화 (nemo_toolkit 별도 설치, requirements 파일 없음) |
| distil-whisper | `--model distil-whisper` | faster-whisper 별칭. `--model-size distil-large-v3` 처럼 distil 모델 이름과 함께 쓴다 |

GPU 로드 실패 시 CPU로 자동 폴백된다.

## Segmentation Modes

| 모드 | 설명 |
|------|------|
| `basic` | PySBD 문장 경계 기반 분할 |
| `advanced` | Pause + Clause 복합 신호 + 동적 임계값 (기본값) |
| `extreme` | advanced + semantic 유사도 기반 추가 분할 (sentence-transformers 별도 설치, 없으면 advanced) |
| `llm-assisted` | advanced + LLM 분할점 재조정 |

CLI `--segmentation-mode` 또는 설정 파일 `segmentation.mode` 로 고른다.

### llm-assisted 모드 사용

```bash
# config.yaml에서 mode를 llm-assisted로 변경하거나 CLI에서 (Claude 구독: 서버 계정에서 claude auth login)
python -m src.main transcribe video.mp4 --segmentation-mode llm-assisted \
  --llm-provider claude-code
```

구독(ChatGPT·Claude, 공식 CLI), 클라우드 API 키(Gemini·OpenRouter·NVIDIA NIM, `.env`), 로컬(Ollama·OpenAI 호환 서버)을 지원하고 분당 요청 수(`llm.rpm`)를 제한할 수 있다. 프로바이더별 준비는 [docs/USAGE_GUIDE.md](docs/USAGE_GUIDE.md) 의 llm-assisted 모드 절을 본다. 로컬 Ollama 예:

```bash
python -m src.main transcribe video.mp4 --segmentation-mode llm-assisted \
  --llm-provider ollama \
  --llm-model llama3
```

## Output Formats

| 포맷 | 용도 |
|------|------|
| **SRT** | 범용 자막 (대부분의 플레이어 호환) |
| **ASS** | 화자별 색상/위치 스타일 (diarize 시 기본) |
| **VTT** | 웹 자막 (HTML5 video) |
| **JSON** | 메타데이터 + 단어 타임스탬프 + 품질 요약 (후처리/분석용) |

## Configuration

YAML 설정 파일로 전체 동작을 제어한다. 설정 우선순위:

```
코드 기본값 → 설정 파일 → 환경변수 → CLI 인자
```

- 설정 파일은 `--config` 로 준 파일이다. CLI 는 `--config` 가 없으면 설정 파일을 읽지 않는다 (프로젝트 폴더의 `config.yaml` 도 자동으로 읽지 않는다)
- `serve` 는 `--config` 가 없으면 `<data>/config.yaml` 을 쓰고, 없으면 만든다

### 환경변수

환경 변수는 프로젝트 폴더의 `.env` 파일에 둘 수 있다. `cp .env.example .env` 후 쓰는 값만 채운다. CLI·`serve` 가 시작할 때 읽고(다른 파일은 `--env-file`), 셸에 이미 있는 값이 우선하며, 값이 빈 줄은 무시한다. `.env` 는 git 에 올리지 않는다 (`.gitignore`).

| 변수 | 용도 |
|------|------|
| `HF_TOKEN` | HuggingFace 인증 (화자 분리) |
| `SUBWYRHTA_MODEL_DIR` | 모델 캐시 경로 |
| `SUBWYRHTA_DEVICE` | 디바이스 오버라이드 |
| `GEMINI_API_KEY` | Google Gemini API 키 |
| `OPENROUTER_API_KEY` | OpenRouter API 키 |
| `NVIDIA_API_KEY` | NVIDIA NIM API 키 (호스팅) |
| `SUBWYRHTA_GLOSSARY` | LLM 오인식 교정의 용어 목록 경로 (파일, 또는 `common.txt`·`<입력 이름>.txt` 가 있는 폴더). 없으면 교정하지 않고 검수 리포트에 제안만 남긴다 |
| `SUBWYRHTA_LLM_EXTRA_BODY` | OpenAI 호환 LLM 요청 본문에 합칠 JSON. 모델마다 다른 추론 끄기 (예: llama-server Qwen3 `{"chat_template_kwargs": {"enable_thinking": false}}`, NIM glm `{"reasoning_effort": "low"}`, 모델별 `{"models": {...}}`). docs/OPERATIONS.md 3.1 |

### 설정 파일 예시

```yaml
stt:
  model: "faster-whisper"
  model_size: "large-v3"
  device: "cuda"
  source_lang: "ko"

segmentation:
  mode: "advanced"
  max_cps: 9              # 한국어 기본값
  min_duration_s: 1.0

output:
  format: "srt"
  dir: "./subtitles"
```

CLI에서 설정 파일을 지정할 수 있다:

```bash
python -m src.main --config my_config.yaml transcribe video.mp4
```

## Python API

```python
from src.api import SubWyrhta, TranscribeConfig, save_result

stt = SubWyrhta(model="whisperx", model_size="large-v3")
result = stt.transcribe("video.mp4", TranscribeConfig(source_lang="en"))

# 결과 접근
for seg in result.segments:
    print(f"[{seg.start:.1f}s] {seg.text}")

# 파일 저장
save_result(result, "output.srt")
save_result(result, "output.json")
```

지정하지 않은 값은 설정 파일(`SubWyrhta(config_path=…)`), 없으면 코드 기본값을 쓴다. `TranscribeConfig(output_dir=…)` 나 `output_format` 을 주면 자막 파일도 쓰고, 아니면 결과만 돌려준다 (저장은 `save_result`, 확장자로 포맷을 고른다). 자세한 옵션은 [docs/USAGE_GUIDE.md](docs/USAGE_GUIDE.md) 의 Python API 절을 본다.

## Supported Input Formats

**영상:** `.mp4`, `.mkv`, `.avi`, `.mov`, `.webm`, `.mpeg`, `.mpg`, `.wmv`, `.flv`, `.m4v`

**음성:** `.mp3`, `.wav`, `.m4a`, `.aac`, `.flac`, `.ogg`, `.opus`, `.wma`, `.mka`

## Development

```bash
# 테스트
pytest                            # 전체
pytest tests/unit/                # 단위 테스트만

# 웹 화면 (web/, 빌드에는 web/vendor/vue-smartview 가 필요하다)
cd web && npm run typecheck && npm run lint && npm test && npm run build
cd web && npm run test:e2e        # Playwright E2E·녹화 (로컬 serve, 실제 STT. ffmpeg 필요)
                                  # 테스트 미디어는 저장소에 없다: 처음 실행할 때 web/e2e/server/make-fixtures.sh 가
                                  # espeak-ng 합성 음성으로 만든다 (없으면 podman 컨테이너로)
                                  # 브라우저 확장(Salesforce Sandbox Bar Toggle) 폴더를 E2E_EXTENSION_PATH 환경 변수나
                                  # web/e2e/.env.local (E2E_EXTENSION_PATH=<폴더>) 에 넣어야 한다

# 린트 & 타입 체크
ruff check src/ tests/ --line-length 100
ruff format src/ tests/ --line-length 100
mypy src/ --strict
```

## Project Structure

```
src/
├── main.py              # CLI (click)
├── api.py               # Python API
├── pipeline.py          # 전체 파이프라인 (CLI·API·웹 공통)
├── checkpoint.py        # Pre-VAD 청크 체크포인트
├── stage_cache.py       # 단계 결과 보관 (전처리 음성·Pre-VAD·화자 분리, 다시 시도할 때 건너뛴다)
├── batch_processor.py   # 배치 입력 수집·병렬 처리 (transcribe)
├── progress.py          # 배치 진행 표시
├── version.py           # 버전 (pyproject.toml)
├── web/                 # 웹 서버 (FastAPI, 인증, 작업 큐·워커, 설정, 로그)
├── audio/               # 오디오 추출, VAD, 향상
├── engines/             # STT 엔진 (WhisperX, faster-whisper, turbo, Parakeet)
├── segmentation/        # 4가지 분할 모드 + CPS 최적화
├── postprocess/         # 노이즈 필터, 정규화, 구두점 복원
├── quality/             # 품질 검사·리포트, 장면 감지
├── diarization/         # 화자 분리, 임베딩 관리
├── writers/             # SRT/ASS/VTT/JSON 출력
├── models/              # 데이터 모델
└── core/                # 설정, 예외, 로깅
web/                     # 웹 화면 소스 (Vue 3, Playwright E2E). web/dist 는 빌드한 화면
```

## License

[MIT](LICENSE) — Copyright (c) 2026 good-ghost

`web/dist/` 에 번들된 제3자 소프트웨어(Vue, Vuetify, vue-i18n, Material Design Icons)의 고지와, 별도로 설치하는 의존성의 라이선스는 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 에 있다.
