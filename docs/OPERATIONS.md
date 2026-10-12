# SubWyrhta 로컬 운영 절차

웹 화면(`serve`)을 로컬 서버로 운영할 때의 준비, 실행·재시작, 상태 확인, 문제 해결 절차다.
컨테이너 이미지는 만들지 않는다. 프로젝트의 `.venv` 로 직접 실행한다.

- 설치 명령과 웹 화면 기능은 [README.md](../README.md), 설정 값의 뜻은 [USAGE_GUIDE.md](USAGE_GUIDE.md) 를 본다.
- 명령은 프로젝트 폴더에서 실행한다. 개발 컨테이너를 쓰면 `distrobox enter gen-dev -- bash -lic 'cd <프로젝트> && …'` 처럼 컨테이너 안에서 실행한다.

## 1. 처음 준비

### 1.1 패키지

| 파일 | 내용 |
|------|------|
| `requirements.txt` | 전부: 코어, STT 엔진, 웹 서버, 화자 분리(pyannote), 오디오 향상(demucs, noisereduce, 구두점 복원), LLM 프로바이더(openai SDK). `pip install -r requirements.txt` 하나로 설치한다 |
| `requirements-dev.txt` | 개발: 위 전부 + pytest, ruff, mypy |

- ffmpeg 가 PATH 에 있어야 한다.
- 웹 화면은 빌드한 `web/dist` 를 쓴다 (저장소에 들어 있다). 화면 소스를 바꾸면 `cd web && npm ci && npm run build` 로 다시 빌드한다. 빌드에는 `web/vendor/vue-smartview` 가 필요하다 (비공개 부분 빌드, 공개 저장소에 없다. `scripts/update-smartview.sh`).
- 대시보드의 "엔진·기능 패키지" 카드가 패키지마다 설치됨/없음을 보인다. 설치한 뒤에는 서버를 다시 시작해야 바뀐다 (시스템 점검은 서버 시작 때 한 번 한다).
- requirements 파일이 없는 선택 패키지: Parakeet 엔진의 `nemo_toolkit`(크고 torch·numpy 버전과 부딪힐 수 있다), extreme 모드의 `sentence-transformers`(없으면 advanced 로 대신한다).

### 1.2 `.env` (토큰·API 키)

```bash
cp .env.example .env    # 쓰는 값만 채운다
```

| 변수 | 쓰는 곳 |
|------|---------|
| `HF_TOKEN` | 화자 분리 (2절) |
| `SUBWYRHTA_MODEL_DIR`, `SUBWYRHTA_DEVICE` | 설정 파일의 `stt.model_dir`, `stt.device` 를 덮어쓴다 |
| `GEMINI_API_KEY`, `OPENROUTER_API_KEY`, `NVIDIA_API_KEY` | llm-assisted 클라우드 프로바이더 (3절) |
| `SUBWYRHTA_GLOSSARY` | LLM 오인식 교정의 용어 목록 (파일, 또는 `common.txt`·`<입력 이름>.txt` 가 있는 폴더). 바꾸면 서버를 다시 시작한다 |
| `SUBWYRHTA_LLM_EXTRA_BODY` | OpenAI 호환 LLM 요청 본문에 합칠 JSON (모델별 추론 끄기, 3.1절) |

- CLI·`serve` 가 **시작할 때** 현재 폴더의 `.env` 를 읽는다 (다른 파일은 `--env-file`). **`.env` 를 고치면 서버를 다시 시작한다.**
- 셸에 이미 있는 환경 변수가 `.env` 보다 우선한다. 값이 빈 줄은 무시한다.
- 서버를 실행한 터미널 출력에 읽은 키 이름이 남는다 (`환경 변수 파일 읽음 keys=[…]`). 값은 남지 않는다. 이 줄은 로그 파일(`data/logs/subwyrhta.log`)과 화면의 로그에는 없다.
- `.env` 는 git 에 올리지 않는다 (`.gitignore`).

### 1.3 이전 이름(BySTT)에서 옮길 때

2026-10-09 에 이름을 SubWyrhta 로 바꿨다. 예전 이름을 읽는 호환 코드는 결과 JSON 의 버전 키 하나뿐이다.

| 바뀐 것 | 이전 | 지금 | 할 일 |
|---------|------|------|-------|
| 환경 변수 | `BYSTT_MODEL_DIR`, `BYSTT_DEVICE` | `SUBWYRHTA_MODEL_DIR`, `SUBWYRHTA_DEVICE` | `.env`·셸의 키 이름을 바꾼다. 예전 이름은 읽지 않는다 |
| 모델 캐시 기본값 | `~/.cache/bystt` | `~/.cache/subwyrhta` | 설정에 `stt.model_dir` 를 적지 않았다면 폴더를 옮기거나 다시 받는다 |
| 로그 파일 | `data/logs/bystt.log` | `data/logs/subwyrhta.log` | 화면의 로그는 새 파일만 보여 준다. 예전 기록은 예전 파일에 남는다 |
| 로그인 쿠키·화면 설정(언어·테마) | `bystt_session`, `bystt.*` | `subwyrhta_session`, `subwyrhta.*` | 다시 로그인하고 언어·테마를 다시 고른다 |
| 결과 JSON 메타데이터 | `bystt_version` | `subwyrhta_version` | 없음. 예전 JSON 도 읽는다 |
| 패키지 이름 | `bystt` | `subwyrhta` | 설치본을 쓰면 다시 설치한다 (`pip install -e .`) |

## 2. 화자 분리 준비 (Hugging Face)

화자 분리는 pyannote 모델을 Hugging Face 에서 받는다. 모델 셋 모두 사용 조건 동의가 필요하다.

1. https://huggingface.co/settings/tokens 에서 **Read** 토큰을 만든다. 값은 `hf_` 로 시작한다.
   fine-grained 토큰이면 "public gated repos 읽기" 권한을 켠다.
2. 토큰을 만든 계정으로 로그인한 채 아래 **세 페이지 모두** 사용 조건에 동의한다.
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/segmentation-3.0
   - https://huggingface.co/pyannote/speaker-diarization-community-1 (pyannote.audio 4.x 가 함께 받는다)
3. `.env` 의 `HF_TOKEN` 에 넣고 서버를 다시 시작한다.
4. 확인: 대시보드 "HF_TOKEN" 이 설정됨으로 보인다. 모델까지 받을 수 있는지는 아래 명령으로 확인한다 (파이프라인을 실제로 불러온 뒤 내린다. 처음에는 모델을 내려받는다).

```bash
python -c "from src.core.config import AppConfig, load_env_file; load_env_file(); \
from src.diarization.diarizer import check_diarization_access; \
check_diarization_access(AppConfig.load('data/config.yaml').diarization); print('화자 분리 준비됨')"
```

화자 분리를 켠 작업은 시작할 때(음성 추출 전) 같은 확인을 한다. 토큰·동의 문제가 있으면 음성 인식을 하기 전에 바로 실패한다.

## 3. llm-assisted 프로바이더 준비

설정 화면 **LLM 탭**에서 프로바이더를 고르면 준비 상태 배너가 보인다 (구독은 CLI 설치 여부, 클라우드는 키 설정 여부).

| 프로바이더 | 준비 |
|------------|------|
| `claude-code` (Claude 구독) | 서버를 실행하는 계정에 Claude Code 설치 후 `claude auth login` (Pro/Max 계정) |
| `codex` (ChatGPT 구독) | 서버를 실행하는 계정에 Codex CLI 설치(`npm i -g @openai/codex`) 후 `codex login` |
| `gemini`, `openrouter`, `nvidia-nim` | `.env` 에 키, 서버 재시작 (openai SDK 는 `requirements.txt` 에 들어 있다) |
| `ollama`, `local` | 해당 서버 실행. 엔드포인트 URL (`local` 은 필수) |

- OpenAI·Claude 는 구독만, NVIDIA NIM 은 클라우드만 지원한다.
- 구독 CLI 의 로그인 만료는 배너로 알 수 없다 (작업할 때 실패로 알린다). 서버 계정에서 `claude auth status` 로 확인한다.
- 구독 CLI 는 시작 시간이 있어 요청 제한 시간을 120초 이상으로 둔다. 분당 요청 수 한도가 있으면 RPM 칸에 넣는다.
- LLM 호출이 실패하면 작업은 advanced 분할 결과로 계속한다 (로그에 경고).

### 3.1 모델별 추론 끄기 (`SUBWYRHTA_LLM_EXTRA_BODY`)

추론(thinking) 모델은 묶음 하나에 수백 초가 걸려 시간 초과가 난다. 끄는 방법이 모델·서버마다 달라, `.env` 의 `SUBWYRHTA_LLM_EXTRA_BODY` 에 OpenAI 호환 요청 본문에 합칠 JSON 을 넣는다 (코드 수정 없음, 2026-10-11 추가). 설정 화면의 "추론 끄기"·"추론 수준" 위에 합치고, 이 값이 우선한다. 값을 `null` 로 주면 설정이 만든 항목을 뺀다. **고친 뒤 서버를 다시 시작한다.**

| 서버·모델 | 값 |
|-----------|-----|
| NIM `z-ai/glm-5.3-flash` | `{"reasoning_effort": "low"}` (`chat_template_kwargs` 로는 꺼지지 않았다. low 면 문장 20개 묶음 13초) |
| llama-server + Qwen3 계열 (`qwen3.8-27b`, `qwen3.8-35b-a3b`) | `{"chat_template_kwargs": {"enable_thinking": false}}` |
| 모델별로 한 번에 | `{"models": {"z-ai/glm-5.3-flash": {"reasoning_effort": "low"}, "qwen3.8-27b": {"chat_template_kwargs": {"enable_thinking": false}}, "*": {}}}` (지금 모델의 항목, 없으면 `"*"`) |

```bash
# .env (한 줄 JSON, 작은따옴표로 감싼다)
SUBWYRHTA_LLM_EXTRA_BODY='{"models": {"z-ai/glm-5.3-flash": {"reasoning_effort": "low"}, "qwen3.8-27b": {"chat_template_kwargs": {"enable_thinking": false}}, "qwen3.8-35b-a3b": {"chat_template_kwargs": {"enable_thinking": false}}}}'
```

- 설정의 "추론 끄기"는 `chat_template_kwargs.thinking=false` 를 보낸다. Qwen3 템플릿이 쓰는 키는 `enable_thinking` 이라, llama-server 에서는 위 값을 쓴다 (설정의 `thinking` 도 함께 가지만 템플릿이 모르는 키라 무시된다. 빼려면 `"thinking": null`)
- JSON 이 아니면 그 작업의 LLM 단계가 설정 오류로 실패하고 (재시도 없음), 검수 리포트의 `llm_error` 에 남는다
- 단계 캐시 키에 이 값이 들어간다 (바꾸면 LLM 단계를 다시 호출한다)

### 3.2 llama-server (로컬 `local` 프로바이더)

- 설정: 프로바이더 `local`, 엔드포인트 `http://<호스트>:<포트>/v1`, 모델은 서버의 `--alias` 이름 (`--alias` 를 주지 않으면 gguf 전체 경로를 넣어야 한다)
- `chat_template_kwargs` 를 쓰려면 llama-server 를 `--jinja` 로 띄운다 (Qwen3 계열 템플릿의 `enable_thinking`)
- 실행 예: `llama-server -m <qwen3.8-27b gguf> --alias qwen3.8-27b --jinja --port 8080 -c <컨텍스트>` (값은 모델·장비에 맞춘다)
- 이 노트북은 장시간 부하에서 전원이 꺼진 이력이 있다 (6.3). 같은 장비에서 STT 와 로컬 LLM 을 함께 돌리면 클럭 상한(4절)을 둔다
- 2026-10-11 기준 llama-server 로 실제 호출은 확인하지 않았다 (로컬 서버가 떠 있지 않았다)

## 4. 서버 실행·재시작

```bash
python -m src.main serve --data ./data                  # http://127.0.0.1:8950
OMP_NUM_THREADS=8 python -m src.main serve --data ./data  # 다른 서비스와 CPU 를 나눌 때
```

- 처음 접속하면 관리 계정을 만든다.
- `--config` 를 주지 않으면 `<data>/config.yaml` 을 쓰고, 없으면 **코드 기본값**으로 만든다 (저장소의 `config.yaml` 이 아니다. 예: `stt.model_dir` 는 `~/.cache/subwyrhta`). 저장소의 설정을 쓰려면 `python -m src.main --config ./config.yaml serve --data ./data`.
- 끄기: 실행한 터미널에서 Ctrl+C. 백그라운드로 띄웠으면 `serve` 의 python 프로세스에 SIGTERM 을 보낸다 (`pgrep -f "src.main serve"` 로 찾는다).
- 긴 작업(1시간 이상 음성, demucs·화자 분리)을 이 노트북(AMD Ryzen AI MAX+ PRO 395)에서 돌릴 때는 부하를 낮춘다 (2026-10-09 작업 #1: `balanced` 에서 두 번 컴퓨터가 꺼졌고, 아래 설정에서는 끝까지 돌았다. 6.3절):
  - 전원 프로필: `echo low-power | sudo tee /sys/firmware/acpi/platform_profile` (끝나면 `balanced` 로 되돌린다)
  - 연산 스레드: `OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python -m src.main serve --data ./data`
  - 처리 시간은 늘어난다 (2시간 음성: 약 2시간 37분)
- 서버는 띄운 터미널(세션)이 닫히면 함께 끝난다. 실행 중이던 작업은 오류 기록 없이 `실행 중` 으로 남았다가, 다음 시작 때 실패(서버 중단)로 바뀐다. 오래 걸리는 작업을 맡길 때는 세션과 떼어 띄운다: `setsid nohup python -m src.main serve --data ./data > serve.log 2>&1 &`
- **다시 시작해야 하는 경우**: `.env` 를 고쳤을 때, 패키지를 설치했을 때, 서버 코드(`src/`)가 바뀌었을 때. 화면 소스가 바뀌면 `web` 을 다시 빌드한다.
- 설정 화면에서 저장한 값은 재시작 없이 다음 작업부터 적용된다. 예외: `web.session_hours`, 서버 프로세스의 로그 수준(`logging.level`, 작업 프로세스에는 바로 적용)은 재시작 후.
- 서버를 다시 시작하면 실행 중이던 작업은 실패(서버 중단)로 표시된다. 다시 시도하면 끝낸 단계를 건너뛰고 이어서 한다 (6.1).

## 5. 상태 확인

| 무엇 | 어디서 |
|------|--------|
| 워커·대기 작업·패키지·HF_TOKEN·디스크 | 대시보드 |
| LLM 프로바이더 준비 상태 | 설정 → LLM 탭 배너 |
| 작업 단계·오류 문구 | 작업 대기열 → 작업 상세 |
| 서버·작업 로그 | 화면의 로그, 또는 `data/logs/subwyrhta.log` (JSON 줄) |
| 작업 기록 | `data/jobs.db` (SQLite) |

## 6. 문제 해결

### 6.1 작업이 실패했을 때

1. 작업 상세의 오류 문구와 실패한 단계를 본다. 로그에서 같은 작업 번호(`job_id`)의 `error`·`warning` 을 찾는다.
2. 원인을 고친다 (아래 표). `.env`·패키지를 바꿨으면 서버를 다시 시작한다.
3. 작업 대기열에서 **다시 시도**를 누른다. 실패·취소한 작업의 업로드 원본은 남아 있다. 끝낸 단계는 건너뛴다: 전처리 음성·Pre-VAD·화자 분리 결과는 작업 폴더 `work/` 에서 읽고, 음성 인식은 체크포인트에서 이어서 한다 (Pre-VAD 를 켠 작업). 작업 옵션이나 설정을 바꿨으면 바뀐 설정이 영향을 주는 단계부터 다시 한다.

| 증상 (오류 문구) | 원인 | 조치 |
|------------------|------|------|
| `401 … Invalid user token`, `토큰이 필요` | `HF_TOKEN` 이 없거나 Hugging Face 토큰이 아니다 (`hf_` 로 시작하지 않음) | 2절 1·3 |
| `403 … not in the authorized list` (`speaker-diarization-3.1`, `segmentation-3.0`, `speaker-diarization-community-1` 중 하나) | 그 모델의 사용 조건에 동의하지 않았다 | 2절 2. 문구에 나온 모델 페이지에서 동의 |
| `… CLI 를 찾을 수 없습니다` | 구독 CLI 가 서버 계정 PATH 에 없다 | 3절 |
| `claude 호출 실패: Failed to authenticate …` | Claude 구독 로그인 만료 | 서버 계정에서 `claude auth login` |
| `… API 키가 없습니다: … GEMINI_API_KEY …` | `.env` 에 키가 없다 | 1.2절, 서버 재시작 |
| `ffmpeg 오디오 추출 실패` | 미디어가 아니거나 손상된 파일 | 원본 파일 확인 |
| 작업 프로세스가 비정상 종료 | 메모리 부족 등 | 설정에서 `compute_type: int8`, 작은 모델, `OMP_NUM_THREADS` |

### 6.2 경고만 남고 기능이 빠졌을 때

작업은 끝났는데 기능이 적용되지 않은 경우가 있다. 로그의 `warning` 을 본다.

| 경고 | 의미 |
|------|------|
| `보컬 분리 실패, 원본 사용` | 배경음 분리를 건너뛰었다. demucs 설치 여부와 경고의 `error` 를 본다 |
| `LLM 세그멘테이션 실패, 폴백 사용` | llm-assisted 대신 advanced 결과를 썼다. 3절 |
| `노이즈 억제 실패, 원본 사용` | 노이즈 억제를 건너뛰었다. noisereduce 설치 여부 |

### 6.3 기록: 2026-10-09 작업 #1 실패

| 순서 | 내용 |
|------|------|
| 증상 | 화자 분리를 켠 작업이 약 1시간 음성 인식 뒤 화자 분리 단계에서 401 로 실패. 같은 작업에서 배경음 분리도 경고만 남기고 건너뜀 |
| 원인 1 | `.env` 의 `HF_TOKEN` 이 Hugging Face 토큰이 아니었다 (32자, `hf_` 아님) → 새 Read 토큰으로 교체 |
| 원인 2 | 새 토큰은 유효했지만 모델 사용 조건에 동의하지 않아 403. `speaker-diarization-3.1`·`segmentation-3.0` 동의 뒤에도 pyannote.audio 4.x 가 함께 받는 `speaker-diarization-community-1` 이 남아 있었다 |
| 원인 3 | 보컬 저장(`torchaudio.save`)이 저장 백엔드 없이 실패해 배경음 분리가 건너뛰어졌다 |
| 고친 것 | 화자 분리를 켠 작업은 시작할 때 파이프라인을 불러와 확인한다 (토큰·동의 문제를 음성 인식 전에 알린다). 보컬은 16kHz mono WAV 로 직접 저장한다 |
| 확인 방법 | 2절 4 의 확인 명령, 실제 음성으로 demucs 분리 → VAD 읽기 |
| 결과 | 세 모델 동의 뒤 2절 4 의 확인 명령 통과. 짧은 음성(16초)으로 화자 분리를 끝까지 실행해 확인 (로드 4초, 분리 2초). 작업 #1 은 화면에서 다시 시도 |
| 2차 시도 | 발화 구간이 516 → 590 개로 바뀌어(배경음 분리를 고쳐 전처리 결과가 달라짐) 체크포인트를 쓰지 못하고 음성 인식을 처음부터 다시 했다 (약 1시간). 화자 분리를 시작한 12:44 에 오류 로그 없이 멈췄다. 처음에는 서버를 띄운 세션이 끝난 탓으로 봤으나, 부팅 기록으로 보면 **컴퓨터 전체가 12:58 에 꺼졌다** (아래 원인 5) |
| 3차 시도 | 같은 코드·설정인데 발화 구간이 590 → 600 개로 또 바뀌어 음성 인식을 처음부터 다시 했다. 14:39 에 화자 분리를 시작했고 **14:53 에 컴퓨터 전체가 꺼졌다** (15:13 재부팅) |
| 원인 5 (확인 중) | 두 번 모두 화자 분리를 시작하고 약 14분 뒤 컴퓨터가 꺼졌다 (`journalctl --list-boots`: 12:58·14:53 에 부팅 종료, `last -x`: crash). 시스템 로그는 평범한 메시지 중간에서 끊겼다 |
| 제외한 원인 | 메모리 부족: 커널 로그에 OOM 기록 없음. 자동 절전: 시스템 로그에 절전 기록 없음. 배터리 방전: 꺼질 때 100% (UPower 기록). 화자 분리 자체: 같은 음성(약 2시간)으로 화자 분리만 돌리면 1555초에 정상 완료 (최대 메모리 2.7GB, CPU 최고 92.2°C, 측정 2026-10-09 15:21~15:47) |
| 남은 의심 | 하드웨어 보호(전원·발열)에 의한 강제 꺼짐. 실제 작업은 화자 분리 전에 약 75분 고부하(demucs·음성 인식)가 이어졌다. 측정 중 AC 어댑터가 39% 시간 동안 끊기고 꽂힌 채 배터리가 약 45W 씩 줄었다 (어댑터 공급 전력 부족 의심) |
| 4·5차 시도 | 서버를 `OMP_NUM_THREADS=8` 로 다시 띄워도 `balanced` 프로필에서는 demucs 시작 20초 만에 CPU 가 100°C 근처에 붙어 클럭을 조절했다 (이 CPU 의 정상 동작). 측정용 안전장치 기준(95°C, 100°C 5초)이 낮아 작업을 멈췄다 |
| 6차 시도 (완료) | 전원 프로필 `low-power` + `OMP_NUM_THREADS=8` 로 2026-10-09 16:04~18:41 (약 2시간 37분: 전처리 26분, 음성 인식 84분, 화자 분리 46분). CPU 최고 84.8°C·평균 65.7°C, AC 끊김 0초, 작업 메모리 최대 30.5GB. 결과: 2시간 음성, 자막 2124개, 화자 2명, 품질 B. 단계 결과(`work/`)를 남기고 성공 뒤 지웠다 |
| 결론 | 꺼진 정확한 원인은 기록이 없어 확정하지 못했다. `balanced` 에서는 장시간 고부하에 CPU 가 계속 100°C 근처에 있고 AC 어댑터가 자주 끊긴다. `low-power` 에서는 같은 작업이 끝까지 돌았다. 이 노트북에서 긴 작업은 4절의 저전력 실행을 따른다 |
| 원인 4 | demucs 는 기본값(`shifts=1`)에서 소리를 0~0.5초 무작위로 밀어 처리해 실행마다 결과가 조금씩 다르다 → 발화 구간 경계·개수가 바뀐다. 또 다시 시도할 때마다 추출·demucs(약 14분 30초)·정규화·노이즈 억제·VAD 를 다시 했다 |
| 고친 것 2 | 전처리 음성·발화 구간·화자 분리 결과를 작업 폴더 `work/` 에 남겨 다시 시도할 때 건너뛴다. 체크포인트는 청크 경계까지 같아야 쓴다 (PHASE-PLAN-09 9.9). 3차 시도는 고치기 전 코드로 돌고 있어 적용되지 않는다 |
