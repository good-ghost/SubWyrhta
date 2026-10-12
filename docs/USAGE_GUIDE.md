# SubWyrhta 사용 가이드

영상/음성 파일에서 고품질 자막을 자동 생성하는 독립형 STT 도구.

---

## 목차

1. [설치](#1-설치)
2. [빠른 시작](#2-빠른-시작)
3. [CLI 명령어](#3-cli-명령어)
   - [transcribe](#31-transcribe)
   - [diarize](#32-diarize)
   - [validate](#33-validate)
   - [enhance](#34-enhance)
   - [info](#35-info)
   - [list-models](#36-list-models)
4. [Python API](#4-python-api)
5. [STT 엔진](#5-stt-엔진)
6. [세그멘테이션](#6-세그멘테이션)
7. [화자 분리](#7-화자-분리)
8. [오디오 전처리](#8-오디오-전처리)
9. [출력 포맷](#9-출력-포맷)
10. [설정 파일](#10-설정-파일)
11. [여러 파일 처리와 재개](#11-여러-파일-처리와-재개)
12. [품질 관리](#12-품질-관리)
13. [활용 예시](#13-활용-예시)
14. [문제 해결](#14-문제-해결)
15. [웹 화면 (serve)](#15-웹-화면-serve)

---

## 1. 설치

### 시스템 요구사항

- Python 3.12 이상
- ffmpeg·ffprobe (오디오 추출용)
- Node.js·npm (웹 화면을 소스에서 다시 빌드할 때만. 빌드한 화면 `web/dist` 가 저장소에 들어 있다)

### 기본 설치

```bash
# 가상환경 생성
python -m venv .venv
source .venv/bin/activate    # Linux/macOS
# .venv\Scripts\activate     # Windows

# 전체 설치 (코어, STT 엔진, 웹 화면, 화자 분리, 오디오 향상, LLM 프로바이더)
pip install -r requirements.txt
# CPU 만 쓰면 torch CPU 휠: pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# 개발 (전체 + pytest, ruff, mypy)
pip install -r requirements-dev.txt
```

requirements 파일에 없는 선택 패키지: Parakeet 엔진의 `nemo_toolkit`, extreme 모드의 `sentence-transformers` (없으면 advanced 로 대신한다).

### pip 설치 (저장소에서 패키지로)

```bash
pip install .                  # 코어만 (명령 이름 subwyrhta)
pip install ".[stt]"           # STT 엔진 포함
pip install ".[all]"           # stt + diarize + enhance (llm·웹은 따로: ".[llm]". requirements.txt 는 전부 설치)
```

### ffmpeg 설치 확인

```bash
ffmpeg -version
```

Ubuntu/Debian에서 설치되지 않은 경우:

```bash
sudo apt install ffmpeg
```

snap으로 설치된 ffmpeg도 자동 감지한다.

---

## 2. 빠른 시작

### 가장 간단한 사용법

```bash
python -m src.main transcribe video.mp4
```

`./output/video.en.srt` 파일이 생성된다.

### 한국어 자막 생성

```bash
python -m src.main transcribe interview.mp4 --source-lang ko
```

### 화자 분리 자막

```bash
export HF_TOKEN="hf_xxxxx"
python -m src.main diarize meeting.mp4 --num-speakers 3
```

### 실행 전 계획 확인

```bash
python -m src.main transcribe video.mp4 --dry-run
```

실제 처리 없이 적용될 설정(엔진, 모델, 디바이스, 언어 등)을 미리 확인한다.

---

## 3. CLI 명령어

모든 명령어는 `python -m src.main` 또는 패키지 설치 후 `subwyrhta`로 실행한다.

### 전역 옵션

| 옵션 | 설명 |
|------|------|
| `--config <path>` | YAML 설정 파일 경로 지정 (없으면 설정 파일을 읽지 않는다. `serve` 는 `<data>/config.yaml`) |
| `--env-file <path>` | 환경 변수 파일 (기본: 현재 폴더의 `.env`, 없으면 건너뛴다). 셸 환경 변수가 우선한다 |
| `--verbose` | 상세 로그 출력 (DEBUG 레벨) |
| `--quiet` | 최소 로그 출력 (WARNING 레벨) |
| `--version` | 버전 정보 |
| `--help` | 도움말 |

```bash
# 커스텀 설정 파일 사용
python -m src.main --config my_config.yaml transcribe video.mp4

# 상세 로그
python -m src.main --verbose transcribe video.mp4
```

---

### 3.1 transcribe

음성/영상 파일을 자막으로 변환한다.

```bash
python -m src.main transcribe <입력> [옵션]
```

`<입력>` 은 파일, 디렉토리, glob 패턴(`"videos/**/*.mp4"`, 셸이 펼치지 않게 따옴표로 감싼다) 중 하나다. 파일이 여럿이면 배치로 처리한다 ([11. 여러 파일 처리와 재개](#11-여러-파일-처리와-재개)).

#### 배치 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--recursive` | 디렉토리·glob 을 하위 폴더까지 찾는다 | - |
| `--file-list <경로>` | 처리할 파일 목록 (한 줄에 하나, `#` 주석). `<입력>` 없이도 쓴다 | - |
| `--workers <N>` | 병렬 worker 수. worker 마다 모델을 한 번 불러온다 | `1` |
| `--workers-device <디바이스>` | 모든 worker 의 디바이스 (`auto` / `cuda` / `cpu` / `mps`). 없으면 GPU 는 worker 하나만 | - |
| `--failed-list <경로>` | 실패한 파일 목록 저장 | - |

#### 입력/출력 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--output-dir <경로>` | 출력 디렉토리 | `./output` |
| `--output-format <포맷>` | `srt` / `ass` / `vtt` / `json` | `srt` |
| `--output-name <이름>` | 출력 파일명 직접 지정 (파일 하나일 때만) | 입력 파일 기반 자동 |
| `--keep-audio` | 추출된 중간 오디오 파일 보존 | 삭제 |
| `--audio-track <N>` | 오디오 트랙 번호 (0-based) | 기본 트랙 |
| `--start <HH:MM:SS>` | 처리 시작 시간 | 처음부터 |
| `--end <HH:MM:SS>` | 처리 종료 시간 | 끝까지 |

#### STT 엔진 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--model <엔진>` | STT 엔진 (`whisperx` / `faster-whisper` / `whisper-turbo` / `parakeet` / `distil-whisper`) | `whisperx` |
| `--model-size <크기>` | 모델 크기 | `large-v3` |
| `--model-dir <경로>` | 모델 캐시 디렉토리 (`SUBWYRHTA_MODEL_DIR` 가 우선) | `~/.cache/subwyrhta` |
| `--source-lang <언어>` | 음성 언어 코드 | `auto` |
| `--device <디바이스>` | `auto` / `cuda` / `cpu` / `mps` | `auto` |
| `--compute-type <타입>` | `auto` / `float16` / `float32` / `int8` | `auto` |
| `--batch-size <N>` | 배치 추론 크기 | `16` |
| `--beam-size <N>` | Beam search 크기 | `5` |
| `--temperature <값>` | 샘플링 온도 (0.0 = greedy) | `0.0` |

#### LLM 옵션 (llm-assisted 세그멘테이션용)

| 옵션 | 설명 |
|------|------|
| `--llm-provider <프로바이더>` | `codex` / `claude-code` / `gemini` / `openrouter` / `nvidia-nim` / `ollama` / `local` |
| `--llm-model <모델>` | 모델명 (예: `sonnet`, `llama3`). 구독(`claude-code`, `codex`)은 생략하면 CLI 기본 모델 |
| `--llm-base-url <URL>` | 로컬 서버 엔드포인트 (`local` 필수, 원격 Ollama). 클라우드 프로바이더는 고정 주소 |

#### 전처리·VAD 옵션 (지정하지 않으면 설정 파일 값)

| 옵션 | 설명 |
|------|------|
| `--denoise` / `--no-denoise` | 노이즈 억제 (noisereduce) |
| `--separate-vocals` / `--no-separate-vocals` | demucs 배경음 분리 |
| `--normalize-audio` / `--no-normalize-audio` | 음량 정규화 |
| `--vad` / `--no-vad` | VAD 사용 |
| `--pre-vad` / `--no-pre-vad` | Pre-VAD (끄면 전체 추론 후 사후 재분할) |
| `--vad-export` | VAD 타임라인 JSON 저장 (`{입력}.vad.json`) |

#### 세그멘테이션·화자·품질 옵션 (지정하지 않으면 설정 파일 값)

| 옵션 | 설명 |
|------|------|
| `--segmentation-mode <모드>` | `basic` / `advanced` / `extreme` / `llm-assisted` |
| `--max-cps <값>` | 최대 CPS (미지정 시 언어별 자동) |
| `--diarize` / `--no-diarize` | 화자 분리 (켜면 파일명에 `.diarized.` 가 붙는다) |
| `--num-speakers <N>` | 화자 수 (`auto` 또는 정수) |
| `--hf-token <토큰>` | HuggingFace 토큰 (`HF_TOKEN`·`.env` 권장) |
| `--qc` / `--no-qc` | 품질 검사 |
| `--quality-report` | 품질 리포트 JSON 저장 (`{입력}.quality.json`) |
| `--export-words` | 단어 타임스탬프 JSON 을 함께 저장 |

#### 기타 옵션

| 옵션 | 설명 |
|------|------|
| `--resume` / `--no-resume` | 체크포인트(`<입력>.checkpoint.json`, Pre-VAD 청크 단위)와 단계 결과(`<입력>.work/`: 전처리 음성·Pre-VAD·화자 분리)에서 재개 / 남은 것을 지우고 처음부터. 둘 다 없으면 남기기만 하고 쓰지 않는다 |
| `--split-by-chapters` | 챕터별 독립 자막 파일 생성 (ffprobe 로 읽은 챕터. 챕터가 없으면 일반 처리) |
| `--dry-run` | 실행 계획만 출력 (실제 처리 안 함) |

#### 사용 예시

```bash
# GPU + 한국어 + VTT 출력
python -m src.main transcribe lecture.mp4 \
  --source-lang ko \
  --output-format vtt \
  --device cuda

# 특정 구간만 처리
python -m src.main transcribe movie.mkv \
  --start 00:10:00 \
  --end 00:20:00

# 경량 모델로 빠르게
python -m src.main transcribe audio.mp3 \
  --model faster-whisper \
  --model-size small \
  --compute-type int8

# 챕터별 분할 자막
python -m src.main transcribe movie.mkv --split-by-chapters

# 두 번째 오디오 트랙 사용
python -m src.main transcribe movie.mkv --audio-track 1
```

---

### 3.2 diarize

화자 분리 포함 자막을 생성한다. transcribe 의 입력/출력 옵션, STT 옵션 중 `--model`·`--model-size`·`--source-lang`·`--device`, LLM 옵션, `--dry-run` 을 받고 아래 화자 분리 옵션이 더해진다. 전처리·VAD·세그멘테이션 모드·품질 검사 옵션은 없으며 설정 파일 값을 쓴다.

```bash
python -m src.main diarize <입력파일> [옵션]
```

#### 화자 분리 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--num-speakers <N\|auto>` | 화자 수 지정 | `auto` |
| `--hf-token <토큰>` | HuggingFace 인증 토큰 | 환경변수 `HF_TOKEN` |
| `--speaker-map <매핑>` | 화자 이름 매핑 | - |
| `--speaker-profile <경로>` | 화자 프로파일 JSON 파일 | - |
| `--speaker-embeddings <경로>` | 이전 파일의 화자 임베딩(`--export-embeddings` 결과). 같은 화자에게 이전과 같은 화자 ID 를 붙인다 | - |
| `--export-embeddings` | 화자 임베딩 JSON 내보내기 (`{입력}.embeddings.json`) | - |
| `--no-speaker-split` | 화자 전환 분할 비활성화 | 활성화 |

**참고:** diarize 명령어의 기본 출력 포맷은 `ass` (화자별 색상/위치 지원)이고, 파일명은 `{입력}.{언어}.diarized.{포맷}` 이다.

#### 사용 예시

```bash
# 2명 대화 + 화자 이름 지정
python -m src.main diarize interview.mp4 \
  --num-speakers 2 \
  --speaker-map "SPEAKER_00:김기자,SPEAKER_01:이교수"

# 화자 임베딩 저장 (output/podcast_ep01.embeddings.json)
python -m src.main diarize podcast_ep01.mp4 \
  --export-embeddings

# 다음 회차: 같은 사람에게 같은 화자 ID (이름 매핑도 같이 맞는다)
python -m src.main diarize podcast_ep02.mp4 \
  --speaker-embeddings output/podcast_ep01.embeddings.json \
  --speaker-map "SPEAKER_00:진행자,SPEAKER_01:게스트"

# SRT로 출력 (기본은 ASS)
python -m src.main diarize meeting.mp4 \
  --output-format srt \
  --num-speakers 4
```

#### HuggingFace 토큰 설정

pyannote 모델 사용에 HuggingFace 토큰과 모델 사용 조건 동의가 필요하다 (절차·확인 명령: [OPERATIONS.md](OPERATIONS.md) 2절):

1. https://huggingface.co/settings/tokens 에서 **Read** 토큰 발급 (`hf_` 로 시작)
2. 토큰을 만든 계정으로 아래 세 모델 페이지에서 사용 조건 동의
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/segmentation-3.0
   - https://huggingface.co/pyannote/speaker-diarization-community-1 (pyannote.audio 4.x 가 함께 받는다)

화자 분리를 켜면 음성 추출 전에 파이프라인을 한 번 불러와 토큰·동의 문제를 먼저 알린다 (긴 음성 인식 뒤에 실패하지 않게).

우선순위: 설정 파일 `diarization.hf_token` < 환경 변수 `HF_TOKEN`(`.env` 포함) < CLI `--hf-token`.

```bash
# .env 또는 환경변수로 설정 (권장)
export HF_TOKEN="hf_xxxxx"

# 또는 CLI 옵션으로 전달
python -m src.main diarize video.mp4 --hf-token "hf_xxxxx"

# 또는 config.yaml에 설정
# diarization:
#   hf_token: "hf_xxxxx"
```

---

### 3.3 validate

생성된 자막 파일의 품질을 검사하고 선택적으로 자동 교정한다.

```bash
python -m src.main validate <자막파일> [옵션]
```

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--auto-fix` | 자동 교정 후 결과 저장 | - |
| `--output <경로>` | 교정 결과 저장 경로 | `원본.fixed.확장자` |
| `--max-chars <N>` | 줄당 최대 글자 수 | `42` |
| `--max-cps <값>` | 최대 CPS (초당 글자 수) | 언어별 자동 |
| `--min-duration <초>` | 최소 표시 시간 | `1.0` |

#### 지원 입력 포맷

`.srt`, `.ass`, `.vtt`, `.json`

#### 사용 예시

```bash
# 품질 검사만
python -m src.main validate output/video.en.srt

# 자동 교정
python -m src.main validate output/video.ko.srt --auto-fix

# 한국어 기준 적용 (CPS 9)
python -m src.main validate subtitle.srt --max-cps 9

# 교정 결과를 별도 파일로
python -m src.main validate subtitle.srt --auto-fix --output corrected.srt
```

#### 검사 항목

- **타임코드 역전** — 끝 시간이 시작보다 앞서는지
- **타임코드 겹침** — 자막 간 시간이 겹치는지
- **줄 길이 초과** — 한 줄이 너무 긴지
- **CPS 초과** — 자막이 너무 빠르게 표시되는지
- **최소 표시 시간** — 자막이 너무 짧게 표시되는지
- **빈 텍스트** — 내용 없는 자막이 있는지

심각도와 자동 교정 내용은 [12. 품질 관리](#12-품질-관리) 를 본다. `--auto-fix` 결과는 `--output` 이 없으면 `{원본}.fixed.{확장자}` 로 저장한다.

---

### 3.4 enhance

오디오 품질 향상만 수행한다 (자막 생성 없음).

```bash
python -m src.main enhance <입력파일> [옵션]
```

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--output-dir <경로>` | 출력 디렉토리 (파일 `{입력}.enhanced.wav`) | `./output` |
| `--denoise` | 노이즈 억제 적용 | - |
| `--separate-vocals` | demucs 배경음 분리 | - |
| `--normalize` / `--no-normalize` | 음량 정규화 | 켬 |
| `--device <디바이스>` | demucs 디바이스 (`auto` / `cuda` / `cpu` / `mps`) | `auto` |

```bash
# 배경 음악 제거 + 노이즈 억제
python -m src.main enhance noisy_video.mp4 \
  --denoise \
  --separate-vocals
```

---

### 3.5 info

입력 파일의 미디어 정보를 확인한다.

```bash
python -m src.main info video.mp4
```

출력 예시:

```
파일: video.mp4
경로: /home/user/videos/video.mp4
크기: 512.3 MB
길이: 01:23:45.67 (5025.7초)
오디오 트랙: 2개
```

---

### 3.6 list-models

사용 가능한 STT 엔진, 모델 크기, 디바이스 정보를 출력한다.

```bash
python -m src.main list-models
```

출력 예시:

```
=== 사용 가능한 STT 엔진 ===
  - distil-whisper
  - faster-whisper
  - parakeet
  - whisper-turbo
  - whisperx

=== 모델 크기 ===
  - tiny
  - base
  - small
  - medium
  - large-v3
  - turbo

=== distil-whisper 모델 ===
  - distil-large-v2
  - distil-large-v3
  - distil-medium.en
  - distil-small.en

=== 디바이스 정보 ===
  사용 가능: cpu, cuda
  PyTorch: 2.3.0+cu121
  GPU: NVIDIA GeForce RTX 4090
  VRAM: 24.0 GB
```

---

## 4. Python API

CLI 외에 Python 코드에서 직접 호출할 수 있다.

### 기본 사용법

```python
from src.api import SubWyrhta, TranscribeConfig, save_result

# 인스턴스 생성
stt = SubWyrhta(
    model="whisperx",
    model_size="large-v3",
    device="auto",
)

# 자막 생성
result = stt.transcribe("video.mp4")

# 결과 접근
print(f"세그먼트 수: {result.segment_count}")
print(f"언어: {result.metadata.language}")
print(f"길이: {result.metadata.duration_s:.1f}초")

for seg in result.segments:
    print(f"[{seg.start:.1f}s → {seg.end:.1f}s] {seg.text}")

# 파일 저장
save_result(result, "output.srt")
save_result(result, "output.json")
```

### TranscribeConfig로 상세 설정

```python
config = TranscribeConfig(
    source_lang="ko",
    segmentation_mode="advanced",
    max_cps=9.0,
    min_duration_s=1.0,
    max_duration_s=7.0,
)

result = stt.transcribe("interview.mp4", config=config)
save_result(result, "./subtitles/interview.ko.srt")

# 출력 위치나 포맷을 주면 CLI 와 같은 이름 규칙으로 파일도 쓴다 (./subtitles/interview.ko.vtt)
stt.transcribe("interview.mp4", TranscribeConfig(source_lang="ko", output_dir="./subtitles", output_format="vtt"))
```

`output_dir`·`output_format` 을 주지 않으면 `transcribe` 는 결과만 돌려준다. 저장은 `save_result` 로 한다.

### 화자 분리

```python
config = TranscribeConfig(
    diarize=True,
    hf_token="hf_xxxxx",
    num_speakers="2",
    source_lang="ko",
)

result = stt.transcribe("meeting.mp4", config=config)

# 화자별 세그먼트 확인
for seg in result.segments:
    print(f"[{seg.speaker}] {seg.text}")
```

### 설정 파일 사용

```python
stt = SubWyrhta(
    config_path="my_config.yaml",
    log_level="DEBUG",
)
```

`SubWyrhta` 인자(`model`, `model_size`, `device`, `compute_type`)와 `TranscribeConfig` 의 값은 **지정한 것만** 설정 파일의 같은 항목을 덮어쓴다. 지정하지 않으면(None) 설정 파일 값, 설정 파일도 없으면 코드 기본값(`whisperx`, `large-v3`, `auto` …)을 쓴다.

### 결과 저장 포맷

```python
save_result(result, "output.srt")    # SRT
save_result(result, "output.ass")    # ASS (화자 색상)
save_result(result, "output.vtt")    # WebVTT
save_result(result, "output.json")   # JSON (메타데이터 + 단어 타임스탬프)
```

확장자에 따라 자동으로 적절한 Writer가 선택된다.

### TranscribeConfig 파라미터

`None` 인 항목은 덮어쓰지 않는다 (설정 파일 → 코드 기본값). 괄호 안은 코드 기본값이다.

| 파라미터 | 타입 | 기본값 | 설명 |
|----------|------|--------|------|
| `source_lang` | `str \| None` | `None` (`"auto"`) | 음성 언어 코드 |
| `diarize` | `bool \| None` | `None` (`False`) | 화자 분리 활성화 |
| `hf_token` | `str \| None` | `None` | HuggingFace 토큰 (없으면 `HF_TOKEN`) |
| `num_speakers` | `str \| None` | `None` (`"auto"`) | 화자 수 |
| `segmentation_mode` | `str \| None` | `None` (`"advanced"`) | 세그멘테이션 모드 |
| `max_cps` | `float \| None` | `None` (언어별 자동) | 최대 CPS |
| `max_duration_s` | `float \| None` | `None` (`7.0`) | 자막 최대 표시 시간(초) |
| `min_duration_s` | `float \| None` | `None` (`1.0`) | 자막 최소 표시 시간(초) |
| `denoise` | `bool \| None` | `None` (`False`) | 노이즈 억제 |
| `keep_audio` | `bool` | `False` | 추출 오디오 보존 |
| `audio_track` | `int \| None` | `None` | 오디오 트랙 인덱스 |
| `start` | `str \| None` | `None` | 처리 시작 시간 |
| `end` | `str \| None` | `None` | 처리 종료 시간 |
| `output_dir` | `str \| None` | `None` | 출력 디렉토리. 주면 자막 파일도 쓴다 |
| `output_format` | `str \| None` | `None` | 출력 포맷. 주면 자막 파일도 쓴다 |

----------|------|--------|------|
| `source_lang` | `str` | `"auto"` | 음성 언어 코드 |
| `diarize` | `bool` | `False` | 화자 분리 활성화 |
| `hf_token` | `str \| None` | `None` | HuggingFace 토큰 |
| `num_speakers` | `str` | `"auto"` | 화자 수 |
| `segmentation_mode` | `str` | `"advanced"` | 세그멘테이션 모드 |
| `max_cps` | `float \| None` | `None` | 최대 CPS (None=언어별 자동) |
| `max_duration_s` | `float` | `7.0` | 자막 최대 표시 시간(초) |
| `min_duration_s` | `float` | `1.0` | 자막 최소 표시 시간(초) |
| `denoise` | `bool` | `False` | 노이즈 억제 |
| `keep_audio` | `bool` | `False` | 추출 오디오 보존 |
| `audio_track` | `int \| None` | `None` | 오디오 트랙 인덱스 |
| `start` | `str \| None` | `None` | 처리 시작 시간 |
| `end` | `str \| None` | `None` | 처리 종료 시간 |
| `output_dir` | `str \| None` | `None` | 출력 디렉토리. 주면 자막 파일도 쓴다 |
| `output_format` | `str \| None` | `None` | 출력 포맷. 주면 자막 파일도 쓴다 |

---

## 5. STT 엔진

### 엔진 비교

| 엔진 | CLI 옵션 | 특징 | 권장 용도 |
|------|----------|------|-----------|
| **WhisperX** | `--model whisperx` | 배치 추론 + forced alignment, 정밀 타임스탬프 | 일반적인 자막 생성 (기본값) |
| **faster-whisper** | `--model faster-whisper` | CTranslate2 기반 고속 추론, 낮은 메모리 | 빠른 처리, 저사양 환경 |
| **whisper-turbo** | `--model whisper-turbo` | OpenAI turbo 모델 (faster-whisper 백엔드). `--model-size` 는 무시하고 항상 turbo | 고속 처리 |
| **Parakeet** | `--model parakeet` | NVIDIA NeMo 기반, 영어 전용 (`nemo_toolkit` 별도 설치) | 영어 고정밀 |
| **distil-whisper** | `--model distil-whisper` | faster-whisper 별칭. `--model-size distil-large-v3` 처럼 distil 모델 이름과 함께 쓴다 | 속도 우선 |

### 모델 크기

| 크기 | 파라미터 | VRAM | 특징 |
|------|----------|------|------|
| `tiny` | 39M | ~1GB | 매우 빠름, 낮은 정확도 |
| `base` | 74M | ~1GB | 빠름, 보통 정확도 |
| `small` | 244M | ~2GB | 균형 |
| `medium` | 769M | ~5GB | 좋은 정확도 |
| `large-v3` | 1550M | ~10GB | 최고 정확도 (기본값) |
| `turbo` | 809M | ~6GB | large 수준 정확도, medium 수준 속도 |

### compute_type 선택

| 타입 | 설명 |
|------|------|
| `auto` | GPU 있으면 `float16`, 없으면 `float32` |
| `float16` | GPU 전용, 빠르고 메모리 절약 |
| `float32` | CPU 기본, 가장 정확 |
| `int8` | 최소 메모리, 약간의 정확도 손실 |

```bash
# VRAM 부족 시 int8로 메모리 절약
python -m src.main transcribe video.mp4 \
  --model-size large-v3 \
  --compute-type int8
```

### GPU 폴백

GPU 로드에 실패하면 자동으로 CPU 모드로 전환된다. `--verbose` 옵션으로 로그를 확인할 수 있다.

---

## 6. 세그멘테이션

자막 분할 방식을 제어한다. CLI `--segmentation-mode` 또는 설정 파일 `segmentation.mode` 로 고른다.

### 모드 비교

| 모드 | 설명 | 적합한 상황 |
|------|------|-------------|
| `basic` | PySBD 문장 경계 기반 분할 | 단순한 내레이션, 뉴스 |
| `advanced` | Pause + Clause 복합 신호, 동적 임계값 | 일반적인 자막 (기본값) |
| `extreme` | advanced + semantic 유사도 기반 추가 분할 (`sentence-transformers` 별도 설치, 없으면 advanced) | 강연, 학술 콘텐츠 |
| `llm-assisted` | advanced + LLM 분할점 재조정 | 최고 품질 자막 |

### 핵심 파라미터

```yaml
segmentation:
  mode: "advanced"
  max_chars_per_line: null  # 한 줄 최대 글자 수 (null = 언어별 자동: ja·zh 16, 그 밖 42)
  max_lines: 2              # 최대 줄 수
  max_cps: null             # 초당 글자 수 (null = 언어별 자동)
  min_duration_s: 1.0       # 최소 표시 시간 (초)
  max_duration_s: 7.0       # 최대 표시 시간 (초) — Hard Cap
  force_split_after_s: 5.0  # 이 시간 초과 시 다음 절 경계에서 분할 (advanced)
  min_gap_s: 0.04           # 자막 간 최소 간격
  dynamic_threshold: true   # advanced: 경과 시간에 따라 분할 임계값을 낮춘다
  dynamic_threshold_start_s: 3.0
  dynamic_threshold_min_ratio: 0.4
```

### 시간 제약 순서

```
force_split_after_s (5초) → 절 경계 탐색 → max_duration_s (7초) — 최종 강제 분할
```

1. 5초 초과 시 다음 절 경계(접속사, 쉼표 등)에서 분할 시도
2. 7초까지 절 경계를 못 찾으면 강제 분할

### 언어별 CPS 기본값

| 언어 | CPS | 설명 |
|------|-----|------|
| 영어 (en) | 17 | 알파벳 기준 |
| 한국어 (ko) | 9 | 한글 1자 = 약 2자 분량 |
| 일본어 (ja) | 7 | 한자 혼용 |
| 중국어 (zh) | 5 | 한자 기준 |

`--max-cps` 또는 `config.yaml`로 명시적으로 오버라이드할 수 있다.

CPS 를 넘는 자막은 절 경계·쉼표·단어 경계에서 나누되, 나눈 양쪽이 모두 `min_duration_s` 이상일 때만 나눈다. 그런 자리가 없으면 나누지 않는다 (품질 검사에서 CPS 초과 경고로 남는다).

### 일본어·중국어

WhisperX 는 일본어·중국어를 글자 단위로 정렬한다. 자막이 글자 단위로 쪼개지지 않도록 다음을 한다.

- 글자를 공백 없이 잇는다 (한자·가나·전각 문자 사이. 라틴 문자·한글은 공백을 둔다)
- 일본어는 형태소 분석(janome, `requirements.txt`)으로 글자를 문절 단위로 묶는다: 조사·조동사·기호·접미어는 앞 단어에, 접두어는 뒤 단어에 붙인다. 자막이 `ございまし` / `た` 처럼 단어 중간에서 나뉘지 않는다. janome 이 없으면 글자 단위로 처리한다 (경고 로그)
- 글자 수에 비해 지나치게 긴 단어(정렬이 무음·신음 구간에 시간을 몰아준 경우)는 끝을 자른다: 상한 `max(1초, 글자 수 x 0.5초)`
- 중국어는 아직 단어 묶기가 없다 (글자 단위)

자막 텍스트 마무리 (일본어 기준, 다른 언어도 해당하는 것만 적용):

- 신음·감탄사만 있는 단어(ああ, んん, はぁ)는 지운다 (`noise_filter.remove_interjections`, 기본 켬). Whisper 가 무음·신음 구간에서 지어내는 문구(`ご視聴ありがとうございました`, `チャンネル登録…`, `Thanks for watching`)는 세그먼트 전체가 그 문구일 때 지운다. 대화에도 나오는 인사(おやすみなさい 등)는 지우지 않는다
- 줄당 최대 글자 수(`max_chars_per_line`, 비우면 ja·zh 16, 그 밖 42)를 넘으면 두 줄로 나눈다. 일본어는 문절 경계에서, 발화 구분·문장 부호 뒤를 먼저 고른다
- 서로 다른 발화가 한 자막에 합쳐지면 일본어·중국어는 사이에 전각 공백을 넣는다 (`初めてです　あ、はい`)
- 자막이 `?` `。` `ー` 로 시작하면 앞 자막 끝으로 옮기고, 부호만 남은 자막은 지운다

최소 표시 시간(`min_duration_s`)보다 짧은 자막은 이웃 자막과 합친다. 합치면 최대 표시 시간(`max_duration_s`)을 넘는 경우(멀리 떨어진 자막)에는 합치지 않고 앞뒤 빈 시간으로 표시 시간을 늘린다.

### llm-assisted 모드

LLM이 분할점을 의미 단위로 재조정한다.

```bash
# Claude 구독 (Pro/Max): 서버 계정에서 claude auth login 후. 모델을 생략하면 CLI 기본 모델
python -m src.main transcribe video.mp4 --segmentation-mode llm-assisted \
  --llm-provider claude-code --llm-model sonnet

# ChatGPT 구독: 서버 계정에서 codex login 후
python -m src.main transcribe video.mp4 --segmentation-mode llm-assisted --llm-provider codex

# Google Gemini: .env 에 GEMINI_API_KEY
python -m src.main transcribe video.mp4 --segmentation-mode llm-assisted \
  --llm-provider gemini --llm-model gemini-3.8-flash

# 로컬 LLM (Ollama, 기본 주소 http://localhost:11434/v1)
python -m src.main transcribe video.mp4 --segmentation-mode llm-assisted \
  --llm-provider ollama \
  --llm-model llama3

# OpenAI 호환 로컬 서버 (llama.cpp·LM Studio·vLLM 등)
python -m src.main transcribe video.mp4 --segmentation-mode llm-assisted \
  --llm-provider local \
  --llm-model qwen3 \
  --llm-base-url http://localhost:8080/v1
```

#### 프로바이더

OpenAI·Claude 는 구독만, NVIDIA NIM 은 클라우드만 지원한다 (2026-10-09 결정). 클라우드 API 키·로컬 프로바이더는 openai SDK 가 필요하다 (`requirements.txt` 에 들어 있다).

| provider | 방식 | 준비 |
|----------|------|------|
| `codex` | ChatGPT 구독 | 서버를 실행하는 계정에 Codex CLI 설치 후 `codex login` |
| `claude-code` | Claude 구독 (Pro/Max) | 서버를 실행하는 계정에 Claude Code 설치 후 `claude auth login` |
| `gemini` | API 키 | `.env` 에 `GEMINI_API_KEY` |
| `openrouter` | API 키 | `.env` 에 `OPENROUTER_API_KEY`. 모델은 OpenRouter 의 `<제공자>/<모델>` 이름 |
| `nvidia-nim` | API 키 (클라우드) | `.env` 에 `NVIDIA_API_KEY` |
| `ollama` | 로컬 | Ollama 실행. 원격이면 `base_url` |
| `local` | 로컬 | OpenAI 호환 서버의 `base_url` (필수) |

- 키는 설정 파일에 쓰지 않고 위의 정해진 환경 변수(`.env` 가능)로만 준다. 설정 화면 LLM 탭은 고른 프로바이더의 준비 상태(키 설정 여부, CLI 설치 여부)를 보인다.
- 클라우드 프로바이더는 고정 주소를 쓴다. `base_url` 은 로컬 서버(ollama, local)만 쓰고, 설정 화면도 그때만 엔드포인트 칸을 보인다.
- 구독 프로바이더는 공식 CLI 를 비대화 모드로 호출한다 (`claude -p`, `codex exec`). 구독 한도는 사용자가 CLI 를 직접 쓸 때와 함께 줄어든다. 호출할 때 `ANTHROPIC_API_KEY`·`ANTHROPIC_AUTH_TOKEN`·`CODEX_API_KEY`·`OPENAI_API_KEY` 는 CLI 에 넘기지 않는다 (넘기면 구독 대신 API 로 과금된다). CLI 는 시작 시간이 있어 `timeout_s` 를 120 이상으로 둔다.
- LLM 호출이 실패하면 advanced 결과를 그대로 쓴다 (로그에 경고).

#### RPM (분당 요청 수)

`llm.rpm` 을 정하면 최근 60초 동안 보낸 요청 수(재시도 포함)가 한도에 닿을 때 다음 요청을 기다린다. 0 이면 제한 없음. 무료 등급처럼 분당 요청 수가 정해진 프로바이더에서 429 오류를 피할 때 쓴다. 작업 하나 안에서 센다 (웹 작업은 한 번에 하나씩 처리한다).

`config.yaml`에서 설정:

```yaml
segmentation:
  mode: "llm-assisted"

llm:
  provider: "gemini"
  model: "gemini-3.8-flash"
  base_url: ""           # 로컬 서버(ollama, local)만. 키는 .env 의 GEMINI_API_KEY
  timeout_s: 30.0
  max_retries: 3
  rpm: 10                # 분당 10회까지
```

---

## 7. 화자 분리

pyannote.audio 기반 화자 분리를 지원한다.

### 기본 사용

토큰과 모델 세 개의 사용 조건 동의가 필요하다 ([3.2 diarize](#32-diarize), [OPERATIONS.md](OPERATIONS.md) 2절).

```bash
export HF_TOKEN="hf_xxxxx"   # 또는 .env

# 자동 화자 수 감지
python -m src.main diarize meeting.mp4

# 화자 수 지정 (정확도 향상)
python -m src.main diarize meeting.mp4 --num-speakers 3
```

### 화자 이름 매핑

화자 ID(SPEAKER_00, SPEAKER_01 등)를 실제 이름으로 교체한다.

```bash
# CLI에서
python -m src.main diarize interview.mp4 \
  --speaker-map "SPEAKER_00:진행자,SPEAKER_01:게스트"

# JSON 프로파일로
python -m src.main diarize interview.mp4 \
  --speaker-profile speakers.json
```

`speakers.json` 예시 (단순 형식):

```json
{
  "SPEAKER_00": "김진행",
  "SPEAKER_01": "이게스트",
  "SPEAKER_02": "박패널"
}
```

상세 형식도 읽는다 (`name` 만 쓴다):

```json
{
  "speakers": {
    "SPEAKER_00": {"name": "김진행", "color": "#FF0000"},
    "SPEAKER_01": {"name": "이게스트"}
  }
}
```

### 화자 임베딩 재사용

같은 사람이 나오는 여러 영상에서 화자 ID 를 맞춘다. `--export-embeddings` 로 화자별 임베딩(256차원 벡터)을 `{입력}.embeddings.json` 으로 저장하고, 다음 영상에서 `--speaker-embeddings` 로 그 파일을 주면 코사인 유사도로 같은 사람을 찾아 이전과 같은 화자 ID 를 붙인다. 닮은 사람이 없는 화자는 겹치지 않는 새 ID(`SPEAKER_nn`)를 받는다. 그래서 `--speaker-map`·프로파일의 이름도 회차마다 같은 사람에게 붙는다.

```bash
# 임베딩 저장
python -m src.main diarize episode01.mp4 --export-embeddings

# 다른 영상에서 재사용
python -m src.main diarize episode02.mp4 \
  --speaker-embeddings output/episode01.embeddings.json
```

임베딩은 pyannote.audio 4.x 가 준다 (3.x 는 임베딩을 주지 않아 내보내기·재사용을 건너뛰고 경고를 남긴다).

### ASS 출력 시 화자 스타일

`diarize` 명령어는 기본적으로 ASS 포맷으로 출력하며, 화자별로 다른 색상과 위치가 적용된다.

---

## 8. 오디오 전처리

입력 오디오의 품질을 향상시키는 파이프라인이다. 처리 순서: **배경음 분리 → 음량 정규화 → 노이즈 억제**.
각 단계가 실패하면 경고(`보컬 분리 실패, 원본 사용`, `노이즈 억제 실패, 원본 사용`)를 남기고 원본으로 계속한다. 작업은 실패로 표시되지 않으므로 로그를 확인한다.

### 배경음 분리 (demucs)

음악이나 효과음에서 보컬만 추출한다 (demucs `htdemucs`, 처음 실행 때 모델을 받는다). 결과는 16kHz mono WAV 로 저장해 다음 단계로 넘긴다.

```bash
# enhance 명령어로 단독 사용
python -m src.main enhance noisy_video.mp4 --separate-vocals

# config.yaml에서 활성화
# audio:
#   separate_vocals: true
```

### 음량 정규화

음량을 표준 레벨(-16 LUFS)로 조정한다. 기본 활성화 상태.

### 노이즈 억제

배경 소음을 줄인다.

```bash
python -m src.main enhance interview.mp4 --denoise

# config.yaml에서 활성화
# audio:
#   denoise:
#     enabled: true
```

### Pre-VAD

VAD(Voice Activity Detection)를 STT 이전에 적용하여 발화 구간별로 독립 추론한다. 타임스탬프 정밀도가 향상된다. Silero VAD 를 쓰고, 실패하면 WebRTC VAD 로 대신한다. 2.0초 미만 구간은 forced alignment 를 건너뛴다. 구간마다 체크포인트를 남겨 `--resume`(웹은 다시 시도)으로 이어서 처리한다.

```yaml
audio:
  vad:
    pre_vad: true           # 기본 활성화
    merge_gap_s: 0.5        # 0.5초 미만 묵음은 병합
    min_chunk_s: 1.0        # STT 최소 구간 길이
    max_chunk_s: 30.0       # STT 최대 구간 길이
```

`--no-pre-vad` 플래그로 비활성화하면 전체 오디오를 한 번에 추론 후 사후 분할한다.

---

## 9. 출력 포맷

### SRT

가장 범용적인 자막 포맷. 대부분의 미디어 플레이어에서 지원한다.

```
1
00:00:01,200 --> 00:00:04,800
안녕하세요, 오늘 발표를 시작하겠습니다.

2
00:00:05,100 --> 00:00:08,300
먼저 프로젝트 개요부터 살펴보겠습니다.
```

### ASS

화자별 색상, 위치, 폰트 스타일을 지원한다. 화자 분리 시 기본 포맷.

### VTT (WebVTT)

웹 브라우저의 `<video>` 태그와 호환되는 포맷.

```
WEBVTT

00:00:01.200 --> 00:00:04.800
안녕하세요, 오늘 발표를 시작하겠습니다.
```

### JSON

메타데이터, 단어별 타임스탬프, 신뢰도, 품질 요약을 포함한다. 후처리나 분석에 유용하다.

```json
{
  "metadata": {
    "source_file": "video.mp4",
    "duration_s": 5025.7,
    "language": "ko",
    "model": "whisperx/large-v3",
    "created_at": "2026-10-09T10:00:00+00:00",
    "subwyrhta_version": "0.8.0"
  },
  "segments": [
    {
      "index": 1,
      "text": "안녕하세요",
      "start": 1.2,
      "end": 4.8,
      "confidence": 0.95,
      "speaker": "SPEAKER_00",
      "words": [
        {"word": "안녕하세요", "start": 1.2, "end": 2.1, "confidence": 0.97}
      ]
    }
  ]
}
```

### 출력 파일명 규칙

- 기본: `{입력파일명}.{언어코드}.{확장자}` (예: `video.en.srt`). 언어를 모르면 `unknown`
- 화자 분리: `{입력파일명}.{언어코드}.diarized.{확장자}`
- 충돌 시 자동 넘버링: `video.en_2.srt`, `video.en_3.srt`
- `--output-name` 옵션으로 직접 지정 가능
- 함께 만드는 파일: `{입력}.vad.json`(`--vad-export`), `{입력}.quality.json`(`--quality-report`), `{입력}.embeddings.json`(`--export-embeddings`)

---

## 10. 설정 파일

### 설정 우선순위

```
코드 기본값 → 설정 파일 → 환경변수(.env 포함) → CLI 인자
```

CLI 인자가 가장 높은 우선순위를 가진다.

- 설정 파일은 `--config` 로 준 파일이다. **CLI 는 `--config` 가 없으면 설정 파일을 읽지 않는다** (프로젝트 폴더의 `config.yaml` 도 자동으로 읽지 않는다).
- `serve` 는 `--config` 가 없으면 `<data>/config.yaml` 을 쓰고, 없으면 코드 기본값으로 만든다. 웹 화면의 설정에서 편집한다.
- Python API 의 `SubWyrhta(config_path=…)` 는 인자 값이 설정 파일을 덮어쓴다 ([4. Python API](#4-python-api) 주의).

### config.yaml 전체 구조

저장소의 `config.yaml` 과 같다 (코드 기본값과 다른 값은 `stt.model_dir` 뿐).

```yaml
# SubWyrhta 기본 설정 파일

# STT 엔진
stt:
  model: "whisperx"             # whisperx / faster-whisper / whisper-turbo / parakeet / distil-whisper
  model_size: "large-v3"        # tiny / base / small / medium / large-v3 / turbo
  model_dir: "./models"         # 이 파일을 쓰지 않으면 코드 기본값 ~/.cache/subwyrhta (SUBWYRHTA_MODEL_DIR 가 우선)
  device: "auto"                # auto / cuda / cpu / mps
  compute_type: "auto"          # auto / float16 / float32 / int8
  batch_size: 16
  beam_size: 5
  temperature: 0.0
  source_lang: "auto"

# 오디오 전처리
audio:
  normalize: true
  vad:
    enabled: true
    threshold: 0.5
    min_speech_ms: 250
    min_silence_ms: 100
    # Pre-VAD 구조 설정 (STT 이전 발화 구간 분리)
    pre_vad: true               # true: 발화 구간별 STT / false: 전체 추론 후 사후 재분할
    merge_gap_s: 0.5            # 이 값 미만의 묵음은 인접 발화 구간으로 병합
    min_chunk_s: 1.0            # STT에 전달하는 구간 최소 길이
    max_chunk_s: 30.0           # STT에 전달하는 구간 최대 길이 (강제 분리 기준)
  denoise:
    enabled: false
    model: "noisereduce"        # noisereduce (현재 유일한 값)
  separate_vocals: false        # demucs 배경음 분리

# 세그멘테이션
segmentation:
  mode: "advanced"              # basic / advanced / extreme / llm-assisted
  max_chars_per_line: null      # null이면 언어별 자동 (ja·zh 16, 그 밖 42). 넘으면 두 줄로 나눈다
  max_lines: 2
  max_cps: null                 # null이면 source_lang에 따라 자동 결정 (en:17, ko:9, ja:7, zh:5)
  min_duration_s: 1.0           # 자막 최소 표시 시간 + 최소 표시 시간 보장 (통합 파라미터)
  max_duration_s: 7.0           # Hard Cap: 이 시간 초과 시 신호 무관하게 강제 분할
  min_gap_s: 0.04
  force_split_after_s: 5.0      # advanced 전용: 이 시간 초과 시 다음 절 경계에서 무조건 분할
  dynamic_threshold: true       # advanced 전용: 경과 시간 기반 임계값 동적 감소 활성화
  dynamic_threshold_start_s: 3.0
  dynamic_threshold_min_ratio: 0.4
  # 화자 기반 분할 (diarize 활성화 시에만 적용)
  speaker_split: true
  overlap_text_separator: " / "
  min_chars: 2

# LLM (llm-assisted 세그멘테이션용)
llm:
  # 구독 CLI: claude-code (Claude Pro/Max, claude 로그인) / codex (ChatGPT, codex 로그인)
  # 클라우드 API 키 (.env): gemini (GEMINI_API_KEY) / openrouter (OPENROUTER_API_KEY) / nvidia-nim (NVIDIA_API_KEY)
  # 로컬: ollama (기본 http://localhost:11434/v1) / local (OpenAI 호환 서버, base_url 필수)
  provider: "claude-code"
  model: ""                       # 프로바이더별 모델명. 구독 CLI 는 비우면 CLI 기본 모델
  base_url: ""                    # 로컬 서버(ollama·local)의 엔드포인트. 클라우드는 고정 주소
  timeout_s: 30.0                 # 요청 타임아웃 (초). 구독 CLI 는 시작 시간이 있어 120 이상 권장
  max_retries: 3                  # 재시도 횟수 (exponential backoff)
  rpm: 0                          # 분당 요청 수 한도 (재시도 포함). 0 이면 제한 없음
  disable_thinking: false         # 추론 끄기 (NIM·vLLM 의 chat_template_kwargs). 추론 모델이 느리면 켠다
  reasoning_effort: ""            # 추론 수준 (low / medium / high). 추론 끄기가 먹히지 않는 모델(NIM glm 등)은 low. 비우면 보내지 않는다

# 화자 분리
diarization:
  enabled: false
  num_speakers: "auto"
  hf_token: ""                  # 비워 두고 HF_TOKEN(.env)을 쓴다. 환경 변수가 이 값보다 우선
  speaker_map: {}               # {"SPEAKER_00": "John", ...}

# 노이즈 필터링
noise_filter:
  enabled: true
  languages: ["en", "ja", "ko"]
  confidence_threshold: 0.7     # 저신뢰도 기준 (기본은 품질 리포트에 개수만 남긴다)
  remove_low_confidence: false  # true 면 기준 미만 단어를 지운다 (작은 모델에서 정상 단어도 빠진다)
  remove_interjections: true    # 감탄사·신음만 있는 단어(ああ, んん, はぁ)를 지운다 (일본어)
  custom_patterns: []

# 후처리
postprocess:
  restore_punctuation: false    # 구두점을 내지 않는 엔진에서 켠다 (Whisper 계열은 끄는 것을 권장)
  punctuation_model: false      # deepmultilingualpunctuation 모델 사용
  llm_correction: false         # 소리는 맞고 표기가 틀린 말을 LLM 이 고치고 그 문장을 다시 정렬한다 (용어 목록: .env 의 SUBWYRHTA_GLOSSARY)
  llm_punctuation: false        # 자막을 나누기 전에 LLM(llm 섹션)이 음성 인식 문장에 부호를 넣는다. 분할·줄바꿈이 이 부호를 쓴다
  llm_review: false             # 줄바꿈·문장 부호를 LLM(llm 섹션)이 검토. 글자는 바꾸지 않고 검증한 수정만 받는다
  review_language: ko           # 검수 리포트의 메모·교정 이유를 쓸 언어 (ko, en). 자막 언어가 아니다. 웹 작업은 등록한 화면의 언어를 쓴다

# 장면 감지 (자막 사이가 gap_s 이상 비면 장면 경계, 경계 앞뒤에 boundary_gap_s 보장)
scene:
  enabled: true
  gap_s: 6.0
  min_segments: 2               # 이보다 작은 장면은 이웃과 합친다
  boundary_gap_s: 0.5
  use_video: false              # PySceneDetect(scenedetect 별도 설치)로 영상 컷도 경계로 (느리다)

# 품질 관리 (max_cps, max_chars_per_line은 segmentation 섹션 값을 사용)
quality:
  enabled: true
  auto_fix: true
  export_report: false

# 출력
output:
  dir: "./output"
  format: "srt"                 # srt / ass / vtt / json
  export_words: false

# 로깅
logging:
  level: "INFO"                 # DEBUG / INFO / WARNING / ERROR

# 웹 화면 (python -m src.main serve)
web:
  host: "127.0.0.1"
  port: 8950
  session_hours: 24             # 로그인 유지 시간 (서버 재시작 후 적용)
  max_upload_mb: 4096
  delete_input_after_done: true # 완료한 작업의 업로드 원본 삭제 (자막만 남긴다)
  static_dir: "./web/dist"
```

### 환경변수

환경 변수는 프로젝트 폴더의 `.env` 파일에 둘 수 있다. `cp .env.example .env` 후 쓰는 값만 채운다. CLI·`serve` 가 시작할 때 읽고(다른 파일은 `--env-file`), 셸에 이미 있는 값이 우선하며, 값이 빈 줄은 무시한다. `.env` 는 git 에 올리지 않는다 (`.gitignore`).

| 변수 | 설명 |
|------|------|
| `HF_TOKEN` | HuggingFace 인증 토큰 (화자 분리) |
| `SUBWYRHTA_MODEL_DIR` | 모델 캐시 디렉토리 오버라이드 |
| `SUBWYRHTA_DEVICE` | 디바이스 오버라이드 |
| `GEMINI_API_KEY` | Google Gemini API 키 |
| `OPENROUTER_API_KEY` | OpenRouter API 키 |
| `NVIDIA_API_KEY` | NVIDIA NIM API 키 (호스팅) |
| `SUBWYRHTA_GLOSSARY` | LLM 오인식 교정의 용어 목록 경로 (파일, 또는 `common.txt`·`<입력 이름>.txt` 가 있는 폴더. 한 줄에 용어 하나, `#` 주석). 없으면 교정하지 않고 검수 리포트(`suggested`)에 제안만 남긴다 |
| `SUBWYRHTA_LLM_EXTRA_BODY` | OpenAI 호환 LLM 요청 본문에 합칠 JSON. 모델마다 다른 추론 끄기 (예: llama-server Qwen3 `{"chat_template_kwargs": {"enable_thinking": false}}`, NIM glm `{"reasoning_effort": "low"}`, 모델별 `{"models": {...}}`). docs/OPERATIONS.md 3.1 |

### 용도별 설정 예시

#### 한국어 강연 자막

```yaml
stt:
  model: "whisperx"
  model_size: "large-v3"
  source_lang: "ko"
segmentation:
  mode: "advanced"
  max_cps: 9
  max_chars_per_line: 30
output:
  format: "srt"
```

#### 영어 팟캐스트 + 화자 분리

```yaml
stt:
  model: "whisperx"
  source_lang: "en"
diarization:
  enabled: true
  num_speakers: "2"
segmentation:
  speaker_split: true
output:
  format: "ass"
```

#### 저사양 환경 (CPU only)

```yaml
stt:
  model: "faster-whisper"
  model_size: "small"
  device: "cpu"
  compute_type: "int8"
  batch_size: 4
```

---

## 11. 여러 파일 처리와 재개

### 입력 방식

```bash
# 디렉토리 (하위 폴더까지: --recursive)
python -m src.main transcribe ./videos/ --recursive

# glob 패턴 (셸이 펼치지 않게 따옴표)
python -m src.main transcribe "./videos/**/*.mkv" --recursive

# 파일 목록 (한 줄에 하나, # 은 주석)
python -m src.main transcribe --file-list files.txt
```

`files.txt`:

```
/path/to/video1.mp4
/path/to/video2.mkv
# /path/to/skip.mp4
```

지원하지 않는 확장자는 건너뛴다. 파일을 하나 직접 줄 때만 `--output-name` 을 쓸 수 있다.

### 병렬 처리와 GPU

`--workers N` 이면 worker 프로세스 N 개가 파일을 나눠 처리한다. worker 마다 모델을 한 번 불러와 다음 파일에 다시 쓴다. GPU 는 worker 하나만 쓰고 나머지는 CPU 로 돈다 (설정 디바이스가 `cuda`/`mps` 이거나 `auto` 에서 GPU 를 찾았을 때). `--workers-device` 를 주면 모든 worker 가 그 디바이스를 쓴다. worker 마다 모델이 메모리에 올라가므로 메모리를 확인하고 늘린다.

```bash
python -m src.main transcribe ./videos/ --workers 3 --failed-list failed.txt
```

진행 막대가 전체 파일 수·완료 수·실패 수·남은 시간을 보인다. 실패한 파일은 건너뛰고 나머지를 계속 처리하며, 끝에 실패 목록을 출력하고 `--failed-list` 에 저장한다. 하나라도 실패하면 종료 코드는 1 이다. 실패 목록은 그대로 `--file-list` 로 다시 줄 수 있다.

### 체크포인트와 재개

처리 중 남기는 것은 두 가지다.

| 무엇 | 위치 | 다시 할 때 건너뛰는 단계 |
|------|------|--------------------------|
| 체크포인트 | `<입력파일>.checkpoint.json` | 음성 인식 (Pre-VAD 발화 구간(청크) 단위로 끝낸 구간) |
| 단계 결과 | `<입력파일>.work/` | 오디오 추출·전처리(demucs·음량 정규화·노이즈 억제), Pre-VAD, 화자 분리 |

- 중단된 뒤 같은 입력으로 `--resume` 을 주면 끝낸 단계와 구간을 건너뛰고 이어서 처리한다. `--no-resume` 은 둘 다 지우고 처음부터 처리한다. 처리를 마치면 둘 다 지운다.
- 입력 파일(크기·수정 시각)이나 그 단계에 영향을 주는 설정이 바뀌면 그 단계부터 다시 한다. 예: VAD 설정만 바꾸면 전처리 음성은 그대로 쓰고 발화 구간부터 다시 찾는다.
- 체크포인트는 발화 구간 경계까지 맞아야 쓴다. 전처리 결과가 달라져 구간이 바뀌면 음성 인식을 처음부터 한다.
- `<입력파일>.work/` 에는 전처리한 음성(16kHz mono WAV)이 들어 있어 1시간 영상이면 약 115MB 를 쓴다.

```bash
python -m src.main transcribe ./videos/ --resume
```

웹 화면은 작업 폴더의 `checkpoint.json` 과 `work/` 를 쓰고, 실패·취소한 작업을 **다시 시도**하면 이어서 처리한다 (서버가 멈춰 중단된 작업 포함). 작업을 지우면 함께 지워진다.

---

## 12. 품질 관리

### 자동 품질 검사

transcribe/diarize 실행 시 자동으로 품질을 검사한다.

```yaml
quality:
  enabled: true       # 품질 검사 활성화
  auto_fix: true      # 자동 교정 (아래 표)
  export_report: false # 품질 리포트 JSON 내보내기 ({입력}.quality.json)
```

CPS·줄 길이 기준은 `segmentation` 섹션 값(`max_cps`, `max_chars_per_line`)을 쓴다.

### 검사 항목

| 항목 | 심각도 | 자동 교정 |
|------|--------|-----------|
| 타임코드 역전 (끝 < 시작) | Error | 시작·끝을 맞바꾼다 |
| 타임코드 겹침 | Warning | 앞 자막의 끝을 다음 자막 시작으로 당긴다 |
| 줄 길이 초과 | Error | 줄 바꿈 |
| CPS 초과 | Warning | 없음 (분할은 세그멘테이션의 CPS 최적화가 한다) |
| 최소 표시 시간 미달 | Warning | 없음 (병합은 세그멘테이션의 최소 표시 시간 보장이 한다) |
| 빈 자막 | Error | 세그먼트 제거 |

### 품질 등급

세그먼트 평균 신뢰도로 매긴다.

| 등급 | 평균 신뢰도 |
|------|-------------|
| A | 0.9 이상 |
| B | 0.8 이상 |
| C | 0.7 이상 |
| D | 0.7 미만 |

---

### 장면 감지

세그멘테이션 뒤에 장면 경계를 찾아 경계 앞뒤 자막 사이에 간격을 둔다 (기본 켬).

- 자막 사이가 `scene.gap_s`(기본 6초) 이상 비면 장면 경계로 본다. 자막이 `scene.min_segments`(기본 2)개보다 적은 장면은 이웃 장면과 합친다.
- 경계 앞뒤 자막 사이에 `scene.boundary_gap_s`(기본 0.5초)를 보장한다.
- `scene.use_video: true` 이면 PySceneDetect(`scenedetect` 별도 설치)로 영상의 컷도 경계로 쓴다. 영상 전체를 읽어 느리고, `--start` 로 앞을 자른 실행에서는 건너뛴다.

```yaml
scene:
  enabled: true
  gap_s: 6.0
  min_segments: 2
  boundary_gap_s: 0.5
  use_video: false
```

---

## 13. 활용 예시

### YouTube 영상 자막

```bash
# yt-dlp로 다운로드 후 자막 생성
yt-dlp -o video.mp4 "https://youtube.com/watch?v=xxxxx"
python -m src.main transcribe video.mp4 --source-lang en
```

### 회의록 자동 생성

```bash
# 화자 분리 + JSON 출력 (후처리용)
python -m src.main diarize meeting.mp4 \
  --num-speakers 4 \
  --speaker-map "SPEAKER_00:김팀장,SPEAKER_01:이대리,SPEAKER_02:박과장,SPEAKER_03:최사원" \
  --output-format json
```

### 강의 자막 (한국어)

```bash
python -m src.main transcribe lecture.mp4 \
  --source-lang ko \
  --output-format vtt
```

### 시리즈 영상 일괄 처리

```bash
# 폴더 안의 영상 모두 (2개씩 병렬, 실패 목록 저장)
python -m src.main transcribe ./season1/ \
  --source-lang ja \
  --output-dir ./subtitles/season1/ \
  --workers 2 --failed-list ./subtitles/season1/failed.txt
```

### 노이즈 많은 현장 영상

```bash
python -m src.main transcribe field_recording.mp4 \
  --source-lang ko \
  --denoise \
  --separate-vocals
```

### 최고 품질 자막 (LLM 분할)

```bash
# Claude 구독 (서버 계정에서 claude auth login)
python -m src.main transcribe documentary.mp4 \
  --source-lang en \
  --model whisperx \
  --model-size large-v3 \
  --segmentation-mode llm-assisted \
  --llm-provider claude-code
```

---

## 14. 문제 해결

### CUDA 메모리 부족

```
RuntimeError: CUDA out of memory
```

해결 방법:

```bash
# 1. 모델 크기 줄이기
python -m src.main transcribe video.mp4 --model-size medium

# 2. compute_type 변경
python -m src.main transcribe video.mp4 --compute-type int8

# 3. 배치 크기 줄이기
python -m src.main transcribe video.mp4 --batch-size 4

# 4. CPU 사용
python -m src.main transcribe video.mp4 --device cpu
```

### ffmpeg를 찾을 수 없음

```
AudioExtractionError: ffmpeg를 찾을 수 없습니다 …
```

해결 방법:

```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# snap 설치 확인
snap list ffmpeg
```

### HuggingFace 인증 오류

```
HuggingFace 토큰이 필요합니다 …
화자 분리 모델을 불러올 수 없습니다. HF_TOKEN(.env)이 유효한 Hugging Face 토큰(hf_…)인지 … 원인: …
```

해결 방법:

1. https://huggingface.co/settings/tokens 에서 **Read** 토큰 발급 (`hf_` 로 시작)
2. 토큰을 만든 계정으로 세 모델(`pyannote/speaker-diarization-3.1`, `pyannote/segmentation-3.0`, `pyannote/speaker-diarization-community-1`) 페이지에서 사용 동의. 원인 줄의 `401 … Invalid user token` 은 토큰 문제, `403 … not in the authorized list` 는 그 모델의 동의 문제다
3. `.env` 의 `HF_TOKEN` 에 넣고 (웹은 서버 재시작) 다시 실행

확인 명령과 자세한 절차는 [OPERATIONS.md](OPERATIONS.md) 2절·6절.

### 언어 자동 감지 실패

`--source-lang auto` 이고 Pre-VAD 를 켜면, 3초 이상인 발화 구간 중 시간 순서로 고르게 5개를 골라 각각 언어를 감지하고 다수결로 정한다. 이 표본의 인식 결과 중 정한 언어와 같은 것은 그대로 쓰고, 다른 것만 다시 인식한다. 그래도 짧은 오디오나 신음·노래가 많은 오디오에서는 틀릴 수 있다.

```bash
# 언어를 명시적으로 지정
python -m src.main transcribe video.mp4 --source-lang ko
```

### 자막 타이밍이 부정확한 경우

```bash
# 1. Pre-VAD 활성화 확인 (기본 활성화)
# config.yaml에서 audio.vad.pre_vad: true

# 2. WhisperX 엔진 사용 (forced alignment)
python -m src.main transcribe video.mp4 --model whisperx

# 3. beam_size 증가
python -m src.main transcribe video.mp4 --beam-size 10
```

### 지원 파일 형식

**영상:** `.mp4`, `.mkv`, `.avi`, `.mov`, `.webm`, `.mpeg`, `.mpg`, `.wmv`, `.flv`, `.m4v`

**음성:** `.mp3`, `.wav`, `.m4a`, `.aac`, `.flac`, `.ogg`, `.opus`, `.wma`, `.mka`

---

## 15. 웹 화면 (serve)

로컬 서버로 웹 화면을 띄워 브라우저에서 파일을 올리고 자막을 받는다. 준비·재시작·문제 해결 절차는 [OPERATIONS.md](OPERATIONS.md) 를 본다.

```bash
pip install -r requirements.txt
python -m src.main serve --data ./data          # http://127.0.0.1:8950
```

저장소에는 빌드한 화면(`web/dist`)이 들어 있어 Node.js 없이 바로 쓸 수 있다. 화면 소스(`web/src`)를 고쳐 다시 빌드하려면 `web/vendor/vue-smartview` 가 있어야 한다. 비공개 vue-smartview 의 부분 빌드라 공개 저장소에는 없다 (관리자는 `scripts/update-smartview.sh <VUE-SMARTVIEW 경로>` 로 만든다).

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--data <폴더>` | 데이터 폴더 (작업·업로드·로그·계정·설정) | 필수 |
| `--host` / `--port` | 바인딩 주소, 포트 | 설정 `web.host` / `web.port` (`127.0.0.1` / `8950`) |
| `--static-dir <폴더>` | 화면 빌드 폴더 | 설정 `web.static_dir` (`./web/dist`) |

전역 `--config` 가 없으면 `<data>/config.yaml` 을 쓰고 없으면 만든다. 작업은 등록 순서대로 한 번에 하나씩, 작업마다 자식 프로세스로 처리한다.

- 실패·취소한 작업, 서버가 멈춰 중단된 작업은 **다시 시도**하면 끝낸 단계를 건너뛰고 이어서 처리한다 (11절 "체크포인트와 재개"). 작업 폴더 `<data>/jobs/<id>/` 에 `checkpoint.json` 과 `work/` 가 남는다.
- 서버는 띄운 터미널이 닫히면 함께 끝난다. 오래 걸리는 작업을 맡길 때는 `setsid nohup python -m src.main serve --data ./data > serve.log 2>&1 &` 처럼 떼어 띄운다.

| 화면 | 하는 일 |
|------|---------|
| 처음 설정·로그인 | 처음 접속하면 관리 계정을 만든다 |
| 대시보드 | 워커 상태, 대기·완료·실패 수, 현재 작업 진행, 시스템(디바이스·ffmpeg·HF_TOKEN·디스크), 패키지 설치 여부, 기본 처리 설정 |
| 작업 대기열 | 새 작업(파일 업로드 + 작업별 옵션: 엔진, 모델 크기, 언어, 연산 정밀도, 출력 포맷 여러 개, 세그멘테이션 모드, 최대 CPS, Pre-VAD, 음량 정규화, 노이즈 억제, 배경음 분리, 화자 분리·화자 수, 구두점 복원, 품질 리포트, 단어 JSON), 진행 상황, 취소, 다시 시도(실패 모두), 삭제, 상세 |
| 완료 목록 | 출력 파일 미리 보기·내려받기, 자막 검수, 삭제, 끝난 작업 모두 지우기 |
| 자막 검수 | 완료 목록의 행에서 연다 (SRT 출력이 있는 작업). SRT 자막과 LLM 검수 리포트의 메모를 표로 보고 자막을 고친다 |
| 로그 | 서버·작업 로그 (수준·검색 필터, 자동 갱신, 일시 정지, 항목 복사) |
| 설정 | config.yaml 편집 (탭: 음성 인식, 오디오·VAD, 세그멘테이션, 화자·후처리, 품질·출력(장면 감지 포함), LLM, 웹), 계정 변경. LLM 탭은 고른 프로바이더의 준비 상태(CLI 설치·키 설정)를 보인다 |

- 화면 언어 한국어/영어, 테마 라이트/다크.
- 설정에서 저장한 값은 다음 작업부터 적용된다 (`web.session_hours`·서버 로그 수준은 서버 재시작 후). 설정 파일에 쓸 수 없으면 읽기 전용으로 보인다.
- 토큰·API 키는 화면에서 받지 않는다. `.env` 에 넣고 서버를 다시 시작한다.

### 자막 검수

완료 목록에서 작업 행의 **자막 검수** 아이콘을 누른다. LLM 마지막 검토(`postprocess.llm_review`)를 켠 작업은 검수 리포트(`<이름>.review.json`)의 항목이 자막에 붙어 보인다. 리포트가 없는 작업도 자막은 고칠 수 있다. 리포트의 메모는 자막 언어가 아니라 작업을 등록한 화면의 언어(한국어·영어)로 쓰인다 (자막에서 인용한 낱말은 그대로). 2026-10-12 이전에 만든 리포트는 메모가 자막 언어로 남아 있을 수 있다.

| 열 | 내용 |
|----|------|
| 시작 / 끝 | 자막 시각 (`HH:MM:SS.mmm`) |
| 자막 | SRT 자막 글자 (줄바꿈은 `↵` 로 보인다) |
| 검수 종류 | 오인식 의심, 문장 부호, 기타, 교정 제안(반영하지 않은 LLM 교정), 반영한 교정 |
| 상태 | 검수 완료(저장한 자막), 확인 필요(검수 항목이 있고 아직 저장하지 않음), 미검수 |
| 메모 | 리포트의 메모. 교정은 `바꾸기 전 → 바꾼 뒤 (이유)` |

- 자막 전체가 시간 순서로 보인다 (고칠지 말지는 앞뒤 자막을 봐야 정할 수 있다). **검수 항목만** 스위치를 켜면 메모가 있는 자막만 보이고, 검수 종류 필터와 검색으로 좁힌다.
- 자막을 누르면 수정 창이 열린다. 메모를 보고 글자를 고쳐 저장하면 SRT 파일에 바로 쓴다 (번호와 시각은 그대로). 빈 줄은 빼고, 500자를 넘거나 `-->` 가 든 글자는 받지 않는다.
- **저장을 누른 자막은 상태가 검수 완료로 바뀐다.** 고칠 것이 없으면 글자를 그대로 두고 저장한다 (이때는 SRT 파일을 다시 쓰지 않는다). 검수 상태는 작업 폴더의 `review-state.json` 에 남아 화면을 다시 열어도 유지되고, 작업을 지우면 함께 지워진다.
- 처음 고칠 때 원본을 `<이름>.orig.srt` 로 남긴다. 상세 창의 출력 파일에서 내려받을 수 있다.
- 고친 글자는 SRT 에만 들어간다. 같은 작업의 VTT·ASS·JSON 출력과 자막 시각은 바꾸지 않는다.
- 리포트의 시각은 마지막 검토 전 자막의 것이라, 메모는 시각이 가장 많이 겹치는 자막에 붙는다.

