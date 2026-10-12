# IMPLEMENTED.md — SubWyrhta 구현 현황 문서

> 영상/음성 파일에서 고품질 자막(SRT/ASS/VTT/JSON)을 자동 생성하는 독립형 STT 도구
> 버전 0.8.0 (pyproject), Python 3.12+, CLI(click) + Python API + 웹 화면(serve, FastAPI + Vue 3)

---

## 1. 프로젝트 구조

```
src/
├── main.py                      # CLI 진입점 (click 서브커맨드, serve 포함)
├── api.py                       # Python API
├── pipeline.py                  # 전체 파이프라인 오케스트레이터 (CLI·API·웹 공통)
├── batch_processor.py           # 배치 입력 수집(디렉토리·glob·목록), 병렬 처리(spawn, GPU worker 하나)
├── checkpoint.py                # Pre-VAD 청크 체크포인트 저장/복원 (<입력>.checkpoint.json)
├── stage_cache.py               # 단계 결과 보관: 전처리 음성·Pre-VAD·화자 분리 (<입력>.work/, 웹은 작업 폴더 work/)
├── progress.py                  # 배치 진행 표시 (tqdm)
├── version.py                   # 버전 (pyproject.toml → 패키지 메타데이터)
├── core/
│   ├── config.py                # 설정 로더 (YAML + 환경변수 + CLI), .env 로더 (load_env_file)
│   ├── exceptions.py            # 예외 계층 구조
│   ├── logging.py               # structlog 설정
│   └── text.py                  # 단어 연결 (한자·가나는 공백 없이, 라틴·한글은 공백)
├── models/
│   ├── word.py                  # WordSegment 데이터 모델
│   ├── segment.py               # TranscriptionSegment 데이터 모델
│   └── result.py                # TranscriptionResult, Metadata, QualitySummary
├── audio/
│   ├── extractor.py             # ffmpeg 오디오 추출 + 챕터 감지
│   ├── vad.py                   # Silero VAD (실패 시 WebRTC VAD) + Pre-VAD 구조
│   ├── enhancer.py              # 음량 정규화 + 노이즈 억제
│   └── vocal_separator.py       # demucs 배경음 분리 (보컬을 16kHz mono WAV 로 저장)
├── engines/
│   ├── base.py                  # STTEngine ABC + 디바이스 감지
│   ├── engine_factory.py        # 엔진 팩토리
│   ├── whisperx_engine.py       # WhisperX (배치 추론 + forced alignment)
│   ├── faster_whisper_engine.py # faster-whisper (CTranslate2)
│   ├── whisper_turbo_engine.py  # whisper-turbo
│   └── parakeet_engine.py       # NVIDIA NeMo Parakeet
├── segmentation/
│   ├── base.py                  # SegmentationStrategy ABC
│   ├── segmentation_factory.py  # 세그멘테이션 팩토리
│   ├── basic.py                 # basic 모드 (PySBD 문장 경계)
│   ├── advanced.py              # advanced 모드 (Pause+Clause 신호)
│   ├── extreme.py               # extreme 모드 (semantic 유사도)
│   ├── llm_assisted.py          # llm-assisted 모드 (LLM 재조정)
│   ├── llm_client.py            # LLM 프로바이더 (구독 CLI·클라우드 API 키·로컬) + RPM 제한 + 준비 상태
│   ├── word_boundary_snapper.py # 단어 경계 스냅
│   ├── speaker_splitter.py      # 화자 기반 분할
│   ├── speed_optimizer.py       # CPS 최적화
│   └── min_display_guard.py     # 최소 표시 시간 보장
├── postprocess/
│   ├── noise_filter.py          # 노이즈 필터 (필러워드, 반복, 패턴)
│   ├── normalizer.py            # 텍스트 정규화
│   └── punctuation.py           # 구두점 복원
├── quality/
│   ├── checker.py               # 품질 검사 + 자동 교정
│   ├── reporter.py              # 품질 리포트 생성
│   └── scene_detector.py        # 장면 감지 (시간 간격, PySceneDetect) + 경계 자막 간격
├── diarization/
│   ├── diarizer.py              # pyannote 화자 분리 + 사전 확인 (check_diarization_access)
│   ├── embedding_store.py       # 화자 임베딩 저장/로드 + 이전 화자 대조 (코사인 유사도)
│   └── speaker_mapper.py        # 화자 이름 매핑
├── web/
│   ├── server.py                # FastAPI 앱, 세션 라우터, serve 실행
│   ├── api.py                   # 상태·작업·업로드·다운로드·설정·로그·계정 API
│   ├── auth.py                  # 관리 계정(scrypt), 세션 쿠키 서명
│   ├── context.py               # 요청 컨텍스트 (WebContext), 세션 확인 의존성
│   ├── jobs.py                  # SQLite 작업 큐, 작업 폴더
│   ├── worker.py                # 작업 워커 (한 번에 하나, 작업마다 spawn 자식 프로세스)
│   ├── runner.py                # 자식 프로세스 진입점 (파이프라인 실행, 진행 보고)
│   ├── job_options.py           # 작업별 옵션 검증 → 설정 오버라이드
│   ├── settings_io.py           # config.yaml 편집 값 검증·저장 (주석 유지)
│   ├── logs.py                  # JSON 로그 이어 읽기 (20MB 넘으면 회전)
│   ├── secret_store.py          # secrets.json (권한 600)
│   └── system_probe.py          # 디바이스·패키지·ffmpeg 점검 (별도 프로세스)
└── writers/
    ├── base.py                  # SubtitleWriter ABC
    ├── writer_factory.py        # 출력 팩토리
    ├── srt_writer.py            # SRT 출력
    ├── ass_writer.py            # ASS 출력 (화자별 색상/위치)
    ├── vtt_writer.py            # WebVTT 출력
    └── json_writer.py           # JSON 출력 (메타데이터 + 단어 타임스탬프)
```

---

## 2. 핵심 파이프라인

`src/pipeline.py`의 `TranscriptionPipeline`이 아래 순서를 실행한다. CLI `transcribe`·`diarize`·`--split-by-chapters`, Python API, 웹 작업이 모두 이 클래스를 쓴다. Pre-VAD 를 켜면 청크별로 추론하고 청크마다 체크포인트를 남긴다. 체크포인트 키에는 설정과 청크 경계가 들어간다 (경계가 바뀌면 쓰지 않는다). 전처리 음성·Pre-VAD·화자 분리 결과는 단계 결과 폴더에 입력·설정 키와 함께 남겨, 다시 시도할 때 키가 같으면 그 단계를 건너뛴다 (`--resume`, 웹의 다시 시도). 성공하면 둘 다 지운다. 구두점 복원은 `postprocess.restore_punctuation`(기본 끔)일 때 STT 세그먼트 단계에서 문장 끝 부호만 반영한다.

```
입력 파일
  ↓
[0] 화자 분리 사전 확인 — 화자 분리를 켰으면 pyannote 파이프라인을 한 번 불러와 내린다 (토큰·사용 조건 문제를 STT 전에)
  ↓
[1] AudioExtractor — ffmpeg 16kHz mono WAV 추출, 트랙 선택, 시간 범위, 챕터
  ↓
[2] AudioEnhancer — demucs 배경음 분리 → 음량 정규화 → 노이즈 억제 (선택, 실패하면 경고 후 원본)
  ↓
[3] VAD — Silero(실패 시 WebRTC) 발화 구간 감지 (Pre-VAD: STT 이전 적용)
  ↓
[4] STT Engine — 모델 로드 → 청크(또는 전체) 추론 → WordSegment 생성
  ↓
[5] Diarization — pyannote 화자 분리 (선택)
  ↓
[6] STT 후처리 — 단어 보완, 일본어 문절 묶기, 늘어진 단어 시간 자르기, 몰린 단어 시각 다시 나누기, NoiseFilter(필러워드/반복/패턴, 저신뢰도 단어는 기본 마킹만), LLM 오인식 교정과 다시 정렬(선택, postprocess.llm_correction), LLM 부호 넣기(선택, postprocess.llm_punctuation, 결과는 단계 캐시), 구두점 복원(선택)
  ↓
[7] Segmentation — 화자 할당·분할 → 모드별 분할 → CPS 최적화 → 최소 표시 시간 보장 → 읽기 시간 확보 → 발화 구분 → 텍스트 정규화 → 앞에 붙은 부호 옮기기 → 줄바꿈 → LLM 마지막 검토(선택, postprocess.llm_review: 경계 옮기기·줄바꿈·검수 리포트 <이름>.review.json)
  ↓
[8] 화자 이름 적용 (화자 분리 시, speaker_map·프로파일. --speaker-embeddings 면 그 전에 이전 화자 ID 로 맞춘다)
  ↓
[8.5] 장면 감지 — 긴 공백(선택: 영상 컷)을 경계로, 경계 앞뒤 자막 간격 보장 (scene 설정)
  ↓
[9] QualityChecker — 역전/겹침/줄길이/CPS/초단/빈 자막 검사 + 자동 교정 (역전·겹침·줄길이·빈 자막)
  ↓
[10] SubtitleWriter — SRT/ASS/VTT/JSON 출력 (+ 품질 리포트, 화자 임베딩)
```

### 시간 제약 순서

| 제약 | 기본값 | 동작 |
|------|--------|------|
| `force_split_after_s` | 5.0초 | 이 시간 초과 시 다음 절 경계에서 분할 |
| `max_duration_s` | 7.0초 | Hard Cap — 신호 무관하게 강제 분할 |
| `min_duration_s` | 1.0초 | 이 미만 세그먼트 → 인접 세그먼트 병합 |

---

## 3. STT 엔진

| 엔진 | CLI 옵션 | 특징 |
|------|----------|------|
| WhisperX | `--model whisperx` | 배치 추론 + phoneme-level forced alignment |
| faster-whisper | `--model faster-whisper` | CTranslate2 고속 추론, distil-whisper 호환 |
| whisper-turbo | `--model whisper-turbo` | OpenAI turbo 모델 (faster-whisper 백엔드, model_size 무시) |
| Parakeet | `--model parakeet` | NVIDIA NeMo 영어 특화 ASR |
| distil-whisper | `--model distil-whisper` | faster-whisper 별칭. distil 모델은 `--model-size distil-large-v3` 등으로 고른다 |

### 디바이스 폴백 체인

```
cuda/float16 → cuda/int8 → cpu/float32
mps/float16  → cpu/float32
cpu/float16  → cpu/float32
```

---

## 4. 세그멘테이션 모드

### basic
PySBD 문장 경계 감지 + 단어 경계 스냅. 빠른 규칙 기반 분할.

### advanced (기본값)
Pause(발화 휴지) + Clause(구두점) 복합 신호 기반 분할.
- 동적 임계값 감소 (3초부터 선형 감소)
- `force_split_after_s` (5초) + Hard Cap `max_duration_s` (7초)
- 글자 수 트리거 (`max_chars_per_line × max_lines`)

### extreme
advanced 1차 분할 + sentence-transformers semantic 유사도 2차 분할.
의미적 단절 지점에서 추가 분할. 미설치 시 advanced 폴백.

### llm-assisted
advanced 1차 분할 결과를 윈도우(10세그먼트, 2 overlap) 단위로 LLM에 전달.
분할점 재조정 후 단어 경계 스냅 + 제약 조건 재검증.
LLM 실패 시 advanced 자동 폴백.

| provider | 방식 |
|----------|------|
| `codex`, `claude-code` | 구독 (OpenAI·Claude 는 구독만): 서버에 로그인한 CLI 를 비대화 모드로 호출 (`codex exec`, `claude -p`). API 키 환경 변수는 CLI 에 넘기지 않는다 |
| `gemini`, `openrouter`, `nvidia-nim` | 클라우드 API 키 (`.env` 의 정해진 이름), 고정 OpenAI 호환 엔드포인트 (NIM 은 클라우드만) |
| `ollama`, `local` | 로컬 OpenAI 호환 서버 (`base_url`, `local` 은 필수) |

설정 화면 LLM 탭은 고른 프로바이더의 준비 상태(`GET /api/settings` 의 `llm`: CLI 설치·키 설정 여부, 값은 내보내지 않음)를 배너로 보이고, 엔드포인트 칸은 로컬 서버일 때만 보인다.

`llm.rpm`: 분당 요청 수 한도 (재시도 포함, 0 = 제한 없음). 작업 하나 안에서 최근 60초 요청 시각을 세어 기다린다.

### CPS 언어별 기본값

| 언어 | max_cps |
|------|---------|
| English | 17 |
| Korean | 9 |
| Japanese | 7 |
| Chinese | 5 |

---

## 5. 출력 포맷

| 포맷 | 확장자 | 특징 |
|------|--------|------|
| SRT | `.srt` | `HH:MM:SS,mmm` 표준 자막 |
| ASS | `.ass` | 화자별 색상 8종 + 2인 좌/우 배치 |
| WebVTT | `.vtt` | `HH:MM:SS.mmm` 웹 자막 |
| JSON | `.json` | 메타데이터 + 단어 타임스탬프 + 품질 요약 |

파일명: `{입력}.{언어}.{포맷}` (화자 분리 시 `{입력}.{언어}.diarized.{포맷}`, 언어를 모르면 `unknown`). 충돌 시 자동 넘버링: `video.en_2.srt`.
함께 만드는 파일: `{입력}.vad.json`(`--vad-export`), `{입력}.quality.json`(`--quality-report`), `{입력}.embeddings.json`(`--export-embeddings`).

---

## 6. CLI 커맨드

### `transcribe` — 음성/영상 → 자막 변환

```bash
python -m src.main transcribe <input> [options]
```

주요 옵션:
- `--output-dir`, `--output-format`, `--output-name`
- `--model`, `--model-size`, `--model-dir`, `--source-lang`, `--device`, `--compute-type`
- `--batch-size`, `--beam-size`, `--temperature`
- `--llm-provider`(codex/claude-code/gemini/openrouter/nvidia-nim/ollama/local), `--llm-model`, `--llm-base-url`
- 전처리·VAD: `--denoise`, `--separate-vocals`, `--normalize-audio`, `--vad`, `--pre-vad` (각각 `--no-…`), `--vad-export`
- `--segmentation-mode`, `--max-cps`, `--diarize`, `--num-speakers`, `--hf-token`
- `--qc`/`--no-qc`, `--quality-report`, `--export-words`, `--resume`/`--no-resume`
- 배치: 입력은 파일·디렉토리·glob, `--recursive`, `--file-list`, `--workers`, `--workers-device`, `--failed-list` (실패 파일은 건너뛰고 하나라도 실패하면 종료 코드 1)
- `--split-by-chapters`, `--keep-audio`, `--audio-track`
- `--start`, `--end`, `--dry-run`

### `diarize` — 화자 분리 포함 자막 생성

```bash
python -m src.main diarize <input> [options]
```

출력·구간 옵션, STT 옵션 일부(`--model`, `--model-size`, `--source-lang`, `--device`), LLM 옵션, `--dry-run` 에 더해:
`--num-speakers`, `--hf-token`, `--speaker-map`, `--speaker-profile`, `--export-embeddings`, `--speaker-embeddings`(이전 화자와 같은 ID), `--no-speaker-split`. 기본 포맷 ASS.

### `validate` — 자막 품질 검사

```bash
python -m src.main validate <subtitle_file> [--auto-fix] [--output PATH] [--max-chars N] [--max-cps N] [--min-duration S]
```

### `enhance` — 오디오 품질 향상 전용

```bash
python -m src.main enhance <input> [--output-dir DIR] [--denoise] [--separate-vocals] [--normalize/--no-normalize] [--device D]
```

### `info` — 파일 정보 출력

```bash
python -m src.main info <input>
```

### `list-models` — 엔진/모델/디바이스 목록

```bash
python -m src.main list-models
```

### `serve` — 웹 화면 서버

```bash
python -m src.main serve --data DIR [--host H] [--port P] [--static-dir DIR]
```

14절을 본다. 전역 옵션: `--config`, `--env-file`, `--verbose`, `--quiet`, `--version`.

---

## 7. 설정 체계

### 병합 순서

```
코드 기본값 → 설정 파일 → 환경변수 → CLI 인자
```

- 설정 파일: CLI 는 `--config` 로 준 파일만 읽는다 (없으면 읽지 않음). `serve` 는 `--config` 가 없으면 `<data>/config.yaml` (없으면 만든다)
- 환경변수: 셸 값 또는 `.env` (CLI·serve 시작 때 `load_env_file`, 셸 값 우선, 빈 값 무시, `--env-file` 로 다른 파일)
- Python API: `SubWyrhta` 인자와 `TranscribeConfig` 값은 지정한 것(None 이 아닌 것)만 설정 파일을 덮어쓴다. `output_dir`·`output_format` 을 주면 파일도 쓴다

### 환경변수

| 변수 | 용도 | 설정 경로 |
|------|------|-----------|
| `HF_TOKEN` | HuggingFace 인증 | `diarization.hf_token` |
| `SUBWYRHTA_MODEL_DIR` | 모델 캐시 경로 | `stt.model_dir` |
| `SUBWYRHTA_DEVICE` | 디바이스 오버라이드 | `stt.device` |
| `GEMINI_API_KEY` | Google Gemini API 키 | llm `gemini` |
| `OPENROUTER_API_KEY` | OpenRouter API 키 | llm `openrouter` |
| `NVIDIA_API_KEY` | NVIDIA NIM API 키 | llm `nvidia-nim` |

### 설정 데이터클래스

| 클래스 | 섹션 | 주요 필드 |
|--------|------|-----------|
| `STTConfig` | `stt` | model, model_size, device, compute_type, batch_size |
| `AudioConfig` | `audio` | normalize, vad(VADConfig), denoise(DenoiseConfig), separate_vocals |
| `VADConfig` | `audio.vad` | enabled, threshold, pre_vad, merge_gap_s, min/max_chunk_s |
| `DenoiseConfig` | `audio.denoise` | enabled, model |
| `SegmentationConfig` | `segmentation` | mode, max_chars_per_line, max_cps, min/max_duration_s |
| `LLMConfig` | `llm` | provider, model, base_url, timeout_s, max_retries, rpm |
| `DiarizationConfig` | `diarization` | enabled, num_speakers, hf_token, speaker_map |
| `NoiseFilterConfig` | `noise_filter` | enabled, languages, confidence_threshold, remove_low_confidence, custom_patterns |
| `PostprocessConfig` | `postprocess` | restore_punctuation, punctuation_model, llm_correction, llm_punctuation, llm_review |
| `SceneConfig` | `scene` | enabled, gap_s, min_segments, boundary_gap_s, use_video |
| `QualityConfig` | `quality` | enabled, auto_fix, export_report |
| `OutputConfig` | `output` | dir, format, export_words |
| `LoggingConfig` | `logging` | level |
| `WebConfig` | `web` | host, port, session_hours, max_upload_mb, delete_input_after_done, static_dir |

---

## 8. 데이터 모델

### WordSegment
```
word: str, start: float, end: float, confidence: float
```

### TranscriptionSegment
```
index: int, text: str, start: float, end: float,
confidence: float, speaker: str|list|None, words: list[WordSegment]
프로퍼티: duration, char_count, cps
```

### TranscriptionResult
```
metadata: Metadata, segments: list[TranscriptionSegment], quality: QualitySummary
메서드: reindex(), save(path), load(path), to_dict(), from_dict()
```

---

## 9. 예외 계층

```
SubWyrhtaError
├── AudioError
│   ├── AudioExtractionError
│   ├── AudioEnhancementError
│   └── VADError
├── STTError
│   ├── ModelLoadError
│   ├── TranscriptionError
│   └── DeviceError
├── SegmentationError
│   └── LLMSegmentationError
├── DiarizationError
├── WriterError
└── ConfigError
```

---

## 10. 디자인 패턴

### Strategy Pattern
- `STTEngine` ABC → WhisperX, faster-whisper, whisper-turbo, Parakeet
- `SegmentationStrategy` ABC → basic, advanced, extreme, llm-assisted
- `SubtitleWriter` ABC → SRT, ASS, VTT, JSON

### Pre-VAD 구조
VAD를 STT **이전**에 적용하여 발화 구간별 독립 추론. 타임스탬프 오프셋 보정.
2.0초 미만 청크는 forced alignment 건너뛰기.

### 언어 자동 감지
`source_lang: auto` 이고 Pre-VAD 청크가 있으면 3초 이상 청크 중 고르게 5개를 각각 인식해 다수결로 언어를 정하고, 나머지 청크는 그 언어로 고정한다 (`TranscriptionPipeline._detect_language`). 엔진은 첫 청크의 언어를 이후 청크에 고정하므로, 첫 청크가 잡음이면 전체가 엉뚱한 언어가 되던 문제를 막는다.

### STT 단어 보정 (`src/postprocess/word_units.py`)
- 일본어: janome 형태소 분석으로 글자 단위 단어를 문절로 묶는다 (조사·조동사·기호·접미어·비자립어는 앞에, 접두어는 뒤에)
- 모든 언어: 단어 시간이 `max(1초, 글자 수 x 0.5초)` 를 넘으면 끝을 자른다
- 단어 연결은 `src/core/text.py` `join_words` (한자·가나·전각 문자 사이 공백 없음), 발화 연결은 `join_utterances` (일본어·중국어는 전각 공백)
- 노이즈 필터: 감탄사·신음 단어(`remove_interjections`), 세그먼트 전체가 Whisper 가 지어낸 문구(`HALLUCINATION_PHRASES`)면 제거

### 자막 텍스트 마무리 (`src/postprocess/subtitle_text.py`)
세그멘테이션 뒤: 발화 구분(전각 공백) → 텍스트 정규화 → 앞에 붙은 문장 부호를 앞 자막으로 → 줄바꿈 (줄당 글자 수 `get_chars_per_line`: ja·zh 16, 그 밖 42, 일본어는 문절 경계)

### 세그멘테이션 파이프라인 순서
1. 화자 기반 분할 (diarize 시)
2. 모드별 분할 (basic/advanced/extreme/llm-assisted)
3. CPS 최적화
4. 최소 표시 시간 보장 (`min_duration_s`): 같은 화자 이웃(gap 짧은 쪽) → 다른 화자면 다음 세그먼트와 병합. 병합하면 `max_duration_s` 를 넘으면 병합하지 않고 빈 시간으로 늘린다. 반복 상한 없음

### 에러 폴백
- 디바이스: cuda → cpu 자동 폴백
- VAD: Silero 실패 → WebRTC VAD
- 배경음 분리·노이즈 억제: 실패 → 경고 후 원본 사용 (작업은 계속)
- llm-assisted: LLM 실패 → advanced 폴백
- extreme: sentence-transformers 미설치 → advanced 폴백
- Parakeet: nemo_toolkit 미설치 → ModelLoadError
- 화자 분리: 토큰 없음·무효·사용 조건 미동의 → 음성 추출 전에 DiarizationError (사전 확인)

---

## 11. 테스트

| 파일 | 대상 |
|------|------|
| `test_api.py` | Python API (지정한 값만 설정 파일 덮어쓰기) |
| `test_ass_writer.py` | ASS 출력 |
| `test_batch.py` | 배치 (파일 수집, 순차·병렬 처리와 worker 디바이스, CLI 입력 해석·디바이스 규칙, 작업 객체 재사용·--no-resume(체크포인트·단계 결과 삭제), transcribe 재개 옵션) |
| `test_checkpoint.py` | 체크포인트 저장/복원 |
| `test_stage_cache.py` | 단계 결과 보관 (키, 저장·읽기, 깨진 목록 파일, 정리) |
| `test_cli.py` | CLI 파싱, 배치 옵션 |
| `test_config.py` | 설정 로드/병합 |
| `test_diarization.py` | 화자 분리, 사전 확인, 화자 임베딩 채우기·대조·재사용 |
| `test_engines.py` | 엔진 팩토리/베이스 |
| `test_env_file.py` | `.env` 로드, `--env-file` |
| `test_extractor.py` | 오디오 추출 |
| `test_models.py` | 데이터 모델 직렬화 |
| `test_phase5.py` | demucs, enhancer, 구두점, extreme |
| `test_phase6.py` | LLM 클라이언트, llm-assisted, 챕터, 장면, 엔진 |
| `test_llm_client.py` | LLM 프로바이더 (구독 CLI, 클라우드·로컬 OpenAI 호환, 준비 상태, RPM, 설정 검증) |
| `test_subtitle_cleanup.py` | 자막 텍스트 정리: 신음·지어낸 문구 제거, 발화 구분, 앞에 붙은 부호, 줄바꿈 |
| `test_cjk_segmentation.py` | 일본어(글자 단위 단어) 과분할 회귀: 글자 연결, CPS 분할, 최소 표시 시간, 화자 틈·겹침 |
| `test_pipeline.py` | 파이프라인 (Pre-VAD, 전체 추론, 전처리, 화자 분리·사전 확인, 장면 감지, 단계 재개·청크 경계 검증) |
| `test_postprocess.py` | 후처리 |
| `test_quality.py` | 품질 검사/교정 |
| `test_segmentation.py` | 세그멘테이션 전 모드 |
| `test_speaker_segmentation.py` | 화자 분할 |
| `test_vad.py` | VAD |
| `test_web_api.py` | 웹 API (세션, 업로드·작업, 설정·로그, 워커, 작업 폴더의 단계 결과 경로) |
| `test_web_runner.py` | 작업 자식 프로세스 진입 (단계 결과 폴더 전달, 재시작 전 서버가 보낸 작업 정보) |
| `test_web_store.py` | 웹 작업 저장소, 인증, 설정 IO, 로그 읽기, 작업 옵션 |
| `test_writers.py` | SRT 출력, 출력 경로, Writer ABC·팩토리 |
| `test_writers_phase4.py` | VTT·JSON 출력 |

**pytest 883 테스트 (880 passed, 3 skipped — 건너뛰는 수는 설치된 선택 패키지에 따라 다르다)**, 웹 vitest 32, 웹 E2E 12장면 (`cd web && npm run test:e2e`, 결과는 `docs/e2e/` 에 만들어지고 저장소에는 올리지 않는다)

---

## 12. 의존성

| 파일 | 용도 |
|------|------|
| `requirements.txt` | 전부 (2026-10-11 하나로 합침): 코어 (pyyaml, structlog, click, tqdm, ffmpeg-python, python-dotenv), STT 엔진 (whisperx, faster-whisper, av<19, silero-vad, pysbd, janome, torch, torchaudio), 웹 서버 (fastapi, uvicorn, ruamel.yaml), 화자 분리 (pyannote.audio), 오디오 향상 (demucs, noisereduce, deepmultilingualpunctuation), LLM 프로바이더 (openai) |
| `requirements-dev.txt` | 개발 (pytest, pytest-cov, ruff, mypy, types-PyYAML, types-tqdm, httpx) + `requirements.txt` |

requirements 에 없는 선택 패키지: `nemo_toolkit`(Parakeet), `sentence-transformers`(extreme), `webrtcvad`(VAD 대체 경로), `scenedetect`(장면 감지의 영상 컷, `scene.use_video`).

---

## 13. 개발 Phase 완료 현황

| Phase | 내용 | Work Items | 상태 |
|-------|------|-----------|------|
| Phase 1 | 코어 STT | 10 | ✅ |
| Phase 2 | 품질 파이프라인 | 9 | ✅ |
| Phase 3 | 화자 분리 | 7 | ✅ |
| Phase 4 | 다중 포맷/배치 | 9 | ✅ |
| Phase 5 | 오디오 향상 | 7 | ✅ |
| Phase 6 | 고도화 | 7 | ✅ |
| Phase 7 | 전체 파이프라인 연결 + 웹 화면 | 7 | ✅ |
| Phase 8 | LLM 프로바이더 확장 + `.env` | 7 | 🔲 (8.5 구독 CLI 실제 호출 검증 남음) |
| Phase 9 | 미연결 기능 연결과 코드 정리 (배치·장면 감지·화자 임베딩 연결, API 설정, 버전), 이름 변경, 단계별 재개 | 9 | 🔲 (9.8 E2E 재녹화 남음) |
| **합계** | | **72** | |

Phase 4 의 배치 처리와 Phase 6 의 장면 감지는 Phase 9 에서 CLI·파이프라인에 연결했다 (그 전에는 모듈과 테스트만 있었다).

---

## 14. 웹 화면 (serve)

```bash
python -m src.main serve --data ./data [--host 127.0.0.1] [--port 8950] [--static-dir web/dist]
```

- 로컬 서버로만 운영한다 (컨테이너 이미지 없음, 2026-10-08 결정)
- 인증: 처음 접속 때 만드는 관리 계정 하나, HMAC 서명 세션 쿠키 (HttpOnly, SameSite=Strict). 쓰기 요청은 JSON 만 (업로드는 application/octet-stream)
- 작업: 브라우저 업로드 → SQLite 대기열 → 워커 스레드가 한 번에 하나씩 spawn 자식 프로세스로 파이프라인 실행. 진행(단계·청크 수) 기록, 실행 중 취소(프로세스 종료), 비정상 종료 감지(`process_crashed`), 서버 재시작 때 실행 중이던 작업은 실패(`interrupted`)로 표시. 다시 시도하면 작업 폴더의 단계 결과(`work/`)와 체크포인트(`checkpoint.json`)에서 이어서 한다 (작업을 지우면 함께 지운다)
- 화면: 대시보드, 작업 대기열(작업별 옵션, 출력 포맷 여러 개, 취소·다시 시도·실패 모두 다시 시도·삭제·상세), 완료 목록(미리 보기·내려받기·자막 검수·끝난 작업 모두 지우기), 로그, 설정(config.yaml 편집, LLM 준비 상태 배너, 계정). ko/en, light/dark
- 설정 저장은 다음 작업부터 적용 (`web.session_hours`·서버 로그 수준은 재시작 후). 설정 파일에 쓸 수 없으면 읽기 전용
- 자막 검수 (`src/web/review.py`, `GET`·`PUT /api/jobs/{id}/review`, `web/src/views/ReviewView.vue`): SRT 자막에 검수 리포트 항목(issues·suggested·corrected)을 시각 겹침으로 붙여 자막 전체를 표로 보이고, 자막 하나의 글자를 고쳐 SRT 에 쓴다 (번호·시각 줄은 그대로). 저장한 자막은 검수 완료로 적는다 (작업 폴더 `review-state.json`, 글자가 그대로면 SRT 는 다시 쓰지 않는다). 처음 고칠 때 원본을 `*.orig.srt` 로 남기고 출력 목록에 더한다. 다른 출력 포맷은 바꾸지 않는다
- 보안: 쓰기 요청 Content-Type 검사(415), 보안 헤더, 로그인 실패 지연
- 운영 절차: [OPERATIONS.md](OPERATIONS.md)
- 화면 소스: `web/` (Vue 3, vue-smartview 부분 빌드 `web/vendor/vue-smartview`, `scripts/update-smartview.sh`). vendor 는 비공개라 공개 저장소에서 빼고, 빌드한 `web/dist` 를 올린다

