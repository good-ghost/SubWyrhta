# SubWyrhta

[한국어](README.md) | English

A standalone speech-to-text tool that generates high-quality subtitles (SRT/ASS/VTT/JSON) from video and audio files.

It does not translate. It focuses on speech recognition and subtitle generation.

## Features

- **Multiple STT engines** — WhisperX, faster-whisper, whisper-turbo, Parakeet
- **Four segmentation modes** — basic, advanced, extreme, llm-assisted
- **Speaker diarization** — pyannote based, with per-speaker color and position styles (ASS)
- **Pre-VAD** — splits speech regions before STT for precise timestamps
- **Audio enhancement** — vocal separation (demucs), loudness normalization, noise suppression
- **Quality control** — checks and auto-fixes CPS, line length, timecodes, and empty subtitles
- **llm-assisted segmentation** — subscriptions (ChatGPT and Claude CLIs), cloud API keys (Gemini, OpenRouter, NVIDIA NIM), local servers (Ollama, OpenAI compatible), with a requests-per-minute (RPM) limit
- **Batch processing** — directory, glob, and file-list input, parallel processing with `--workers` (one worker on the GPU), failed files are skipped
- **Scene detection** — treats long gaps (optionally video cuts) as scene boundaries and keeps a gap between subtitles around them
- **Resume from where it stopped** — keeps the results of preprocessing, Pre-VAD, and diarization plus a recognition checkpoint (per Pre-VAD chunk) so finished stages are skipped (`--resume`, Retry in the web UI)
- **Web UI (serve)** — upload, job queue (one at a time), completed list and preview, subtitle review (edit the SRT while reading the review notes), logs, settings
- **Python API** — call the same pipeline from code

## Requirements

- Python 3.12+
- ffmpeg (including ffprobe)
- Node.js and npm (only to rebuild the web UI from source. The built UI, `web/dist`, is included in the repository)

## Installation

```bash
# Clone the repository
git clone https://github.com/good-ghost/SubWyrhta.git
cd SubWyrhta

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies (core, STT engines, web, diarization, audio enhancement, LLM)
pip install -r requirements.txt
# CPU-only server: use the CPU wheel of torch (several GB smaller than the CUDA wheel)
#   pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Development (pytest, ruff, mypy)
pip install -r requirements-dev.txt
```

## Quick Start

### Generate subtitles

```bash
python -m src.main transcribe video.mp4
```

This creates `./output/video.en.srt`.

### Choose the output format and language

```bash
python -m src.main transcribe video.mp4 --output-format vtt --source-lang ko
```

### Speaker diarization

```bash
export HF_TOKEN="hf_..."   # or put it in .env

python -m src.main diarize video.mp4 --num-speakers 2
```

You need a token and must accept the terms of the model set (`speaker-diarization-3.1`, `segmentation-3.0`, `speaker-diarization-community-1`): see section 2 of [docs/OPERATIONS.md](docs/OPERATIONS.md) (Korean).

This creates ASS subtitles with a color per speaker (`./output/video.en.diarized.ass`).

### Show file information

```bash
python -m src.main info video.mp4
```

## CLI Commands

### transcribe

Converts an audio or video file to subtitles.

```bash
python -m src.main transcribe <input> [options]
```

`<input>` is a file, a directory, or a glob pattern (`"videos/**/*.mp4"`, in quotes). Several files are processed as a batch.

| Option | Description | Default |
|--------|-------------|---------|
| `--recursive` | Search directories and globs recursively | - |
| `--file-list` | List of files to process (one per line, `#` comments). Works without `<input>` | - |
| `--workers` | Number of parallel workers. Only one worker uses the GPU, the rest use the CPU | `1` |
| `--workers-device` | Set the device of every worker | - |
| `--failed-list` | Where to save the list of failed files (failed files are skipped and processing continues; exit code 1 if any failed) | - |
| `--output-dir` | Output directory | `./output` |
| `--output-format` | `srt` / `ass` / `vtt` / `json` | `srt` |
| `--output-name` | Output file name (single file only) | `{input}.{language}.{format}` |
| `--model` | STT engine (`whisperx` / `faster-whisper` / `whisper-turbo` / `parakeet` / `distil-whisper`) | `whisperx` |
| `--model-size` | Model size | `large-v3` |
| `--model-dir` | Model cache path | `~/.cache/subwyrhta` |
| `--source-lang` | Spoken language (`auto` detects it) | `auto` |
| `--device` | `auto` / `cuda` / `cpu` / `mps` | `auto` |
| `--compute-type` | `auto` / `float16` / `float32` / `int8` | `auto` |
| `--batch-size` / `--beam-size` / `--temperature` | Inference batch size, beam search size, sampling temperature | `16` / `5` / `0.0` |
| `--audio-track` | Audio track number (0-based) | default track |
| `--start` / `--end` | Time range to process (HH:MM:SS) | whole file |
| `--split-by-chapters` | Create separate subtitles per chapter | - |
| `--keep-audio` | Keep the extracted audio | - |
| `--denoise` / `--separate-vocals` / `--normalize-audio` | Preprocessing (turn each off with `--no-…`) | config file |
| `--vad` / `--pre-vad` | VAD, Pre-VAD (`--no-pre-vad`: transcribe the whole file, then re-split) | config file |
| `--vad-export` | Save the VAD timeline as JSON | - |
| `--segmentation-mode` / `--max-cps` | Segmentation mode, maximum CPS | config file |
| `--diarize` / `--num-speakers` / `--hf-token` | Speaker diarization | config file |
| `--qc` / `--no-qc`, `--quality-report` | Quality check, quality report JSON | config file |
| `--export-words` | Also save word timestamps as JSON | - |
| `--resume` / `--no-resume` | Resume from the checkpoint (`<input>.checkpoint.json`) and stage results (`<input>.work/`) / delete what is left and start over | kept but not used |
| `--llm-provider` / `--llm-model` / `--llm-base-url` | llm-assisted provider (`codex` / `claude-code` / `gemini` / `openrouter` / `nvidia-nim` / `ollama` / `local`), model, local server address | config file |
| `--dry-run` | Print the plan only | - |

`transcribe`, `diarize`, the Python API, and the web UI all run the same full pipeline (`src/pipeline.py`):
(model access check when diarization is on) → extraction → preprocessing (demucs → loudness normalization → noise suppression) → Pre-VAD → STT → diarization →
noise filter (and punctuation restoration) → segmentation (speaker split → mode-specific split → CPS optimization → minimum display time → text normalization) → speaker names → scene boundary gaps → quality check → output.

### diarize

Generates subtitles with speaker diarization (default format ASS, file name `{input}.{language}.diarized.{format}`). It takes the output and range options of transcribe (`--output-*`, `--keep-audio`, `--audio-track`, `--start`/`--end`), some STT options (`--model`, `--model-size`, `--source-lang`, `--device`), the LLM options, and `--dry-run`, plus the options below. Preprocessing, VAD, segmentation mode, and quality checks use the config file values.

```bash
python -m src.main diarize <input> [options]
```

| Option | Description |
|--------|-------------|
| `--num-speakers` | Number of speakers (`auto` or an integer) |
| `--hf-token` | HuggingFace token |
| `--speaker-map` | Speaker name mapping (`SPEAKER_00:John,SPEAKER_01:Jane`) |
| `--speaker-profile` | Speaker profile JSON file |
| `--export-embeddings` | Export speaker embeddings (`{input}.embeddings.json`) |
| `--speaker-embeddings` | Speaker embeddings from an earlier file. The same speaker gets the same speaker ID as before (to keep speaker names consistent across a series) |
| `--no-speaker-split` | Do not split at speaker changes |

### validate

Checks the quality of an existing subtitle file.

```bash
python -m src.main validate subtitle.srt --auto-fix
```

| Option | Description |
|--------|-------------|
| `--auto-fix` | Fix automatically and save |
| `--output` | Where to save the fixed file (default `{original}.fixed.{extension}`) |
| `--max-chars` | Maximum characters per line (default 42) |
| `--max-cps` | Maximum CPS |
| `--min-duration` | Minimum display time (default 1.0 s) |

### enhance

Runs audio enhancement only.

```bash
python -m src.main enhance audio.mp3 --denoise --separate-vocals
```

| Option | Description |
|--------|-------------|
| `--output-dir` | Output directory (default `./output`, file `{input}.enhanced.wav`) |
| `--denoise` | Noise suppression |
| `--separate-vocals` | Vocal separation with demucs |
| `--normalize` / `--no-normalize` | Loudness normalization (on by default) |
| `--device` | demucs device (`auto` / `cuda` / `cpu` / `mps`) |

### info / list-models

```bash
python -m src.main info video.mp4        # file information
python -m src.main list-models           # engines, models, devices
```

## Web UI (serve)

Starts the web UI as a local server. Video and audio files uploaded from the browser go into a job queue and are processed **one at a time**, in the order they were registered (each job runs in a child process: all model memory is released when it ends, and a running job can be cancelled).

```bash
pip install -r requirements.txt

python -m src.main serve --data ./data          # http://127.0.0.1:8950
python -m src.main --config ./config.yaml serve --data ./data --host 0.0.0.0 --port 8950
```

- The repository includes the built UI (`web/dist`), so it works without Node.js. Rebuilding the UI from its source (`web/src`) needs `web/vendor/vue-smartview`. That is a partial build of the private vue-smartview library and is not in the public repository (maintainers create it with `scripts/update-smartview.sh <path to VUE-SMARTVIEW>`)
- On the first visit you create the admin account, then sign in with it (session cookie; the password is stored as a scrypt hash in `data/secrets.json`, mode 600)
- Without `--config` the server uses `data/config.yaml` and creates it from the code defaults if it is missing. Values saved on the Settings screen apply from the next job (`web.session_hours` and the server log level apply after a restart)
- Data folder: `config.yaml`, `secrets.json` (admin account), `jobs.db` (job records), `jobs/<id>/input` (uploads), `jobs/<id>/output` (subtitles), `jobs/<id>/checkpoint.json` (recognition checkpoint), `jobs/<id>/work/` (stage results: preprocessed audio, Pre-VAD, diarization), `logs/subwyrhta.log` (JSON log), `uploads/`
- A new job takes per-job options (engine, model size, language, several output formats, segmentation, preprocessing, diarization, and more). You can retry all failed jobs, clear all finished jobs, and preview subtitles
- **Subtitle review** on the completed list shows every SRT subtitle in a table with the notes from the LLM review report (`*.review.json`): start, end, subtitle, review type, status, note. Clicking a subtitle opens an editor with its notes. Saving writes straight to the SRT file and marks that subtitle as reviewed (saving without changes is fine). The first edit keeps the original as `*.orig.srt`. The VTT, ASS, and JSON outputs of the same job are not changed
- The uploaded source of a completed job is deleted by default (`web.delete_input_after_done`). Failed and cancelled jobs keep it so they can be retried (finished stages are skipped and recognition continues from the checkpoint)
- For long jobs, start the server detached from the terminal session (`setsid nohup python -m src.main serve --data ./data > serve.log 2>&1 &`). If the server stops because the terminal closed, Retry continues from the finished stages
- The diarization token is not entered in the UI. Pass it as the `HF_TOKEN` environment variable when starting the server (`.env` works)
- For preparation (`.env`, Hugging Face model terms, LLM providers), starting and restarting, status checks, and what to do when a job fails, follow [docs/OPERATIONS.md](docs/OPERATIONS.md) (Korean)

### Load when running locally

| Item | Details |
|------|---------|
| Concurrency | One job. Queued jobs wait in order |
| CPU threads | The job process uses every core it can see. When sharing the machine with other services, limit it, for example `OMP_NUM_THREADS=8 python -m src.main serve …` |
| Memory (estimate, not measured) | large-v3 on CPU: about 9 to 11 GB with float32, about 4 to 6 GB with int8. `compute_type: int8` is recommended on a CPU server |
| Model cache | `stt.model_dir` (or `SUBWYRHTA_MODEL_DIR`), `~/.cache/huggingface`, `~/.cache/torch` |
| Web server | Does not import torch (the device check runs once at startup in a separate process) |

## STT Engines

| Engine | Option | Notes |
|--------|--------|-------|
| WhisperX | `--model whisperx` | Batched inference + forced alignment (default) |
| faster-whisper | `--model faster-whisper` | Fast inference with CTranslate2 |
| whisper-turbo | `--model whisper-turbo` | OpenAI turbo model |
| Parakeet | `--model parakeet` | NVIDIA NeMo, English-focused (install nemo_toolkit separately; not in the requirements files) |
| distil-whisper | `--model distil-whisper` | Alias of faster-whisper. Use it with a distil model name such as `--model-size distil-large-v3` |

If loading on the GPU fails, the engine falls back to the CPU automatically.

## Segmentation Modes

| Mode | Description |
|------|-------------|
| `basic` | Splits at PySBD sentence boundaries |
| `advanced` | Combined pause and clause signals with a dynamic threshold (default) |
| `extreme` | advanced plus extra splits by semantic similarity (install sentence-transformers separately; falls back to advanced without it) |
| `llm-assisted` | advanced plus split points adjusted by an LLM |

Choose with the CLI option `--segmentation-mode` or `segmentation.mode` in the config file.

### Using llm-assisted mode

```bash
# Set mode to llm-assisted in config.yaml, or on the CLI (Claude subscription: run claude auth login as the server account)
python -m src.main transcribe video.mp4 --segmentation-mode llm-assisted \
  --llm-provider claude-code
```

It supports subscriptions (ChatGPT and Claude, through their official CLIs), cloud API keys (Gemini, OpenRouter, NVIDIA NIM, in `.env`), and local servers (Ollama, OpenAI compatible), and can limit requests per minute (`llm.rpm`). For provider setup, see the llm-assisted section of [docs/USAGE_GUIDE.md](docs/USAGE_GUIDE.md) (Korean). A local Ollama example:

```bash
python -m src.main transcribe video.mp4 --segmentation-mode llm-assisted \
  --llm-provider ollama \
  --llm-model llama3
```

## Output Formats

| Format | Use |
|--------|-----|
| **SRT** | General-purpose subtitles (works in most players) |
| **ASS** | Per-speaker color and position styles (default for diarize) |
| **VTT** | Web subtitles (HTML5 video) |
| **JSON** | Metadata, word timestamps, and a quality summary (for post-processing and analysis) |

## Configuration

A YAML config file controls the whole behavior. Precedence:

```
code defaults → config file → environment variables → CLI arguments
```

- The config file is the one passed with `--config`. Without `--config` the CLI reads no config file (it does not pick up `config.yaml` in the project folder either)
- `serve` without `--config` uses `<data>/config.yaml` and creates it if it is missing

### Environment variables

Environment variables can go in a `.env` file in the project folder. Run `cp .env.example .env` and fill in only the values you use. The CLI and `serve` read it at startup (use `--env-file` for another file), values already set in the shell win, and lines with an empty value are ignored. `.env` is not committed (`.gitignore`).

| Variable | Use |
|----------|-----|
| `HF_TOKEN` | HuggingFace authentication (diarization) |
| `SUBWYRHTA_MODEL_DIR` | Model cache path |
| `SUBWYRHTA_DEVICE` | Device override |
| `GEMINI_API_KEY` | Google Gemini API key |
| `OPENROUTER_API_KEY` | OpenRouter API key |
| `NVIDIA_API_KEY` | NVIDIA NIM API key (hosted) |
| `SUBWYRHTA_GLOSSARY` | Path of the glossary for LLM misrecognition correction (a file, or a folder with `common.txt` and `<input name>.txt`). Without it nothing is corrected and suggestions only go into the review report |
| `SUBWYRHTA_LLM_EXTRA_BODY` | JSON merged into OpenAI-compatible LLM request bodies. Turning off reasoning differs per model (for example llama-server Qwen3 `{"chat_template_kwargs": {"enable_thinking": false}}`, NIM glm `{"reasoning_effort": "low"}`, per model `{"models": {...}}`). See docs/OPERATIONS.md 3.1 |

### Config file example

```yaml
stt:
  model: "faster-whisper"
  model_size: "large-v3"
  device: "cuda"
  source_lang: "ko"

segmentation:
  mode: "advanced"
  max_cps: 9              # default for Korean
  min_duration_s: 1.0

output:
  format: "srt"
  dir: "./subtitles"
```

Pass a config file on the CLI:

```bash
python -m src.main --config my_config.yaml transcribe video.mp4
```

## Python API

```python
from src.api import SubWyrhta, TranscribeConfig, save_result

stt = SubWyrhta(model="whisperx", model_size="large-v3")
result = stt.transcribe("video.mp4", TranscribeConfig(source_lang="en"))

# Read the result
for seg in result.segments:
    print(f"[{seg.start:.1f}s] {seg.text}")

# Save to files
save_result(result, "output.srt")
save_result(result, "output.json")
```

Values you do not set come from the config file (`SubWyrhta(config_path=…)`), or from the code defaults when there is none. With `TranscribeConfig(output_dir=…)` or `output_format` it also writes the subtitle file; otherwise it only returns the result (save with `save_result`, which picks the format from the extension). For all options, see the Python API section of [docs/USAGE_GUIDE.md](docs/USAGE_GUIDE.md) (Korean).

## Supported Input Formats

**Video:** `.mp4`, `.mkv`, `.avi`, `.mov`, `.webm`, `.mpeg`, `.mpg`, `.wmv`, `.flv`, `.m4v`

**Audio:** `.mp3`, `.wav`, `.m4a`, `.aac`, `.flac`, `.ogg`, `.opus`, `.wma`, `.mka`

## Development

```bash
# Tests
pytest                            # all
pytest tests/unit/                # unit tests only

# Web UI (web/; building needs web/vendor/vue-smartview)
cd web && npm run typecheck && npm run lint && npm test && npm run build
cd web && npm run test:e2e        # Playwright E2E and recording (local serve, real STT; needs ffmpeg)
                                  # Test media is not in the repository: on the first run
                                  # web/e2e/server/make-fixtures.sh creates it from espeak-ng synthetic speech
                                  # (in a podman container when espeak-ng is missing)
                                  # Put the folder of the browser extension (Salesforce Sandbox Bar Toggle) in the
                                  # E2E_EXTENSION_PATH environment variable or in web/e2e/.env.local
                                  # (E2E_EXTENSION_PATH=<folder>)

# Lint and type check
ruff check src/ tests/ --line-length 100
ruff format src/ tests/ --line-length 100
mypy src/ --strict
```

## Project Structure

```
src/
├── main.py              # CLI (click)
├── api.py               # Python API
├── pipeline.py          # Full pipeline (shared by CLI, API, and web)
├── checkpoint.py        # Pre-VAD chunk checkpoint
├── stage_cache.py       # Stage results (preprocessed audio, Pre-VAD, diarization; skipped on retry)
├── batch_processor.py   # Batch input collection and parallel processing (transcribe)
├── progress.py          # Batch progress display
├── version.py           # Version (pyproject.toml)
├── web/                 # Web server (FastAPI, authentication, job queue and worker, settings, logs)
├── audio/               # Audio extraction, VAD, enhancement
├── engines/             # STT engines (WhisperX, faster-whisper, turbo, Parakeet)
├── segmentation/        # Four segmentation modes + CPS optimization
├── postprocess/         # Noise filter, normalization, punctuation restoration
├── quality/             # Quality check and report, scene detection
├── diarization/         # Speaker diarization, embedding management
├── writers/             # SRT/ASS/VTT/JSON output
├── models/              # Data models
└── core/                # Configuration, exceptions, logging
web/                     # Web UI source (Vue 3, Playwright E2E). web/dist is the built UI
```

## License

[MIT](LICENSE) — Copyright (c) 2026 good-ghost

Notices for the third-party software bundled in `web/dist/` (Vue, Vuetify, vue-i18n, Material Design Icons) and the licenses of the separately installed dependencies are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
