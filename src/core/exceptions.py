"""SubWyrhta 예외 계층 구조."""


class SubWyrhtaError(Exception):
    """SubWyrhta 최상위 예외. 모든 프로젝트 예외의 기반 클래스."""


# ── 오디오 관련 ──────────────────────────────


class AudioError(SubWyrhtaError):
    """오디오 처리 관련 예외."""


class AudioExtractionError(AudioError):
    """ffmpeg 오디오 추출 실패."""


class AudioEnhancementError(AudioError):
    """오디오 향상(노이즈 억제, 음량 정규화 등) 실패."""


class VADError(AudioError):
    """VAD(Voice Activity Detection) 처리 실패."""


# ── STT 엔진 관련 ────────────────────────────


class STTError(SubWyrhtaError):
    """STT 엔진 관련 예외."""


class ModelLoadError(STTError):
    """모델 로드 실패."""


class TranscriptionError(STTError):
    """음성 인식 추론 실패."""


class DeviceError(STTError):
    """디바이스 감지 또는 초기화 실패."""


# ── 세그멘테이션 관련 ────────────────────────


class SegmentationError(SubWyrhtaError):
    """세그멘테이션 처리 관련 예외."""


class LLMSegmentationError(SegmentationError):
    """LLM 기반 세그멘테이션 실패."""


# ── 화자 분리 관련 ───────────────────────────


class DiarizationError(SubWyrhtaError):
    """화자 분리 처리 관련 예외."""


# ── 출력 관련 ────────────────────────────────


class WriterError(SubWyrhtaError):
    """자막 출력 관련 예외."""


# ── 설정 관련 ────────────────────────────────


class ConfigError(SubWyrhtaError):
    """설정 파일 로드 또는 검증 실패."""
