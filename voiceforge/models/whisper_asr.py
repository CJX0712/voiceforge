"""WhisperASR：可选 SOTA 语音识别后端（openai-whisper）。需网络下载权重，离线不可用。"""
from ..core.errors import err_model


class WhisperASR:
    def __init__(self, model_size: str = "base"):
        self.model_size = model_size
        self._model = None

    @staticmethod
    def available() -> bool:
        try:
            import whisper  # noqa: F401

            return True
        except Exception:
            return False

    def name(self) -> str:
        return f"whisper-{self.model_size}"

    def _ensure(self):
        if self._model is None:
            try:
                import whisper
            except Exception as e:
                raise err_model("openai-whisper 不可用", cause=e)
            self._model = whisper.load_model(self.model_size)

    def transcribe(self, sample) -> str:
        self._ensure()
        import numpy as np

        audio = np.asarray(sample.waveform, dtype=np.float32)
        result = self._model.transcribe(audio, sr=sample.sample_rate)
        return result.get("text", "")
