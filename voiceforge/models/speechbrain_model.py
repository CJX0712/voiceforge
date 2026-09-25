"""SpeechBrainClassifier：可选 SOTA 音频分类/嵌入后端。需网络下载预训练权重，离线不可用。"""
from ..core.errors import err_model


class SpeechBrainClassifier:
    def __init__(self, source: str = "speechbrain/urbansound8k_ecapa"):
        self.source = source
        self._model = None

    @staticmethod
    def available() -> bool:
        try:
            import speechbrain  # noqa: F401

            return True
        except Exception:
            return False

    def name(self) -> str:
        return "speechbrain-ecapa"

    def _ensure(self):
        if self._model is None:
            try:
                from speechbrain.inference.classifiers import EncoderClassifier
            except Exception as e:
                raise err_model("speechbrain 不可用", cause=e)
            self._model = EncoderClassifier.from_hparams(source=self.source)

    def classify(self, sample) -> str:
        self._ensure()
        import torch

        w = torch.tensor(sample.waveform).unsqueeze(0)
        out, _score, _, _ = self._model.classify_batch(w)
        return str(out[0])
