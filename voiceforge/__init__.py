"""VoiceForge - 语音/音频机器学习系统（随机交付，作者：晨星）。

模块化 pipeline：合成音频 -> 特征提取(numpy/librosa) -> 分类(sklearn) -> 评测 -> pitch 跟踪。
SOTA 后端(whisper/speechbrain/librosa) 不可用时自动降级为纯 numpy + scikit-learn 实现。
"""

__version__ = "1.0.0"
__author__ = "晨星"
