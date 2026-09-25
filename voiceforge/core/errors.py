"""统一错误码体系（E100~E500）。所有异常经 VoiceForgeError 抛出，便于调用方按 code 处理。"""


class VoiceForgeError(Exception):
    def __init__(self, code: str, message: str, cause: Exception = None):
        self.code = code
        self.message = message
        self.cause = cause
        super().__init__(f"[{code}] {message}")

    def __str__(self):
        base = f"[{self.code}] {self.message}"
        if self.cause is not None:
            return f"{base} (caused by {type(self.cause).__name__}: {self.cause})"
        return base


def err_config(message: str, cause: Exception = None) -> VoiceForgeError:
    return VoiceForgeError("E100", message, cause=cause)


def err_data(message: str, cause: Exception = None) -> VoiceForgeError:
    return VoiceForgeError("E200", message, cause=cause)


def err_model(message: str, cause: Exception = None) -> VoiceForgeError:
    return VoiceForgeError("E300", message, cause=cause)


def err_eval(message: str, cause: Exception = None) -> VoiceForgeError:
    return VoiceForgeError("E400", message, cause=cause)


def err_pipeline(message: str, cause: Exception = None) -> VoiceForgeError:
    return VoiceForgeError("E500", message, cause=cause)
