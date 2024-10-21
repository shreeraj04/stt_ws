from .faster_whisper_asr import FasterWhisperASR
from .whisper_asr import WhisperASR
from .distil_whisper_asr import DistilWhisperASR

class ASRFactory:
    @staticmethod
    def create_asr_pipeline(asr_type, **kwargs):
        if asr_type == "whisper":
            return WhisperASR(**kwargs)
        elif asr_type == "faster_whisper":
            return FasterWhisperASR(**kwargs)
        elif asr_type == "distil_whisper":
            return DistilWhisperASR(**kwargs)
        else:
            raise ValueError(f"Unknown ASR pipeline type: {asr_type}")
