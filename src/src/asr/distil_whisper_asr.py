import os
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torch
from src.audio_utils import save_audio_to_file
from .asr_interface import ASRInterface

class DistilWhisperASR(ASRInterface):
    def __init__(self, **kwargs):
        model_id = kwargs.get("model_id", "distil-whisper/distil-small.en")
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True, use_safetensors=True
        )
        model.to(self.device)

        processor = AutoProcessor.from_pretrained(model_id)

        self.asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=128,
            chunk_length_s=30,
            batch_size=16,
            return_timestamps=True,
            torch_dtype=torch.float16,
            device=self.device,
        )

    async def transcribe(self, client):
        file_path = await save_audio_to_file(
            client.scratch_buffer, client.get_file_name()
        )

        result = self.asr_pipeline(file_path)
        os.remove(file_path)

        to_return = {
            "language": result.get("language", ""),
            "text": result["text"],
        }
        return to_return