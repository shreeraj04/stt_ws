import time
import torch
import torchaudio
from torchaudio.datasets import LIBRISPEECH
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, WhisperProcessor, WhisperForConditionalGeneration
from faster_whisper import WhisperModel as FasterWhisperModel
from vosk import Model, KaldiRecognizer
import numpy as np
from jiwer import wer, cer
import pandas as pd
import psutil
import os

# Load models globally
vosk_model = Model("./src/vosk-model-en-us-0.22", lang="en-us")
faster_whisper_model = FasterWhisperModel("base", device="cpu", compute_type="int8")
distil_whisper_processor = AutoProcessor.from_pretrained("distil-whisper/distil-small.en")
distil_whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained("distil-whisper/distil-small.en").to('cpu').eval()
whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-base")
whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base").to('cpu').eval()

def load_librispeech_sample(num_samples=10):
    dataset = LIBRISPEECH(".", url="test-clean", download=True)
    return [dataset[i] for i in range(min(num_samples, len(dataset)))]

def transcribe_faster_whisper(audio, sampling_rate):
    audio_float32 = audio.numpy().flatten().astype(np.float32)
    segments, _ = faster_whisper_model.transcribe(audio_float32)
    return " ".join([s.text.strip() for s in segments])

@torch.no_grad()
def transcribe_distil_whisper(audio, sampling_rate):
    input_features = distil_whisper_processor(audio.numpy(), sampling_rate=sampling_rate, return_tensors="pt").input_features
    predicted_ids = distil_whisper_model.generate(input_features)
    return distil_whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

def transcribe_vosk(audio, sampling_rate):
    rec = KaldiRecognizer(vosk_model, sampling_rate)
    audio_int16 = (audio.numpy() * 32768.0).astype(np.int16).tobytes()
    rec.AcceptWaveform(audio_int16)
    result = rec.FinalResult()
    return eval(result)['text']

@torch.no_grad()
def transcribe_openai_whisper(audio, sampling_rate):
    input_features = whisper_processor(audio.numpy(), sampling_rate=sampling_rate, return_tensors="pt").input_features
    predicted_ids = whisper_model.generate(input_features)
    return whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

def get_resource_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024, process.cpu_percent(interval=None)  # Memory in MB

def evaluate_model(model_func, audio, transcript, sampling_rate, model_name):
    try:
        start_time = time.time()
        start_memory, _ = get_resource_usage()
        
        transcription = model_func(audio, sampling_rate)
        
        end_time = time.time()
        end_memory, cpu_used = get_resource_usage()

        processing_time = end_time - start_time
        memory_used = end_memory - start_memory

        error_rate = wer(transcript, transcription)
        char_error_rate = cer(transcript, transcription)

        return transcription, error_rate, char_error_rate, processing_time, memory_used, cpu_used
    except Exception as e:
        print(f"Error in {model_name}: {e}")
        return None, None, None, None, None, None

def main():
    samples = load_librispeech_sample(num_samples=100)  # Adjust sample size as needed
    models = {
        "Faster Whisper": transcribe_faster_whisper,
        "Distil Whisper": transcribe_distil_whisper,
        "Vosk": transcribe_vosk,
        "OpenAI Whisper": transcribe_openai_whisper
    }
    
    results = []
    overall_start_memory, _ = get_resource_usage()
    max_memory_usage = 0
    max_cpu_usage = 0
    
    for i, (audio, sampling_rate, transcript, _, _, _) in enumerate(samples, 1):
        for model_name, model_func in models.items():
            transcription, error_rate, char_error_rate, processing_time, memory_used, cpu_used = evaluate_model(model_func, audio, transcript, sampling_rate, model_name)
            
            if transcription is not None:
                results.append({
                    "Model": model_name,
                    "Sample": i,
                    "Original": transcript,
                    "Transcribed": transcription,
                    "WER": error_rate,
                    "CER": char_error_rate,
                    "Processing Time": processing_time
                })
                
                print(f"Sample {i} - {model_name}:")
                print(f"Original: {transcript}")
                print(f"Transcribed: {transcription}")
                print(f"WER: {error_rate:.4f}")
                print(f"CER: {char_error_rate:.4f}")
                print(f"Processing time: {processing_time:.2f} seconds")
                print(f"Memory used: {memory_used:.2f} MB")
                print(f"CPU used: {cpu_used:.2f}%")
                print()

                max_memory_usage = max(max_memory_usage, memory_used)
                max_cpu_usage = max(max_cpu_usage, cpu_used)
        
        if i % 20 == 0 and input("Continue to next sample? (y/n): ").lower() != 'y':
            break
    
    overall_end_memory, overall_cpu_usage = get_resource_usage()
    overall_memory_usage = overall_end_memory - overall_start_memory

    df = pd.DataFrame(results)
    df.to_excel("stt_model_comparison_results.xlsx", index=False)
    print("Results saved to stt_model_comparison_results.xlsx")
    
    print("\nOverall Resource Usage:")
    print(f"Total Memory Usage: {overall_memory_usage:.2f} MB")
    print(f"Max Memory Usage: {max_memory_usage:.2f} MB")
    print(f"Max CPU Usage: {max_cpu_usage:.2f}%")
    print(f"Final CPU Usage: {overall_cpu_usage:.2f}%")
    
    for model_name in models:
        model_df = df[df['Model'] == model_name]
        print(f"\n{model_name} - Average Metrics:")
        print(f"WER: {model_df['WER'].mean():.4f}")
        print(f"CER: {model_df['CER'].mean():.4f}")
        print(f"Processing Time: {model_df['Processing Time'].mean():.2f} seconds")

if __name__ == "__main__":
    main()