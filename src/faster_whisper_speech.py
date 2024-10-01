import argparse
import os
import numpy as np
import speech_recognition as sr
from faster_whisper import WhisperModel
import torch
from datetime import datetime, timedelta
from queue import Queue
from time import sleep
from sys import platform
import psutil
import time

def get_resource_usage():
    process = psutil.Process(os.getpid())
    cpu_percent = process.cpu_percent(interval=1)
    memory_info = process.memory_info()
    return cpu_percent, memory_info.rss / (1024 * 1024)  # RSS in MB

import time

def transcribe_file(audio_file, audio_model):
    print("Starting transcription")
    
    # Record the start time
    start_time = time.time()

    # Call the transcription method
    segments, info = audio_model.transcribe(audio_file)

    # Print the detected language
    print(f"Detected language: {info.language}")
    
    # Compile the transcription
    transcription = " ".join([segment.text for segment in segments])
    end_time = time.time()

    # Calculate the processing time
    processing_time = end_time - start_time
    print(f"Done. Processing time: {processing_time:.2f} seconds")
    
    return transcription

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="base", help="Model to use",
                        choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--energy_threshold", default=1000,
                        help="Energy level for mic to detect.", type=int)
    parser.add_argument("--record_timeout", default=2,
                        help="How real time the recording is in seconds.", type=float)
    parser.add_argument("--phrase_timeout", default=3,
                        help="How much empty space between recordings before we "
                             "consider it a new line in the transcription.", type=float)
    parser.add_argument("--file", help="Path to audio file to transcribe", type=str)
    if 'linux' in platform:
        parser.add_argument("--default_microphone", default='pulse',
                            help="Default microphone name for SpeechRecognition. "
                                 "Run this with 'list' to view available Microphones.", type=str)
    args = parser.parse_args()

    # Resource monitoring variables
    start_time = time.time()
    max_cpu_usage = 0
    max_memory_usage = 0

    # Load faster-whisper model
    audio_model = WhisperModel(args.model, device="cuda" if torch.cuda.is_available() else "cpu", compute_type="int8")

    if args.file:
        # Transcribe file
        transcription = transcribe_file(args.file, audio_model)
        print("Transcription:")
        print(transcription)
    else:
        # Microphone transcription
        phrase_time = None
        data_queue = Queue()
        recorder = sr.Recognizer()
        recorder.energy_threshold = args.energy_threshold
        recorder.dynamic_energy_threshold = False

        if 'linux' in platform:
            mic_name = args.default_microphone
            if not mic_name or mic_name == 'list':
                print("Available microphone devices are: ")
                for index, name in enumerate(sr.Microphone.list_microphone_names()):
                    print(f"Microphone with name \"{name}\" found")
                return
            else:
                for index, name in enumerate(sr.Microphone.list_microphone_names()):
                    if mic_name in name:
                        source = sr.Microphone(sample_rate=16000, device_index=index)
                        break
        else:
            source = sr.Microphone(sample_rate=16000)

        record_timeout = args.record_timeout
        phrase_timeout = args.phrase_timeout

        transcription = []
        current_phrase = ""

        with source:
            recorder.adjust_for_ambient_noise(source)

        def record_callback(_, audio: sr.AudioData) -> None:
            data = audio.get_raw_data()
            data_queue.put(data)

        recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)

        print("Model loaded. Ready to transcribe from microphone.\n")

        try:
            while True:
                now = datetime.utcnow()
                if not data_queue.empty():
                    phrase_complete = False
                    if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                        phrase_complete = True
                    phrase_time = now

                    audio_data = b''.join(data_queue.queue)
                    data_queue.queue.clear()

                    audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

                    segments, info = audio_model.transcribe(audio_np)
                    text = " ".join([segment.text for segment in segments])

                    current_phrase += text + " "

                    if phrase_complete:
                        transcription.append(current_phrase.strip())
                        current_phrase = ""

                    os.system('cls' if os.name == 'nt' else 'clear')
                    print(f"Detected language: {info.language}")
                    for line in transcription:
                        print(line)
                    print(current_phrase, end='', flush=True)

                    cpu_usage, memory_usage = get_resource_usage()
                    max_cpu_usage = max(max_cpu_usage, cpu_usage)
                    max_memory_usage = max(max_memory_usage, memory_usage)

                else:
                    sleep(0.25)

        except KeyboardInterrupt:
            print("\n\nResource Usage:")
            print(f"Total runtime: {time.time() - start_time:.2f} seconds")
            print(f"Peak CPU usage: {max_cpu_usage:.2f}%")
            print(f"Peak memory usage: {max_memory_usage:.2f} MB")
            
            print("\n\nFinal Transcription:")
            for line in transcription:
                print(line)
            if current_phrase:
                print(current_phrase)

if __name__ == "__main__":
    main()