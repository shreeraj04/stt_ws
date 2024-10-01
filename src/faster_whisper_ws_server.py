import argparse
import os
import numpy as np
from faster_whisper import WhisperModel
import torch
from datetime import datetime, timedelta
from queue import Queue
import asyncio
import websockets
import psutil
import time
import uuid
import logging
import wave
 
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
 
clients = {}
audio_model = None  # Global model instance
 
def get_resource_usage():
    process = psutil.Process(os.getpid())
    cpu_percent = process.cpu_percent(interval=1)
    memory_info = process.memory_info()
    return cpu_percent, memory_info.rss / (1024 * 1024)  # RSS in MB
 
def is_speech(audio_chunk, energy_threshold):
    """
    Check if the audio chunk contains speech based on energy threshold.
    """
    audio_data = np.frombuffer(audio_chunk, dtype=np.int16)
    energy = np.sum(audio_data.astype(float) ** 2) / len(audio_data)
    return energy > energy_threshold
 
def save_session_audio(audio_data, client_id, sample_rate=16000):
    """
    Save the entire session's audio data as a single .wav file.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"session_audio_{client_id}_{timestamp}.wav"
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 2 bytes for int16
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data)
    logger.info(f"Saved session audio: {filename}")
 
def filter_false_positives(text, min_words=3, min_confidence=0.5):
    """
    Filter out potential false positives like isolated "thank you" messages.
    """
    words = text.split()
    if len(words) < min_words:
        common_phrases = ["thank you", "thanks", "okay", "bye"]
        if any(phrase in text.lower() for phrase in common_phrases):
            return ""
    return text
 
async def transcribe_audio(websocket, path, args):
    client_id = str(uuid.uuid4())
    clients[client_id] = {
        "data_queue": Queue(),
        "phrase_time": None,
        "last_sample": bytes(),
        "transcription": [],
        "current_phrase": "",
        "session_audio": bytes()
    }
    logger.info(f"New client connected: {client_id}")
 
    start_time = time.time()
    max_cpu_usage = 0
    max_memory_usage = 0
 
    try:
        while True:
            now = datetime.utcnow()
           
            try:
                chunk = await asyncio.wait_for(websocket.recv(), timeout=0.1)
                if isinstance(chunk, bytes):
                    clients[client_id]["data_queue"].put(chunk)
                    clients[client_id]["session_audio"] += chunk
            except asyncio.TimeoutError:
                logger.debug("Receive timeout - no data received")
                pass
            except Exception as e:
                logger.error(f"Error receiving data: {e}")
                break
           
            if not clients[client_id]["data_queue"].empty():
                phrase_complete = False
               
                if clients[client_id]["phrase_time"] and now - clients[client_id]["phrase_time"] > timedelta(seconds=args.phrase_timeout):
                    phrase_complete = True
               
                clients[client_id]["phrase_time"] = now
 
                while not clients[client_id]["data_queue"].empty():
                    clients[client_id]["last_sample"] += clients[client_id]["data_queue"].get()
               
                if len(clients[client_id]["last_sample"]) > args.record_timeout * 16000 * 1: #changed
                    audio_data = clients[client_id]["last_sample"]
                    clients[client_id]["last_sample"] = bytes()
 
                    if is_speech(audio_data, args.energy_threshold):
                        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0 #
 
                        try:
                            segments, info = audio_model.transcribe(audio_np, language="en")
                            text = " ".join([segment.text for segment in segments])
                            filtered_text = filter_false_positives(text)
                            logger.debug(f"Transcribed text: {text}")
                            logger.debug(f"Filtered text: {filtered_text}")
                        except Exception as e:
                            logger.error(f"Transcription error: {e}")
                            continue
 
                        if filtered_text.strip():
                            clients[client_id]["current_phrase"] += filtered_text.strip() + " "
 
                        if phrase_complete:
                            if clients[client_id]["current_phrase"].strip():
                                clients[client_id]["transcription"].append(clients[client_id]["current_phrase"].strip())
                                await websocket.send(clients[client_id]["current_phrase"].strip())
                                logger.info(f"Sent complete phrase: {clients[client_id]['current_phrase'].strip()}")
                            clients[client_id]["current_phrase"] = ""
                        elif filtered_text.strip():
                            await websocket.send(filtered_text.strip())
                            logger.debug(f"Sent interim result: {filtered_text.strip()}")
 
                        logger.info(f"Client {client_id}: {clients[client_id]['current_phrase']}")
 
                    else:
                        logger.debug("Audio chunk below energy threshold - no transcription attempted")
 
                    cpu_usage, memory_usage = get_resource_usage()
                    max_cpu_usage = max(max_cpu_usage, cpu_usage)
                    max_memory_usage = max(max_memory_usage, memory_usage)
            else:
                await asyncio.sleep(0.1)
    except websockets.exceptions.ConnectionClosed as e:
        logger.info(f"Connection closed for client {client_id}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error for client {client_id}: {e}")
    finally:
        # Save the entire session's audio
        save_session_audio(clients[client_id]["session_audio"], client_id)
 
        logger.info(f"Resource Usage for client {client_id}:")
        logger.info(f"Total runtime: {time.time() - start_time:.2f} seconds")
        logger.info(f"Peak CPU usage: {max_cpu_usage:.2f}%")
        logger.info(f"Peak memory usage: {max_memory_usage:.2f} MB")
       
        logger.info(f"Final Transcription for client {client_id}:")
        for line in clients[client_id]["transcription"]:
            logger.info(line)
        if clients[client_id]["current_phrase"]:
            logger.info(clients[client_id]["current_phrase"])
       
        del clients[client_id]
 
async def main(args):
    global audio_model
 
    try:
        audio_model = WhisperModel(args.model, device="cuda" if torch.cuda.is_available() else "cpu", compute_type="float16" if torch.cuda.is_available() else "int8")
        logger.info(f"Loaded Whisper model: {args.model} on {'GPU' if torch.cuda.is_available() else 'CPU'}")
    except Exception as e:
        logger.error(f"Failed to load Whisper model: {e}")
        return
 
    while True:
        try:
            server = await websockets.serve(
                lambda ws, path: transcribe_audio(ws, path, args),
                args.host,
                args.port
            )
            logger.info(f"WebSocket server started on ws://{args.host}:{args.port}")
            await server.wait_closed()
        except Exception as e:
            logger.error(f"Server error: {e}")
            logger.info("Attempting to restart server in 5 seconds...")
            await asyncio.sleep(5)
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="base", help="Model to use",
                        choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--energy_threshold", default=800, type=int,
                        help="Energy level for mic to detect. Increase this for noisy environments.")
    parser.add_argument("--record_timeout", default=0.5,
                        help="How real time the recording is in seconds.", type=float)
    parser.add_argument("--phrase_timeout", default=1,
                        help="How much empty space between recordings before we "
                             "consider it a new line in the transcription.", type=float)
    parser.add_argument("--host", default="0.0.0.0", help="WebSocket server host")
    parser.add_argument("--port", default=8765, type=int, help="WebSocket server port")
 
    args = parser.parse_args()
   
    asyncio.run(main(args))