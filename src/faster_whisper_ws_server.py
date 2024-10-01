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
import webrtcvad
import collections

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

clients = {}
audio_model = None  # Global model instance
vad = webrtcvad.Vad(3)  # Create a VAD instance with aggressiveness level 3 (0-3)

def frame_generator(frame_duration_ms, audio, sample_rate):
    """Generates audio frames from PCM audio data.
    Takes the desired frame duration in milliseconds, the PCM data, and
    the sample rate.
    Yields Frames of the requested duration.
    """
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield (audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n

def vad_collector(sample_rate, frame_duration_ms, padding_duration_ms, vad, frames):
    """Filters out non-voiced audio frames.
    Given a webrtcvad.Vad and a source of audio frames, yields only
    the voiced audio.
    Uses a padded, sliding window algorithm over the audio frames.
    When more than 90% of the frames in the window are voiced (as
    reported by the VAD), the collector triggers and begins yielding
    audio frames. Then the collector waits until 90% of the frames in
    the window are unvoiced to detrigger.
    The window is padded at the front and back to provide a small
    amount of silence or the beginnings/endings of speech around the
    voiced frames.
    Arguments:
    sample_rate - The audio sample rate, in Hz.
    frame_duration_ms - The frame duration in milliseconds.
    padding_duration_ms - The amount to pad the window, in milliseconds.
    vad - An instance of webrtcvad.Vad.
    frames - a source of audio frames (sequence or generator).
    Returns: A generator that yields PCM audio data.
    """
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    # We use a deque for our sliding window/ring buffer.
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    # We have two states: TRIGGERED and NOTTRIGGERED. We start in the
    # NOTTRIGGERED state.
    triggered = False

    voiced_frames = []
    for frame, timestamp, duration in frames:
        is_speech = vad.is_speech(frame, sample_rate)

        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            # If we're NOTTRIGGERED and more than 90% of the frames in
            # the ring buffer are voiced frames, then enter the
            # TRIGGERED state.
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                # We want to yield all the audio we see from now until
                # we are NOTTRIGGERED, but we have to start with the
                # audio that's already in the ring buffer.
                for f, s in ring_buffer:
                    voiced_frames.append(f)
                ring_buffer.clear()
        else:
            # We're in the TRIGGERED state, so collect the audio data
            # and add it to the ring buffer.
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            # If more than 90% of the frames in the ring buffer are
            # unvoiced, then enter NOTTRIGGERED and yield whatever
            # audio we've collected.
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                triggered = False
                yield b''.join(voiced_frames)
                ring_buffer.clear()
                voiced_frames = []
    if voiced_frames:
        yield b''.join(voiced_frames)

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
               
                if len(clients[client_id]["last_sample"]) > args.record_timeout * 16000 * 1:
                    audio_data = clients[client_id]["last_sample"]
                    clients[client_id]["last_sample"] = bytes()

                    # Apply VAD
                    frames = frame_generator(30, audio_data, 16000)
                    frames = list(frames)
                    segments = vad_collector(16000, 30, 300, vad, frames)

                    for segment in segments:
                        audio_np = np.frombuffer(segment, dtype=np.int16).astype(np.float32) / 32768.0

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

# The rest of the script (main function, argument parsing, etc.) remains the same
 
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