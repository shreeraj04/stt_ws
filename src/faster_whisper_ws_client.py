import asyncio
import websockets
import pyaudio
import argparse

async def send_audio(uri):
    async with websockets.connect(uri) as websocket:
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=16000,
                        input=True,
                        frames_per_buffer=4000) #change
        
        print("Connected to the server. Start speaking...")
        
        try:
            while True:
                data = stream.read(4000)
                await websocket.send(data)
                
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=0.1)
                    print(f"Received transcription: {response}")
                except asyncio.TimeoutError:
                    pass
        except KeyboardInterrupt:
            print("Stopping...")
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost", help="WebSocket server host")
    parser.add_argument("--port", default=8765, type=int, help="WebSocket server port")
    args = parser.parse_args()
    
    uri = f"ws://{args.host}:{args.port}"
    asyncio.get_event_loop().run_until_complete(send_audio(uri))

if __name__ == "__main__":
    main()