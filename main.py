import os
import argparse
import asyncio
import logging
import time
from collections import deque
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from modules.processors.frame.core import get_frame_processors_modules, load_frame_processor_module
from modules.face_analyser import get_one_face
import modules.globals
from modules.core import encode_execution_providers, decode_execution_providers, suggest_execution_threads, suggest_max_memory
from pydantic import BaseModel
import base64
from fastapi.responses import JSONResponse
from asyncio import Queue
import json

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Deep Live Cam Server')
parser.add_argument('--execution-provider', type=str, default='cpu',
                    choices=encode_execution_providers(['CPUExecutionProvider', 'CUDAExecutionProvider', 'CoreMLExecutionProvider']),
                    help='Execution provider (default: cpu)')
parser.add_argument('--source-image', type=str, default='le.jpg',  
                    help='Path to the source image (default: le.jpg)')
parser.add_argument('--execution-threads', help='number of execution threads', dest='execution_threads', type=int, default=suggest_execution_threads())
parser.add_argument('--max-memory', help='maximum amount of RAM in GB', dest='max_memory', type=int, default=suggest_max_memory())
args = parser.parse_args()

# Set the execution provider
modules.globals.execution_providers = decode_execution_providers([args.execution_provider])
modules.globals.execution_threads = args.execution_threads
modules.globals.max_memory = args.max_memory
logger.info(f"Using execution provider: {modules.globals.execution_providers}")

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update to your frontend's origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store current source image
CURRENT_SOURCE_IMAGE = None

# Load the source image
SOURCE_IMAGE_PATH = os.path.join(os.path.dirname(__file__), args.source_image)
try:
    source_image = cv2.imread(SOURCE_IMAGE_PATH)
    if source_image is None:
        raise FileNotFoundError(f"Could not read the image file: {SOURCE_IMAGE_PATH}")
    source_face = get_one_face(source_image)
    logger.info(f"Loaded source image from {SOURCE_IMAGE_PATH}")
    logger.info(f"Source face detected: {source_face is not None}")
except Exception as e:
    logger.error(f"Error loading source image: {e}")
    source_face = None

# Load all available frame processors once
ALL_FRAME_PROCESSORS = ['face_swapper', 'face_enhancer']
ALL_FRAME_PROCESSOR_MODULES  =[]
for frame_processor in ALL_FRAME_PROCESSORS:
    ALL_FRAME_PROCESSOR_MODULES.append(load_frame_processor_module(frame_processor))
# ALL_FRAME_PROCESSOR_MODULES = get_frame_processors_modules(ALL_FRAME_PROCESSORS)
print(f'ALL_FRAME_PROCESSOR_MODULES: {ALL_FRAME_PROCESSOR_MODULES}')
ACTIVE_FRAME_PROCESSORS = ['DLC.FACE-SWAPPER']  # Initially active processors --'DLC.FACE-ENHANCER'
MAINTAIN_FPS = False

# FPS calculation
FPS_WINDOW = 100  # Calculate FPS over this many frames
frame_times = deque(maxlen=FPS_WINDOW)

FRAME_WIDTH = 320 # 320
FRAME_HEIGHT = 240 # 240
WEBSOCKET_TIMEOUT = 60  # Increase timeout to 60 seconds

def time_function(func):
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
        end_time = time.time()
        # logger.info(f"{func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    return wrapper

@time_function
async def receive_frame(websocket):
    try:
        data = await asyncio.wait_for(websocket.receive_bytes(), timeout=1.0)
        nparr = np.frombuffer(data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return frame
    except asyncio.TimeoutError:
        logger.warning("Timeout while receiving frame")
        return None

@time_function
async def process_frame(frame, frame_count):
    if frame is None or CURRENT_SOURCE_IMAGE is None:
        return None
    if frame.shape[0] != FRAME_HEIGHT or frame.shape[1] != FRAME_WIDTH:
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    
    for frame_processor in ALL_FRAME_PROCESSOR_MODULES:
        if frame_processor.NAME in ACTIVE_FRAME_PROCESSORS:
            start_time = time.time()
            frame = frame_processor.process_frame(CURRENT_SOURCE_IMAGE["face"], frame)
            end_time = time.time()
            logger.info(f"{frame_processor.NAME} for frame {frame_count} took {end_time - start_time:.4f} seconds")
    
    return frame

@time_function
async def send_frame(websocket, frame):
    if frame is None:
        return
    _, buffer = cv2.imencode('.jpg', frame)
    processed_data = buffer.tobytes()
    await websocket.send_bytes(processed_data)

class Config(BaseModel):
    frame_processors: list[str]
    maintain_fps: bool = False  # Configuration option

@app.post("/set_config")
async def set_config(config: Config):
    global ACTIVE_FRAME_PROCESSORS, MAINTAIN_FPS
    try:
        # Validate frame processors
        valid_processors = {'face_swapper': 'DLC.FACE-SWAPPER', 'face_enhancer': 'DLC.FACE-ENHANCER'}
        active_processors = []
        for processor in config.frame_processors:
            if processor not in valid_processors:
                raise ValueError(f"Invalid frame processor: {processor}")
            active_processors.append(valid_processors[processor])
        
        # Update active frame processors and FPS maintenance setting
        ACTIVE_FRAME_PROCESSORS = active_processors
        MAINTAIN_FPS = config.maintain_fps
        
        logger.info(f"Updated active frame processors: {ACTIVE_FRAME_PROCESSORS}")
        logger.info(f"Maintain FPS: {MAINTAIN_FPS}")
        
        return {"message": "Configuration updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.websocket("/ws/video")
async def websocket_endpoint(websocket: WebSocket):
    try:
        await websocket.accept()
        logger.info("WebSocket connection accepted")
        frame_count = 0
        processed_count = 0
        skipped_count = 0
        frame_times = deque(maxlen=30)  # Store the last 30 frame times for FPS calculation
        frame_queue = Queue(maxsize=1)  # Queue to hold frames
        last_process_time = time.time()
        MIN_PROCESS_INTERVAL = 0.033  # 30 FPS when not maintaining FPS
        MAX_QUEUE_AGE = 0.5  # Drop frames older than 500ms

        async def receive_frames():
            nonlocal frame_count
            while True:
                try:
                    frame = await receive_frame(websocket)
                    frame_count += 1
                    current_time = time.time()
                    
                    if MAINTAIN_FPS:
                        await frame_queue.put((frame, current_time))
                    else:
                        # Clear old frames
                        cleared_frames = 0
                        while not frame_queue.empty():
                            try:
                                old_frame, old_time = frame_queue.get_nowait()
                                if current_time - old_time > MAX_QUEUE_AGE:
                                    logger.debug(f"Dropped frame that was {current_time - old_time:.3f}s old")
                                cleared_frames += 1
                            except asyncio.QueueEmpty:
                                break
                        
                        await frame_queue.put((frame, current_time))
                        if cleared_frames > 0:
                            logger.debug(f"Cleared {cleared_frames} old frames from queue")
                except Exception as e:
                    logger.error(f"Error in receive_frames: {e}")
                    break

        async def process_frames():
            nonlocal last_process_time, processed_count, skipped_count
            last_fps_log_time = time.time()
            while True:
                try:
                    start_time = time.time()
                    frame, frame_time = await frame_queue.get()
                    current_time = time.time()  # Get current time after queue.get()
                    
                    # Skip processing if not enough time has passed and not maintaining FPS
                    if not MAINTAIN_FPS:
                        time_since_last = current_time - last_process_time
                        frame_age = current_time - frame_time
                        logger.debug(f"Frame timing: age={frame_age:.3f}s, time_since_last={time_since_last:.3f}s, last_process_time={last_process_time:.3f}s, current_time={current_time:.3f}s")
                        if time_since_last < MIN_PROCESS_INTERVAL:
                            skipped_count += 1
                            if skipped_count % 30 == 0:  # Log every 30th skip to avoid spam
                                logger.info(f"Skipped frame - time since last: {time_since_last:.3f}s < {MIN_PROCESS_INTERVAL:.3f}s (Total skipped: {skipped_count})")
                            continue
                        else:
                            logger.debug(f"Processing frame - time since last: {time_since_last:.3f}s >= {MIN_PROCESS_INTERVAL:.3f}s")
                    
                    before_process = time.time()
                    processed_frame = await process_frame(frame, processed_count)
                    after_process = time.time()
                    process_duration = after_process - before_process
                    logger.debug(f"Frame {processed_count} processing duration: {process_duration:.3f}s")
                    
                    await send_frame(websocket, processed_frame)
                    processed_count += 1
                    last_process_time = before_process  # Set last_process_time to when we started processing
                    
                    # Log processing stats every second
                    end_time = time.time()
                    frame_times.append(process_duration)  # Only store the actual processing time
                    if end_time - last_fps_log_time >= 1:
                        if len(frame_times) > 1:
                            avg_process_time = sum(frame_times) / len(frame_times)
                            fps = 1 / avg_process_time if avg_process_time > 0 else 0
                            logger.info(
                                f"Stats: FPS={fps:.1f} (avg_process_time={avg_process_time:.3f}s), "
                                f"Received={frame_count}, "
                                f"Processed={processed_count}, "
                                f"Skipped={skipped_count}, "
                                f"Queue Size={frame_queue.qsize()}"
                            )
                        last_fps_log_time = end_time
                        frame_times.clear()

                except Exception as e:
                    logger.error(f"Error in process_frames: {e}")
                    break

        # Start both tasks
        receive_task = asyncio.create_task(receive_frames())
        process_task = asyncio.create_task(process_frames())

        # Wait for both tasks to complete
        done, pending = await asyncio.wait(
            [receive_task, process_task],
            return_when=asyncio.FIRST_COMPLETED
        )

        # Cancel any pending tasks
        for task in pending:
            task.cancel()

    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")
    finally:
        logger.info(f"WebSocket connection closed. Final stats: Received={frame_count}, Processed={processed_count}, Skipped={skipped_count}")

# Update FastAPI app configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Increase WebSocket timeout
@app.middleware("http")
async def add_websocket_timeout(request, call_next):
    if "websocket" in request.url.path:
        request.state.websocket_timeout = WEBSOCKET_TIMEOUT
    response = await call_next(request)
    return response

@app.post("/set_source_image")
async def set_source_image(file: UploadFile = File(...)):
    global CURRENT_SOURCE_IMAGE
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    face = get_one_face(img)
    if face is None:
        return JSONResponse(content={"error": "No face detected in the image"}, status_code=400)
    
    CURRENT_SOURCE_IMAGE = {
        "image": img,
        "face": face
    }
    
    _, buffer = cv2.imencode('.jpg', img)
    base64_image = base64.b64encode(buffer).decode('utf-8')
    
    return {"message": "Source image updated successfully", "image": base64_image}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, timeout_keep_alive=WEBSOCKET_TIMEOUT)