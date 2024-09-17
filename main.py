import os
import argparse
import asyncio
import logging
import time
from collections import deque
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from modules.processors.frame.core import get_frame_processors_modules
from modules.face_analyser import get_one_face
import modules.globals
from modules.core import encode_execution_providers, decode_execution_providers

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Deep Live Cam Server')
parser.add_argument('--execution-provider', type=str, default='cpu',
                    choices=encode_execution_providers(['CPUExecutionProvider', 'CUDAExecutionProvider', 'CoreMLExecutionProvider']),
                    help='Execution provider (default: cpu)')
parser.add_argument('--source-image', type=str, default='srk.jpg',
                    help='Path to the source image (default: srk.jpg)')
args = parser.parse_args()

# Set the execution provider
modules.globals.execution_providers = decode_execution_providers([args.execution_provider])
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

# Set up frame processors
modules.globals.frame_processors = ['face_swapper', 'face_enhancer']
# modules.globals.frame_processors = ['face_swapper']
frame_processors = get_frame_processors_modules(modules.globals.frame_processors)
logger.info(f"Initialized frame processors: {[fp.NAME for fp in frame_processors]}")

# FPS calculation
FPS_WINDOW = 30  # Calculate FPS over this many frames
frame_times = deque(maxlen=FPS_WINDOW)

FRAME_WIDTH = 320*2
FRAME_HEIGHT = 240*2
WEBSOCKET_TIMEOUT = 60  # Increase timeout to 60 seconds

def time_function(func):
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"{func.__name__} took {end_time - start_time:.4f} seconds")
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
    if frame is None:
        return None
    if frame.shape[0] != FRAME_HEIGHT or frame.shape[1] != FRAME_WIDTH:
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    
    for frame_processor in frame_processors:
        start_time = time.time()
        frame = frame_processor.process_frame(source_face, frame)
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

@app.websocket("/ws/video")
async def websocket_endpoint(websocket: WebSocket):
    try:
        await websocket.accept()
        logger.info("WebSocket connection accepted")
        frame_count = 0
        last_fps_log_time = time.time()
        frame_times = deque(maxlen=30)  # Store the last 30 frame times for FPS calculation
        last_processed_time = time.time()

        next_frame = await receive_frame(websocket)
        while True:
            start_time = time.time()

            # Start processing next frame
            process_task = asyncio.create_task(process_frame(next_frame, frame_count))
            
            # Receive next frame while processing current frame
            next_frame = await receive_frame(websocket)
            
            # Wait for processing to complete and send the frame
            processed_frame = await process_task
            await send_frame(websocket, processed_frame)

            frame_count += 1

            # Calculate and log FPS
            end_time = time.time()
            frame_times.append(end_time - start_time)
            if end_time - last_fps_log_time >= 5:
                if len(frame_times) > 1:
                    fps = len(frame_times) / sum(frame_times)
                    logger.info(f"Processed {frame_count} frames. Current FPS: {fps:.2f}")
                last_fps_log_time = end_time
                frame_times.clear()

            # Implement frame skipping if processing is falling behind
            if end_time - last_processed_time < 0.1:  # Aim for max 10 FPS
                await asyncio.sleep(0.1 - (end_time - last_processed_time))
            last_processed_time = end_time

    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")
    finally:
        logger.info("WebSocket connection closed")

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, timeout_keep_alive=WEBSOCKET_TIMEOUT)