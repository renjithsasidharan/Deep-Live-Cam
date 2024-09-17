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
# modules.globals.frame_processors = ['face_swapper', 'face_enhancer']
modules.globals.frame_processors = ['face_swapper']
frame_processors = get_frame_processors_modules(modules.globals.frame_processors)
logger.info(f"Initialized frame processors: {[fp.NAME for fp in frame_processors]}")

# FPS calculation
FPS_WINDOW = 30  # Calculate FPS over this many frames
frame_times = deque(maxlen=FPS_WINDOW)

FRAME_WIDTH = 320*2
FRAME_HEIGHT = 240*2
PIPELINE_SIZE = 5  # Number of frames to process in parallel

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
    data = await websocket.receive_bytes()
    nparr = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return frame

@time_function
async def process_frames(frames, frame_counts):
    processed_frames = []
    for frame, count in zip(frames, frame_counts):
        if frame.shape[0] != FRAME_HEIGHT or frame.shape[1] != FRAME_WIDTH:
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        
        for frame_processor in frame_processors:
            start_time = time.time()
            frame = frame_processor.process_frame(source_face, frame)
            end_time = time.time()
            logger.info(f"{frame_processor.NAME} for frame {count} took {end_time - start_time:.4f} seconds")
        
        processed_frames.append(frame)
    return processed_frames

@time_function
async def send_frame(websocket, frame):
    if frame.shape[0] != FRAME_HEIGHT or frame.shape[1] != FRAME_WIDTH:
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
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

        # Initialize the pipeline
        pipeline = []
        for _ in range(PIPELINE_SIZE):
            frame = await receive_frame(websocket)
            frame_count += 1
            pipeline.append((frame, frame_count))

        while True:
            # Process the batch of frames
            frames, counts = zip(*pipeline)
            process_task = asyncio.create_task(process_frames(frames, counts))

            # Start receiving new frames to refill the pipeline
            receive_tasks = [asyncio.create_task(receive_frame(websocket)) for _ in range(PIPELINE_SIZE)]

            # Wait for the current batch to finish processing
            processed_frames = await process_task
            
            # Send processed frames and refill the pipeline
            pipeline = []
            for i, processed_frame in enumerate(processed_frames):
                send_task = asyncio.create_task(send_frame(websocket, processed_frame))
                new_frame = await receive_tasks[i]
                frame_count += 1
                pipeline.append((new_frame, frame_count))
                await send_task

            # Calculate and log FPS
            current_time = time.time()
            frame_times.append(current_time)
            if current_time - last_fps_log_time >= 5:
                if len(frame_times) > 1:
                    fps = (len(frame_times) - 1) * PIPELINE_SIZE / (frame_times[-1] - frame_times[0])
                    logger.info(f"Processed {frame_count} frames. Current FPS: {fps:.2f}")
                last_fps_log_time = current_time
                frame_times.clear()
                frame_times.append(current_time)

    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")

# Add this at the end of the file
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)