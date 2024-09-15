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
frame_processors = get_frame_processors_modules(modules.globals.frame_processors)
logger.info(f"Initialized frame processors: {[fp.NAME for fp in frame_processors]}")

# FPS calculation
FPS_WINDOW = 30  # Calculate FPS over this many frames
frame_times = deque(maxlen=FPS_WINDOW)

@app.websocket("/ws/video")
async def websocket_endpoint(websocket: WebSocket):
    try:
        await websocket.accept()
        logger.info("WebSocket connection accepted")
        frame_count = 0
        last_fps_log_time = time.time()
        while True:
            start_time = time.time()
            
            data = await websocket.receive_bytes()
            frame_count += 1
            
            # Convert bytes to numpy array
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Process the frame
            processed_frame = process_frame(frame, frame_count)
            
            # Convert processed frame back to bytes
            _, buffer = cv2.imencode('.jpg', processed_frame)
            processed_data = buffer.tobytes()
            
            # Send processed frame back to client
            await websocket.send_bytes(processed_data)
            
            # Calculate and log FPS
            end_time = time.time()
            frame_times.append(end_time - start_time)
            
            if end_time - last_fps_log_time >= 5:  # Log FPS every 5 seconds
                fps = len(frame_times) / sum(frame_times)
                logger.info(f"Processed {frame_count} frames. Current FPS: {fps:.2f}")
                last_fps_log_time = end_time
            
    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")

def process_frame(frame, frame_count):
    for i, frame_processor in enumerate(frame_processors):
        logger.debug(f"Applying {frame_processor.NAME} to frame {frame_count}")
        frame = frame_processor.process_frame(source_face, frame)
        logger.debug(f"Applied {frame_processor.NAME} to frame {frame_count}")
    return frame

# Add this at the end of the file
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)