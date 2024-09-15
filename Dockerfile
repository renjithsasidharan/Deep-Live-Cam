# Use NVIDIA CUDA 11.8 runtime as the base image
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    ffmpeg \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as the default python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Clone the repository
RUN git clone https://github.com/hacksider/Deep-Live-Cam.git .

# Download models
RUN mkdir -p models && \
    wget -O models/GFPGANv1.4.pth https://huggingface.co/hacksider/deep-live-cam/resolve/main/GFPGANv1.4.pth && \
    wget -O models/inswapper_128_fp16.onnx https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128_fp16.onnx

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Install CUDA-enabled onnxruntime
RUN pip3 uninstall -y onnxruntime onnxruntime-gpu && \
    pip3 install --no-cache-dir onnxruntime-gpu==1.16.3

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Set a default execution provider
ENV EXECUTION_PROVIDER=cuda

# Run main.py when the container launches, using the specified execution provider
CMD python run.py --execution-provider $EXECUTION_PROVIDER