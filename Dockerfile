# Use NVIDIA CUDA 11.8 runtime as the base image
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

RUN rm -rf "/usr/local/share/boost"
RUN rm -rf "$AGENT_TOOLSDIRECTORY"

# Set the working directory in the container
WORKDIR /app

# Set environment variable to non-interactive (this prevents prompts)
ENV DEBIAN_FRONTEND=noninteractive

# Set the timezone
ENV TZ=Etc/UTC

RUN echo 'Acquire::https::developer.download.nvidia.com::Verify-Peer "false";' | tee -a /etc/apt/apt.conf

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    ffmpeg \
    wget \
    g++ \
    libgl1-mesa-glx \
    python3-tk \
    tzdata \
    vim \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone \
    && dpkg-reconfigure --frontend noninteractive tzdata \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as the default python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Clone the repository
RUN git clone https://github.com/renjithsasidharan/Deep-Live-Cam.git .

# Download models
RUN mkdir -p models && \
    wget -O models/GFPGANv1.4.pth https://huggingface.co/hacksider/deep-live-cam/resolve/main/GFPGANv1.4.pth && \
    wget -O models/inswapper_128.onnx https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128.onnx

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Install CUDA-enabled onnxruntime
RUN pip3 uninstall -y onnxruntime onnxruntime-gpu && \
    pip3 install --no-cache-dir onnxruntime-gpu==1.16.3

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Set CUDA related environment variables
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
ENV PATH=/usr/local/cuda/bin:$PATH

# Create symlinks for CUDA libraries
RUN ln -s /usr/local/cuda/lib64/libnvrtc.so /usr/lib/libnvrtc.so && \
    ln -s /usr/local/cuda/lib64/libnvrtc.so.11.8 /usr/lib/libnvrtc.so.11.8

# Set a default execution provider
ENV EXECUTION_PROVIDER=cuda

# Run main.py when the container launches, using the specified execution provider
CMD python main.py --execution-provider $EXECUTION_PROVIDER