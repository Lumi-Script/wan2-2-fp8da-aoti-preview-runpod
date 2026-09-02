# Use a RunPod PyTorch base image for dramatically faster cold starts
FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Enable lightning-fast downloads from huggingface at runtime
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV HF_HOME="/runpod-volume/huggingface"

# Copy the source code
COPY . .
RUN chmod +x start.sh

# Command to start the RunPod handler via the startup script
CMD ["./start.sh"]
