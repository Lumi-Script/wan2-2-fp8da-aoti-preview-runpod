#!/bin/bash
echo "Installing/Updating requirements at runtime..."
export PIP_CACHE_DIR="/runpod-volume/.pip-cache"
mkdir -p $PIP_CACHE_DIR

pip install --upgrade pip
pip install -r requirements.txt

echo "Starting RunPod handler..."
python -u handler.py
