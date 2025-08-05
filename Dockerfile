# BraTS-Lighthouse 2025 Challenge Submission
# Fast-CWDM: Conditional Wavelet Diffusion Model
FROM --platform=linux/amd64 pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    nibabel==5.1.0 \
    numpy==1.24.3 \
    monai==1.2.0 \
    blobfile==2.0.2 \
    matplotlib==3.7.2 \
    tqdm==4.65.0 \
    PyWavelets \
    tensorboard==2.13.0 \
    wandb==0.15.8 \
    pyyaml \
    scipy

# Copy the entire codebase
# Copy the app directory contents to /app/
COPY app/ /app/
# Copy other necessary files
COPY main.py /app/
COPY run.sh /app/

# Create checkpoints directory (to be populated with your models)
RUN mkdir -p /app/checkpoints

# Copy your best model checkpoints
# YOU NEED TO ADD YOUR CHECKPOINT FILES HERE:
# COPY path/to/your/brats_t1n_*.pt /app/checkpoints/
# COPY path/to/your/brats_t1c_*.pt /app/checkpoints/
# COPY path/to/your/brats_t2w_*.pt /app/checkpoints/
# COPY path/to/your/brats_t2f_*.pt /app/checkpoints/

# Set Python path for imports
ENV PYTHONPATH="/app"

# Ensure scripts are executable
RUN chmod +x /app/main.py

# Set environment variables for reproducibility
ENV CUDA_VISIBLE_DEVICES=0
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=UTF-8

# Disable wandb for submission (no network access)
ENV WANDB_MODE=disabled
ENV WANDB_DISABLED=true

# Runtime configuration
# The challenge will mount:
# --volume /path/to/input:/input:ro (read-only input data)
# --volume /path/to/output:/output:rw (writable output directory)
# --memory=16G --shm-size 4G (memory limits)
# --network none (no network access)

# Set entrypoint
ENTRYPOINT ["python", "/app/main.py"]

# Metadata
LABEL maintainer="Fast-CWDM Team"
LABEL description="Fast Conditional Wavelet Diffusion Model for BraTS Challenge 2025"
LABEL challenge="BraTS-Lighthouse 2025"
LABEL method="Fast-CWDM"