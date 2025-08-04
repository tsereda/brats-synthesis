# BraTS Challenge Submission - Quick Start

## 🚀 3 Steps to Submit

### Step 1: Add Your Checkpoints (5 minutes)
```bash
# Put your 4 best model files in checkpoints/
mkdir -p checkpoints/
cp /path/to/your/brats_t1n_*.pt checkpoints/
cp /path/to/your/brats_t1c_*.pt checkpoints/
cp /path/to/your/brats_t2w_*.pt checkpoints/
cp /path/to/your/brats_t2f_*.pt checkpoints/

# Update Dockerfile to copy them:
# Uncomment and fix the COPY lines in Dockerfile around line 25
```

### Step 2: Build & Test (10 minutes)
```bash
# Make test script executable
chmod +x build_and_test.sh

# Build Docker image
./build_and_test.sh

# Test with sample data (optional but recommended)
# 1. Create test_input/ with sample BraTS cases
# 2. Run: ./build_and_test.sh test
```

### Step 3: Submit to Synapse (15 minutes)
```bash
# Login to Synapse Docker registry
docker login docker.synapse.org --username YOUR_SYNAPSE_USERNAME
# Use your Personal Access Token as password

# Tag for your Synapse project
docker tag fast-cwdm-brats2025 docker.synapse.org/YOUR_PROJECT_ID/fast-cwdm-brats2025:latest

# Upload
docker push docker.synapse.org/YOUR_PROJECT_ID/fast-cwdm-brats2025:latest

# Submit via Synapse web interface:
# 1. Go to your project Docker tab
# 2. Click "Docker Repository Tools" 
# 3. Select "Submit Docker Repository to Challenge"
# 4. Choose the appropriate task queue
```

## 🔧 What Each File Does

- **`main.py`** - Challenge entrypoint that calls your existing code
- **`Dockerfile`** - Container definition with all dependencies  
- **`build_and_test.sh`** - Helper script for building and testing
- **`requirements.txt`** - Python dependencies list

## 💡 Key Points

- Your existing `complete_dataset.py` does all the work
- The Docker container just wraps it for the challenge
- No need to change your model code at all
- Fast sampling with 100 steps for speed

## 🆘 Troubleshooting

**"Checkpoint not found"** → Check your checkpoint filenames match the pattern in `find_checkpoint()`

**"CUDA out of memory"** → Your code already uses batch_size=1, should be fine

**"Import errors"** → All dependencies are in the Dockerfile

**"No outputs created"** → Check that your checkpoints are properly copied into the container

## 📋 Submission Checklist

- [ ] Checkpoints copied to `checkpoints/` directory
- [ ] Dockerfile COPY lines uncommented and pointing to your checkpoints
- [ ] Docker builds successfully: `./build_and_test.sh`
- [ ] Container runs without network: `./build_and_test.sh test`
- [ ] Uploaded to Synapse Docker registry
- [ ] Submitted to challenge queue

**Total time needed: ~30 minutes** ⚡