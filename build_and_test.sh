#!/bin/bash
# Quick build and test script for BraTS submission
set -e  # Exit on any error

echo "🐳 Building BraTS Challenge Docker Container..."

# Build the Docker image
docker build -t fast-cwdm-brats2025 .

echo "✅ Docker build completed!"

# Check if test data exists
TEST_INPUT="./test_input"
TEST_OUTPUT="./test_output"

if [ ! -d "$TEST_INPUT" ]; then
    echo "⚠️  No test input directory found at $TEST_INPUT"
    echo "📁 Create test data structure like:"
    echo "   $TEST_INPUT/"
    echo "     └── BraTS-GLI-00001-000/"
    echo "         ├── BraTS-GLI-00001-000-t1n.nii.gz"
    echo "         ├── BraTS-GLI-00001-000-t1c.nii.gz"
    echo "         └── BraTS-GLI-00001-000-t2f.nii.gz  # missing t2w"
    echo ""
    echo "🚀 To test when ready, run:"
    echo "   ./build_and_test.sh test"
    exit 0
fi

# If 'test' argument provided, run the container
if [ "$1" = "test" ]; then
    echo "🧪 Testing Docker container..."
   
    # Create output directory
    mkdir -p "$TEST_OUTPUT"
   
    # Run the container (simulating challenge environment)
    echo "🏃 Running container with challenge settings..."
    docker run --rm \
        --network none \
        --gpus=all \
        --volume "$(pwd)/$TEST_INPUT:/input:ro" \
        --volume "$(pwd)/$TEST_OUTPUT:/output:rw" \
        --memory=16G \
        --shm-size=4G \
        fast-cwdm-brats2025
   
    echo "🔍 Checking outputs..."
    
    # Check for .nii.gz files recursively in output directory
    OUTPUT_FILES=$(find "$TEST_OUTPUT" -name "*.nii.gz" 2>/dev/null | wc -l)
    
    if [ "$OUTPUT_FILES" -gt 0 ]; then
        echo "✅ Success! Found $OUTPUT_FILES output files:"
        find "$TEST_OUTPUT" -name "*.nii.gz" -exec ls -lh {} \;
        echo ""
        echo "📁 Directory structure:"
        ls -la "$TEST_OUTPUT"
        if [ -d "$TEST_OUTPUT"/*/ ]; then
            ls -la "$TEST_OUTPUT"/*/
        fi
    else
        echo "❌ No .nii.gz output files found in $TEST_OUTPUT"
        echo "📁 Current output directory contents:"
        ls -la "$TEST_OUTPUT" 2>/dev/null || echo "Output directory is empty or doesn't exist"
        exit 1
    fi
   
    echo "🎉 Docker test completed successfully!"
    echo "📤 Ready for Synapse upload!"
else
    echo "✅ Build completed! To test:"
    echo "   ./build_and_test.sh test"
fi