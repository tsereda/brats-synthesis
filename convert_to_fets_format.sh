#!/bin/bash

# Script to convert BraTS format to FeTS format
# Creates proper directory structure and file naming for FeTS

# Function to display usage
usage() {
    echo "Usage: $0 -i <input_directory> -o <output_directory>"
    echo ""
    echo "Arguments:"
    echo "  -i, --input   Input directory containing BraTS-GLI-* subdirectories"
    echo "  -o, --output  Output directory for FeTS formatted data"
    echo "  -h, --help    Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 -i completed_cases -o fets_formatted"
    exit 1
}

# Initialize variables
INPUT_DIR=""
OUTPUT_DIR=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--input)
            INPUT_DIR="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Validate arguments
if [[ -z "$INPUT_DIR" || -z "$OUTPUT_DIR" ]]; then
    echo "Error: Both input and output directories must be specified."
    usage
fi

# Check if input directory exists
if [[ ! -d "$INPUT_DIR" ]]; then
    echo "Error: Input directory '$INPUT_DIR' does not exist."
    exit 1
fi

echo "Converting BraTS format to FeTS format..."
echo "Input directory: $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Counter for patient numbering
patient_num=1

# Process each case directory
for case_dir in "$INPUT_DIR"/BraTS-GLI-*/; do
    if [ -d "$case_dir" ]; then
        case_name=$(basename "$case_dir")
       
        # Create patient directory with zero-padded number
        patient_dir=$(printf "Patient_%03d" $patient_num)
        mkdir -p "$OUTPUT_DIR/$patient_dir"
       
        echo "Processing $case_name → $patient_dir"
       
        # Check if all required files exist
        if [ -f "$case_dir/t1.nii.gz" ] && [ -f "$case_dir/t1ce.nii.gz" ] && [ -f "$case_dir/t2.nii.gz" ] && [ -f "$case_dir/flair.nii.gz" ]; then
            # Copy and rename files to FeTS format
            cp "$case_dir/t1.nii.gz" "$OUTPUT_DIR/$patient_dir/${patient_dir}_brain_t1.nii.gz"
            cp "$case_dir/t1ce.nii.gz" "$OUTPUT_DIR/$patient_dir/${patient_dir}_brain_t1ce.nii.gz"
            cp "$case_dir/t2.nii.gz" "$OUTPUT_DIR/$patient_dir/${patient_dir}_brain_t2.nii.gz"
            cp "$case_dir/flair.nii.gz" "$OUTPUT_DIR/$patient_dir/${patient_dir}_brain_flair.nii.gz"
           
            echo "  ✅ Converted $case_name to $patient_dir"
            ((patient_num++))
        else
            echo "  ❌ Missing files in $case_name, skipping"
        fi
    fi
done

echo ""
echo "Conversion complete! Created $((patient_num-1)) patient directories in '$OUTPUT_DIR'."
echo ""
echo "Now you can run FeTS segmentation:"
echo "./squashfs-root/usr/bin/fets_cli_segment -d '$OUTPUT_DIR'/ -a fets_singlet,fets_triplet -lF STAPLE,ITKVoting,SIMPLE,MajorityVoting -g 1 -t 0"