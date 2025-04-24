#!/bin/bash

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate backend

# Run the FastAPI application with uvicorn
cd /home/petpooja-504/Documents/fina/backend
uvicorn api:app --reload --host 0.0.0.0 --port 8000