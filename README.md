# AI Image Moderation Service

## Start

1. Create a Python environment.
2. Install dependencies:

   pip install -r requirements.txt

3. Run the API server:

   uvicorn app:app --host 0.0.0.0 --port 8001 --reload

## Endpoints

- GET /health
- POST /moderate-images
  - multipart/form-data
  - field files: image files (multiple)
  - field threshold: optional float in [0, 1]

## Environment Variables

- CHECKPOINT_DIR: path to model checkpoint folder (default: ./checkpoint)
- NSFW_THRESHOLD: default threshold (default: 0.70)
"# MAD_AI" 
