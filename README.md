# Media Service

Microservice for uploading and deleting media files (images, videos, documents) to S3.

## Endpoints

- POST /media/upload
- DELETE /media/delete

### Run app

- uvicorn main:app --reload --host 0.0.0.0 --port 8001 --log-level info
- python3 worker.py
