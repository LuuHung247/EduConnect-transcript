# Media Service - EduConnect Microservice

File upload and storage microservice for EduConnect platform.

## Overview

Media Service handles all media-related operations including:
- Image uploads (thumbnails, avatars)
- Video uploads with optional transcription
- Document uploads (PDF, etc.)
- File deletion from S3
- Batch operations

## Architecture

```
Frontend/Backend → Media Service → AWS S3
                        ↓
                   AWS SQS (for transcripts)
```

## Endpoints

### Upload Operations
- `POST /api/upload/thumbnail` - Upload thumbnail/avatar images
- `POST /api/upload/video` - Upload video with optional transcription
- `POST /api/upload/document` - Upload document file
- `POST /api/upload/documents/batch` - Upload multiple documents

### Delete Operations
- `DELETE /api/delete` - Delete single file
- `DELETE /api/delete/batch` - Delete multiple files

### Health Check
- `GET /health` - Service health check

## Running Locally

### Prerequisites
- Python 3.11+
- AWS Account with S3 access
- MongoDB (for transcript metadata)

### Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure environment:
```bash
cp .env.example .env
# Edit .env with your AWS credentials and S3 bucket
```

3. Run:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8001 --log-level info
```

Service runs on http://localhost:8001

### Run Transcript Worker (Optional)
```bash
python3 worker.py
```

## Running with Docker

```bash
docker build -t media-service .
docker run -p 8001:8001 --env-file .env media-service
```

## Running with Docker Compose

See main docker-compose.yml in project root:
```bash
docker-compose up media-service
```

## Configuration

Environment variables (see `.env.example`):

- `AWS_REGION` - AWS region (default: ap-southeast-1)
- `AWS_ACCESS_KEY_ID` - AWS access key
- `AWS_SECRET_ACCESS_KEY` - AWS secret key
- `AWS_S3_BUCKET` - S3 bucket name
- `AWS_SQS_QUEUE_URL` - SQS queue URL for transcript jobs
- `MONGODB_URI` - MongoDB connection string
- `MONGODB_NAME` - Database name
- `GEMINI_API_KEY` - Gemini API key (optional)
- `DEEPSEEK_API_KEY` - DeepSeek API key (optional)

## Technology Stack

- **Framework**: FastAPI 0.x
- **Server**: Uvicorn
- **File Upload**: python-multipart, aiofiles
- **AWS SDK**: boto3
- **Database**: MongoDB (PyMongo)

## Security

- File type validation
- Size limits enforced
- Non-root user in Docker
- Secure credential handling
