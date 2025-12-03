from uuid import uuid4
from typing import List, Dict, Optional
from fastapi import UploadFile, BackgroundTasks
from pathlib import Path
import os
import logging
from datetime import datetime, timezone

from app.utils.s3 import upload_to_s3, delete_from_s3
from app.utils.sqs import send_transcript_job

logger = logging.getLogger(__name__)

# Temp directory cho video queue
TEMP_VIDEO_DIR = "/tmp/video_queue"
os.makedirs(TEMP_VIDEO_DIR, exist_ok=True)


class StorageService:
    """Service for handling file storage operations"""
    
    async def upload_thumbnail(self, file: UploadFile, user_id: str) -> Dict[str, str]:
        """Upload thumbnail image"""
        filename = f"{uuid4()}_{file.filename}"
        content_type = file.content_type or 'image/jpeg'
        buffer = await file.read()
        
        url = upload_to_s3(
            buffer=buffer,
            key=filename,
            content_type=content_type,
            prefix=f"files/user-{user_id}/thumbnail"
        )
        
        return {
            "url": url,
            "key": f"files/user-{user_id}/thumbnail/{filename}"
        }
    
    async def upload_video(
        self, 
        file: UploadFile, 
        user_id: str,
        background_tasks: Optional[BackgroundTasks] = None,
        create_transcript: bool = True,
        lesson_id: Optional[str] = None,
        series_id: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Upload video và gửi transcript job tới SQS
        Worker sẽ xử lý transcript riêng
        """
        filename = f"{uuid4()}_{file.filename}"
        content_type = file.content_type or 'video/mp4'
        buffer = await file.read()
        
        # Upload video to S3
        url = upload_to_s3(
            buffer=buffer,
            key=filename,
            content_type=content_type,
            prefix=f"files/user-{user_id}/videos"
        )
        
        video_key = f"files/user-{user_id}/videos/{filename}"
        
        logger.info(f"✅ Video uploaded: {video_key}")
        
        result = {
            "url": url,
            "key": video_key,
            "transcript_status": "disabled"
        }
        
        # Gửi job tới SQS (Worker sẽ xử lý)
        if create_transcript and lesson_id and series_id:
            # Lưu video temp để worker đọc
            temp_path = f"{TEMP_VIDEO_DIR}/{filename}"
            with open(temp_path, 'wb') as f:
                f.write(buffer)
            
            # Gửi job tới SQS
            job_sent = send_transcript_job({
                "video_path": temp_path,
                "video_url": url,
                "video_key": video_key,
                "user_id": user_id,
                "lesson_id": lesson_id,
                "series_id": series_id,
                "filename": filename
            })
            
            if job_sent:
                result["transcript_status"] = "queued"
                logger.info(f"📋 Transcript job queued for lesson: {lesson_id}")
            else:
                result["transcript_status"] = "queue_failed"
                logger.warning(f"⚠️ Failed to queue transcript job")
        elif create_transcript and not (lesson_id and series_id):
            logger.warning("⚠️ Transcript requested but lesson_id/series_id not provided")
            result["transcript_status"] = "skipped"
        
        return result
    
    async def upload_document(self, file: UploadFile, user_id: str) -> Dict[str, str]:
        """Upload document file"""
        filename = f"{uuid4()}_{file.filename}"
        content_type = file.content_type or 'application/pdf'
        buffer = await file.read()
        
        url = upload_to_s3(
            buffer=buffer,
            key=filename,
            content_type=content_type,
            prefix=f"files/user-{user_id}/docs"
        )
        
        return {
            "url": url,
            "key": f"files/user-{user_id}/docs/{filename}"
        }
    
    async def upload_documents_batch(self, files: List[UploadFile], user_id: str) -> List[Dict[str, str]]:
        """Upload multiple documents"""
        results = []
        
        for file in files:
            try:
                result = await self.upload_document(file, user_id)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to upload {file.filename}: {e}")
                results.append({
                    "url": None,
                    "key": None,
                    "error": str(e),
                    "filename": file.filename
                })
        
        return results
    
    async def upload_transcript(self, file: UploadFile, user_id: str) -> Dict[str, str]:
        """Upload transcript file manually"""
        filename = f"{uuid4()}_{file.filename}"
        content_type = file.content_type or 'text/plain'
        buffer = await file.read()
        
        url = upload_to_s3(
            buffer=buffer,
            key=filename,
            content_type=content_type,
            prefix=f"files/user-{user_id}/transcripts"
        )
        
        return {
            "url": url,
            "key": f"files/user-{user_id}/transcripts/{filename}"
        }
    
    def delete_file(self, url_or_key: str) -> bool:
        """Delete a file from S3"""
        return delete_from_s3(url_or_key)
    
    def delete_files_batch(self, urls: List[str]) -> Dict[str, List]:
        """Delete multiple files"""
        deleted = []
        failed = []
        
        for url in urls:
            try:
                if delete_from_s3(url):
                    deleted.append(url)
                else:
                    failed.append({"url": url, "error": "Delete failed"})
            except Exception as e:
                failed.append({"url": url, "error": str(e)})
        
        return {"deleted": deleted, "failed": failed}