from uuid import uuid4
from typing import List, Dict, Optional, Callable
from fastapi import UploadFile, BackgroundTasks
from pathlib import Path
import tempfile
import os
import logging
from faster_whisper import WhisperModel
import torch
from bson import ObjectId
from app.utils.mongodb import get_db
from datetime import datetime, timezone
from app.utils.s3 import upload_to_s3, delete_from_s3

logger = logging.getLogger(__name__)

# Global Whisper model (lazy loading)
_WHISPER_MODEL = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"


def get_whisper_model():
    """Lazy load Whisper model"""
    global _WHISPER_MODEL
    if _WHISPER_MODEL is None:
        logger.info(f"Loading Whisper model on {DEVICE}")
        _WHISPER_MODEL = WhisperModel(
            "small",
            device=DEVICE,
            compute_type=COMPUTE_TYPE,
            num_workers=1
        )
        logger.info("✅ Whisper model loaded")
    return _WHISPER_MODEL


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
        # Callback info để update DB sau khi transcript xong
        lesson_id: Optional[str] = None,
        series_id: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Upload video file and optionally create transcript in background
        
        Args:
            file: Video file to upload
            user_id: User ID
            background_tasks: FastAPI BackgroundTasks for async processing
            create_transcript: Whether to create transcript (default: True)
            lesson_id: Lesson ID để update transcript URL sau
            series_id: Series ID để update transcript URL sau
        
        Returns:
            Dict with url, key, and transcript_status
        """
        filename = f"{uuid4()}_{file.filename}"
        content_type = file.content_type or 'video/mp4'
        buffer = await file.read()
        
        # Upload video to S3 first
        url = upload_to_s3(
            buffer=buffer,
            key=filename,
            content_type=content_type,
            prefix=f"files/user-{user_id}/videos"
        )
        
        video_key = f"files/user-{user_id}/videos/{filename}"
        
        result = {
            "url": url,
            "key": video_key,
            "transcript_status": "disabled"
        }
        
        # Add transcript task to background if requested
        if create_transcript and background_tasks and lesson_id and series_id:
            background_tasks.add_task(
                self._create_transcript_and_update_lesson,
                buffer=buffer,
                video_filename=filename,
                user_id=user_id,
                lesson_id=lesson_id,
                series_id=series_id
            )
            result["transcript_status"] = "processing"
            logger.info(f"✅ Video uploaded: {video_key}")
            logger.info(f"🔄 Transcript generation queued for lesson: {lesson_id}")
        elif create_transcript and not (lesson_id and series_id):
            logger.warning("⚠️ Transcript requested but lesson_id/series_id not provided")
            result["transcript_status"] = "skipped"
        
        return result
    
    def _create_transcript_and_update_lesson(
        self,
        buffer: bytes,
        video_filename: str,
        user_id: str,
        lesson_id: str,
        series_id: str
    ):
        """
        Background task: Create transcript từ video, upload S3, và update lesson trong DB
        """
        temp_video_path = None
        
        try:
            # Save video to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(video_filename).suffix) as temp_video:
                temp_video.write(buffer)
                temp_video_path = temp_video.name
            
            logger.info(f"🎬 Starting transcription for lesson: {lesson_id}")
            
            # Generate transcript
            model = get_whisper_model()
            segments_generator, info = model.transcribe(
                temp_video_path,
                language=None,  # Auto-detect
                beam_size=1,
                best_of=1,
                temperature=0,
                vad_filter=True,
                vad_parameters={
                    "threshold": 0.5,
                    "min_speech_duration_ms": 250,
                },
            )
            
            # Collect segments
            full_text = []
            for segment in segments_generator:
                full_text.append(segment.text.strip())
            
            transcript_text = ' '.join(full_text)
            
            # Upload transcript to S3
            transcript_filename = f"{Path(video_filename).stem}_transcript.txt"
            transcript_buffer = transcript_text.encode('utf-8')
            
            transcript_url = upload_to_s3(
                buffer=transcript_buffer,
                key=transcript_filename,
                content_type='text/plain',
                prefix=f"files/user-{user_id}/transcripts"
            )
            
            logger.info(f"✅ Transcript uploaded: {transcript_url}")
            logger.info(f"   Language: {info.language} ({info.language_probability:.2%})")
            logger.info(f"   Duration: {info.duration:.2f}s")
            
            # Update lesson trong DB
            self._update_lesson_transcript(lesson_id, series_id, transcript_url)
            
            logger.info(f"✅ Lesson {lesson_id} updated with transcript")
            
        except Exception as e:
            logger.error(f"❌ Transcript generation failed for lesson {lesson_id}: {str(e)}")
            # Optionally: Update lesson với error status
            self._update_lesson_transcript_error(lesson_id, series_id, str(e))
        
        finally:
            # Cleanup temporary files
            if temp_video_path and os.path.exists(temp_video_path):
                os.unlink(temp_video_path)
    
    def _update_lesson_transcript(self, lesson_id: str, series_id: str, transcript_url: str):
        """Update lesson với transcript URL"""

        
        _, db = get_db()
        db["lessons"].update_one(
            {"_id": ObjectId(lesson_id), "lesson_serie": series_id},
            {
                "$set": {
                    "lesson_transcript": transcript_url,
                    "transcript_status": "completed",
                    "updatedAt":datetime.now(timezone.utc).isoformat() # MongoDB sẽ set timestamp nếu có middleware
                }
            }
        )
    
    def _update_lesson_transcript_error(self, lesson_id: str, series_id: str, error: str):
        """Update lesson với transcript error status"""
        from bson import ObjectId
        from app.utils.mongodb import get_db
        
        _, db = get_db()
        db["lessons"].update_one(
            {"_id": ObjectId(lesson_id), "lesson_serie": series_id},
            {
                "$set": {
                    "transcript_status": "failed",
                    "transcript_error": error,
                    "updatedAt": None
                }
            }
        )
    
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
        
        return {
            "deleted": deleted,
            "failed": failed
        }