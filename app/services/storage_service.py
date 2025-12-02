from uuid import uuid4
from typing import List, Dict, Optional
from fastapi import UploadFile, BackgroundTasks
from pathlib import Path
import tempfile
import os
import logging
from faster_whisper import WhisperModel
import torch

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
        create_transcript: bool = True
    ) -> Dict[str, str]:
        """
        Upload video file and optionally create transcript in background
        
        Args:
            file: Video file to upload
            user_id: User ID
            background_tasks: FastAPI BackgroundTasks for async processing
            create_transcript: Whether to create transcript (default: True)
        
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
        if create_transcript and background_tasks:
            background_tasks.add_task(
                self._create_and_upload_transcript,
                buffer=buffer,
                video_filename=filename,
                user_id=user_id,
                video_url=url
            )
            result["transcript_status"] = "processing"
            logger.info(f"✅ Video uploaded: {video_key}")
            logger.info(f"🔄 Transcript generation queued for: {filename}")
        
        return result
    
    async def _create_and_upload_transcript(
        self,
        buffer: bytes,
        video_filename: str,
        user_id: str,
        video_url: str
    ):
        """
        Background task: Create transcript from video and upload to S3
        
        This runs asynchronously after video upload completes
        """
        temp_video_path = None
        temp_transcript_path = None
        
        try:
            # Save video to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(video_filename).suffix) as temp_video:
                temp_video.write(buffer)
                temp_video_path = temp_video.name
            
            logger.info(f"🎬 Starting transcription for: {video_filename}")
            
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
            segments = []
            full_text = []
            
            for segment in segments_generator:
                segments.append({
                    'start': round(segment.start, 2),
                    'end': round(segment.end, 2),
                    'text': segment.text.strip()
                })
                full_text.append(segment.text.strip())
            
            transcript_text = ' '.join(full_text)
            
            # Save transcript to temporary file
            transcript_filename = f"{Path(video_filename).stem}_transcript.txt"
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8') as temp_transcript:
                temp_transcript.write(transcript_text)
                temp_transcript_path = temp_transcript.name
            
            # Upload transcript to S3
            with open(temp_transcript_path, 'rb') as f:
                transcript_buffer = f.read()
            
            transcript_url = upload_to_s3(
                buffer=transcript_buffer,
                key=transcript_filename,
                content_type='text/plain',
                prefix=f"files/user-{user_id}/transcripts"
            )
            
            transcript_key = f"files/user-{user_id}/transcripts/{transcript_filename}"
            
            logger.info(f"✅ Transcript uploaded: {transcript_key}")
            logger.info(f"   Language: {info.language} ({info.language_probability:.2%})")
            logger.info(f"   Duration: {info.duration:.2f}s")
            logger.info(f"   Segments: {len(segments)}")
            
            # TODO: Optionally update database with transcript info
            # You can call an API or update DB here with:
            # - video_url
            # - transcript_url
            # - transcript_key
            # - language
            # - segments
            
        except Exception as e:
            logger.error(f"❌ Transcript generation failed for {video_filename}: {str(e)}")
            # TODO: Optionally notify user or update status in database
        
        finally:
            # Cleanup temporary files
            if temp_video_path and os.path.exists(temp_video_path):
                os.unlink(temp_video_path)
            if temp_transcript_path and os.path.exists(temp_transcript_path):
                os.unlink(temp_transcript_path)
    
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