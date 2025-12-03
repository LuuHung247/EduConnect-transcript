import os
import sys
import logging
from pathlib import Path
from datetime import datetime, timezone

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s'
)
logger = logging.getLogger('transcript-worker')

# Add app to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.utils.sqs import receive_transcript_job, delete_message
from app.utils.s3 import upload_to_s3
from app.utils.mongodb import get_db
from bson import ObjectId

# Whisper model (singleton)
_WHISPER_MODEL = None


def get_whisper_model():
    """Lazy load Whisper model"""
    global _WHISPER_MODEL
    if _WHISPER_MODEL is None:
        import torch
        from faster_whisper import WhisperModel
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        
        logger.info(f"🔄 Loading Whisper model on {device}...")
        _WHISPER_MODEL = WhisperModel(
            "small",
            device=device,
            compute_type=compute_type,
            num_workers=1
        )
        logger.info("✅ Whisper model loaded")
    return _WHISPER_MODEL


def update_lesson_success(lesson_id: str, series_id: str, transcript_url: str):
    """Update lesson với transcript URL"""
    _, db = get_db()
    db["lessons"].update_one(
        {"_id": ObjectId(lesson_id), "lesson_serie": series_id},
        {
            "$set": {
                "lesson_transcript": transcript_url,
                "transcript_status": "completed",
                "updatedAt": datetime.now(timezone.utc)
            }
        }
    )


def update_lesson_error(lesson_id: str, series_id: str, error: str):
    """Update lesson với error status"""
    _, db = get_db()
    db["lessons"].update_one(
        {"_id": ObjectId(lesson_id), "lesson_serie": series_id},
        {
            "$set": {
                "transcript_status": "failed",
                "transcript_error": error,
                "updatedAt": datetime.now(timezone.utc)
            }
        }
    )


def process_job(job_data: dict) -> bool:
    """Process single transcript job"""
    video_path = job_data['video_path']
    user_id = job_data['user_id']
    lesson_id = job_data['lesson_id']
    series_id = job_data['series_id']
    filename = job_data.get('filename', Path(video_path).name)
    
    try:
        logger.info(f"🎬 Processing transcript for lesson: {lesson_id}")
        logger.info(f"   Video: {video_path}")
        
        # Check file exists
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Transcribe
        model = get_whisper_model()
        segments_generator, info = model.transcribe(
            video_path,
            language=None,
            beam_size=1,
            best_of=1,
            temperature=0,
            vad_filter=True,
            vad_parameters={
                "threshold": 0.5,
                "min_speech_duration_ms": 250,
            },
        )
        
        # Collect text
        full_text = []
        for segment in segments_generator:
            full_text.append(segment.text.strip())
        
        transcript_text = ' '.join(full_text)
        
        logger.info(f"✅ Transcription complete")
        logger.info(f"   Language: {info.language} ({info.language_probability:.2%})")
        logger.info(f"   Duration: {info.duration:.2f}s")
        logger.info(f"   Text length: {len(transcript_text)} chars")
        
        # Upload transcript to S3
        transcript_filename = f"{Path(filename).stem}_transcript.txt"
        transcript_url = upload_to_s3(
            buffer=transcript_text.encode('utf-8'),
            key=transcript_filename,
            content_type='text/plain',
            prefix=f"files/user-{user_id}/transcripts"
        )
        
        logger.info(f"✅ Transcript uploaded: {transcript_url}")
        
        # Update DB
        update_lesson_success(lesson_id, series_id, transcript_url)
        logger.info(f"✅ Lesson {lesson_id} updated with transcript")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Job failed for lesson {lesson_id}: {str(e)}")
        update_lesson_error(lesson_id, series_id, str(e))
        return False
    
    finally:
        # Cleanup temp file
        if os.path.exists(video_path):
            os.unlink(video_path)
            logger.info(f"🗑️ Temp file deleted: {video_path}")


def main():
    """Main worker loop"""
    logger.info("=" * 60)
    logger.info("🚀 Starting Transcript Worker")
    logger.info("=" * 60)
    
    # Pre-load Whisper model
    logger.info("📦 Pre-loading Whisper model...")
    get_whisper_model()
    
    logger.info("👂 Polling SQS for jobs... (Press Ctrl+C to stop)")
    logger.info("")
    
    jobs_processed = 0
    
    while True:
        try:
            # Long polling - wait up to 20 seconds for message
            job = receive_transcript_job(wait_time=20)
            
            if job is None:
                # No message, continue polling
                continue
            
            logger.info("-" * 40)
            logger.info(f"📥 Received job: {job['message_id']}")
            
            # Process job
            success = process_job(job['body'])
            
            if success:
                # Delete message from queue
                delete_message(job['receipt_handle'])
                jobs_processed += 1
                logger.info(f"✅ Job completed successfully (Total: {jobs_processed})")
            else:
                # Don't delete - message will be retried after visibility timeout
                logger.warning("⚠️ Job failed, will be retried")
            
            logger.info("-" * 40)
            logger.info("")
                
        except KeyboardInterrupt:
            logger.info("")
            logger.info("=" * 60)
            logger.info(f"👋 Shutting down... (Processed {jobs_processed} jobs)")
            logger.info("=" * 60)
            break
            
        except Exception as e:
            logger.error(f"❌ Worker error: {e}")
            # Continue polling


if __name__ == '__main__':
    main()