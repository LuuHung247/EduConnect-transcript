import os
import sys
import logging
from pathlib import Path
from datetime import datetime, timezone
import google.generativeai as genai

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

# Gemini API config
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')


def init_gemini():
    """Initialize Gemini API"""
    if not GEMINI_API_KEY:
        logger.warning("⚠️ GEMINI_API_KEY not configured, summary will be skipped")
        return False
    
    genai.configure(api_key=GEMINI_API_KEY)
    logger.info("✅ Gemini API initialized")
    return True


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


def generate_summary(transcript_text: str) -> str:
    """
    Generate structured lesson notes in English using Gemini API.
    Removes icons and forces English output to prevent font issues.
    """
    if not GEMINI_API_KEY:
        logger.error("Gemini API Key is missing.")
        return ""
    
    try:
        logger.info("Generating structured notes with Gemini...")
        
        # Cấu hình model: Temperature thấp để đảm bảo tính chính xác
        generation_config = {
            "temperature": 0.3, 
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 2048,
        }

        model = genai.GenerativeModel(
            model_name='gemini-2.0-flash',
            generation_config=generation_config
        )
        
        # Prompt được tối ưu cho đầu ra tiếng Anh, không icon
        prompt = f"""
You are an expert educational content summarizer and note-taker. 
Your task is to process the following raw video transcript into high-quality, structured study notes in ENGLISH.

INPUT TRANSCRIPT:
\"\"\"
{transcript_text}
\"\"\"

INSTRUCTIONS:
1. Language: The final output must be strictly in ENGLISH, regardless of the input language.
2. Formatting: Use standard Markdown. Do NOT use emojis or special unicode characters (like icons) that might cause font rendering issues.
3. Quality Control: Remove filler words, verbal tics, and repetitive phrasing. Focus on the core educational value.

OUTPUT STRUCTURE (Follow this strictly):

# Lesson Title
[Create a concise, descriptive title]

## Executive Summary
[Provide a 2-3 sentence high-level overview of the entire lesson. What is the main problem and solution?]

## Key Concepts & Details
[Use hierarchical bullet points. Bold key terms.]
- Concept 1:
  - Explanation or details...
  - Example...
- Concept 2:
  - Explanation or details...

## Terminology (If applicable)
[Create a Markdown table for technical terms]
| Term | Definition |
|------|------------|
| ...  | ...        |

## Key Takeaways
[List actionable advice, warnings, or conclusions]
- ...
- ...

"""

        response = model.generate_content(prompt)
        
        if not response.parts:
            logger.warning("Gemini generated empty response.")
            return "Error: Could not generate summary."

        summary = response.text.strip()
        
        logger.info(f"Summary generated ({len(summary)} chars)")
        return summary
        
    except Exception as e:
        logger.error(f"Failed to generate summary: {e}")
        return ""


def update_lesson_success(
    lesson_id: str, 
    series_id: str, 
    transcript_url: str,
    summary_url: str = ""
):
    """Update lesson với transcript URL và summary URL"""
    _, db = get_db()
    
    update_data = {
        "lesson_transcript": transcript_url,
        "transcript_status": "completed",
        "updatedAt": datetime.now(timezone.utc)
    }
    
    if summary_url:
        update_data["lesson_summary"] = summary_url
    
    db["lessons"].update_one(
        {"_id": ObjectId(lesson_id), "lesson_serie": series_id},
        {"$set": update_data}
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
        
        # ===== Step 1: Transcribe =====
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
        
        # ===== Step 2: Upload transcript to S3 =====
        transcript_filename = f"{Path(filename).stem}_transcript.txt"
        transcript_url = upload_to_s3(
            buffer=transcript_text.encode('utf-8'),
            key=transcript_filename,
            content_type='text/plain',
            prefix=f"files/user-{user_id}/transcripts"
        )
        logger.info(f"✅ Transcript uploaded: {transcript_url}")
        
        # ===== Step 3: Generate summary with Gemini =====
        summary_url = ""
        if transcript_text and len(transcript_text) > 50:  # Chỉ tóm tắt nếu có nội dung
            summary_text = generate_summary(transcript_text)
            
            if summary_text:
                # Upload summary to S3
                summary_filename = f"{Path(filename).stem}_summary.txt"
                summary_url = upload_to_s3(
                    buffer=summary_text.encode('utf-8'),
                    key=summary_filename,
                    content_type='text/plain',
                    prefix=f"files/user-{user_id}/summaries"
                )
                logger.info(f"✅ Summary uploaded: {summary_url}")
        
        # ===== Step 4: Update DB =====
        update_lesson_success(lesson_id, series_id, transcript_url, summary_url)
        logger.info(f"✅ Lesson {lesson_id} updated with transcript and summary")
        
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
    
    # Initialize Gemini
    gemini_ready = init_gemini()
    if not gemini_ready:
        logger.warning("⚠️ Running without summary generation")
    
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
