# import os
# import sys
# import logging
# import time
# import random
# import asyncio
# from pathlib import Path
# from datetime import datetime, timezone

# import google.generativeai as genai
# from google.api_core import exceptions as google_exceptions

# # Import OpenAI async client cho DeepSeek
# from openai import AsyncOpenAI

# # Setup logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s %(levelname)s %(name)s: %(message)s'
# )
# logger = logging.getLogger('transcript-worker')

# # Add app to path
# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# from app.utils.sqs import receive_transcript_job, delete_message
# from app.utils.s3 import upload_to_s3
# from app.utils.mongodb import get_db
# from bson import ObjectId

# # Whisper model (singleton)
# _WHISPER_MODEL = None

# # API Keys
# GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')
# DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY', '')

# # DeepSeek async client (singleton)
# _DEEPSEEK_CLIENT = None


# def init_gemini():
#     if not GEMINI_API_KEY:
#         logger.warning("⚠️ GEMINI_API_KEY not configured, timeline will be skipped")
#         return False
#     genai.configure(api_key=GEMINI_API_KEY)
#     logger.info("✅ Gemini API initialized")
#     return True


# def init_deepseek():
#     global _DEEPSEEK_CLIENT
#     if not DEEPSEEK_API_KEY:
#         logger.warning("⚠️ DEEPSEEK_API_KEY not configured, summary will be skipped")
#         return False
#     _DEEPSEEK_CLIENT = AsyncOpenAI(
#         api_key=DEEPSEEK_API_KEY,
#         base_url="https://api.deepseek.com"
#     )
#     logger.info("✅ DeepSeek API initialized")
#     return True


# def get_whisper_model():
#     global _WHISPER_MODEL
#     if _WHISPER_MODEL is None:
#         import torch
#         from faster_whisper import WhisperModel
        
#         device = "cuda" if torch.cuda.is_available() else "cpu"
#         compute_type = "float16" if device == "cuda" else "int8"
        
#         logger.info(f"🔄 Loading Whisper model on {device}...")
#         _WHISPER_MODEL = WhisperModel("small", device=device, compute_type=compute_type, num_workers=1)
#         logger.info("✅ Whisper model loaded")
#     return _WHISPER_MODEL


# def format_timestamp(seconds: float) -> str:
#     minutes = int(seconds // 60)
#     remaining_seconds = int(seconds % 60)
#     return f"{minutes:02d}:{remaining_seconds:02d}"


# def get_gemini_model():
#     generation_config = {
#         "temperature": 0.3, 
#         "top_p": 0.8,
#         "top_k": 40,
#         "max_output_tokens": 8192,
#     }
#     return genai.GenerativeModel(
#         model_name='gemini-2.5-flash', 
#         generation_config=generation_config
#     )


# # --- ASYNC DEEPSEEK RETRY CHO OVERALL SUMMARY ---
# async def call_deepseek_retry_async(prompt: str, max_retries=5, base_delay=10):
#     """
#     Gọi DeepSeek API async với retry cho Overall Summary
#     """
#     if not _DEEPSEEK_CLIENT:
#         logger.warning("⚠️ DeepSeek client chưa được khởi tạo")
#         return ""
    
#     for attempt in range(max_retries):
#         try:
#             response = await _DEEPSEEK_CLIENT.chat.completions.create(
#                 model="deepseek-chat",
#                 messages=[
#                     {"role": "system", "content": "You are an expert educational content creator."},
#                     {"role": "user", "content": prompt}
#                 ],
#                 temperature=0.3,
#                 max_tokens=8192
#             )
            
#             if response.choices and response.choices[0].message.content:
#                 return response.choices[0].message.content.strip()
#             return ""
            
#         except Exception as e:
#             error_msg = str(e).lower()
#             if "429" in error_msg or "rate" in error_msg or "quota" in error_msg:
#                 wait_time = base_delay * (attempt + 1) + random.uniform(2, 5)
#                 logger.warning(f"⚠️ DeepSeek Rate Limit. Đang chờ {wait_time:.1f}s để thử lại lần {attempt+1}/{max_retries}...")
#                 await asyncio.sleep(wait_time)
#             else:
#                 logger.error(f"❌ DeepSeek Error: {e}")
#                 return ""
                
#     logger.error("❌ DeepSeek: Thất bại sau nhiều lần thử lại")
#     return ""


# # --- ASYNC GEMINI RETRY CHO TIMELINE SUMMARY ---
# async def call_gemini_retry_async(prompt: str, max_retries=5, base_delay=10):
#     """
#     Gọi Gemini async với retry cho Timeline Summary
#     """
#     if not GEMINI_API_KEY:
#         logger.warning("⚠️ Gemini API key không có")
#         return ""
    
#     model = get_gemini_model()
    
#     for attempt in range(max_retries):
#         try:
#             # Gemini's generate_content is sync, wrap in executor
#             loop = asyncio.get_event_loop()
#             response = await loop.run_in_executor(None, model.generate_content, prompt)
            
#             if response.parts:
#                 return response.text.strip()
#             return ""
            
#         except google_exceptions.ResourceExhausted as e:
#             wait_time = base_delay * (attempt + 1) + random.uniform(2, 5)
#             logger.warning(f"⚠️ Gemini Quota (429). Đang chờ {wait_time:.1f}s để thử lại lần {attempt+1}/{max_retries}...")
#             await asyncio.sleep(wait_time)
            
#         except Exception as e:
#             error_msg = str(e).lower()
#             if "429" in error_msg or "quota" in error_msg or "resource exhausted" in error_msg:
#                 wait_time = base_delay * (attempt + 1) + random.uniform(2, 5)
#                 logger.warning(f"⚠️ Gemini Quota (Generic). Đang chờ {wait_time:.1f}s để thử lại lần {attempt+1}/{max_retries}...")
#                 await asyncio.sleep(wait_time)
#             else:
#                 logger.error(f"❌ Gemini Error: {e}")
#                 return ""
                
#     logger.error("❌ Gemini: Thất bại sau nhiều lần thử lại")
#     return ""


# async def generate_overall_summary_async(transcript_text: str) -> str:
#     """
#     Sử dụng DeepSeek cho Overall Summary (async)
#     """
#     prompt = f"""
# You are an expert educational content creator. Read the video transcript below and create a HIGH-QUALITY, EASY-TO-UNDERSTAND executive summary for learners. 
# Requirements:
# 1. Use simple, clear, and engaging language suitable for beginners and non-experts.
# 2. Preserve all main points, key concepts, and learning objectives in logical order.
# 3. Include practical examples or real-world applications where possible.
# 4. Explain technical terms or jargon if used, to ensure comprehension.
# 5. Organize the summary into clear sections for readability.

# INPUT TRANSCRIPT:
# \"\"\"
# {transcript_text}
# \"\"\"

# OUTPUT FORMAT (Markdown):
# # Video Summary
# [A structured 3-5 paragraph summary, covering all main points and logical flow, easy to follow for learners.]

# ## Key Concepts
# - [List and briefly explain the most important concepts covered in the video]

# ## Practical Applications / Examples
# - [Include relevant examples, use cases, or applications that help learners understand how to apply the concepts]

# ## Key Takeaways
# - [3-5 concise and actionable points summarizing the core insights, lessons, or skills learners should retain]
# """
#     return await call_deepseek_retry_async(prompt)


# async def generate_timeline_summary_async(transcript_with_timestamps: str) -> str:
#     """
#     Sử dụng Gemini cho Timeline Summary (async)
#     """
#     prompt = f"""
# You are a video chapter generator. Group transcript segments into LOGICAL CHAPTERS.
# INPUT TRANSCRIPT (with timestamps):
# \"\"\"
# {transcript_with_timestamps}
# \"\"\"
# OUTPUT FORMAT (Markdown):
# ## Timeline & Topics
# **[Start] - [End] : [Topic Name]**
# > [Summary]
# """
#     return await call_gemini_retry_async(prompt)


# def update_lesson_success(lesson_id, series_id, transcript_url, summary_url="", timeline_url=""):
#     _, db = get_db()
#     update_data = {
#         "lesson_transcript": transcript_url,
#         "transcript_status": "completed",
#         "updatedAt": datetime.now(timezone.utc)
#     }
#     if summary_url: update_data["lesson_summary"] = summary_url
#     if timeline_url: update_data["lesson_timeline"] = timeline_url
    
#     db["lessons"].update_one(
#         {"_id": ObjectId(lesson_id), "lesson_serie": series_id},
#         {"$set": update_data}
#     )

# def update_lesson_error(lesson_id, series_id, error):
#     _, db = get_db()
#     db["lessons"].update_one(
#         {"_id": ObjectId(lesson_id), "lesson_serie": series_id},
#         {"$set": {"transcript_status": "failed", "transcript_error": error, "updatedAt": datetime.now(timezone.utc)}}
#     )

# async def process_job_async(job_data: dict) -> bool:
#     video_path = job_data['video_path']
#     user_id = job_data['user_id']
#     lesson_id = job_data['lesson_id']
#     series_id = job_data['series_id']
#     filename = job_data.get('filename', Path(video_path).name)
    
#     try:
#         logger.info(f"🎬 Processing lesson: {lesson_id}")
#         if not os.path.exists(video_path): 
#             raise FileNotFoundError(f"Video file not found")
        
#         # 1. Transcribe (sync operation)
#         model = get_whisper_model()
#         segments_generator, info = model.transcribe(video_path, beam_size=1, temperature=0, vad_filter=True)
        
#         full_text_clean = []
#         full_text_timed = []
#         for segment in segments_generator:
#             ts = format_timestamp(segment.start)
#             txt = segment.text.strip()
#             full_text_clean.append(txt)
#             full_text_timed.append(f"[{ts}] {txt}")
        
#         clean_str = ' '.join(full_text_clean)
#         timed_str = '\n'.join(full_text_timed)
        
#         logger.info(f"✅ Transcribed ({info.duration:.2f}s)")
        
#         # 2. Upload Transcript
#         transcript_url = upload_to_s3(
#             clean_str.encode('utf-8'), 
#             f"{Path(filename).stem}_transcript.txt", 
#             'text/plain', 
#             f"files/user-{user_id}/transcripts"
#         )
        
#         # 3. AI Generation (PARALLEL - Gọi song song)
#         summary_url = ""
#         timeline_url = ""
        
#         if len(clean_str) > 50:
#             logger.info("🤖 Generating Summary & Timeline in PARALLEL...")
            
#             # Chạy song song 2 API calls
#             summary_task = generate_overall_summary_async(clean_str)
#             timeline_task = generate_timeline_summary_async(timed_str)
            
#             # Đợi cả 2 hoàn thành
#             summary_content, timeline_content = await asyncio.gather(
#                 summary_task, 
#                 timeline_task,
#                 return_exceptions=True  # Không crash nếu 1 trong 2 lỗi
#             )
            
#             # Upload Summary nếu thành công
#             if isinstance(summary_content, str) and summary_content:
#                 summary_url = upload_to_s3(
#                     summary_content.encode('utf-8'), 
#                     f"{Path(filename).stem}_summary.txt", 
#                     'text/plain', 
#                     f"files/user-{user_id}/summaries"
#                 )
#                 logger.info("✅ [DeepSeek] Summary created")
#             elif isinstance(summary_content, Exception):
#                 logger.error(f"❌ [DeepSeek] Summary failed: {summary_content}")
            
#             # Upload Timeline nếu thành công
#             if isinstance(timeline_content, str) and timeline_content:
#                 timeline_url = upload_to_s3(
#                     timeline_content.encode('utf-8'), 
#                     f"{Path(filename).stem}_timeline.txt", 
#                     'text/plain', 
#                     f"files/user-{user_id}/summaries"
#                 )
#                 logger.info("✅ [Gemini] Timeline created")
#             elif isinstance(timeline_content, Exception):
#                 logger.error(f"❌ [Gemini] Timeline failed: {timeline_content}")

#         # 4. Update DB
#         update_lesson_success(lesson_id, series_id, transcript_url, summary_url, timeline_url)
#         logger.info(f"✅ Done! Summary: {'OK' if summary_url else 'Miss'}, Timeline: {'OK' if timeline_url else 'Miss'}")
#         return True
        
#     except Exception as e:
#         logger.error(f"❌ Job Failed: {e}")
#         update_lesson_error(lesson_id, series_id, str(e))
#         return False
#     finally:
#         if os.path.exists(video_path): 
#             os.unlink(video_path)


# def process_job(job_data: dict) -> bool:
#     """Sync wrapper for async process_job_async"""
#     return asyncio.run(process_job_async(job_data))


# def main():
#     logger.info("🚀 Worker Started (Async DeepSeek + Gemini Mode)")
#     init_deepseek()  # Khởi tạo DeepSeek async client
#     init_gemini()    # Khởi tạo Gemini
#     get_whisper_model()
    
#     while True:
#         try:
#             job = receive_transcript_job(wait_time=20)
#             if job:
#                 if process_job(job['body']): 
#                     delete_message(job['receipt_handle'])
#         except KeyboardInterrupt: 
#             break
#         except Exception as e: 
#             logger.error(f"Worker Error: {e}")

# if __name__ == '__main__':
#     main()