from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Request
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional, List
import uvicorn
from faster_whisper import WhisperModel
import torch
import os
from pathlib import Path
import time
import aiofiles
import asyncio
from datetime import datetime
import logging

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Faster-Whisper Transcription API",
    description="Production-ready AI transcription service",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Templates
templates = Jinja2Templates(directory="templates")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
UPLOAD_FOLDER = Path("uploads")
OUTPUT_FOLDER = Path("outputs")
UPLOAD_FOLDER.mkdir(exist_ok=True)
OUTPUT_FOLDER.mkdir(exist_ok=True)

MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB
ALLOWED_EXTENSIONS = {'.mp4', '.mp3', '.wav', '.avi', '.mov', '.webm', '.m4a', '.flac', '.ogg'}

# Global model
MODEL = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"

# Pydantic models
class TranscribeRequest(BaseModel):
    filename: str
    language: Optional[str] = None
    fast_mode: bool = True

class TranscribeResponse(BaseModel):
    success: bool
    text: str
    language: str
    language_probability: float
    segments: List[dict]
    processing_time: float
    video_duration: float
    speedup: float
    output_file: str

class HealthResponse(BaseModel):
    status: str
    device: str
    compute_type: str
    gpu_name: Optional[str]
    cuda_version: Optional[str]
    model_loaded: bool
    uptime_seconds: float

# Global state
START_TIME = time.time()

def load_model(model_size: str = "small") -> WhisperModel:
    """Load Faster-Whisper model (thread-safe)"""
    global MODEL
    if MODEL is None:
        logger.info(f"Loading Faster-Whisper model '{model_size}' on {DEVICE}")
        logger.info(f"Compute type: {COMPUTE_TYPE}")
        
        model_kwargs = {
            "device": DEVICE,
            "compute_type": COMPUTE_TYPE,
            "num_workers": 1,
        }
        
        if DEVICE == "cpu":
            model_kwargs["cpu_threads"] = 4
        
        MODEL = WhisperModel(model_size, **model_kwargs)
        logger.info("✅ Model loaded successfully!")
    return MODEL

async def transcribe_audio(
    video_path: str,
    language: Optional[str] = None,
    use_fast: bool = True
) -> dict:
    """Async transcription wrapper"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        _transcribe_sync,
        video_path,
        language,
        use_fast
    )

def _transcribe_sync(video_path: str, language: Optional[str], use_fast: bool) -> dict:
    """Synchronous transcription logic"""
    model = load_model("small")
    
    if use_fast:
        beam_size = 1
        best_of = 1
        vad_filter = True
        vad_parameters = {
            "threshold": 0.5,
            "min_speech_duration_ms": 250,
            "max_speech_duration_s": float('inf'),
            "min_silence_duration_ms": 100,
            "window_size_samples": 512,
        }
    else:
        beam_size = 5
        best_of = 5
        vad_filter = False
        vad_parameters = None
    
    logger.info(f"Transcribing: {os.path.basename(video_path)}")
    logger.info(f"Mode: {'Fast' if use_fast else 'Accurate'}, VAD: {vad_filter}")
    
    segments_generator, info = model.transcribe(
        video_path,
        language=language,
        beam_size=beam_size,
        best_of=best_of,
        temperature=0,
        condition_on_previous_text=False,
        vad_filter=vad_filter,
        vad_parameters=vad_parameters,
        no_speech_threshold=0.6,
        log_prob_threshold=-1.0,
        compression_ratio_threshold=2.4,
    )
    
    segments = []
    full_text = []
    
    for segment in segments_generator:
        segments.append({
            'start': round(segment.start, 2),
            'end': round(segment.end, 2),
            'text': segment.text.strip()
        })
        full_text.append(segment.text.strip())
    
    result = {
        'text': ' '.join(full_text),
        'language': info.language,
        'language_probability': round(info.language_probability, 4),
        'duration': round(info.duration, 2),
        'segments': segments
    }
    
    logger.info(f"✅ Completed! Found {len(segments)} segments")
    return result

async def cleanup_old_files():
    """Background task to cleanup old files"""
    while True:
        await asyncio.sleep(3600)  # Run every hour
        current_time = time.time()
        
        for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER]:
            for file in folder.glob("*"):
                if file.is_file():
                    file_age = current_time - file.stat().st_mtime
                    if file_age > 86400:  # 24 hours
                        file.unlink()
                        logger.info(f"Cleaned up old file: {file.name}")

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    logger.info("=" * 60)
    logger.info("🚀 FASTER-WHISPER TRANSCRIPTION API")
    logger.info("=" * 60)
    
    if torch.cuda.is_available():
        logger.info(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"   CUDA: {torch.version.cuda}")
        logger.info(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        logger.info(f"   Compute Type: {COMPUTE_TYPE}")
    else:
        logger.info("⚠️  No GPU detected, using CPU")
        logger.info(f"   Compute Type: {COMPUTE_TYPE}")
    
    # Pre-load model
    logger.info("🔥 Pre-loading model...")
    load_model("small")
    
    # Start cleanup task
    asyncio.create_task(cleanup_old_files())
    
    logger.info("=" * 60)
    logger.info("✅ API is ready!")
    logger.info("=" * 60)

@app.get("/", response_class=HTMLResponse, tags=["UI"])
async def root(request: Request):
    """Serve main UI"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api", tags=["API Info"])
async def api_root():
    """API root endpoint"""
    return {
        "message": "Faster-Whisper Transcription API",
        "ui": "/",
        "docs": "/api/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint for k8s probes"""
    return HealthResponse(
        status="healthy",
        device=DEVICE,
        compute_type=COMPUTE_TYPE,
        gpu_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        cuda_version=torch.version.cuda if torch.cuda.is_available() else None,
        model_loaded=MODEL is not None,
        uptime_seconds=round(time.time() - START_TIME, 2)
    )

@app.post("/upload", tags=["Transcription"])
async def upload_file(file: UploadFile = File(...)):
    """Upload audio/video file"""
    # Validate file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File type not allowed. Supported: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    # Generate unique filename
    timestamp = int(time.time())
    unique_filename = f"{timestamp}_{file.filename}"
    filepath = UPLOAD_FOLDER / unique_filename
    
    # Save file asynchronously
    try:
        async with aiofiles.open(filepath, 'wb') as out_file:
            content = await file.read()
            
            # Check file size
            if len(content) > MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=400,
                    detail=f"File too large. Max size: {MAX_FILE_SIZE / 1024 / 1024}MB"
                )
            
            await out_file.write(content)
        
        logger.info(f"Uploaded: {unique_filename} ({len(content) / 1024 / 1024:.2f}MB)")
        
        return {
            "success": True,
            "filename": unique_filename,
            "message": "File uploaded successfully"
        }
    
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/transcribe", response_model=TranscribeResponse, tags=["Transcription"])
async def transcribe(request: TranscribeRequest):
    """Transcribe uploaded audio/video file"""
    filepath = UPLOAD_FOLDER / request.filename
    
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    try:
        start_time = time.time()
        
        # Transcribe (runs in thread pool to not block event loop)
        result = await transcribe_audio(
            str(filepath),
            language=request.language,
            use_fast=request.fast_mode
        )
        
        elapsed_time = time.time() - start_time
        
        # Save transcript
        output_filename = f"{Path(request.filename).stem}_transcript.txt"
        output_path = OUTPUT_FOLDER / output_filename
        
        async with aiofiles.open(output_path, 'w', encoding='utf-8') as f:
            await f.write(result['text'])
        
        video_duration = result['duration']
        speedup = video_duration / elapsed_time if elapsed_time > 0 else 0
        
        return TranscribeResponse(
            success=True,
            text=result['text'],
            language=result['language'],
            language_probability=result['language_probability'],
            segments=result['segments'],
            processing_time=round(elapsed_time, 2),
            video_duration=video_duration,
            speedup=round(speedup, 2),
            output_file=output_filename
        )
    
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download/{filename}", tags=["Files"])
async def download_file(filename: str):
    """Download transcript file"""
    filepath = OUTPUT_FOLDER / filename
    
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        filepath,
        media_type="text/plain",
        filename=filename
    )

@app.get("/video/{filename}", tags=["Files"])
async def serve_video(filename: str):
    """Serve video file"""
    filepath = UPLOAD_FOLDER / filename
    
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    
    return FileResponse(filepath)

@app.delete("/cleanup", tags=["Admin"])
async def cleanup_files(background_tasks: BackgroundTasks):
    """Manually trigger file cleanup"""
    async def cleanup():
        count = 0
        for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER]:
            for file in folder.glob("*"):
                if file.is_file():
                    file.unlink()
                    count += 1
        logger.info(f"Cleaned up {count} files")
    
    background_tasks.add_task(cleanup)
    return {"message": "Cleanup initiated"}

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Set to False for production
        log_level="info",
        access_log=True
    )