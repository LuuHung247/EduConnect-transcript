from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from typing import List, Optional
from pydantic import BaseModel
from app.services.storage_service import StorageService

router = APIRouter()
storage_service = StorageService()


class DeleteRequest(BaseModel):
    url_or_key: str


class DeleteBatchRequest(BaseModel):
    urls: List[str]


@router.get('/health')
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "media-storage-service",
        "version": "1.0.0"
    }


@router.post('/api/upload/thumbnail')
async def upload_thumbnail(
    file: UploadFile = File(...),
    user_id: str = Form(...)
):
    """Upload thumbnail image"""
    try:
        result = await storage_service.upload_thumbnail(file, user_id)
        
        return {
            "success": True,
            "url": result['url'],
            "key": result['key']
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post('/api/upload/video')
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    user_id: str = Form(...),
    create_transcript: bool = Form(True)
):
    """
    Upload video file with optional transcript generation
    
    Args:
        file: Video file
        user_id: User ID
        create_transcript: Generate transcript in background (default: True)
    
    Returns:
        {
            "success": true,
            "url": "s3://...",
            "key": "files/user-123/videos/...",
            "transcript_status": "processing" | "disabled"
        }
    """
    try:
        result = await storage_service.upload_video(
            file=file,
            user_id=user_id,
            background_tasks=background_tasks,
            create_transcript=create_transcript
        )
        
        return {
            "success": True,
            "url": result['url'],
            "key": result['key'],
            "transcript_status": result.get('transcript_status', 'disabled')
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post('/api/upload/document')
async def upload_document(
    file: UploadFile = File(...),
    user_id: str = Form(...)
):
    """Upload document file"""
    try:
        result = await storage_service.upload_document(file, user_id)
        
        return {
            "success": True,
            "url": result['url'],
            "key": result['key']
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post('/api/upload/documents/batch')
async def upload_documents_batch(
    files: List[UploadFile] = File(...),
    user_id: str = Form(...)
):
    """Upload multiple documents"""
    try:
        results = await storage_service.upload_documents_batch(files, user_id)
        
        return {
            "success": True,
            "urls": [r['url'] for r in results if r.get('url')],
            "results": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete('/api/delete')
async def delete_file(request: DeleteRequest):
    """Delete a file from S3"""
    try:
        storage_service.delete_file(request.url_or_key)
        
        return {
            "success": True,
            "deleted": request.url_or_key
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete('/api/delete/batch')
async def delete_files_batch(request: DeleteBatchRequest):
    """Delete multiple files"""
    try:
        result = storage_service.delete_files_batch(request.urls)
        
        return {
            "success": True,
            "deleted": result['deleted'],
            "failed": result['failed']
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))