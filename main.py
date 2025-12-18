from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import router
import logging
from dotenv import load_dotenv
import os
load_dotenv() 

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)

logger = logging.getLogger("media-service")
logger.info("Starting Media Storage Service...")

app = FastAPI(
    title="Media Storage Service",
    description="File upload and management service",
    version="1.0.0"
)

# CORS middleware - configure from environment
cors_origins = os.getenv('CORS_ORIGINS', 'http://localhost:5173').split(',')
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include router
app.include_router(router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)
    
# uvicorn main:app --reload --host 0.0.0.0 --port 8001 --log-level info
    
