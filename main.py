from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import router
import logging


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

# CORS middleware (điều chỉnh origins theo nhu cầu)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Thay đổi trong production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include router
app.include_router(router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
    
# uvicorn main:app --reload --host 0.0.0.0 --port 8001 --log-level info

