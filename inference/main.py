from contextlib import asynccontextmanager
from fastapi import FastAPI
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from inference.routes import predict
from inference.helpers.logger import logger
from inference.helpers.onnx_session import get_onnx_session


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting SpoofFormer ONNX Inference API...")

    try:
        session = get_onnx_session()
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        logger.info("ONNX model loaded successfully.")
        logger.info(f"ONNX input name: {input_name}")
        logger.info(f"ONNX output name: {output_name}")

    except Exception as e:
        logger.exception(f"Failed to load ONNX model during startup: {e}")
        raise

    yield

    logger.info("Shutting down SpoofFormer ONNX Inference API...")






app = FastAPI(
    title="Valify Task Anti-Spoof Detection",
    lifespan=lifespan,
    description="AI Tool that Classify an image weather it's Live or Spoof",
    version="1.0.0",
    contact={
        "name":"Ayman M. Ibrahim",
        "email":"ayman.m.ibrahim.1994@gmail.com",
    }
)



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_app():
    pass

@app.on_event("shutdown")
async def shutdown_app():
    pass

app.include_router(predict.router, prefix="/predict", tags=["Predict Image Weather it's Live or Spoof"])



if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)