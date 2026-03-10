from fastapi import APIRouter, File, HTTPException, UploadFile

from inference.helpers.logger import logger
from inference.helpers.predict import predict_image_bytes


router = APIRouter()


@router.post("/")
async def predict(file: UploadFile = File(...)):
    logger.info(
        f"Received prediction request | filename={file.filename} | content_type={file.content_type}"
    )

    if not file.content_type or not file.content_type.startswith("image/"):
        logger.warning(
            f"Rejected non-image upload | filename={file.filename} | content_type={file.content_type}"
        )
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    try:
        image_bytes = await file.read()
        logger.info(f"Read uploaded file successfully | size_bytes={len(image_bytes)}")

        result = predict_image_bytes(image_bytes)

        logger.info(
            f"Prediction completed | filename={file.filename} | "
            f"predicted_label={result['predicted_label']} | confidence={result['confidence']:.4f}"
        )

        return {
            "filename": file.filename,
            "content_type": file.content_type,
            "result": result,
        }

    except Exception as e:
        logger.exception(f"Inference failed for file={file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")