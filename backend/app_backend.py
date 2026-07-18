import sys
import os
import io
import numpy as np
import cv2
import base64

# Добавляем родительскую директорию в путь поиска модулей
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Annotated
from model import apply_clahe_lab, detect_fracture
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response, JSONResponse

app = FastAPI()


@app.post("/files/")
async def create_file(file: Annotated[bytes, File()]):
    return {"file_size": len(file)}


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    image_array = np.frombuffer(
        file.file.read(), np.uint8
    )  # получаем numpy array байтов
    image = cv2.imdecode(
        image_array, cv2.IMREAD_COLOR
    )  # переводим из байтового формата в BGR
    clahe_image = apply_clahe_lab(image)
    detected_image, fractures = detect_fracture(clahe_image)
    success, buffer = cv2.imencode(".jpg", detected_image)
    image_for_json = base64.b64encode(buffer.tobytes()).decode("utf-8")
    return JSONResponse(
        status_code=200, content={"image": image_for_json, "fractures": fractures}
    )
