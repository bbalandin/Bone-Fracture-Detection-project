import cv2
import numpy as np
import gdown
import os
import shutil
import streamlit as st
from ultralytics import YOLO
import sys
from pathlib import Path

# Добавляем корневую директорию проекта в sys.path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))
from file_paths import MODEL_CACHE_DIR, FILE_ID_ACCURATE, FILE_ID_FAST


def apply_clahe_lab(image, clip_limit=3.0, tile_grid_size=(8, 8)):
    """Применяет CLAHE через LAB цветовое пространство"""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    img_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    return img_clahe


def load_model(model_type):
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
    if model_type == "точная":
        model_path = os.path.join(MODEL_CACHE_DIR, "accurate.pt")
        file_id = FILE_ID_ACCURATE
        # путь к точной модели на google drive
    else:
        model_path = os.path.join(MODEL_CACHE_DIR, "fast.pt")
        file_id = FILE_ID_FAST
        # путь к быстрой модели на google drive
    if not os.path.exists(model_path):
        st.info(f"В данный момент модель скачивается, это займёт некоторое время")
        download_url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(download_url, model_path, quiet=False)
        st.success(f"Модель скачана!")
    else:
        st.success(f"Модель скачана!")
    return YOLO(model_path)


def detect_fracture(image, model_type, is_obb=True):  # добавить для obb
    model = load_model(model_type)
    bbox_color = (204, 204, 0)  # цвет нашего bounding box'
    # инференс на данном изображении
    results = model(image, conf=0.5, iou=0.4)  # list of Results objects
    # TODO пока not is_obb работает некорректно(это задел на будущее, если захочется использовать модель
    # в обычном формате YOLO, а не obb,
    # но так как на данный момент у нас обе модели работают с obb, этот код закомментирован)
    # if not is_obb:
    #     for result in results:
    #         boxes = result.boxes  # Bounding boxes
    #         if boxes:
    #             for box in boxes:  # переделать на oriented boundin box'ы, так как у них качество лучше
    #                 x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
    #                 # 2. Draw the rectangle
    #                 cv2.rectangle(image, (x1, y1), (x2, y2), bbox_color,
    #                               thickness=2)
    # Добавить здесь функционал с нахождением перелома
    if is_obb:
        fractures = []
        for result in results:  # используем формат obb для отрисовки bounding box'ов
            obbs = result.obb
            if obbs is not None and len(obbs) > 0:
                for obb in obbs:
                    fractures.append((obb.conf.item(), obb.cls.item()))
                    # получаем 4 координаты
                    points = obb.xyxyxyxy[0].numpy().astype(np.int64)
                    # отрисовываем bounding box
                    cv2.polylines(
                        image, [points], isClosed=True, color=bbox_color, thickness=2
                    )
    return image, fractures
