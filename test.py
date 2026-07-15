import streamlit as st
from PIL import Image
import io
import numpy as np
import os
import sys

# Импортируем функции из backend
from backend.model import apply_clahe_lab, detect_fracture

fracture_classes = ['elbow positive', 'fingers positive', 'forearm fracture', 
                    'humerus fracture', 'humerus', 'shoulder fracture', 'wrist positive']


def load_css(file_path):
    """Загрузка CSS из файла"""
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ""


def inject_custom_css():
    """Внедрение CSS"""
    css_files = [
        'frontend/styles/main.css',
        'frontend/styles/sidebar.css',
        'frontend/styles/components.css'
    ]
    
    css_combined = ""
    for css_file in css_files:
        css_combined += load_css(css_file) + "\n"
    
    if css_combined.strip():
        st.markdown(f"<style>{css_combined}</style>", unsafe_allow_html=True)


def process_image_directly(uploaded_file):
    """
    Прямая обработка изображения через PIL (без cv2)
    """
    # 1. Получаем байты файла
    file_bytes = uploaded_file.getvalue()
    
    # 2. Открываем через PIL (вместо cv2.imdecode)
    pil_image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    
    # 3. Конвертируем RGB → BGR (так как backend ожидает формат OpenCV)
    image_bgr = np.array(pil_image)[:, :, ::-1]
    
    # 4. Применяем CLAHE и детектируем
    clahe_image = apply_clahe_lab(image_bgr)
    detected_image_bgr, fractures = detect_fracture(clahe_image)
    
    # 5. Конвертируем результат BGR → RGB для отображения (вместо cv2.imencode)
    detected_image_rgb = detected_image_bgr[:, :, ::-1]
    result_image = Image.fromarray(detected_image_rgb)
    
    return result_image, fractures


def uploading_detecting():
    st.markdown("### Загрузка рентгеновского снимка")

    uploaded_file = st.file_uploader(
        "Перетащите файл сюда или нажмите для выбора",
        type=["jpg", "jpeg", "png"],
        help="Поддерживаемые форматы: JPG, JPEG, PNG • Макс. размер: 200MB"
    )

    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Оригинал")
            st.image(uploaded_file, caption="Загруженный снимок", use_container_width=True)

        if st.button("Запустить анализ", use_container_width=True):
            with st.spinner("Анализ снимка..."):
                result_image, fractures = process_image_directly(uploaded_file)
            
            with col2:
                st.markdown("#### Результат детекции")
                st.image(result_image, caption="С обнаруженными переломами", use_container_width=True)

            if not fractures:
                st.success("Поздравляем! На Вашем снимке нет перелома!")
            else:
                st.warning(f"Обнаружено переломов: {len(fractures)}")
                st.markdown("### Детальная информация")

                for i, fracture in enumerate(fractures):
                    confidence = fracture[0]
                    class_name = fracture_classes[int(fracture[1])]

                    st.markdown(f"""
                    <div class='fracture-result'>
                        <div style='display: flex; justify-content: space-between; align-items: center;'>
                            <div>
                                <strong style='color: #14b8a6; font-size: 16px;'>Перелом #{i + 1}</strong><br>
                                <span style='color: #94a3b8; font-size: 14px;'>Класс: {class_name}</span>
                            </div>
                            <div style='text-align: right;'>
                                <div style='color: #fbbf24; font-size: 20px; font-weight: 700;'>{confidence:.1%}</div>
                                <div style='color: #64748b; font-size: 12px;'>Уверенность</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)


def navigation_menu():
    try:
        image = Image.open("data/Пример.jpg")
    except:
        image = "🦴"

    st.set_page_config(
        layout="wide",
        initial_sidebar_state="auto",
        page_title="Bone Fracture Detection",
        page_icon=image if image != "🦴" else "🦴",
    )
    
    inject_custom_css()

    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'Главная'

    with st.sidebar:
        st.markdown("""
        <div style='text-align: center; padding: 20px 0;'>
            <h2 style='color: #14b8a6; margin: 0;'>Fracture Detection</h2>
            <p style='color: #64748b; font-size: 12px; margin-top: 8px;'>X-Ray Analysis</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### Навигация")

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Главная"):
                st.session_state.current_page = 'Главная'
        with col2:
            if st.button("Информация"):
                st.session_state.current_page = 'Информация'
        with col3:
            if st.button("Примеры"):
                st.session_state.current_page = 'Примеры'

    if st.session_state.current_page == 'Главная':
        st.markdown("""
        <div class='medical-header'>
            <h1>Детекция переломов</h1>
            <p style='color: #94a3b8; font-size: 16px; margin-top: 12px;'>
                Система для анализа рентгеновских снимков на основе YOLO
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        uploading_detecting()
        
    elif st.session_state.current_page == 'Информация':
        st.markdown("### О модели")
        st.info("Модель: YOLO11 с CLAHE предобработкой")
        st.write(f"Классы детекции: {len(fracture_classes)}")
        
    elif st.session_state.current_page == 'Примеры':
        st.markdown("### Примеры переломов")
        st.info("Здесь будут примеры")


if __name__ == "__main__":
    navigation_menu()