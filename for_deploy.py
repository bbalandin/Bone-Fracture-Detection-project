import pandas as pd
import streamlit as st
from PIL import Image
import requests
import io
import numpy as np
import os

# TODO передалать структуру
# TODO отделить текст markdown от основного кода
from backend.model import apply_clahe_lab, detect_fracture

fracture_classes = ['elbow positive', 'fingers positive', 'forearm fracture', 'humerus fracture', 'humerus', 'shoulder fracture', 'wrist positive']
MODEL_TYPE = 'быстрая'


def load_css(file_path):
    """Загрузка CSS из файла"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def inject_custom_css():
    """Внедрение всех CSS файлов"""
    css_files = [
        'frontend/styles/main.css',
        'frontend/styles/sidebar.css',
        'frontend/styles/components.css'
    ]

    css_combined = ""
    for css_file in css_files:
        if os.path.exists(css_file):
            css_combined += load_css(css_file) + "\n"
    st.markdown(f"<style>{css_combined}</style>", unsafe_allow_html=True)


def navigation_menu():
    image = Image.open("data\\Пример.jpg")  # для примера

    st.set_page_config(
        layout="wide",
        initial_sidebar_state="auto",
        page_title="Bone Fracture Detection",
        page_icon=image,
    )
    # Внедрение CSS
    inject_custom_css()

    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'Главная'

    with st.sidebar:
        # Логотип и название
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
            if st.button("Главная страница"):
                st.session_state.current_page = 'Главная'
        with col2:
            if st.button("Подробная информация"):
                st.session_state.current_page = 'Подробная информация'
        with col3:
            if st.button("Примеры переломов"):
                st.session_state.current_page = 'Примеры переломов'

        st.markdown("---")

        # Статистика
        # st.markdown("### Статистика")
        # st.markdown("""
        # <div class='info-card'>
        #     <div style='color: #94a3b8; font-size: 13px;'>
        #         Обработано снимков: <strong style='color: #14b8a6;'>1,247</strong><br>
        #         Точность модели: <strong style='color: #14b8a6;'>94%</strong><br>
        #         Среднее время: <strong style='color: #14b8a6;'>2.3s</strong>
        #     </div>
        # </div>
        # """, unsafe_allow_html=True)
        # TODO доделать статистику

    # Отображение контента в зависимости от выбранной страницы
    if st.session_state.current_page == 'Главная':
        show_main_page()
    elif st.session_state.current_page == 'Подробная информация':
        show_info_page()
    elif st.session_state.current_page == 'Примеры переломов':
        show_examples_page()


def uploading_detecting():
    st.markdown("### Загрузка рентгеновского снимка")

    uploaded_file = st.file_uploader(
        "Перетащите файл сюда или нажмите для выбора",
        type=["jpg", "jpeg", "png"],
        help="Поддерживаемые форматы: JPG, JPEG, PNG • Макс. размер: 200MB"
    )

    if uploaded_file is not None:
        # Показываем оригинал
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Оригинал")
            st.image(uploaded_file, caption="Загруженный снимок", use_container_width=True)

        if st.button("Запустить анализ", use_container_width=True):
            with st.spinner("Анализ снимка..."):
                file_bytes = uploaded_file.getvalue()
                pil_image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
                image_bgr = np.array(pil_image)[:, :, ::-1]
                clahe_image = apply_clahe_lab(image_bgr)
                detected_image_bgr, fractures = detect_fracture(clahe_image, MODEL_TYPE)
                detected_image_rgb = detected_image_bgr[:, :, ::-1]
                result_image = Image.fromarray(detected_image_rgb)
            
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


def show_main_page():
    # Заголовок с медицинским акцентом
    st.markdown("""
    <div class='medical-header'>
        <h1>Детекция переломов</h1>
        <p style='color: #94a3b8; font-size: 16px; margin-top: 12px;'>
            Система для анализа рентгеновских снимков на основе YOLO
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='info-card'>
        <p style='color: #cbd5e1; font-size: 15px; line-height: 1.7; margin: 0;'>
            Загрузите рентгеновский снимок для анализа на наличие переломов.
            Наша модель обучена обнаруживать даже незаметные на первый взгляд трещины.
            Перед анализом изображение проходит предварительную обработку с помощью CLAHE
            для улучшения контрастности и детализации.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Выбор модели - улучшенный UI
    st.markdown("### Выберите модель анализа")

    # Инициализация состояния

    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = "быстрая"

    st.markdown("""
        <style>
            /* Увеличиваем кнопки до размера карточек */
            button[key="fast_btn"], button[key="slow_btn"] {
                min-height: 600px !important;
                height: 600px !important;
                padding: 0 !important;
                margin: 0 !important;
            }
        </style>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        is_selected = st.session_state.selected_model == "быстрая"
        border_color = "#14b8a6" if is_selected else "#334155"
        bg_color = "#1a3a3a" if is_selected else "#1e293b"

        # Невидимая кнопка
        if st.button("", key="fast_btn", use_container_width=True):
            st.session_state.selected_model = "быстрая"
            st.rerun()
        # Карточка с отрицательным margin поверх кнопки
        st.markdown(f"""
        <div style='
            background: {bg_color};
            border: 3px solid {border_color};
            border-radius: 12px;
            padding: 20px;
            margin-top: -48px;
            pointer-events: none;
            min-height: 100px;
        '>
            <div style='font-weight: 600; font-size: 18px; color: #e2e8f0; margin-bottom: 8px;'>
                Быстрая модель (YOLO11n-obb с CLAHE)
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        is_selected = st.session_state.selected_model == "точная"
        border_color = "#14b8a6" if is_selected else "#334155"
        bg_color = "#1a3a3a" if is_selected else "#1e293b"

        # Невидимая кнопка
        if st.button("", key="slow_btn", use_container_width=True):
            st.session_state.selected_model = "точная"
            st.rerun()

        # Карточка с отрицательным margin поверх кнопки
        st.markdown(f"""
        <div style='
            background: {bg_color};
            border: 3px solid {border_color};
            border-radius: 12px;
            padding: 20px;
            margin-top: -48px;
            pointer-events: none;
            min-height: 100px;
        '>
            <div style='font-weight: 600; font-size: 18px; color: #e2e8f0; margin-bottom: 8px;'>
                Точная модель - (YOLO11m-obb с CLAHE)
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.write(f"Выбрана: {st.session_state.selected_model} модель")
    st.markdown("---")
    global MODEL_TYPE
    MODEL_TYPE = st.session_state.selected_model
    uploading_detecting()


def show_info_page():
    st.markdown("""
    <div class='medical-header'>
        <h1>Подробная информация</h1>
        <p style='color: #94a3b8; font-size: 16px; margin-top: 12px;'>
            Техническая документация и описание системы
        </p>
    </div>
    """, unsafe_allow_html=True)

    try:
        response = requests.get("http://localhost:8000/info")
        if response.status_code == 200:
            data = response.json()

            st.markdown("### О модели")
            st.markdown(f"""
            <div class='info-card'>
                <p style='color: #cbd5e1; margin: 0;'>
                    <strong>Тип модели:</strong> {data.get('model_type', 'YOLOv8')}<br>
                    <strong>Обучающая выборка:</strong> {data.get('dataset_size', 'N/A')} изображений<br>
                    <strong>Классы детекции:</strong> {len(fracture_classes)} типов переломов
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("### Классы переломов")
            for i, class_name in enumerate(fracture_classes, 1):
                st.markdown(f"""
                <div class='fracture-result'>
                    <strong style='color: #14b8a6;'>{i}.</strong> {class_name}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.error("Не удалось получить информацию с сервера")
    except Exception as e:
        st.error(f"Ошибка подключения к серверу: {str(e)}")


def show_examples_page():
    st.markdown("""
    <div class='medical-header'>
        <h1>Примеры детекции переломов на train выборке</h1>
        <p style='color: #94a3b8; font-size: 16px; margin-top: 12px;'>
            Выберите класс переломов
        </p>
    </div>
    """, unsafe_allow_html=True)
    # genre = st.radio(
    #     "Классы переломов",
    #     [classes for classes in fracture_classes]
    # )
    # if genre == "***elbow positive***":
    #     st.write("Отличный выбор, перелом локтевого сустава")
    # else:

    #     st.write("Прекрасный выбор, модель медленная, но у неё высокая точность")
    type_of_fracture = st.selectbox(
        "Классы переломов",
        fracture_classes
    )
    # TODO придумать как красиво отображать кнопки
    # BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    dir_gt = f"{type_of_fracture}_gt"
    dir_pred = f"{type_of_fracture}_pred"
    DIR_EXAMPLES_GT = f"data\Examples\{dir_gt}"
    DIR_EXAMPLES_PRED = f"data\Examples\{dir_pred}"
    col1, col2 = st.columns(2)
    for i, file in enumerate(os.listdir(DIR_EXAMPLES_GT)):
        # image_gt = Image.open(os.path.join(DIR_EXAMPLES_GT, file))
        image_pred = Image.open(os.path.join(DIR_EXAMPLES_PRED, file))
        image_gt = Image.open(os.path.join(DIR_EXAMPLES_GT, file))
        with col1:
            st.image(image_pred, caption="Предсказанный перелом", use_container_width=True)
        with col2:
            st.image(image_gt, caption="Реальный перелом(выделен при разметке)",
                      use_container_width=True)


if __name__ == "__main__":
    navigation_menu()
