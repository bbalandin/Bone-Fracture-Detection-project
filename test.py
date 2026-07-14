import pandas as pd
import streamlit as st
from PIL import Image
import requests
import io
import base64
from requests_toolbelt import MultipartEncoder, MultipartEncoderMonitor
import time
# from model import open_data, preprocess_data, split_data, load_model_and_predict


# TODO передалать структуру
PATH_TO_MODEL = "data\\best (4).pt"

fracture_classes = ['elbow positive', 'fingers positive', 'forearm fracture', 'humerus fracture', 'humerus', 'shoulder fracture', 'wrist positive']

def navigation_menu():
    image = Image.open('data/Пример.jpg')  # для примера

    st.set_page_config(
        layout="wide",
        initial_sidebar_state="auto",
        page_title="Bone Fracture Detection",
        page_icon=image,
    )

    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'Главная'

    with st.sidebar:
        st.title("Навигация")
        if st.button("🏠 Главная"):
            st.session_state.current_page = 'Главная'
        if st.button("📋 Подробная информация"):
            st.session_state.current_page = 'Подробная информация'
        if st.button("🖼️ Примеры переломов"):
            st.session_state.current_page = 'Примеры переломов'


    # Отображение контента в зависимости от выбранной страницы
    if st.session_state.current_page == 'Главная':
        show_main_page()
    elif st.session_state.current_page == 'Подробная информация':
        show_info_page()
    elif st.session_state.current_page == 'Примеры переломов':
        show_examples_page()


def uploading_detecting():
    uploaded_file = st.file_uploader("Выберите картинку...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Показываем оригинал
        col1, col2 = st.columns(2)
        with col1:
            st.image(uploaded_file, caption="Оригинал", use_container_width=True)

        if st.button("Обработать"):
            # Отправляем файл в FastAPI
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            progress_bar = st.progress(0)
            status_text = st.empty()
            response = requests.post("http://localhost:8000/uploadfile/", files=files)

            def create_monitor(encoder):
                def callback(monitor):
                    progress = (monitor.bytes_read / monitor.len) * 100
                    progress_bar.progress(progress / 100)
                    status_text.text(f"Загрузка: {progress:.1f}%")
                return callback # исправить
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            encoder = MultipartEncoder(fields=files)
            monitor = MultipartEncoderMonitor(encoder, create_monitor(encoder))
            response = requests.post(
                "http://localhost:8000/uploadfile/",
                data=monitor,
                headers={"Content-Type": monitor.content_type}
            )   # TODO добавить выбор модели
            data_response = response.json()
            if response.status_code == 200:
                # Читаем байты ответа как изображение
                img_bytes = base64.b64decode(data_response['image'])
                result_image = Image.open(io.BytesIO(img_bytes))
                with col2:
                    st.image(result_image, caption="После обработки", use_container_width=True)
                fractures = data_response['fractures']
                if not fractures:
                    st.success("Поздравляем! На Вашем снимке нет перелома!")
                else:
                    st.warning("Наша модель считает, что на Вашем снимке есть переломы, вот более подробная информация")
                    for i, fracture in enumerate(fractures):
                        print(i, fracture)
                        st.markdown(f"#### Вероятность наличия перелома №{i + 1} -"
                                    f"{fracture[0]:.2f}, это перелом класса {fracture_classes[int(fracture[1])]}")
            else:
                st.error(f"Ошибка: {response.status_code} — {response.text}")
        # Проверяем успешность ответа
        # if response.status_code != 200:
        #     st.error("Ошибка при обработке изображения на сервере.")
        #     result = response.json()
        #     st.success(result["message"])
        #     st.json(result)
        # else:


def show_main_page():
    st.write(
        """
        # Детекция переломов
        Загрузите сюда любой свой рентген снимок и проверьте, есть ли на нём переломы,
        наша модель обучена так, чтобы замечать даже незаметные на первый взгляд трещины.
        После загрузки ваша рентген будет преобразован при помоши CLAHE, чтобы модель могла
        лучше различать все детали, заитригованы, попробуйте!
        """
    )
    genre = st.radio(
        "Выберите модель",
        ["***Быстрая модель***", "***Медленная модель***"]
    )
    if genre == "***Быстрая модель***":
        st.write("Отличный выбор, но у быстрой модель ниже точность")
    else:

        st.write("Прекрасный выбор, модель медленная, но у неё высокая точность")
    uploading_detecting()
    # st.image(image)
    # st.success("Фотография успешно обработана")


def show_info_page():
    st.write("# Подробная информация")
    response = requests.get("http://localhost:8000/info")
    if response.status_code == 200:
        data = response.json()
    else:
        st.write("Произошла ошибка")


def show_examples_page():
    st.write("Примеры детектированных переломов")
    response = requests.get("http://localhost:8000/fracture_examples")
    if response.status_code == 200:
        data = response.json()
    else:
        st.write("Произошла ошибка")


def write_user_data(df):
    st.write("## Ваши данные")
    st.write(df)


if __name__ == "__main__":
    hide_menu_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibily: hidden;}
    </style>
    """  # убираем встроенное меню streamlit
    st.markdown(hide_menu_style, unsafe_allow_html=True)
    navigation_menu()
