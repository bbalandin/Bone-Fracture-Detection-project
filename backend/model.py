# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score
# from pickle import dump, load
# import pandas as pd
import cv2
import numpy as np
import os
import shutil
from ultralytics import YOLO


def apply_clahe_lab(image, clip_limit=3.0, tile_grid_size=(8, 8)):
    """Применяет CLAHE через LAB цветовое пространство"""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    img_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    return img_clahe


def detect_fracture(image, model_type, is_obb=True):  # добавить для obb
    # TODO вынести все пути в константы
    # MODEL_TYPE_FILE = os.path.normpath(MODEL_TYPE_FILE)
    if model_type == "точная":
        MODEL_PATH = "../data/best (4).pt"  # TODO заменить модель
    else:
        MODEL_PATH = "../data/best (5).pt"  # TODO заменить модель
    model = YOLO(MODEL_PATH)
    bbox_color = (204, 204, 0)  # цвет нашего bounding box'
    # инференс на данном изображении
    results = model(image, conf=0.5, iou=0.4)  # list of Results objects
    # TODO пока not is_obb работает некорректно
    if not is_obb:
        for result in results:
            boxes = result.boxes  # Bounding boxes
            if boxes:
                for box in boxes:  # переделать на oriented boundin box'ы, так как у них качество лучше
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    # 2. Draw the rectangle
                    cv2.rectangle(image, (x1, y1), (x2, y2), bbox_color,
                                  thickness=2)
        # Добавить здесь функционал с нахождением перелома
    else:
        fractures = []
        for result in results:  # используем формат obb для отрисовки bounding box'ов
            obbs = result.obb
            if obbs is not None and len(obbs) > 0:
                for obb in obbs:
                    fractures.append((obb.conf.item(), obb.cls.item()))
                    # получаем 4 координаты
                    points = obb.xyxyxyxy[0].numpy().astype(np.int64)
                    # отрисовываем bounding box
                    cv2.polylines(image, [points], isClosed=True,
                                  color=bbox_color, thickness=2)
    return image, fractures

# def using_sahi(image):
    # in progress
# def split_data(df: pd.DataFrame):
#     y = df['Survived']
#     X = df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Embarked"]]

#     return X, y


# def open_data(path="data/titanic_dataset_train.csv"):
#     df = pd.read_csv(path)
#     df = df[['Survived', "Pclass", "Sex", "Age", "SibSp", "Parch", "Embarked"]]

#     return df


# def preprocess_data(df: pd.DataFrame, test=True):
#     df.dropna(inplace=True)

#     if test:
#         X_df, y_df = split_data(df)
#     else:
#         X_df = df

#     to_encode = ['Sex', 'Embarked']
#     for col in to_encode:
#         dummy = pd.get_dummies(X_df[col], prefix=col)
#         X_df = pd.concat([X_df, dummy], axis=1)
#         X_df.drop(col, axis=1, inplace=True)

#     if test:
#         return X_df, y_df
#     else:
#         return X_df


# def fit_and_save_model(X_df, y_df, path="data/model_weights.mw"):
#     model = RandomForestClassifier()
#     model.fit(X_df, y_df)

#     test_prediction = model.predict(X_df)
#     accuracy = accuracy_score(test_prediction, y_df)
#     print(f"Model accuracy is {accuracy}")

#     with open(path, "wb") as file:
#         dump(model, file)

#     print(f"Model was saved to {path}")


# def load_model_and_predict(df, path="data/model_weights.mw"):
#     with open(path, "rb") as file:
#         model = load(file)

#     prediction = model.predict(df)[0]
#     # prediction = np.squeeze(prediction)

#     prediction_proba = model.predict_proba(df)[0]
#     # prediction_proba = np.squeeze(prediction_proba)

#     encode_prediction_proba = {
#         0: "Вам не повезло с вероятностью",
#         1: "Вы выживете с вероятностью"
#     }

#     encode_prediction = {
#         0: "Сожалеем, вам не повезло",
#         1: "Ура! Вы будете жить"
#     }

#     prediction_data = {}
#     for key, value in encode_prediction_proba.items():
#         prediction_data.update({value: prediction_proba[key]})

#     prediction_df = pd.DataFrame(prediction_data, index=[0])
#     prediction = encode_prediction[prediction]

#     return prediction, prediction_df


# if __name__ == "__main__":
#     df = open_data()
#     X_df, y_df = preprocess_data(df)
#     fit_and_save_model(X_df, y_df)