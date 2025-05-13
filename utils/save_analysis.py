import os
import pandas as pd

def save_analysis_to_csv(
    id, age, sex, fov,
    image_paths,
    predicted_diameters,
    predicted_positions,
    modified_diameters,
    modified_positions,
    csv_path="data/analysis.csv"
):
    # Проверка наличия директории
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    # Проверка существования CSV
    if not os.path.exists(csv_path):
        df = pd.DataFrame(columns=[
            "Id", "Возраст", "Пол", "FOV",
            "Пути к изображениям",
            "Предсказанные диаметры артерий",
            "Предсказанные позиции артерий",
            "Измененные диаметры артерий",
            "Измененные позиции артерий"
        ])
    else:
        df = pd.read_csv(csv_path)

    # Добавление новой строки
    new_row = {
        "Id": id,
        "Возраст": age,
        "Пол": sex,
        "FOV": str(fov),
        "Пути к изображениям": str(image_paths),
        "Предсказанные диаметры артерий": str(predicted_diameters),
        "Предсказанные позиции артерий": str(predicted_positions),
        "Измененные диаметры артерий": str(modified_diameters),
        "Измененные позиции артерий": str(modified_positions)
    }

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    df.to_csv(csv_path, index=False)