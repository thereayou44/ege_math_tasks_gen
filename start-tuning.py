#!/usr/bin/env python3

from __future__ import annotations
import os
import sys
import uuid
import time
from dotenv import load_dotenv
from yandex_cloud_ml_sdk import YCloudML

# Загружаем переменные окружения из .env файла
load_dotenv()

# Получаем значения из переменных окружения
YANDEX_API_KEY = os.getenv('YANDEX_API_KEY')
YANDEX_FOLDER_ID = os.getenv('YANDEX_FOLDER_ID')

def main():
    if len(sys.argv) > 1:
        dataset_id = sys.argv[1]
    else:
        # Если ID датасета не передан через аргумент командной строки, пробуем взять из .env
        dataset_id = os.getenv('DATASET_ID')
        if not dataset_id:
            # Если ID нет и в .env, запрашиваем у пользователя
            dataset_id = input("Введите ID датасета для дообучения: ")
    
    if not dataset_id:
        print("ID датасета не указан. Выход.")
        return

    # Настройка максимального числа попыток
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            print(f"Инициализация SDK с ID каталога: {YANDEX_FOLDER_ID}")
            sdk = YCloudML(
                folder_id=YANDEX_FOLDER_ID,
                auth=YANDEX_API_KEY,
            )
            print("SDK инициализирован успешно!")

            # Получаем датасет для обучения
            print(f"Получение датасета с ID: {dataset_id}")
            train_dataset = sdk.datasets.get(dataset_id)
            print(f"Датасет получен успешно: {train_dataset.name}")

            # Задаем базовую модель
            print("Получение базовой модели yandexgpt-lite")
            base_model = sdk.models.completions("yandexgpt-lite")

            # Генерируем уникальное имя для дообученной модели или используем указанное
            model_name = input("Введите название для дообученной модели (Enter для генерации случайного имени): ")
            if not model_name:
                model_name = f"ege_math_model_{str(uuid.uuid4())[:8]}"
            print(f"Будет использовано имя модели: {model_name}")
            
            # Задаем количество семплов для обучения
            n_samples_input = input("Введите количество примеров для обучения (Enter для использования 10000): ")
            n_samples = 10000
            if n_samples_input.strip() and n_samples_input.isdigit():
                n_samples = int(n_samples_input)
            print(f"Количество примеров для обучения: {n_samples}")
            

            # Запускаем дообучение
            print("\nЗапускаем процесс дообучения...")
            tuned_model = base_model.tune(
                train_dataset, name=model_name, n_samples=n_samples
            )
            
            print(f"Задача дообучения завершена!")
            print(f"URI модели: {tuned_model.uri}")
            
            # Сохраняем URI модели в файл для удобства
            with open("tuned_model_info.txt", "w") as f:
                f.write(f"URI модели: {tuned_model.uri}\n")
                f.write(f"Модель: {model_name}\n")
                f.write(f"Завершено: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            print(f"Информация о модели сохранена в файл tuned_model_info.txt")
            
            # Сохраняем URI модели в .env файл
            uri = tuned_model.uri
            print(f"URI дообученной модели: {uri}")
            
            save_uri = input("Сохранить URI модели в .env файл? (Y/n): ").lower() != 'n'
            if save_uri:
                with open(".env", "a") as f:
                    f.write(f"\nTUNED_MODEL_URI={uri}\n")
                    if '@' in uri:
                        suffix = uri.split('@')[-1]
                        f.write(f"TUNING_SUFFIX={suffix}\n")
                print("URI модели сохранен в .env файл")
            
            # Выводим список всех доступных моделей
            print("\nПолучение списка всех доступных моделей...")
            try:
                # Используем альтернативный способ получения моделей
                from yandex_cloud_ml_sdk._models.completions.list import list_models
                
                all_models = list_models(sdk)
                print(f"Найдено {len(all_models)} моделей:")
                for model in all_models:
                    print(f"- {model.name} (URI: {model.uri if hasattr(model, 'uri') else 'Н/Д'})")
            except Exception as e:
                print(f"Ошибка при получении списка моделей: {e}")
                print("Вы можете проверить список доступных моделей в консоли Yandex Cloud")
            
            return tuned_model
            
        except Exception as e:
            retry_count += 1
            print(f"Ошибка при выполнении операции (попытка {retry_count}/{max_retries}): {e}")
            if retry_count < max_retries:
                wait_time = 10 * retry_count
                print(f"Повторная попытка через {wait_time} секунд...")
                time.sleep(wait_time)
            else:
                print("Превышено максимальное количество попыток. Завершение.")
                print("Проверьте подключение к интернету и доступность сервисов Yandex Cloud.")
                print("Также можно проверить статус дообучения в консоли Yandex Cloud.")

if __name__ == "__main__":
    main() 