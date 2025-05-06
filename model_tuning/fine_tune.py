#!/usr/bin/env python3

from __future__ import annotations

import os
import sys
import time
from dotenv import load_dotenv
from yandex_cloud_ml_sdk import YCloudML
import argparse

# Загружаем переменные окружения из .env файла
load_dotenv()

# Получаем значения из переменных окружения
YANDEX_API_KEY = os.getenv('YANDEX_API_KEY')
YANDEX_FOLDER_ID = os.getenv('YANDEX_FOLDER_ID')

if not YANDEX_API_KEY or not YANDEX_FOLDER_ID:
    raise ValueError("Не найдены необходимые переменные окружения YANDEX_API_KEY или YANDEX_FOLDER_ID в файле .env")

def start_fine_tuning(dataset_id, model_name="yandexgpt-lite"):
    """
    Запускает процесс дообучения модели с использованием загруженного датасета.
    
    Args:
        dataset_id: ID датасета, загруженного в Яндекс.Облако
        model_name: Имя базовой модели для дообучения (yandexgpt-lite или yandexgpt)
        
    Returns:
        str: ID созданной операции дообучения
    """
    try:
        print(f"Запуск дообучения модели {model_name} с использованием датасета {dataset_id}...")
        
        # Инициализация SDK
        sdk = YCloudML(
            folder_id=YANDEX_FOLDER_ID,
            auth=YANDEX_API_KEY,
        )
        
        # Получаем список существующих моделей
        print("Список существующих моделей:")
        try:
            for model in sdk.models.list():
                print(f"Модель: {model}, ID: {model.id}")
        except Exception as e:
            print(f"Ошибка при получении списка моделей: {e}")
            print("Продолжаем процесс дообучения...")
        
        # Запрашиваем, хочет ли пользователь удалить существующие модели
        delete_models = input("\nУдалить все существующие модели? (y/N): ").lower() == 'y'
        if delete_models:
            try:
                for model in sdk.models.list():
                    print(f"Удаление модели {model}")
                    model.delete()
                print("Все модели удалены")
            except Exception as e:
                print(f"Ошибка при удалении моделей: {e}")
                print("Продолжаем процесс дообучения...")
        
        # Запускаем дообучение
        print(f"Запуск дообучения модели {model_name} с датасетом {dataset_id}...")
        
        # Формируем URI модели
        base_model_uri = f"gpt://{YANDEX_FOLDER_ID}/{model_name}"
        
        # Параметры дообучения
        operation = sdk.models.start_fine_tuning(
            dataset_id=dataset_id,
            name="EGE Math Model",
            description="Модель для генерации задач ЕГЭ по математике и их решений",
            base_model_uri=base_model_uri
        )
        
        print(f"Дообучение запущено. ID операции: {operation.id}")
        print("Процесс дообучения может занять несколько часов...")
        print("Вы можете проверить статус дообучения в консоли Яндекс.Облака.")
        
        # Опционально: ожидание завершения дообучения
        wait_for_completion = input("\nОжидать завершения дообучения? (y/N): ").lower() == 'y'
        if wait_for_completion:
            print("Ожидание завершения дообучения...")
            print("Это может занять несколько часов.")
            print("Вы можете прервать ожидание, нажав Ctrl+C.")
            
            try:
                # Периодически проверяем статус операции
                while True:
                    status = sdk.operations.get(operation.id)
                    if status.done:
                        print("\nДообучение завершено!")
                        
                        # Получаем ID модели
                        try:
                            models = list(sdk.models.list())
                            for model in models:
                                print(f"Модель: {model}, ID: {model.id}")
                                
                                # Сохраняем ID в .env файл
                                save_model_id = input(f"\nСохранить ID модели {model.id} в .env файл? (Y/n): ").lower() != 'n'
                                if save_model_id:
                                    with open(".env", "a") as f:
                                        f.write(f"\nTUNED_MODEL_ID={model.id}\n")
                                    print(f"ID модели {model.id} сохранен в .env файл.")
                                    print("Теперь вы можете использовать test_tuned.py для тестирования модели.")
                        except Exception as e:
                            print(f"Ошибка при получении списка моделей: {e}")
                        
                        break
                    
                    print(".", end="", flush=True)
                    time.sleep(30)  # Проверяем каждые 30 секунд
            
            except KeyboardInterrupt:
                print("\nОжидание прервано пользователем.")
                print(f"Вы можете проверить статус дообучения позже через консоль Яндекс.Облака.")
        
        return operation.id
        
    except Exception as e:
        print(f"Ошибка при запуске дообучения: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    parser = argparse.ArgumentParser(description="Запуск дообучения модели YandexGPT")
    parser.add_argument("dataset_id", help="ID датасета, загруженного в Яндекс.Облако")
    parser.add_argument("--model", choices=["yandexgpt-lite", "yandexgpt"], default="yandexgpt-lite", 
                      help="Базовая модель для дообучения (по умолчанию yandexgpt-lite)")
    
    args = parser.parse_args()
    
    # Запускаем дообучение
    operation_id = start_fine_tuning(args.dataset_id, args.model)
    
    if operation_id:
        print(f"\nПроцесс дообучения запущен. ID операции: {operation_id}")
        print("По завершении дообучения вы сможете использовать модель в приложении.")
        print("Для тестирования дообученной модели используйте: python test_tuned.py")
    else:
        print("\nНе удалось запустить процесс дообучения. Проверьте ошибки выше.")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Если аргументы не указаны, запрашиваем ID датасета интерактивно
        dataset_id = input("Введите ID датасета для дообучения: ")
        
        model_options = {
            "1": "yandexgpt-lite",
            "2": "yandexgpt"
        }
        print("\nВыберите базовую модель для дообучения:")
        print("1. YandexGPT Lite (меньше и быстрее)")
        print("2. YandexGPT (больше и точнее)")
        
        model_choice = input("Введите номер (по умолчанию 1): ") or "1"
        model_name = model_options.get(model_choice, "yandexgpt-lite")
        
        # Запускаем дообучение
        operation_id = start_fine_tuning(dataset_id, model_name)
        
        if operation_id:
            print(f"\nПроцесс дообучения запущен. ID операции: {operation_id}")
            print("По завершении дообучения вы сможете использовать модель в приложении.")
            print("Для тестирования дообученной модели используйте: python test_tuned.py")
        else:
            print("\nНе удалось запустить процесс дообучения. Проверьте ошибки выше.")
            sys.exit(1)
    else:
        main() 