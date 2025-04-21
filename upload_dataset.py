#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import time
import sys
import pathlib
import re
from dotenv import load_dotenv
from yandex_cloud_ml_sdk import YCloudML
from yandex_cloud_ml_sdk.exceptions import DatasetValidationError

# Загружаем переменные окружения из .env файла
load_dotenv()

# Получаем значения из переменных окружения
YANDEX_API_KEY = os.getenv('YANDEX_API_KEY')
YANDEX_FOLDER_ID = os.getenv('YANDEX_FOLDER_ID')

if not YANDEX_API_KEY or not YANDEX_FOLDER_ID:
    raise ValueError("Не найдены необходимые переменные окружения YANDEX_API_KEY или YANDEX_FOLDER_ID в файле .env")

def check_dataset_format(dataset_path):
    """
    Проверяет формат датасета и конвертирует его в нужный формат при необходимости.
    
    YandexGPT ожидает формат:
    {"request": [{"role": "system", "text": "..."}, {"role": "user", "text": "..."}], "response": "..."}
    
    Args:
        dataset_path: Путь к файлу датасета
        
    Returns:
        str: Путь к проверенному/конвертированному файлу или None в случае ошибки
    """
    try:
        if not os.path.exists(dataset_path):
            print(f"Файл {dataset_path} не найден")
            return None
        
        # Проверяем, является ли файл JSONL
        is_jsonl = dataset_path.endswith('.jsonl')
        is_json = dataset_path.endswith('.json')
        
        if not (is_jsonl or is_json):
            print("Предупреждение: файл должен иметь расширение .jsonl или .json")
            proceed = input("Продолжить проверку? (y/N): ").lower().startswith('y')
            if not proceed:
                return None
        
        # Открываем файл и читаем первую строку для проверки формата
        is_correct_format = True
        is_array = False
        lines_checked = 0
        format_errors = []
        
        if is_jsonl:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f, 1):
                    if i > 10:  # Проверяем до 10 строк
                        break
                    
                    if not line.strip():
                        continue
                        
                    lines_checked += 1
                    try:
                        data = json.loads(line.strip())
                        # Проверяем формат
                        if not isinstance(data, dict):
                            format_errors.append(f"Строка {i}: ожидается объект JSON")
                            is_correct_format = False
                            continue
                            
                        if "request" not in data or "response" not in data:
                            format_errors.append(f"Строка {i}: отсутствуют обязательные поля 'request' и 'response'")
                            is_correct_format = False
                            continue
                            
                        if not isinstance(data["request"], list):
                            format_errors.append(f"Строка {i}: поле 'request' должно быть массивом")
                            is_correct_format = False
                            continue
                            
                        for msg in data["request"]:
                            if not isinstance(msg, dict) or "role" not in msg or "text" not in msg:
                                format_errors.append(f"Строка {i}: сообщения в 'request' должны содержать поля 'role' и 'text'")
                                is_correct_format = False
                                break
                    except json.JSONDecodeError:
                        format_errors.append(f"Строка {i}: невалидный JSON")
                        is_correct_format = False
        else:  # JSON file
            with open(dataset_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    if isinstance(data, list):
                        is_array = True
                        for i, item in enumerate(data[:10]):  # Проверяем до 10 элементов
                            lines_checked += 1
                            
                            if not isinstance(item, dict):
                                format_errors.append(f"Элемент {i+1}: ожидается объект JSON")
                                is_correct_format = False
                                continue
                                
                            if "request" not in item or "response" not in item:
                                format_errors.append(f"Элемент {i+1}: отсутствуют обязательные поля 'request' и 'response'")
                                is_correct_format = False
                                continue
                                
                            if not isinstance(item["request"], list):
                                format_errors.append(f"Элемент {i+1}: поле 'request' должно быть массивом")
                                is_correct_format = False
                                continue
                                
                            for msg in item["request"]:
                                if not isinstance(msg, dict) or "role" not in msg or "text" not in msg:
                                    format_errors.append(f"Элемент {i+1}: сообщения в 'request' должны содержать поля 'role' и 'text'")
                                    is_correct_format = False
                                    break
                    else:
                        format_errors.append("JSON файл должен содержать массив объектов")
                        is_correct_format = False
                except json.JSONDecodeError:
                    format_errors.append("Невалидный JSON файл")
                    is_correct_format = False
        
        # Выводим результаты проверки
        if lines_checked == 0:
            print("Предупреждение: файл пуст или не содержит данных для дообучения")
            return None
            
        if is_correct_format:
            print(f"✅ Формат датасета проверен и соответствует требованиям YandexGPT.")
            print(f"   Проверено строк/элементов: {lines_checked}")
            
            # Если JSON массив, конвертируем в JSONL
            if is_array and is_json:
                jsonl_path = dataset_path.replace('.json', '.jsonl')
                convert = input(f"Конвертировать JSON в JSONL для оптимальной загрузки? (Y/n): ").lower() != 'n'
                if convert:
                    if convert_json_to_jsonl(dataset_path, jsonl_path):
                        return jsonl_path
            
            return dataset_path
        else:
            print("❌ Формат датасета не соответствует требованиям:")
            for error in format_errors:
                print(f"   - {error}")
            
            print("\nОжидаемый формат для дообучения YandexGPT:")
            print("""
{
  "request": [
    {"role": "system", "text": "Системная инструкция"},
    {"role": "user", "text": "Запрос пользователя"}
  ],
  "response": "Ответ модели"
}
            """)
            
            proceed = input("\nПродолжить с загрузкой в текущем формате? (не рекомендуется) (y/N): ").lower().startswith('y')
            if proceed:
                return dataset_path
            else:
                print("Рекомендуется исправить формат датасета перед загрузкой.")
                print("Можно использовать скрипт prepare_dataset.py для создания корректного датасета.")
                return None
            
    except Exception as e:
        print(f"Ошибка при проверке формата датасета: {e}")
        import traceback
        traceback.print_exc()
        return None

def upload_dataset(dataset_path, wait_for_completion=True):
    """
    Загружает датасет в Yandex.Cloud для дообучения модели.
    
    Args:
        dataset_path: Путь к файлу датасета в формате JSONL
        wait_for_completion: Ожидать ли завершения загрузки
        
    Returns:
        str: Идентификатор загруженного датасета
    """
    try:
        print(f"Инициализация SDK с ID каталога: {YANDEX_FOLDER_ID}")
        sdk = YCloudML(
            folder_id=YANDEX_FOLDER_ID,
            auth=YANDEX_API_KEY,
        )
        print("SDK инициализирован успешно!")
        
        # Проверяем, существует ли файл датасета
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Файл датасета не найден по пути: {dataset_path}")
            
        
        file_size = os.path.getsize(dataset_path) / (1024 * 1024)  # Размер в МБ
        print(f"Размер файла датасета: {file_size:.2f} МБ")
        
        # Проверяем формат датасета
        print("\nПроверка формата датасета...")
        checked_path = check_dataset_format(dataset_path)
        if not checked_path:
            return None
        
        if checked_path != dataset_path:
            print(f"Будет использован конвертированный файл: {checked_path}")
            dataset_path = checked_path
        
        # Получаем список существующих датасетов
        print("\nПолучение списка существующих датасетов...")
        datasets = list(sdk.datasets.list())
        if datasets:
            print("Найдены существующие датасеты:")
            for i, dataset in enumerate(datasets, 1):
                print(f"{i}. {dataset.name} (ID: {dataset.id}, Статус: {dataset.status})")
            
            delete_datasets = input("\nУдалить существующие датасеты перед загрузкой нового? (y/N): ").lower().startswith('y')
            if delete_datasets:
                for dataset in datasets:
                    print(f"Удаление датасета {dataset.name} (ID: {dataset.id})...")
                    dataset.delete()
                print("Все датасеты удалены.")
        else:
            print("Существующих датасетов не найдено.")
        
        # Подготавливаем имя датасета из имени файла
        dataset_name = os.path.basename(dataset_path).split('.')[0]
        dataset_name = f"{dataset_name}_{int(time.time())}"  # Добавляем timestamp для уникальности
        
        print(f"\nЗагрузка датасета '{dataset_name}' из файла {dataset_path}...")
        
        # Создаем новый датасет используя правильный метод согласно SDK
        try:
            # Создаем черновик датасета и загружаем данные
            dataset_draft = sdk.datasets.draft_from_path(
                task_type="TextToTextGeneration",
                path=dataset_path,
                upload_format="jsonlines",
                name=dataset_name,
            )
            print(f"Создан черновик датасета")
            
            # Загружаем данные из файла
            print("Загрузка данных...")
            operation = dataset_draft.upload_deferred()
            
            if wait_for_completion:
                print("Ожидание завершения загрузки и валидации...")
                print("Это может занять некоторое время в зависимости от размера датасета.")
                print("Вы можете прервать ожидание, нажав Ctrl+C (процесс загрузки продолжится в Яндекс.Облаке).")
                
                try:
                    # Ожидаем завершения операции загрузки
                    tuning_dataset = operation.wait()
                    print("\nЗагрузка завершена!")
                    
                    # Проверяем статус датасета после загрузки
                    print("Проверка статуса датасета...")
                    dataset_id = tuning_dataset.id
                    dataset = sdk.datasets.get(dataset_id)
                    
                    if dataset.status == "READY":
                        print(f"Датасет успешно загружен и валидирован!")
                        print(f"ID датасета: {dataset_id}")
                        
                        # Сохраняем ID датасета в .env файл
                        save_id = input(f"Сохранить ID датасета {dataset_id} в .env файл? (Y/n): ").lower() != 'n'
                        if save_id:
                            with open(".env", "a") as f:
                                f.write(f"\nDATASET_ID={dataset_id}\n")
                            print(f"ID датасета сохранен в .env файл.")
                            
                        # Предлагаем сразу запустить дообучение
                        start_tuning = input(f"Запустить дообучение модели с использованием этого датасета? (y/N): ").lower().startswith('y')
                        if start_tuning:
                            if os.path.exists("start-tuning.py"):
                                print("\nЗапуск скрипта start-tuning.py...")
                                os.system(f"python start-tuning.py {dataset_id}")
                            else:
                                print("Файл start-tuning.py не найден.")
                                print(f"Вы можете запустить дообучение вручную, используя ID датасета: {dataset_id}")
                        
                        return dataset_id
                    else:
                        print(f"Статус датасета после загрузки: {dataset.status}")
                        print("Датасет загружен, но не прошел валидацию.")
                        print("Проверьте формат данных и повторите попытку.")
                        return None
                        
                except KeyboardInterrupt:
                    print("\nОжидание прервано пользователем.")
                    print(f"Загрузка продолжается в Яндекс.Облаке.")
                    print(f"Вы можете проверить статус позже через консоль Яндекс.Облака.")
                    print(f"Для просмотра статуса проверьте операцию с ID: {operation.id}")
                    return None
            else:
                print(f"Загрузка запущена.")
                print(f"Процесс загрузки и валидации продолжится в Яндекс.Облаке.")
                print(f"Для просмотра статуса проверьте операцию с ID: {operation.id}")
                return None
            
        except Exception as e:
            print(f"Ошибка при загрузке датасета: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    except Exception as e:
        print(f"Ошибка при работе с Яндекс.Облаком: {e}")
        import traceback
        traceback.print_exc()
        return None

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
        for model in sdk.models.list():
            print(f"Модель: {model}, ID: {model.id}")
        
        # Запрашиваем, хочет ли пользователь удалить существующие модели
        delete_models = input("\nУдалить все существующие модели? (y/N): ").lower() == 'y'
        if delete_models:
            for model in sdk.models.list():
                print(f"Удаление модели {model}")
                model.delete()
            print("Все модели удалены")
        
        # Запускаем дообучение
        operation = sdk.models.start_fine_tuning(
            dataset_id=dataset_id,
            name="EGE Math Model",
            description="Модель для генерации задач ЕГЭ по математике и их решений",
            base_model_uri=f"gpt://{YANDEX_FOLDER_ID}/{model_name}",
        )
        
        print(f"Дообучение запущено. ID операции: {operation.id}")
        print("Процесс дообучения может занять несколько часов...")
        print("Вы можете проверить статус дообучения в консоли Яндекс.Облака.")
        
        # Для дополнительной информации можно использовать:
        # model = operation.wait()
        # print(f"Дообучение завершено. ID модели: {model.id}")
        
        return operation.id
        
    except Exception as e:
        print(f"Ошибка при запуске дообучения: {e}")
        import traceback
        traceback.print_exc()
        return None

def convert_json_to_jsonl(json_path, jsonl_path):
    """
    Конвертирует файл в формате JSON (массив объектов) в формат JSONL.
    
    Args:
        json_path: Путь к исходному JSON файлу
        jsonl_path: Путь к выходному JSONL файлу
    """
    try:
        print(f"Конвертация {json_path} в формат JSONL...")
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"Файл успешно конвертирован и сохранен как {jsonl_path}")
        return True
    except Exception as e:
        print(f"Ошибка при конвертации файла: {e}")
        return False

def wait_for_dataset_ready(sdk, dataset_id, max_wait_minutes=10):
    """
    Ожидает готовности датасета, периодически проверяя его статус.
    
    Args:
        sdk: Инициализированный объект YCloudML
        dataset_id: Идентификатор датасета
        max_wait_minutes: Максимальное время ожидания в минутах
        
    Returns:
        bool: True, если датасет готов, False в противном случае
    """
    print(f"Проверка статуса датасета {dataset_id}...")
    
    start_time = time.time()
    max_wait_seconds = max_wait_minutes * 60
    
    while time.time() - start_time < max_wait_seconds:
        try:
            dataset = sdk.datasets.get(dataset_id)
            status = dataset.status
            
            print(f"Текущий статус датасета: {status}")
            
            if status == "READY":
                print(f"Датасет готов к использованию!")
                return True
            elif status == "ERROR":
                print(f"Ошибка при подготовке датасета. Проверьте формат данных.")
                return False
            elif status in ["VALIDATING", "CREATING"]:
                print(f"Датасет еще обрабатывается. Повторная проверка через 10 секунд...")
                time.sleep(10)
            else:
                print(f"Неизвестный статус датасета: {status}. Повторная проверка через 10 секунд...")
                time.sleep(10)
                
        except Exception as e:
            print(f"Ошибка при проверке статуса датасета: {e}")
            time.sleep(10)
    
    print(f"Превышено максимальное время ожидания ({max_wait_minutes} минут).")
    print("Датасет все еще не готов. Вы можете проверить его статус позже в консоли Яндекс.Облака.")
    return False

def main():
    """
    Основная функция для загрузки датасета.
    """
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    else:
        # Ищем файлы .jsonl и .json в текущей директории
        jsonl_files = list(pathlib.Path('.').glob('*.jsonl'))
        json_files = list(pathlib.Path('.').glob('*.json'))
        all_files = jsonl_files + json_files
        
        if not all_files:
            print("В текущей директории не найдены файлы .jsonl или .json")
            dataset_path = input("Введите путь к файлу датасета (.jsonl или .json): ")
        else:
            print("Найдены следующие файлы датасетов:")
            for i, file in enumerate(all_files, 1):
                file_size = os.path.getsize(file) / (1024 * 1024)  # Размер в МБ
                print(f"{i}. {file} ({file_size:.2f} МБ)")
            
            choice = input(f"Выберите файл для загрузки (1-{len(all_files)}) или введите другой путь: ")
            try:
                index = int(choice) - 1
                if 0 <= index < len(all_files):
                    dataset_path = str(all_files[index])
                else:
                    dataset_path = choice
            except ValueError:
                dataset_path = choice
    
    if not dataset_path:
        print("Путь к файлу датасета не указан. Выход.")
        return
    
    wait_for_completion = input("Ожидать завершения загрузки и валидации? (Y/n): ").lower() != 'n'
    
    dataset_id = upload_dataset(dataset_path, wait_for_completion)
    
    if dataset_id:
        print(f"\nПроцесс загрузки датасета инициирован.")
        print(f"ID датасета: {dataset_id}")
        if not wait_for_completion:
            print(f"Вы можете проверить статус загрузки через консоль Яндекс.Облака.")
            print(f"После завершения загрузки и валидации вы можете использовать этот ID для дообучения модели.")
    else:
        print("\nНе удалось загрузить датасет. Проверьте ошибки выше.")

if __name__ == "__main__":
    main() 
    print(sdk.datasets.list())