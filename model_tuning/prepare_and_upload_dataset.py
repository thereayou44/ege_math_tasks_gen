#!/usr/bin/env python3

import json
import os
import time
import sys
import random
import glob
import re
import pathlib
import requests
from collections import defaultdict
from bs4 import BeautifulSoup
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

def extract_text_and_formulas(html_content):
    """
    Извлекает текст и формулы из HTML-содержимого задачи.
    
    Args:
        html_content: HTML-код задачи
        
    Returns:
        str: Текст с формулами в формате LaTeX
    """
    if not html_content:
        return ""
        
    soup = BeautifulSoup(html_content, "html.parser")
    
    # Заменяем изображения формул на LaTeX-код из атрибута alt
    for img in soup.find_all("img"):
        alt_text = img.get("alt", "").strip()
        if alt_text:
            latex_span = soup.new_tag("span")
            latex_span["class"] = "math"
            latex_span.string = f"$$ {alt_text} $$"
            img.replace_with(latex_span)
    
    # Получаем текст без лишних пробелов
    text = re.sub(r'\s+', ' ', soup.get_text()).strip()
    
    return text

def get_available_categories():
    """
    Получает список всех доступных категорий задач из директории data/categories/math_catalog_subcategories.
    
    Returns:
        list: Список названий категорий
    """
    base_dir = "data/categories/math_catalog_subcategories"
    try:
        categories = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        print(f"Найдено {len(categories)} категорий задач")
        return categories
    except FileNotFoundError:
        print(f"Каталог {base_dir} не найден.")
        return []

def get_available_subcategories(category):
    """
    Получает список всех доступных подкатегорий для указанной категории задач.
    
    Args:
        category: Название категории
        
    Returns:
        list: Список названий подкатегорий
    """
    base_dir = "data/categories/math_catalog_subcategories"
    category_dir = os.path.join(base_dir, category)
    
    try:
        subcategories = [d for d in os.listdir(category_dir) if os.path.isdir(os.path.join(category_dir, d))]
        print(f"Для категории '{category}' найдено {len(subcategories)} подкатегорий")
        return subcategories
    except FileNotFoundError:
        print(f"Каталог {category_dir} не найден.")
        return []

def yandex_generate(prompt, temperature=0.7, max_tokens=1000):
    """
    Отправляет запрос к API YandexGPT и возвращает ответ.
    Используется для получения примеров для обучения.
    """
    try:
        url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Api-Key {YANDEX_API_KEY}"
        }
        
        payload = {
            "modelUri": f"gpt://{YANDEX_FOLDER_ID}/yandexgpt-lite",
            "completionOptions": {
                "temperature": temperature,
                "maxTokens": max_tokens,
                "stream": False
            },
            "messages": [
                {
                    "role": "system",
                    "text": "Вы - опытный преподаватель математики, специализирующийся на подготовке к ЕГЭ. Давайте точные и краткие ответы без лишних пояснений."
                },
                {
                    "role": "user",
                    "text": prompt
                }
            ]
        }
        
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        
        generated_text = result["result"]["alternatives"][0]["message"]["text"]
        
        # Добавляем задержку, чтобы не превысить лимиты API
        time.sleep(1)
        
        return generated_text
    except Exception as e:
        print(f"Ошибка при генерации через Yandex API: {e}")
        return None

def generate_solution(task):
    """
    Генерирует решение для задачи.
    
    Args:
        task: Текст задачи
        
    Returns:
        str: Решение задачи
    """
    prompt = f"""
Решите следующую математическую задачу из ЕГЭ:

{task}

Предоставьте полное пошаговое решение с пояснениями. 
Математические формулы запишите в формате LaTeX между символами $$...$$
В конце обязательно укажите итоговый ответ.
"""

    solution = yandex_generate(prompt, temperature=0.3, max_tokens=1500)
    return solution if solution else "Не удалось сгенерировать решение."

def generate_hints(task, solution, difficulty_level=3):
    """
    Генерирует серию адаптивных подсказок для задачи.
    
    Args:
        task: Текст задачи
        solution: Полное решение задачи
        difficulty_level: Уровень сложности подсказок
        
    Returns:
        list: Список из 3 подсказок
    """
    prompt = f"""
Создайте 3 последовательные подсказки для решения этой математической задачи ЕГЭ:

Задача: {task}

Полное решение: {solution}

Уровень сложности подсказок: {difficulty_level} (по шкале от 1 до 5, где 1 - самые подробные подсказки, 
5 - минимальная помощь)

Подсказки должны быть адаптивными, постепенно раскрывая ход решения:
1. Первая подсказка: только направление мысли, без конкретных шагов решения
2. Вторая подсказка: ключевой метод или формула, необходимые для решения
3. Третья подсказка: конкретное указание на следующий шаг, почти решение, но без финального ответа

Оформите ответ в виде трех пронумерованных подсказок, разделенных символом ###.
Математические формулы запишите в формате LaTeX между символами $$...$$
"""

    result = yandex_generate(prompt, temperature=0.5, max_tokens=1000)
    
    if not result:
        return [
            "Подумайте о том, как применить основные формулы для этого типа задач.",
            "Начните с анализа условия и определения ключевых величин.",
            "Примените соответствующие формулы и методы решения."
        ]
    
    # Разделение подсказок в список
    hints = [hint.strip() for hint in result.split("###")]
    
    # Очистка от номеров в начале подсказок
    hints = [re.sub(r"^\d+\.\s*", "", hint) for hint in hints if hint]
    
    # Если получили меньше 3 подсказок, дополняем список
    while len(hints) < 3:
        hints.append("Используйте результаты предыдущих шагов для завершения решения.")
    
    # Ограничиваем до 3 подсказок
    return hints[:3]

def prepare_training_dataset(output_file="tuning_dataset.jsonl"):
    """
    Собирает и подготавливает датасет для дообучения модели YandexGPT из всех категорий и подкатегорий.
    Берет ВСЕ задачи без ограничений.
    
    Args:
        output_file: Имя выходного файла
        
    Returns:
        str: Путь к созданному файлу датасета
    """
    print(f"Подготовка датасета для дообучения YandexGPT...")
    
    # Системная инструкция для модели с указанием использовать LaTeX для математических формул
    system_prompt = """Ты — помощник для подготовки к ЕГЭ по математике. 
Ты генерируешь задачи ЕГЭ, их решения и подсказки различного уровня сложности.
Твои ответы должны быть структурированы следующим образом:
УСЛОВИЕ: [текст условия задачи]
РЕШЕНИЕ: [подробное решение с объяснениями]
ПОДСКАЗКА-1: [первая подсказка]
ПОДСКАЗКА-2: [вторая подсказка]
ПОДСКАЗКА-3: [третья подсказка]

Все математические формулы и выражения должны быть записаны в формате LaTeX между символами $$ для многострочных формул или $ для формул в тексте.
Например: $x^2 + 5x + 6 = 0$ или $$\\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}$$

Твои ответы должны быть лаконичными, понятными и точными, без лишней информации."""
    
    # Получаем все категории из директории
    categories = get_available_categories()
    
    # Словарь для хранения задач
    all_tasks = []
    
    # Перебираем все категории и подкатегории
    for category in categories:
        print(f"\nОбработка категории: {category}")
        # Получаем все подкатегории для текущей категории
        subcategories = get_available_subcategories(category)
        
        for subcategory in subcategories:
            print(f"  Обработка подкатегории: {subcategory}")
            
            # Собираем все задачи из подкатегории
            category_tasks = []
            base_dir = "data/categories/math_catalog_subcategories"
            folder = os.path.join(base_dir, category, subcategory)
            
            try:
                files = [f for f in os.listdir(folder) if f.endswith(".json") and f.lower() != "subcategories.json"]
                
                for file in files:
                    filepath = os.path.join(folder, file)
                    try:
                        with open(filepath, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        
                        condition = extract_text_and_formulas(data.get("condition", {}).get("html", ""))
                        solution = data.get("solution", {}).get("text", "")
                        
                        # Если нет решения, но есть условие, генерируем решение
                        if condition and not solution:
                            print(f"    Генерация решения для задачи из файла {file}...")
                            solution = generate_solution(condition)
                            
                        # Формируем подсказки (если их нет в данных или если их меньше 3)
                        hints = []
                        if "hints" in data and isinstance(data["hints"], list) and len(data["hints"]) >= 3:
                            hints = data["hints"][:3]
                        elif condition and solution:
                            # Генерируем подсказки с помощью модели
                            print(f"    Генерация подсказок для задачи из файла {file}...")
                            hints = generate_hints(condition, solution)
                        
                        if condition and solution:
                            category_tasks.append({
                                "condition": condition,
                                "solution": solution,
                                "hints": hints,
                                "category": category,
                                "subcategory": subcategory
                            })
                    except Exception as e:
                        print(f"    Ошибка при обработке файла {filepath}: {e}")
                
                # Добавляем все задачи без ограничений
                print(f"    Добавлено {len(category_tasks)} задач из подкатегории {subcategory}")
                all_tasks.extend(category_tasks)
                
            except FileNotFoundError:
                print(f"    Каталог {folder} не найден.")
    
    print(f"\nВсего собрано задач: {len(all_tasks)}")
    
    # Формируем примеры для дообучения в формате YandexGPT
    examples = []
    
    # Создаем примеры для каждой задачи и типа запроса
    for task in all_tasks:
        # Пример запроса на условие задачи
        examples.append({
            "request": [
                {"role": "system", "text": system_prompt},
                {"role": "user", "text": f"Сгенерируй задачу ЕГЭ по математике из раздела '{task['category']}' на тему '{task['subcategory']}' с запросом 'условие'"}
            ],
            "response": f"УСЛОВИЕ: {task['condition']}\n\nРЕШЕНИЕ: \n\nПОДСКАЗКА-1: \n\nПОДСКАЗКА-2: \n\nПОДСКАЗКА-3: "
        })
        
        # Пример запроса на решение
        examples.append({
            "request": [
                {"role": "system", "text": system_prompt},
                {"role": "user", "text": f"Дай решение для задачи: {task['condition']}"}
            ],
            "response": f"УСЛОВИЕ: {task['condition']}\n\nРЕШЕНИЕ: {task['solution']}\n\nПОДСКАЗКА-1: \n\nПОДСКАЗКА-2: \n\nПОДСКАЗКА-3: "
        })
        
        # Пример запроса на подсказки
        if task['hints'] and len(task['hints']) > 0:
            # Запрос на первую подсказку
            examples.append({
                "request": [
                    {"role": "system", "text": system_prompt},
                    {"role": "user", "text": f"Дай подсказку для задачи: {task['condition']}"}
                ],
                "response": f"УСЛОВИЕ: {task['condition']}\n\nРЕШЕНИЕ: \n\nПОДСКАЗКА-1: {task['hints'][0]}\n\nПОДСКАЗКА-2: \n\nПОДСКАЗКА-3: "
            })
            
            # Если есть больше подсказок
            if len(task['hints']) > 1:
                examples.append({
                    "request": [
                        {"role": "system", "text": system_prompt},
                        {"role": "user", "text": f"Дай вторую подсказку для задачи: {task['condition']}"}
                    ],
                    "response": f"УСЛОВИЕ: {task['condition']}\n\nРЕШЕНИЕ: \n\nПОДСКАЗКА-1: {task['hints'][0]}\n\nПОДСКАЗКА-2: {task['hints'][1]}\n\nПОДСКАЗКА-3: "
                })
            
            if len(task['hints']) > 2:
                examples.append({
                    "request": [
                        {"role": "system", "text": system_prompt},
                        {"role": "user", "text": f"Дай третью подсказку для задачи: {task['condition']}"}
                    ],
                    "response": f"УСЛОВИЕ: {task['condition']}\n\nРЕШЕНИЕ: \n\nПОДСКАЗКА-1: {task['hints'][0]}\n\nПОДСКАЗКА-2: {task['hints'][1]}\n\nПОДСКАЗКА-3: {task['hints'][2]}"
                })
        
        # Пример запроса на полное решение с подсказками
        if task['hints'] and len(task['hints']) > 0:
            # Формируем полный текст ответа
            full_response = f"УСЛОВИЕ: {task['condition']}\n\nРЕШЕНИЕ: {task['solution']}\n\n"
            
            for i, hint in enumerate(task['hints'][:3], 1):
                full_response += f"ПОДСКАЗКА-{i}: {hint}\n\n"
            
            # Дополняем пустыми подсказками, если их меньше 3
            for i in range(len(task['hints']) + 1, 4):
                full_response += f"ПОДСКАЗКА-{i}: \n\n"
            
            examples.append({
                "request": [
                    {"role": "system", "text": system_prompt},
                    {"role": "user", "text": f"Сгенерируй задачу ЕГЭ по математике из раздела '{task['category']}' на тему '{task['subcategory']}' с полным решением и подсказками"}
                ],
                "response": full_response.strip()
            })
    
    # Запись в файл JSONL
    with open(output_file, 'w', encoding='utf-8') as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    print(f"\nДатасет успешно создан! Записано {len(examples)} примеров в файл {output_file}")
    print(f"Размер файла: {os.path.getsize(output_file) / (1024 * 1024):.2f} МБ")
    
    return output_file

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
                print("Рекомендуется исправить формат датасета.")
                return None
            
    except Exception as e:
        print(f"Ошибка при проверке формата датасета: {e}")
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
                    print(f"ID датасета: {dataset_draft.id}")
                    print(f"Вы можете проверить статус позже через консоль Яндекс.Облака.")
                    return dataset_draft.id
            else:
                print(f"Загрузка запущена. ID датасета: {dataset_draft.id}")
                print(f"Процесс загрузки и валидации продолжится в Яндекс.Облаке.")
                return dataset_draft.id
            
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

def main():
    """
    Основная функция для подготовки и загрузки датасета.
    """
    try:
        # Параметры по умолчанию
        output_file = "tuning_dataset.jsonl"
        
        # Проверяем аргументы командной строки
        if len(sys.argv) > 1:
            output_file = sys.argv[1]
            
        print("Будет создан датасет со ВСЕМИ доступными задачами из всех категорий и подкатегорий")
            
        # Шаг 1: Подготовка датасета
        print("\n=== Шаг 1: Подготовка датасета ===")
        dataset_path = prepare_training_dataset(output_file)
        
        if not dataset_path:
            print("Не удалось подготовить датасет. Пожалуйста, проверьте наличие файлов в директории Data.")
            return
            
        # Шаг 2: Загрузка датасета в облако
        print("\n=== Шаг 2: Загрузка датасета в Яндекс.Облако ===")
        upload_dataset_now = input("Загрузить датасет в Яндекс.Облако? (Y/n): ").lower() != 'n'
        
        if upload_dataset_now:
            dataset_id = upload_dataset(dataset_path)
            
            if dataset_id:
                print(f"\nДатасет успешно загружен. ID датасета: {dataset_id}")
                print("Теперь вы можете использовать этот ID для запуска процесса дообучения модели.")
                print(f"Команда для запуска дообучения: python start-tuning.py {dataset_id}")
            else:
                print("\nНе удалось загрузить датасет. Проверьте ошибки выше.")
        else:
            print(f"\nДатасет создан и сохранен в файл {dataset_path}")
            print("Вы можете загрузить его позже с помощью команды:")
            print(f"python upload_dataset.py {dataset_path}")
        
    except Exception as e:
        print(f"Произошла ошибка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 