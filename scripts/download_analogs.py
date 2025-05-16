import os
import json
import re
import sys

# Добавляем явное указание пути к библиотеке
sys.path.append('/home/thereayou/.local/lib/python3.10/site-packages')
from sdamgia import SdamGIA

import time
import glob

sdamgia = SdamGIA()

# Функция для очистки текста
def clean_text(text):
    # Удаление мягких переносов и лишних пробелов
    text = text.replace('\xa0', ' ')  # Неразрывные пробелы
    text = text.replace('\xad', '')  # Мягкие переносы
    text = re.sub(r'\s+', ' ', text)  # Удаление повторяющихся пробелов
    return text.strip()

# Функция для сохранения задачи
def save_problem_to_folder(folder_path, problem_id, problem, subject='math'):
    # Очистка текста задачи если есть
    if 'condition' in problem and 'text' in problem['condition']:
        problem['condition']['text'] = clean_text(problem['condition']['text'])

    # Сохранение задачи в файл JSON
    problem_file = os.path.join(folder_path, f'{problem_id}.json')
    with open(problem_file, 'w', encoding='utf-8') as f:
        json.dump(problem, f, ensure_ascii=False, indent=4)
    print(f"Сохранена задача {problem_id} в {problem_file}")

# Функция для скачивания аналогов задачи
def download_analogs(source_file, subject='math'):
    # Загружаем JSON из файла
    with open(source_file, 'r', encoding='utf-8') as f:
        problem_data = json.load(f)
    
    problem_id = problem_data.get('id')
    analog_ids = problem_data.get('analog_ids', [])
    
    if not analog_ids:
        print(f"Для задачи {problem_id} не найдены аналоги")
        return
    
    # Получаем путь к директории, где находится исходный файл
    folder_path = os.path.dirname(source_file)
    
    # Скачиваем каждый аналог
    for analog_id in analog_ids:
        # Пропускаем, если файл аналога уже существует
        analog_file = os.path.join(folder_path, f'{analog_id}.json')
        if os.path.exists(analog_file):
            print(f"Аналог {analog_id} уже существует, пропускаем")
            continue
        
        try:
            # Получение данных задачи-аналога
            analog_problem = sdamgia.get_problem_by_id(subject, analog_id)
            
            # Сохранение задачи в ту же папку
            save_problem_to_folder(folder_path, analog_id, analog_problem, subject)
            
            # Добавляем небольшую задержку, чтобы не перегружать API
            time.sleep(0.2)
        except Exception as e:
            print(f"Ошибка при скачивании аналога {analog_id} для задачи {problem_id}: {e}")

# Основная функция для обработки всех задач в указанных папках
def process_all_problems():
    # Пути к папкам с задачами
    base_paths = [
        'data/categories/math_base_catalog_subcategories',
        'data/categories/math_catalog_subcategories'
    ]
    
    for base_path in base_paths:
        print(f"Обрабатываем задачи в {base_path}")
        
        # Получаем все JSON файлы с задачами (исключаем subcategories.json)
        problem_files = []
        for root, dirs, files in os.walk(base_path):
            for file in files:
                if file.endswith('.json') and file != 'subcategories.json':
                    problem_files.append(os.path.join(root, file))
        
        print(f"Найдено {len(problem_files)} задач")
        
        # Обрабатываем каждый файл задачи
        for i, problem_file in enumerate(problem_files, 1):
            print(f"[{i}/{len(problem_files)}] Обрабатываем {problem_file}")
            download_analogs(problem_file)

            # Добавляем небольшую задержку между обработкой файлов
            if i % 10 == 0:
                print(f"Обработано {i} задач, делаем паузу...")
                time.sleep(1)  # Пауза после каждых 10 задач

if __name__ == "__main__":
    process_all_problems()
    print("Скачивание аналогов задач завершено!") 