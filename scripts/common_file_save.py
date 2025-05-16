#!/usr/bin/env python3
import os
import re
import json
import time
import glob
import shutil
from datetime import datetime

"""
Скрипт для объединения созданных задач и промптов в общие файлы.
Обрабатывает фиксированные файлы debug_prompt.txt, debug_response.txt и debug_task_info.json
и добавляет их содержимое в общие файлы с сохранением истории.
"""

# Пути к общим файлам
ALL_PROMPTS_FILE = os.path.join(DEBUG_FILES_DIR, "debug_all_prompts.txt")
ALL_RESPONSES_FILE = os.path.join(DEBUG_FILES_DIR, "debug_all_responses.txt")
ALL_TASKS_INFO_FILE = os.path.join(DEBUG_FILES_DIR, "debug_all_tasks_info.json")

# Пути к фиксированным файлам
DEBUG_FILES_DIR = "debug_files"  # Та же директория, что указана в task_generator.py
PROMPT_FILE = os.path.join(DEBUG_FILES_DIR, "debug_prompt.txt")
RESPONSE_FILE = os.path.join(DEBUG_FILES_DIR, "debug_response.txt")
TASK_INFO_FILE = os.path.join(DEBUG_FILES_DIR, "debug_task_info.json")

def process_prompt_files():
    """Обрабатывает файл с промптом и добавляет его в общий файл"""
    if not os.path.exists(PROMPT_FILE):
        print(f"Файл {PROMPT_FILE} не найден")
        return
    
    # Текущая метка времени для записи
    time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    with open(ALL_PROMPTS_FILE, 'a', encoding='utf-8') as out_file:
        try:
            with open(PROMPT_FILE, 'r', encoding='utf-8') as in_file:
                content = in_file.read()
            
            # Записываем в общий файл
            out_file.write(f"\n\n===НОВЫЙ ЗАПРОС ({time_str})===\n\n")
            out_file.write(content)
            
            print(f"Обработан файл {PROMPT_FILE}")
        except Exception as e:
            print(f"Ошибка при обработке файла {PROMPT_FILE}: {e}")
    
    print(f"Промпт добавлен в файл {ALL_PROMPTS_FILE}")

def process_response_files():
    """Обрабатывает файл с ответом и добавляет его в общий файл"""
    if not os.path.exists(RESPONSE_FILE):
        print(f"Файл {RESPONSE_FILE} не найден")
        return
    
    # Текущая метка времени для записи
    time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Получаем информацию о задаче, если она есть
    category = "Неизвестная категория"
    subcategory = ""
    
    if os.path.exists(TASK_INFO_FILE):
        try:
            with open(TASK_INFO_FILE, 'r', encoding='utf-8') as info_f:
                task_info = json.load(info_f)
                category = task_info.get("category", "Неизвестная категория")
                subcategory = task_info.get("subcategory", "")
        except Exception as e:
            print(f"Ошибка при чтении информации о задаче из {TASK_INFO_FILE}: {e}")
    
    with open(ALL_RESPONSES_FILE, 'a', encoding='utf-8') as out_file:
        try:
            with open(RESPONSE_FILE, 'r', encoding='utf-8') as in_file:
                content = in_file.read()
            
            # Записываем в общий файл
            out_file.write(f"\n\n===НОВЫЙ ОТВЕТ ({time_str})===\n")
            out_file.write(f"Категория: {category}")
            if subcategory:
                out_file.write(f", Подкатегория: {subcategory}")
            out_file.write(f"\n===ОТВЕТ МОДЕЛИ===\n{content}")
            
            print(f"Обработан файл {RESPONSE_FILE}")
        except Exception as e:
            print(f"Ошибка при обработке файла {RESPONSE_FILE}: {e}")
    
    print(f"Ответ добавлен в файл {ALL_RESPONSES_FILE}")

def process_task_info_files():
    """Обрабатывает файл с информацией о задаче и добавляет его в общий JSON-файл"""
    if not os.path.exists(TASK_INFO_FILE):
        print(f"Файл {TASK_INFO_FILE} не найден")
        return
    
    all_tasks = []
    
    # Загружаем существующие задачи, если файл уже есть
    if os.path.exists(ALL_TASKS_INFO_FILE):
        try:
            with open(ALL_TASKS_INFO_FILE, 'r', encoding='utf-8') as f:
                all_tasks = json.load(f)
        except Exception as e:
            print(f"Ошибка при чтении существующего файла {ALL_TASKS_INFO_FILE}: {e}")
    
    # Добавляем новую задачу
    try:
        with open(TASK_INFO_FILE, 'r', encoding='utf-8') as in_file:
            task_info = json.load(in_file)
            
            # Добавляем метку времени, если её нет
            if "timestamp" not in task_info:
                task_info["timestamp"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            all_tasks.append(task_info)
            print(f"Обработан файл {TASK_INFO_FILE}")
    except Exception as e:
        print(f"Ошибка при обработке файла {TASK_INFO_FILE}: {e}")
    
    # Сохраняем все задачи в один JSON-файл
    with open(ALL_TASKS_INFO_FILE, 'w', encoding='utf-8') as out_file:
        json.dump(all_tasks, out_file, ensure_ascii=False, indent=4)
    
    print(f"Информация о задаче добавлена в файл {ALL_TASKS_INFO_FILE}")

def backup_debug_files():
    """Создает резервную копию текущих debug-файлов с временной меткой"""
    backup_dir = os.path.join(DEBUG_FILES_DIR, "backups")
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)
    
    timestamp = int(time.time())
    for file_path in [PROMPT_FILE, RESPONSE_FILE, TASK_INFO_FILE]:
        if os.path.exists(file_path):
            base_filename = os.path.basename(file_path)
            name_without_ext = os.path.splitext(base_filename)[0]
            file_ext = os.path.splitext(file_path)[1]
            backup_file = os.path.join(backup_dir, f"{name_without_ext}_{timestamp}{file_ext}")
            try:
                shutil.copy2(file_path, backup_file)
                print(f"Создана резервная копия {file_path} -> {backup_file}")
            except Exception as e:
                print(f"Ошибка при создании резервной копии {file_path}: {e}")

if __name__ == "__main__":
    print("Начинаем добавление файлов в общие файлы...")
    
    # Создаем общие файлы, если их нет
    for file_path in [ALL_PROMPTS_FILE, ALL_RESPONSES_FILE]:
        if not os.path.exists(file_path):
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f"Файл создан {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Спрашиваем, нужно ли создать резервные копии перед обработкой
    backup = input("Создать резервные копии текущих debug-файлов перед обработкой? (y/n): ")
    if backup.lower() == 'y':
        backup_debug_files()
    
    # Обрабатываем все типы файлов
    process_prompt_files()
    process_response_files()
    process_task_info_files()
    
    print("Обработка завершена!") 