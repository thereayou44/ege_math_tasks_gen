import os
import json
import re
from sdamgia import SdamGIA

sdamgia = SdamGIA()

subject = 'mathb'

# Получение каталога
catalog = sdamgia.get_catalog(subject)

# Создание базовой папки для хранения задач
base_folder = f'{subject}_problems_basic'
os.makedirs(base_folder, exist_ok=True)

# Папки для задач с изображениями и без
with_images_folder = os.path.join(base_folder, 'with_images')
without_images_folder = os.path.join(base_folder, 'without_images')

# Создание этих папок
os.makedirs(with_images_folder, exist_ok=True)
os.makedirs(without_images_folder, exist_ok=True)

# Функция для очистки текста
def clean_text(text):
    # Удаление мягких переносов и лишних пробелов
    text = text.replace('\xa0', ' ')  # Неразрывные пробелы
    text = text.replace('\xad', '')  # Мягкие переносы
    text = re.sub(r'\s+', ' ', text)  # Удаление повторяющихся пробелов
    return text.strip()

# Функция для сохранения задачи
def save_problem_to_folder(base_folder, has_images, topic_name, category_name, problem):
    # Удаление недопустимых символов из названий
    topic_name = topic_name.replace('/', '_').replace('\\', '_')
    category_name = category_name.replace('/', '_').replace('\\', '_')

    # Выбираем папку для изображений или без изображений
    if has_images:
        images_folder = os.path.join(with_images_folder, topic_name, category_name)
    else:
        images_folder = os.path.join(without_images_folder, topic_name, category_name)

    # Создание нужных папок для темы и категории
    os.makedirs(images_folder, exist_ok=True)

    # Очистка текста задачи
    problem['condition']['text'] = clean_text(problem['condition']['text'])

    # Удаление поля "solution", если оно есть
    if 'solution' in problem:
        del problem['solution']

    # Сохранение задачи в файл JSON
    problem_id = problem['id']
    problem_file = os.path.join(images_folder, f'{problem_id}.json')
    with open(problem_file, 'w', encoding='utf-8') as f:
        json.dump(problem, f, ensure_ascii=False, indent=4)

# Сбор и сохранение задач
for topic in catalog:
    topic_name = topic['topic_name']
    for category in topic['categories']:
        category_name = category['category_name']
        category_id = category['category_id']

        print(f"Скачиваем задачи из темы: {topic_name}, категории: {category_name}")
        problems = sdamgia.get_category_by_id(subject, category_id)

        for problem_id in problems:
            try:
                # Получение данных задачи
                problem = sdamgia.get_problem_by_id(subject, problem_id)

                # Проверка наличия изображений в задаче
                has_images = len(problem['condition']['images']) > 0

                # Сохранение задачи в соответствующую папку
                save_problem_to_folder(base_folder, has_images, topic_name, category_name, problem)
            except Exception as e:
                print(f"Ошибка при скачивании задачи {problem_id}: {e}")
