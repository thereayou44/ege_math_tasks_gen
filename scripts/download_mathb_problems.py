import os
import json
import re
from sdamgia_api import SdamGIA

sdamgia = SdamGIA()

subject = 'mathb'  # базовый уровень математики

# Получение каталога
catalog = sdamgia.get_catalog(subject)

# Создание базовой папки для хранения задач
base_folder = 'data/categories/math_base_catalog_subcategories'
os.makedirs(base_folder, exist_ok=True)

# Функция для очистки текста
def clean_text(text):
    # Удаление мягких переносов и лишних пробелов
    text = text.replace('\xa0', ' ')  # Неразрывные пробелы
    text = text.replace('\xad', '')  # Мягкие переносы
    text = re.sub(r'\s+', ' ', text)  # Удаление повторяющихся пробелов
    return text.strip()

# Функция для сохранения задачи
def save_problem_to_folder(topic_name, category_name, problem):
    # Удаление недопустимых символов из названий
    topic_name = topic_name.replace('/', '_').replace('\\', '_')
    category_name = category_name.replace('/', '_').replace('\\', '_')

    # Путь к папке категории
    category_folder = os.path.join(base_folder, topic_name, category_name)
    
    # Создание папок для темы и категории
    os.makedirs(category_folder, exist_ok=True)

    # Очистка текста задачи
    problem['condition']['text'] = clean_text(problem['condition']['text'])

    # Удаление поля "solution", если оно есть
    if 'solution' in problem:
        del problem['solution']

    # Сохранение задачи в файл JSON
    problem_id = problem['id']
    problem_file = os.path.join(category_folder, f'{problem_id}.json')
    with open(problem_file, 'w', encoding='utf-8') as f:
        json.dump(problem, f, ensure_ascii=False, indent=4)

# Создание файла для хранения структуры каталога
catalog_data = []

# Сбор и сохранение задач
for topic in catalog:
    topic_name = topic['topic_name']
    topic_data = {"name": topic_name, "categories": []}
    
    for category in topic['categories']:
        category_name = category['category_name']
        category_id = category['category_id']
        
        # Добавление категории в данные каталога
        topic_data["categories"].append({"name": category_name})

        print(f"Скачиваем задачи из темы: {topic_name}, категории: {category_name}")
        problems = sdamgia.get_category_by_id(subject, category_id)

        for problem_id in problems:
            try:
                # Получение данных задачи
                problem = sdamgia.get_problem_by_id(subject, problem_id)
                
                # Сохранение задачи в соответствующую папку
                save_problem_to_folder(topic_name, category_name, problem)
            except Exception as e:
                print(f"Ошибка при скачивании задачи {problem_id}: {e}")
    
    catalog_data.append(topic_data)

# Сохранение структуры каталога в JSON файл
catalog_file = os.path.join("Data", "math_base_catalog.json")
with open(catalog_file, 'w', encoding='utf-8') as f:
    json.dump(catalog_data, f, ensure_ascii=False, indent=4)

# После загрузки задач запускаем генерацию списка категорий
print("Обновляем списки категорий...")
os.system("python generate_categories_list.py")

print("Загрузка задач и обновление категорий завершены!") 