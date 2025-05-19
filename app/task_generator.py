import requests
from PIL import Image
import io
import json
import os
import random
from bs4 import BeautifulSoup
import re
from dotenv import load_dotenv
import logging
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Используем не-интерактивный бэкенд
import uuid
import base64
import traceback
import time
from io import BytesIO
from PIL import Image, ImageEnhance

try:
    from app.prompts import HINT_PROMPTS, SYSTEM_PROMPT, create_complete_task_prompt, REGEX_PATTERNS, DEFAULT_VISUALIZATION_PARAMS
    from app.visualization import process_bar_chart, process_pie_chart, process_chart_visualization
    from app.visualization.chart_utils import normalize_function_expression
    from app.visualization.renderer import GeometryRenderer
    from app.visualization.detector import needs_visualization, determine_visualization_type

except ImportError:
    from visualization import process_multiple_function_plots, process_bar_chart, process_pie_chart, process_chart_visualization
    from visualization.renderer import GeometryRenderer
    from visualization.detector import needs_visualization, determine_visualization_type
import traceback
import matplotlib.patches as patches
# Импортируем utils.converters так, чтобы работало как из корня проекта, так и из папки app
try:
    from app.utils.converters import convert_html_to_markdown as html_to_markdown
except ImportError:
    from utils.converters import convert_html_to_markdown as html_to_markdown
    
from app.visualization.processors import process_visualization_params as process_viz_params


# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Загрузка переменных окружения из .env файла
load_dotenv()

# Инициализация Yandex API
YANDEX_API_KEY = os.getenv("YANDEX_API_KEY")
YANDEX_FOLDER_ID = os.getenv("YANDEX_FOLDER_ID")

# Директория для хранения отладочных файлов
DEBUG_FILES_DIR = "debug_files"
if not os.path.exists(DEBUG_FILES_DIR):
    os.makedirs(DEBUG_FILES_DIR)
    logging.info(f"Создана папка для отладочных файлов: {DEBUG_FILES_DIR}")

# Инициализируем файлы для отладки, если они не существуют
debug_files = {
    "debug_prompt.txt": "# Файл для хранения промптов\n",
    "debug_response.txt": "# Файл для хранения ответов модели\n",
    "debug_task_info.json": "{}"
}

for filename, content in debug_files.items():
    filepath = os.path.join(DEBUG_FILES_DIR, filename)
    if not os.path.exists(filepath):
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        logging.info(f"Создан файл для отладки: {filepath}")

# Кэш для хранения результатов запросов API
_task_cache = {}

# Используем готовые инструкции для подсказок из модуля prompts
from app.prompts import HINT_PROMPTS, SYSTEM_PROMPT, REGEX_PATTERNS, DEFAULT_VISUALIZATION_PARAMS

def select_file(category, subcategory="", is_basic_level=False):
    """
    Выбирает случайный файл с задачей из указанной категории и подкатегории.
    
    Args:
        category: Название категории задач
        subcategory: Название подкатегории (опционально)
        is_basic_level: Определяет директорию для поиска задач: 
                        True - использовать директорию для базового уровня ЕГЭ,
                        False - использовать директорию для профильного уровня ЕГЭ
        
    Returns:
        dict: JSON-данные выбранной задачи
    """
    # Инициализируем новый seed для генератора случайных чисел,
    # чтобы обеспечить разные результаты при каждом вызове
    random.seed(int(time.time() * 1000) % 10000000)
    
    # Получаем абсолютный путь к корню проекта
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # Выбираем соответствующий каталог в зависимости от уровня
    if is_basic_level:
        base_dir = os.path.join(project_root, "data/categories/math_base_catalog_subcategories")
    else:
        base_dir = os.path.join(project_root, "data/categories/math_catalog_subcategories")
        
    category_dir = os.path.join(base_dir, category)
    
    if not subcategory:
        try:
            subdirs = [d for d in os.listdir(category_dir) if os.path.isdir(os.path.join(category_dir, d))]
        except FileNotFoundError:
            print(f"Каталог {category_dir} не найден.")
            return None
            
        if not subdirs:
            print(f"В каталоге {category_dir} нет подпапок.")
            return None
            
        # Инициализируем новый seed перед выбором подкатегории
        random.seed(int(time.time() * 1000) % 10000000)
        subcategory = random.choice(subdirs)
        print(f"Случайно выбрана подкатегория: {subcategory}")
        
    folder = os.path.join(category_dir, subcategory)
    
    try:
        files = [f for f in os.listdir(folder) if f.endswith(".json") and f.lower() != "subcategories.json"]
    except FileNotFoundError:
        print(f"Каталог {folder} не найден.")
        return None
        
    if not files:
        print("Нет подходящих JSON файлов в каталоге.")
        return None
        
    # Инициализируем новый seed перед выбором файла
    random.seed(int(time.time() * 1000) % 10000000)
    chosen_file = random.choice(files)
    filepath = os.path.join(folder, chosen_file)
    print(f"Выбран файл: {filepath}")
    
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Проверяем новую структуру JSON
    if "condition" in data and "html" in data["condition"]:
        condition_html = data["condition"]["html"]
        if condition_html:
            data["condition_text"] = extract_text_and_formulas(condition_html)
    
    # Для обратной совместимости (старый формат)
    elif "html" in data and data["html"]:
        data["task"] = extract_text_and_formulas(data["html"])

    return data

def extract_text_and_formulas(html_content):
    """
    Извлекает текст и формулы из HTML-содержимого.
    
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

def yandex_gpt_generate(prompt, temperature=0.5, max_tokens=10000, is_basic_level=None):
    """
    Отправляет запрос к API YandexGPT и возвращает ответ.
    
    Args:
        prompt: Текст запроса
        temperature: Температура генерации (от 0 до 1)
        max_tokens: Максимальное количество токенов в ответе (увеличено до 8000)
        is_basic_level: Флаг базового/профильного уровня задачи
        
    Returns:
        str: Сгенерированный текст ответа
    """
    # Для базового уровня добавляем случайный компонент к ключу кэша, чтобы избежать повторов
    random_component = ""
    if is_basic_level:
        random_component = f"_{time.time() * 1000}"
    
    # Создаем ключ кэша на основе параметров запроса
    cache_key = f"{prompt}_{temperature}_{max_tokens}_{is_basic_level}{random_component}"
    
    # Симулируем проверку кэша без фактического использования кэшированных ответов
    if cache_key in _task_cache:
        print("Результат найден в кэше")
        # Удаляем устаревшую задачу из кэша
        del _task_cache[cache_key]
        
    try:
        # Сохраняем промпт в файл для отладки
        prompt_file = os.path.join(DEBUG_FILES_DIR, "debug_prompt.txt")
        # Заменяем секцию с исходной задачей на сокращенную версию для отладочного файла

        with open(prompt_file, 'w', encoding='utf-8') as f:
            f.write(f"===СИСТЕМНЫЙ ПРОМПТ===\n{SYSTEM_PROMPT}\n\n===ПОЛЬЗОВАТЕЛЬСКИЙ ПРОМПТ===\n{prompt}")
        print(f"Промпт сохранен в файл: {prompt_file}")
        
        url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Api-Key {YANDEX_API_KEY}"
        }
        
        payload = {
            "modelUri": f"gpt://{YANDEX_FOLDER_ID}/yandexgpt/rc",
            "completionOptions": {
                "temperature": temperature,
                "maxTokens": max_tokens,
                "stream": False
            },
            "messages": [
                {
                    "role": "system",
                    "text": SYSTEM_PROMPT
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
        
        # Сохраняем результат в кэше
        generated_text = result["result"]["alternatives"][0]["message"]["text"]
        _task_cache[cache_key] = generated_text
        
        # Сохраняем ответ в файл для отладки
        response_file = os.path.join(DEBUG_FILES_DIR, "debug_response.txt")
        with open(response_file, 'w', encoding='utf-8') as f:
            f.write(generated_text)
        print(f"Ответ сохранен в файл: {response_file}")
        
        return generated_text
    except Exception as e:
        print(f"Ошибка при генерации через Yandex API: {e}")
        return None

def extract_answer_with_latex(solution):
    """
    Извлекает ответ из решения и корректирует отображение LaTeX.
    
    Args:
        solution: Полное решение задачи
        
    Returns:
        str: Правильно отформатированный ответ
    """
    if not solution or len(solution.strip()) < 10:
        return "Ответ не найден"
    
    # Проверяем особый формат с ---ОТВЕТ---
    special_pattern = r'[-]{2,}\s*(?:ОТВЕТ|Ответ|ответ)\s*[-]{2,}\s*(.*?)(?:$|\n\s*-|\n\s*\n)'
    special_match = re.search(special_pattern, solution, re.IGNORECASE | re.DOTALL)
    
    if special_match:
        answer = special_match.group(1).strip()
        logging.info(f"Найден ответ в формате ---ОТВЕТ---: {answer}")
        
        # Ищем число в ответе, если это просто число
        number_pattern = r'-?\d+[.,]\d+|-?\d+'
        number_matches = re.findall(number_pattern, answer)
        if number_matches and len(number_matches) == 1:
            answer = number_matches[0]
            
        # Применяем форматирование LaTeX
        return format_latex_answer(answer)
        
    # Ищем "Ответ:" или "Ответ :"
    answer_pattern = r"(?:Ответ|ОТВЕТ|ответ)\s*:(.+?)(?:$|(?<!\.)\n|\n\s*\n|\<\/p\>)"
    answer_match = re.search(answer_pattern, solution, re.IGNORECASE | re.DOTALL)
    
    if answer_match:
        answer = answer_match.group(1).strip()
        logging.info(f"Найден ответ: {answer}")
        return format_latex_answer(answer)
    
    # Если не нашли по первому паттерну, ищем альтернативы
    alternative_patterns = [
        r"(?:Итоговый ответ|итоговый ответ|Итого|итого)\s*:(.+?)(?:$|(?<!\.)\n|\n\s*\n|\<\/p\>)",
        r"(?:Таким образом|Итак|Окончательный ответ|окончательный ответ)\s*:(.+?)(?:$|(?<!\.)\n|\n\s*\n|\<\/p\>)",
        r"(?:В ответе получаем|в ответе получим|В ответе|в ответе)\s*:(.+?)(?:$|(?<!\.)\n|\n\s*\n|\<\/p\>)"
    ]
    
    for pattern in alternative_patterns:
        alt_match = re.search(pattern, solution, re.IGNORECASE | re.DOTALL)
        if alt_match:
            answer = alt_match.group(1).strip()
            logging.info(f"Найден ответ: {answer}")
            return format_latex_answer(answer)
    
    # Прямой поиск чисел в тексте
    # Сначала ищем числа в подходящем формате
    number_pattern = r'-?\d+[.,]\d+'
    number_matches = re.findall(number_pattern, solution)
    
    if number_matches:
        # Берем последнее число с десятичной точкой
        answer = number_matches[-1]
        logging.info(f"Найдено числовое значение с десятичной точкой: {answer}")
        return format_latex_answer(answer)
    
    # Если ответ не найден
    logging.warning("Ответ не найден в решении")
    return "См. решение"

def format_latex_answer(answer):
    """
    Форматирует ответ с правильным LaTeX-окружением.
    
    Args:
        answer: Исходный ответ
        
    Returns:
        str: Отформатированный ответ
    """
    if not answer:
        return "Ответ не найден"
        
    # Удаляем возможные префиксы и суффиксы, такие как "Ответ: "
    answer = re.sub(r'^(?:Ответ|ОТВЕТ|ответ)\s*:\s*', '', answer)
    answer = answer.strip()
    
    # Если ответ уже содержит LaTeX-окружение, сохраняем его как есть
    if answer.startswith('$') and answer.endswith('$'):
        # Проверяем, что $ встречается только в начале и конце ответа
        if answer.count('$') == 2:
            return answer
        
    # Если ответ уже содержит $$, заменяем их на $ для единообразия
    if '$$' in answer:
        # Удаляем все $$
        answer = answer.replace('$$', '')
        # Оборачиваем снова в $
        answer = f"${answer}$"
    
    # Обработка чисел 
    # Специальная обработка для отрицательных чисел
    neg_number_match = re.match(r'^-\s*(\d+[.,]?\d*)$', answer)
    if neg_number_match:
        num = neg_number_match.group(1).replace(',', '.')
        return f"$-{num}$"
    
    # Если у нас просто число, форматируем его как LaTeX
    number_match = re.match(r'^([+-]?\d*[.,]?\d+)$', answer)
    if number_match:
        # Заменяем запятые на точки для единообразия чисел
        num = number_match.group(1).replace(',', '.')
        return f"${num}$"
    
    # Если ответ уже содержит одинарные $, но внутри есть текст, очищаем текст
    if '$' in answer:
        # Извлекаем содержимое между долларами
        latex_parts = re.findall(r'\$(.*?)\$', answer)
        if latex_parts:
            # Берем первое LaTeX выражение
            cleaned_latex = latex_parts[0].strip()
            return f"${cleaned_latex}$"
    
    # Проверяем наличие дробей в форматах a/b или \frac{a}{b}
    frac_patterns = [
        r'\\frac\{([^{}]+)\}\{([^{}]+)\}',  # \frac{a}{b}
        r'(\d+)/(\d+)'                      # a/b
    ]
    
    for pattern in frac_patterns:
        frac_match = re.search(pattern, answer)
        if frac_match:
            if pattern.startswith(r'\\frac'):
                numerator, denominator = frac_match.groups()
                return f"$\\frac{{{numerator}}}{{{denominator}}}$"
            else:
                numerator, denominator = frac_match.groups()
                return f"$\\frac{{{numerator}}}{{{denominator}}}$"
    
    # Если в ответе есть специальные символы LaTeX, оборачиваем в $
    latex_symbols = ['\\', '^', '_', '{', '}', '\\sqrt', '\\pi', '\\cdot', '\\times', '\\div']
    if any(symbol in answer for symbol in latex_symbols):
        return f"${answer}$"
    
    # Для всех остальных случаев проверяем, является ли ответ числом или простым выражением
    answer_cleaned = answer.strip()
    
    # Если ответ выглядит как число (целое или с десятичной точкой)
    if re.match(r'^[+-]?\d*[.,]?\d+$', answer_cleaned):
        # Заменяем запятые на точки для единообразия
        answer_cleaned = answer_cleaned.replace(',', '.')
        return f"${answer_cleaned}$"
    
    # По умолчанию оборачиваем ответ в $, если он не пустой
    return f"${answer_cleaned}$" if answer_cleaned else "Ответ не найден"

def parse_hints(hints_string):
    """
    Разделяет строку с подсказками на 3 отдельные подсказки.
    
    Args:
        hints_string: Строка с подсказками
        
    Returns:
        list: Список из 3 подсказок
    """
    # Пустые подсказки по умолчанию, если что-то пойдет не так
    default_hints = [
        "Подсказка недоступна", 
        "Подсказка недоступна", 
        "Подсказка недоступна"
    ]
    
    if not hints_string:
        return default_hints
    
    # Ищем подсказки в формате "1. [текст]", "2. [текст]", "3. [текст]"
    hint_pattern = r'(?:^|\n)(\d+\.)\s*(.*?)(?=(?:\n\d+\.)|$)'
    hint_matches = re.findall(hint_pattern, hints_string, re.DOTALL)
    
    if not hint_matches:
        # Если не нашли подсказки в нужном формате, пробуем разделить по строкам
        lines = [line.strip() for line in hints_string.split('\n') if line.strip()]
        if len(lines) >= 3:
            # Берем первые 3 непустые строки
            return [lines[0], lines[1], lines[2]]
        elif len(lines) > 0:
            # Если строк меньше 3, возвращаем то что есть, дополняя до 3
            while len(lines) < 3:
                lines.append("Подсказка недоступна")
            return lines
        else:
            return default_hints
    
    hints = []
    
    # Обрабатываем найденные подсказки
    for number, text in hint_matches:
        # Удаляем технические комментарии в квадратных скобках
        cleaned_text = re.sub(r'\[.*?\]', '', text).strip()
        hints.append(cleaned_text)
    
    # Если мы нашли меньше 3 подсказок, добавляем заглушки
    while len(hints) < 3:
        hints.append("Подсказка недоступна")
    
    # Берем только первые 3 подсказки
    return hints[:3]

def save_to_file(content, filename="last_generated_task.txt"):
    """
    Сохраняет сгенерированный контент в файл.
    
    Args:
        content: Сгенерированный контент (текст или словарь)
        filename: Имя файла для сохранения
    """
    try:
        # Если контент - словарь, форматируем его
        if isinstance(content, dict):
            formatted_text = ""
            
            # Добавляем данные о категории и подкатегории, если они есть
            if "category" in content:
                category = content["category"]
                subcategory = content.get("subcategory", "")
                formatted_text += f"===КАТЕГОРИЯ===\n{category}"
                if subcategory:
                    formatted_text += f"\nПодкатегория: {subcategory}"
                formatted_text += "\n\n"
            
            # Добавляем текст задачи
            if "task" in content:
                task = content["task"]
                
                # Очищаем от HTML-тегов
                import re
                task_text = re.sub(r'<[^>]+>', '', task)
                
                formatted_text += f"===ЗАДАЧА===\n{task_text.strip()}\n\n"
            
            # Добавляем решение
            if "solution" in content:
                solution = content["solution"]
                
                # Очищаем от HTML-тегов
                solution_text = re.sub(r'<[^>]+>', '', solution)
                
                # Улучшаем форматирование текста решения
                # Проверяем, есть ли нумерованные шаги в решении
                if re.search(r'^\d+\.\s', solution_text, re.MULTILINE):
                    # Добавляем пустую строку между шагами для лучшей читаемости
                    solution_text = re.sub(r'(\n\d+\.\s)', r'\n\1', solution_text)
                
                # Улучшаем форматирование для задач с характеристиками или интервалами
                if 'интервал' in solution_text.lower() or 'промежуток' in solution_text.lower():
                    # Выделяем строки с интервалами, добавляя пустую строку до и после
                    solution_text = re.sub(r'([^.\n])([\(\[][-\d,;∞\s]+[;,][-\d,;∞\s]+[\)\]])', r'\1\n\2', solution_text)
                    solution_text = re.sub(r'([\(\[][-\d,;∞\s]+[;,][-\d,;∞\s]+[\)\]])([^.\n])', r'\1\n\2', solution_text)
                
                # Улучшаем отображение систем уравнений
                solution_text = re.sub(r'({)([^}]+)(})', r'\n\1\n\2\n\3\n', solution_text)
                
                formatted_text += f"===РЕШЕНИЕ===\n{solution_text.strip()}\n\n"
            
            # Добавляем подсказки
            if "hints" in content and content["hints"]:
                hints = content["hints"]
                formatted_text += "===ПОДСКАЗКИ===\n"
                
                for i, hint in enumerate(hints):
                    # Очищаем от HTML-тегов
                    hint_text = re.sub(r'<[^>]+>', '', hint)
                    formatted_text += f"Подсказка {i+1}: {hint_text.strip()}\n"
                formatted_text += "\n"
            
            # Добавляем ответ отдельно, если он есть
            if "answer" in content and content["answer"]:
                answer = content["answer"]
                
                # Очищаем от HTML-тегов
                answer_text = re.sub(r'<[^>]+>', '', answer)
                
                formatted_text += f"===ОТВЕТ===\n{answer_text.strip()}\n\n"
            
            # Добавляем уровень сложности
            if "difficulty_level" in content:
                difficulty = content["difficulty_level"]
                formatted_text += f"===УРОВЕНЬ СЛОЖНОСТИ===\n{difficulty}\n\n"
            
            # Добавляем информацию о визуализации
            if "image_path" in content:
                image_path = content["image_path"]
                formatted_text += f"===ИЗОБРАЖЕНИЕ===\n{image_path}\n\n"
                
            # Добавляем параметры визуализации
            if "visualization_params" in content and content["visualization_params"]:
                visualization_params = content["visualization_params"]
                formatted_text += f"===ПАРАМЕТРЫ ВИЗУАЛИЗАЦИИ===\n{visualization_params}\n\n"
                
            # Добавляем тип визуализации
            if "visualization_type" in content and content["visualization_type"]:
                visualization_type = content["visualization_type"]
                formatted_text += f"===ТИП ВИЗУАЛИЗАЦИИ===\n{visualization_type}\n\n"
            
            # Сохраняем форматированный текст
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(formatted_text)
                
        else:
            # Если контент - строка, просто сохраняем
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(content)
        
        return True
    except Exception as e:
        logging.error(f"Ошибка при сохранении файла {filename}: {e}")
        return False

def generate_coordinate_system(points=None, functions=None, vectors=None, grid=True, filename=None):
    """
    Эта функция перемещена в модуль app.visualization.coordinates.coordinate_system
    и импортирована выше. Сохраняем функцию для обратной совместимости.
    """
    from app.visualization.coordinates.coordinate_system import generate_coordinate_system as _generate_coordinate_system
    return _generate_coordinate_system(points, functions, vectors, grid, filename)

def generate_graph_image(function_expr, x_range=(-10, 10), y_range=None, filename=None):
    """
    Эта функция перемещена в модуль app.visualization.graphs.function_graphs
    и импортирована выше. Сохраняем функцию для обратной совместимости.
    """
    from app.visualization.graphs.function_graphs import generate_graph_image as _generate_graph_image
    return _generate_graph_image(function_expr, x_range, y_range, filename)

def get_image_base64(image_path):
    """
    Эта функция перемещена в модуль app.visualization.utils.image_utils
    и импортирована выше. Сохраняем функцию для обратной совместимости.
    """
    from app.visualization.utils.image_utils import get_image_base64 as _get_image_base64
    return _get_image_base64(image_path)

def process_visualization_params(params_text):
    """
    Обрабатывает параметры визуализации и создает изображение.
    """
    try:
        # Используем уже импортированную функцию process_viz_params
        return process_viz_params(params_text)
    except Exception as e:
        logging.error(f"Ошибка при обработке параметров визуализации: {e}")
        logging.error(traceback.format_exc())
        return None, "unknown"

def extract_visualization_params(task_text, category, subcategory):
    """
    Извлекает параметры для визуализации из текста задачи.
    
    Args:
        task_text: Текст задачи
        category: Категория задачи
        subcategory: Подкатегория задачи
        
    Returns:
        dict: Словарь с параметрами для визуализации или None, если не удалось извлечь параметры
    """
    # Определяем тип визуализации на основе текста задачи и категории
    visualization_type = determine_visualization_type(task_text, category, subcategory)
    
    if not visualization_type:
        return None
    
    # Извлекаем параметры в зависимости от типа визуализации
    if visualization_type == "graph":
        return extract_graph_params(task_text)
    elif visualization_type == "triangle":
        return extract_triangle_params(task_text)
    elif visualization_type == "circle":
        return extract_circle_params(task_text)
    elif visualization_type == "rectangle":
        return extract_rectangle_params(task_text)
    elif visualization_type == "parallelogram":
        return extract_parallelogram_params(task_text)
    elif visualization_type == "trapezoid":
        return extract_trapezoid_params(task_text)
    elif visualization_type == "coordinate":
        return extract_coordinate_params(task_text)
    else:
        return None

def extract_graph_params(task_text):
    """
    Извлекает параметры для графика функции из текста задачи.
    
    Args:
        task_text: Текст задачи
        
    Returns:
        dict: Словарь с параметрами для визуализации графика
    """
    import re
    import logging
    from app.prompts import DEFAULT_VISUALIZATION_PARAMS
    
    params = {"type": "graph"}
    
    # Ищем особые точки в тексте (например, A(1,0), B(5,0), C(3,-4))
    special_points = []
    special_points_pattern = r'([A-Z])\s*\(\s*(-?\d+(?:[,.]\d+)?)\s*,\s*(-?\d+(?:[,.]\d+)?)\s*\)'
    special_point_matches = re.findall(special_points_pattern, task_text)
    for match in special_point_matches:
        label, x, y = match
        x = float(x.replace(',', '.'))
        y = float(y.replace(',', '.'))
        special_points.append((x, y, label))
    
    # Добавляем найденные точки к параметрам
    if special_points:
        params["special_points"] = special_points
    
    # Ищем конкретные значения параметров, если они заданы
    parameters = {}
    parameter_matches = re.findall(r'([a-zA-Z])\s*=\s*(-?\d+(?:[,.]\d+)?)', task_text)
    for param, value in parameter_matches:
        parameters[param] = float(value.replace(',', '.'))
    
    # Также ищем параметры в прямых условиях, например "при a = 2"
    additional_param_patterns = [
        r'при\s+([a-zA-Z])\s*=\s*(-?\d+(?:[,.]\d+)?)',
        r'значени[ие][^\.]+?параметра\s+([a-zA-Z])[^\.]+?равн[оа][^\.]+?(-?\d+(?:[,.]\d+)?)',
        r'параметр\s+([a-zA-Z])[^\.]+?равен\s+(-?\d+(?:[,.]\d+)?)'
    ]
    
    for pattern in additional_param_patterns:
        param_matches = re.findall(pattern, task_text, re.IGNORECASE)
        for param, value in param_matches:
            if param not in parameters:  # Избегаем перезаписи уже найденных значений
                try:
                    parameters[param] = float(value.replace(',', '.'))
                except ValueError:
                    continue
    
    # Ищем функции в разных форматах
    function_patterns = [
        # Простой формат y = [выражение] или f(x) = [выражение]
        r'y\s*=\s*([-+0-9a-zA-Z\^\*\/\(\)\s\{\}\[\]\.,]+)(?=[,.:;)]|\s*$|\n)',
        r'f\s*\(\s*x\s*\)\s*=\s*([-+0-9a-zA-Z\^\*\/\(\)\s\{\}\[\]\.,]+)(?=[,.:;)]|\s*$|\n)',
        
        # Формат с определённой функцией f(x) = [выражение] или g(x) = [выражение]
        r'([a-zA-Z])\s*\(\s*x\s*\)\s*=\s*([-+0-9a-zA-Z\^\*\/\(\)\s\{\}\[\]\.,]+)(?=[,.:;)]|\s*$|\n|\$)',
        
        # Поиск выражений в LaTeX
        r'\$([a-zA-Z])\s*\(\s*x\s*\)\s*=\s*([-+0-9a-zA-Z\^\*\/\(\)\s\{\}\[\]\.,\\]+)\$'
    ]
    
    functions = []
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, pattern in enumerate(function_patterns):
        matches = re.findall(pattern, task_text)
        for j, match in enumerate(matches):
            if i <= 1:  # Формат без имени функции (y= или f(x)=)
                func_expr = match.strip()
                if i == 0:  # y = expr
                    func_name = 'y'
                else:  # f(x) = expr
                    func_name = 'f(x)'
            else:  # Формат с именем функции
                func_name, func_expr = match
                func_name = f"{func_name}(x)"
                
            # Очищаем выражение от лишних пробелов
            func_expr = func_expr.strip()
            
            # Преобразуем выражение для Python
            func_expr = func_expr.replace('^', '**')
            
            # Заменяем различные LaTeX-символы на Python-эквиваленты
            func_expr = func_expr.replace('\\cdot', '*').replace('\\frac{', '(').replace('}{', ')/(').replace('}', ')')
            func_expr = func_expr.replace('\\sqrt', 'sqrt')
            
            # Выбираем цвет для функции
            color_idx = j % len(colors)
            color = colors[color_idx]
            
            # Проверяем, не добавляли ли мы уже эту функцию
            if not any(f[0] == func_expr for f in functions):
                functions.append((func_expr, color, func_name))
    
    # Ищем функции из секции с параметрами визуализации
    param_section_pattern = r'===ПАРАМЕТРЫ ВИЗУАЛИЗАЦИИ===\s*(.*?)(?===|$)'
    param_section_match = re.search(param_section_pattern, task_text, re.DOTALL)
    
    if param_section_match:
        param_section = param_section_match.group(1)
        
        # Ищем функцию в параметрах визуализации
        func_pattern = r'Функция\s+(\d+):\s*(.*?)(?=Цвет|Название|Диапазон|$)'
        func_matches = re.findall(func_pattern, param_section, re.DOTALL)
        
        # Ищем цвета
        color_pattern = r'Цвет\s+(\d+):\s*(.*?)(?=Функция|Название|Диапазон|$)'
        color_matches = re.findall(color_pattern, param_section, re.DOTALL)
        color_dict = {int(num): color.strip() for num, color in color_matches}
        
        # Ищем названия
        name_pattern = r'Название\s+(\d+):\s*(.*?)(?=Функция|Цвет|Диапазон|$)'
        name_matches = re.findall(name_pattern, param_section, re.DOTALL)
        name_dict = {int(num): name.strip() for num, name in name_matches}
        
        param_functions = []
        for func_match in func_matches:
            num = int(func_match[0])
            func_expr = func_match[1].strip().replace('^', '**')
            
            # Получаем цвет и имя для этой функции
            color = color_dict.get(num, colors[min(num-1, len(colors)-1)])
            
            # Конвертируем русские названия цветов в английские
            color_mapping = {
                'красный': 'red',
                'синий': 'blue',
                'зеленый': 'green',
                'зелёный': 'green',
                'желтый': 'yellow',
                'жёлтый': 'yellow',
                'черный': 'black',
                'чёрный': 'black',
                'фиолетовый': 'purple',
                'оранжевый': 'orange',
                'коричневый': 'brown',
                'розовый': 'pink',
                'серый': 'gray',
                'голубой': 'cyan',
                'малиновый': 'magenta'
            }
            
            # Преобразуем название цвета, если оно на русском
            if color.lower() in color_mapping:
                color = color_mapping[color.lower()]
                
            name = name_dict.get(num, f"f_{num}(x)")
            
            param_functions.append((func_expr, color, name))
        
        # Если нашли хотя бы одну функцию в параметрах, используем её вместо найденных в тексте задачи
        if param_functions:
            functions = param_functions
            
            # Логируем найденные функции для отладки
            for i, (func, color, name) in enumerate(functions):
                logging.info(f"Найдены параметры для функции {i+1}: {func}, цвет: {color}, метка: {name}")
            
            # Добавляем список функций в параметры
            params['functions'] = functions
            
            # Проверяем особые точки
            special_points_pattern = r'Особые точки:\s*(.*?)(?===|$)'
            special_points_match = re.search(special_points_pattern, param_section, re.DOTALL)
            
            if special_points_match:
                try:
                    special_points_str = special_points_match.group(1).strip()
                    # Разделяем по запятым, но не внутри скобок
                    import re
                    # Разделяем точки, учитывая возможные скобки и запятые внутри координат
                    points_list = re.findall(r'\(([^)]+)\)', special_points_str)
                    
                    special_points = []
                    import logging
                    logging.info(f"Найдены точки в параметрах: {points_list}")
                    
                    for point_str in points_list:
                        try:
                            # Разделяем по запятой на x, y, label
                            parts = point_str.split(',', 2)
                            
                            if len(parts) >= 2:
                                x_expr = parts[0].strip()
                                y_expr = parts[1].strip()
                                label = parts[2].strip() if len(parts) > 2 else ""
                                
                                # Обрабатываем выражения вида "1 + sqrt(3)/3"
                                try:
                                    # Заменяем математические выражения на Python-синтаксис
                                    x_expr = x_expr.replace('^', '**').replace('sqrt', 'math.sqrt')
                                    
                                    logging.info(f"Пытаемся вычислить x_expr: {x_expr}")
                                    
                                    # Если выражение содержит математические функции
                                    if any(func in x_expr for func in ['math.', 'sqrt', 'sin', 'cos']):
                                        import math
                                        x_val = eval(x_expr)
                                        logging.info(f"Вычислили x_val = {x_val}")
                                    else:
                                        x_val = float(x_expr)
                                    
                                    # Если y_expr содержит f(x), вычисляем значение функции
                                    if 'f(' in y_expr:
                                        # Получаем функцию из списка
                                        if functions:
                                            func_expr = functions[0][0]
                                            y_val = eval(func_expr.replace('x', f'({x_val})'))
                                            logging.info(f"Вычислили значение функции y_val = {y_val}")
                                        else:
                                            logging.warning(f"Не удалось вычислить значение функции для точки ({x_expr}, {y_expr})")
                                            continue
                                    else:
                                        # Обрабатываем y так же, как x
                                        y_expr = y_expr.replace('^', '**').replace('sqrt', 'math.sqrt')
                                        
                                        if any(func in y_expr for func in ['math.', 'sqrt', 'sin', 'cos']):
                                            import math
                                            y_val = eval(y_expr)
                                        else:
                                            y_val = float(y_expr)
                                    
                                    # Добавляем точку с вычисленными координатами
                                    special_points.append((x_val, y_val, label))
                                    logging.info(f"Добавлена точка: ({x_val}, {y_val}, {label})")
                                except Exception as e:
                                    logging.warning(f"Ошибка при разборе особых точек: {e}")
                                    continue
                                
                        except Exception as e:
                            logging.warning(f"Ошибка при обработке точки '{point_str}': {e}")
                    
                    if special_points:
                        params['special_points'] = special_points
                        logging.info(f"Найдены особые точки: {special_points}")
                except Exception as e:
                    logging.warning(f"Ошибка при обработке списка особых точек: {e}")
            
            # Проверяем диапазоны осей
            x_range_pattern = r'Диапазон X:\s*\$?\[(.*?)\]\$?'
            x_range_match = re.search(x_range_pattern, param_section)
            
            if x_range_match:
                try:
                    x_range_str = x_range_match.group(1).strip()
                    # Убираем все доллары и другие специальные символы
                    x_range_str = x_range_str.replace('$', '').strip()
                    # Разделяем по запятой, обрабатывая возможные пробелы
                    parts = [part.strip() for part in x_range_str.split(',')]
                    if len(parts) == 2:
                        x_min = float(parts[0])
                        x_max = float(parts[1])
                        params['x_range'] = (x_min, x_max)
                except Exception as e:
                    logging.warning(f"Ошибка при разборе диапазона X: {e}")
            
            y_range_pattern = r'Диапазон Y:\s*\$?\[(.*?)\]\$?'
            y_range_match = re.search(y_range_pattern, param_section)
            
            if y_range_match:
                try:
                    y_range_str = y_range_match.group(1).strip()
                    # Убираем все доллары и другие специальные символы
                    y_range_str = y_range_str.replace('$', '').strip()
                    
                    if y_range_str.lower() in ['автоматический', 'auto']:
                        params['y_range'] = None
                    else:
                        # Разделяем по запятой, обрабатывая возможные пробелы
                        parts = [part.strip() for part in y_range_str.split(',')]
                        if len(parts) == 2:
                            y_min = float(parts[0])
                            y_max = float(parts[1])
                            params['y_range'] = (y_min, y_max)
                except Exception as e:
                    logging.warning(f"Ошибка при разборе диапазона Y: {e}")
    
    # Поиск функций в мультифункциональном формате
    multi_func_pattern = r'Рассмотрим\s+две\s+функции\s*:\s*([a-zA-Z])\s*\(\s*x\s*\)\s*=\s*([^,]+)\s*,\s*([a-zA-Z])\s*\(\s*x\s*\)\s*=\s*([^,\.]+)'
    multi_func_match = re.search(multi_func_pattern, task_text, re.IGNORECASE)
    
    # Еще один формат: "На графике изображены две функции: $f(x) = ...$"
    if not multi_func_match:
        graph_func_match = re.search(r'На графике изображены (?:две|несколько) функции:\s*\$([a-zA-Z])\s*\(\s*x\s*\)\s*=\s*([^$]+)\$\s*и\s*\$([a-zA-Z])\s*\(\s*x\s*\)\s*=\s*([^$]+)\$', task_text, re.IGNORECASE | re.DOTALL)
        if not graph_func_match:
            # Попробуем более общее выражение
            func_expr_pattern = r'\$([a-zA-Z])\s*\(\s*x\s*\)\s*=\s*(.+?)\$'
            func_matches = re.findall(func_expr_pattern, task_text)
            if len(func_matches) >= 2 and "две функции" in task_text.lower():
                func1_name, func1_expr = func_matches[0]
                func2_name, func2_expr = func_matches[1]
                
                # Заменяем параметры в выражениях их значениями
                # Также заменяем LaTeX-команды и символы на Python-синтаксис
                func1_expr = func1_expr.replace('^', '**').replace('\\sqrt', 'sqrt')
                func2_expr = func2_expr.replace('^', '**').replace('\\sqrt', 'sqrt')
                
                # Если в функциях есть специальные математические функции, добавляем их импорт
                if 'sqrt' in func1_expr or 'sqrt' in func2_expr:
                    params['imports'] = 'from math import sqrt'
                
                # Поиск значений функций в тексте
                val_pattern = r'\$?%s\s*\(\s*(\d+(?:[,.]\d+)?)\s*\)\s*=\s*(-?\d+(?:[,.]\d+)?)\$?'
                f1_vals = re.findall(val_pattern % func1_name, task_text)
                f2_vals = re.findall(val_pattern % func2_name, task_text)
                
                # Если нашли значения, проверим корректность функций
                if f1_vals:
                    x_val, expected_y = f1_vals[0]
                    x_val = float(x_val.replace(',', '.'))
                    expected_y = float(expected_y.replace(',', '.'))
                    
                    # Преобразуем выражение и подставляем x
                    try:
                        actual_y = eval(func1_expr.replace('x', str(x_val)))
                        if abs(actual_y - expected_y) > 0.01:
                            # Если результат не совпадает, возможно, функция была задана неверно
                            logging.warning(f"Несоответствие значений для функции {func1_name}: ожидается {expected_y}, получено {actual_y}")
                    except Exception as e:
                        logging.warning(f"Ошибка при вычислении функции {func1_name}: {e}")
                
                # Добавляем функции с разными цветами
                functions.append((func1_expr, 'blue', func1_name))
                functions.append((func2_expr, 'red', func2_name))
        else:
            func1_name, func1_expr = graph_func_match.group(1), graph_func_match.group(2).strip()
            func2_name, func2_expr = graph_func_match.group(3), graph_func_match.group(4).strip()
            
            # Заменяем параметры в выражениях их значениями, если они известны
            # Также заменяем LaTeX-команды и символы на Python-синтаксис
            func1_expr = func1_expr.replace('^', '**').replace('\\sqrt', 'sqrt')
            func2_expr = func2_expr.replace('^', '**').replace('\\sqrt', 'sqrt')
            
            # Если в функциях есть специальные математические функции, добавляем их импорт
            if 'sqrt' in func1_expr or 'sqrt' in func2_expr:
                params['imports'] = 'from math import sqrt'
            
            # Добавляем функции с разными цветами
            functions.append((func1_expr, 'blue', func1_name))
            functions.append((func2_expr, 'red', func2_name))
    else:
        func1_name, func1_expr = multi_func_match.group(1), multi_func_match.group(2).strip()
        func2_name, func2_expr = multi_func_match.group(3), multi_func_match.group(4).strip()
        
        # Заменяем параметры в выражениях их значениями, если они известны
        # Также заменяем LaTeX-команды и символы на Python-синтаксис
        func1_expr = func1_expr.replace('^', '**').replace('\\sqrt', 'sqrt')
        func2_expr = func2_expr.replace('^', '**').replace('\\sqrt', 'sqrt')
        
        # Если в функциях есть специальные математические функции, добавляем их импорт
        if 'sqrt' in func1_expr or 'sqrt' in func2_expr:
            params['imports'] = 'from math import sqrt'
        
        # Добавляем функции с разными цветами
        functions.append((func1_expr, 'blue', func1_name))
        functions.append((func2_expr, 'red', func2_name))
    
    # Поиск точек на плоскости
    # Часто в задачах указывают набор точек или интервалы на оси X
    point_labels_pattern = r'точк[иа]\s+([A-Za-z,\s]+)\s+(на оси|на графике)'
    point_labels = re.findall(point_labels_pattern, task_text, re.IGNORECASE)
    
    if not special_points and point_labels:
        special_points = []
        
        if point_labels:
            labels = re.findall(r'([a-zA-Z])', point_labels[0])
            
            # Создаем специальные точки 
            
            # Если это точки на графике функции (A, B, C, D), устанавливаем координаты на графике функции
            is_on_function_graph = False
            for label in labels:
                if label.isupper():  # Если точки обозначены заглавными буквами, считаем, что они на графике
                    is_on_function_graph = True
                    break
            
            if is_on_function_graph:
                # Если нашли функцию, используем её для создания точек
                if functions:
                    func_expr = functions[0][0]
                    
                    # Создаем точки равномерно на интервале
                    for i, label in enumerate(labels):
                        x_val = -5 + i * 2  # Равномерное распределение начиная с x=-5
                        try:
                            # Вычисляем значение функции в этой точке
                            import math
                            y_val = eval(func_expr.replace('x', str(x_val)))
                            special_points.append((x_val, y_val, label))
                        except Exception as e:
                            logging.warning(f"Ошибка при вычислении значения функции для точки {label}: {e}")
            else:
                # Для обозначений на оси x (a, b, c, d, e, f)
                x_values = []
                if 'a' in labels:
                    x_values.append(-1)  # a
                if 'b' in labels:
                    x_values.append(1)   # b
                if 'c' in labels:
                    x_values.append(3)   # c
                if 'd' in labels:
                    x_values.append(5)   # d
                if 'e' in labels:
                    x_values.append(7)   # e
                if 'f' in labels:
                    x_values.append(9)   # f
                
                for i, label in enumerate(labels):
                    if i < len(x_values):
                        x_val = x_values[i]
                        special_points.append((x_val, 0, label))  # Точки на оси X
        
        if special_points:
            params['special_points'] = special_points
    
    # Если нашли хотя бы одну функцию, добавляем их в параметры
    if functions:
        params['functions'] = functions
        # Также добавляем первую функцию как основную для обратной совместимости
        params['function'] = functions[0][0]
    else:
        # Если не нашли функции, но есть явное упоминание функции в тексте
        function_keywords = ['функция', 'график функции', 'парабола', 'гипербола']
        if any(keyword in task_text.lower() for keyword in function_keywords):
            # Ищем любые математические выражения
            math_expr_match = re.search(r'y\s*=\s*([-+0-9a-zA-Z\^\*\/\(\)\s\{\}\[\]\.,]+)', task_text)
            if math_expr_match:
                func_expr = math_expr_match.group(1).strip().replace('^', '**')
                params['function'] = func_expr
            else:
                # Если в тексте есть упоминание квадратичной функции, используем параболу
                if any(keyword in task_text.lower() for keyword in ['квадратичн', 'парабол', 'вверх', 'вниз']):
                    params['function'] = 'x**2' if 'вверх' in task_text.lower() else '-x**2'
                # Если линейная функция
                elif any(keyword in task_text.lower() for keyword in ['линейн', 'прям']):
                    params['function'] = 'x'
                # По умолчанию - парабола
                else:
                    params['function'] = 'x**2'
        else:
            # Если не нашли функции, но мы уже нашли функции в разделе параметров визуализации, 
            # НЕ переопределяем функцию по умолчанию
            if not params.get('functions') and not params.get('function'):
                logging.warning("Не удалось извлечь ни одной функции, использую функцию по умолчанию")
                params['function'] = 'x**2'
    
    # Обработка задач с анализом графиков функций, где функция не задана явно
    if not functions and "график функции" in task_text.lower():
        # Проверяем, есть ли ключевые фразы, указывающие на задачу с анализом графика
        analysis_keywords = [
            "изображ[её]н график функции",
            "точки.+задают.+интервалы",
            "характеристик[аи] функции",
            "производн[ойая] функции",
            "знач[ениея] функции", 
            "интервал[аыу]",
            "[Тт]очк[иа]\s+[ABCD]", # Точки A, B, C, D на графике
            "[Тт]очк[иа]\s+[A-D].*[,\sи][^\\)]*[A-D]"  # Точки A, B, C, или D, перечисленные через запятую или "и"
        ]
        
        is_analysis_task = any(re.search(pattern, task_text, re.IGNORECASE) for pattern in analysis_keywords)
        
        if is_analysis_task:
            # Создаем стандартную функцию для задачи с анализом свойств функции
            # Используем кубическую функцию, которая имеет интересные свойства
            
            # Если задача про точки A, B, C, D, используем функцию с четкими максимумами и минимумами
            if "точк" in task_text.lower() and any(point in task_text for point in "ABCD"):
                # Функция для задач с точками A, B, C, D с четко выраженными экстремумами
                # x^3 - 3x^2 имеет минимум и максимум в точках x=0 и x=2
                func_expr = "x**3 - 3*x**2 + 2*x"
            else:
                # Для других задач с анализом функции
                func_expr = "x**3 - 6*x**2 + 9*x - 4"
            functions.append((func_expr, 'blue', 'f'))
            
            point_labels = re.findall(r'точк[иа]\s+([A-Za-z,\s]+)', task_text, re.IGNORECASE)
            
            special_points = []
            
            if point_labels:
                labels = re.findall(r'([a-zA-Z])', point_labels[0])
                
                # Создаем специальные точки 
                
                # Если это точки на графике функции (A, B, C, D), устанавливаем координаты на графике функции
                is_on_function_graph = False
                for label in labels:
                    if label.isupper():  # Если точки обозначены заглавными буквами, считаем, что они на графике
                        is_on_function_graph = True
                        break
                
                if is_on_function_graph:
                    # Если нашли функцию, используем её для создания точек
                    func_expr = functions[0][0]
                    
                    # Создаем точки равномерно на интервале
                    for i, label in enumerate(labels):
                        x_val = -1 + i * 2  # Равномерное распределение начиная с x=-1
                        try:
                            # Вычисляем значение функции в этой точке
                            import math
                            y_val = eval(func_expr.replace('x', str(x_val)))
                            special_points.append((x_val, y_val, label))
                        except Exception as e:
                            logging.warning(f"Ошибка при вычислении значения функции для точки {label}: {e}")
                else:
                    # Для обозначений на оси x (a, b, c, d, e, f)
                    x_values = []
                    if 'a' in labels:
                        x_values.append(-1)  # a
                    if 'b' in labels:
                        x_values.append(1)   # b
                    if 'c' in labels:
                        x_values.append(3)   # c
                    if 'd' in labels:
                        x_values.append(5)   # d
                    if 'e' in labels:
                        x_values.append(7)   # e
                    if 'f' in labels:
                        x_values.append(9)   # f
                    
                    for i, label in enumerate(labels):
                        if i < len(x_values):
                            x_val = x_values[i]
                            special_points.append((x_val, 0, label))  # Точки на оси X
            
            if special_points:
                params['special_points'] = special_points
    
    # Добавляем результаты обработки в params
    if functions:
        params['functions'] = functions
        
    # Стандартные значения для диапазонов осей
    if not params.get('x_range'):
        params['x_range'] = (-10, 10)
    if not params.get('y_range'):
        params['y_range'] = (-10, 10)
    
    return params

def extract_triangle_params(task_text):
    """
    Извлекает параметры для треугольника из текста задачи.
    
    Args:
        task_text: Текст задачи
        
    Returns:
        dict: Словарь с параметрами для визуализации треугольника
    """
    import re
    from app.prompts import DEFAULT_VISUALIZATION_PARAMS
    
    params = DEFAULT_VISUALIZATION_PARAMS["triangle"].copy()
    params["type"] = "triangle"
    
    # Ищем координаты вершин в формате A(x1,y1), B(x2,y2), C(x3,y3)
    coord_pattern = r'([A-Z])\s*\((-?\d+(?:[,.]\d+)?)\s*[;,]\s*(-?\d+(?:[,.]\d+)?)\)'
    coord_matches = re.findall(coord_pattern, task_text)
    
    if coord_matches and len(coord_matches) >= 3:
        points = []
        vertex_labels = []
        
        for match in coord_matches[:3]:  # Берем только первые три вершины
            label, x, y = match
            vertex_labels.append(label)
            points.append((float(x.replace(',', '.')), float(y.replace(',', '.'))))
        
        params['points'] = points
        params['vertex_labels'] = vertex_labels
    
    # Определяем, является ли треугольник прямоугольным
    is_right = "прямоугольный треугольник" in task_text.lower() or "прямоугольного треугольника" in task_text.lower()
    params['is_right'] = is_right
    
    # Определяем, нужно ли показывать углы
    params['show_angles'] = "угол" in task_text.lower() or "углы" in task_text.lower() or "градус" in task_text.lower()
    
    return params

def extract_circle_params(task_text):
    """
    Извлекает параметры для окружности из текста задачи.
    
    Args:
        task_text: Текст задачи
        
    Returns:
        dict: Словарь с параметрами для визуализации окружности
    """
    import re
    from app.prompts import DEFAULT_VISUALIZATION_PARAMS
    
    params = DEFAULT_VISUALIZATION_PARAMS["circle"].copy()
    params["type"] = "circle"
    
    # Ищем центр окружности
    center_pattern = r'центр\s*([A-Z])\s*\((-?\d+(?:[,.]\d+)?)\s*[;,]\s*(-?\d+(?:[,.]\d+)?)\)'
    center_matches = re.findall(center_pattern, task_text, re.IGNORECASE)
    
    if center_matches:
        label, x, y = center_matches[0]
        params['center'] = (float(x.replace(',', '.')), float(y.replace(',', '.')))
        params['center_label'] = label
    
    # Ищем радиус окружности
    radius_pattern = r'радиус\s*[=:]\s*(\d+(?:[,.]\d+)?)'
    radius_matches = re.findall(radius_pattern, task_text, re.IGNORECASE)
    
    if radius_matches:
        params['radius'] = float(radius_matches[0].replace(',', '.'))
    
    # Определяем, нужно ли показывать радиус или диаметр
    params['show_radius'] = "радиус" in task_text.lower()
    params['show_diameter'] = "диаметр" in task_text.lower()
    
    return params

def extract_rectangle_params(task_text):
    """
    Извлекает параметры для прямоугольника из текста задачи.
    
    Args:
        task_text: Текст задачи
        
    Returns:
        dict: Словарь с параметрами для визуализации прямоугольника
    """
    import re
    from app.prompts import DEFAULT_VISUALIZATION_PARAMS
    
    params = DEFAULT_VISUALIZATION_PARAMS["rectangle"].copy()
    params["type"] = "rectangle"
    
    # Ищем размеры прямоугольника
    dim_patterns = [
        r'прямоугольник[а-я]*\s*со\s*сторонами\s*(\d+(?:[,.]\d+)?)\s*и\s*(\d+(?:[,.]\d+)?)',
        r'длина\s*[=:]\s*(\d+(?:[,.]\d+)?)[^,;.]*ширина\s*[=:]\s*(\d+(?:[,.]\d+)?)',
        r'ширина\s*[=:]\s*(\d+(?:[,.]\d+)?)[^,;.]*длина\s*[=:]\s*(\d+(?:[,.]\d+)?)'
    ]
    
    for pattern in dim_patterns:
        dim_matches = re.findall(pattern, task_text, re.IGNORECASE)
        if dim_matches:
            try:
                width = float(dim_matches[0][0].replace(',', '.'))
                height = float(dim_matches[0][1].replace(',', '.'))
                params['width'] = width
                params['height'] = height
                break
            except:
                pass
    
    # Определяем, нужно ли показывать размеры
    params['show_lengths'] = "длина" in task_text.lower() or "ширина" in task_text.lower() or "сторона" in task_text.lower()
    
    return params

def extract_parallelogram_params(task_text):
    """
    Извлекает параметры для параллелограмма из текста задачи.
    
    Args:
        task_text: Текст задачи
        
    Returns:
        dict: Словарь с параметрами для визуализации параллелограмма
    """
    import re
    from app.prompts import DEFAULT_VISUALIZATION_PARAMS
    
    params = DEFAULT_VISUALIZATION_PARAMS["parallelogram"].copy()
    params["type"] = "parallelogram"
    
    # Ищем размеры параллелограмма
    dim_patterns = [
        r'параллелограмм[а-я]*\s*со\s*сторонами\s*(\d+(?:[,.]\d+)?)\s*и\s*(\d+(?:[,.]\d+)?)',
        r'сторон[а-я]\s*[=:]\s*(\d+(?:[,.]\d+)?)[^,;.]*сторон[а-я]\s*[=:]\s*(\d+(?:[,.]\d+)?)'
    ]
    
    for pattern in dim_patterns:
        dim_matches = re.findall(pattern, task_text, re.IGNORECASE)
        if dim_matches:
            try:
                width = float(dim_matches[0][0].replace(',', '.'))
                height = float(dim_matches[0][1].replace(',', '.'))
                params['width'] = width
                params['height'] = height
                break
            except:
                pass
    
    # Ищем угол наклона
    skew_pattern = r'угол\s*[=:]\s*(\d+(?:[,.]\d+)?)\s*(?:градус|°)'
    skew_matches = re.findall(skew_pattern, task_text, re.IGNORECASE)
    
    if skew_matches:
        params['skew'] = float(skew_matches[0].replace(',', '.'))
    
    # Определяем, нужно ли показывать размеры и углы
    params['show_lengths'] = "сторона" in task_text.lower() or "сторон" in task_text.lower()
    params['show_angles'] = "угол" in task_text.lower() or "углы" in task_text.lower()
    
    return params

def extract_trapezoid_params(task_text):
    """
    Извлекает параметры для трапеции из текста задачи.
    
    Args:
        task_text: Текст задачи
        
    Returns:
        dict: Словарь с параметрами для визуализации трапеции
    """
    import re
    from app.prompts import DEFAULT_VISUALIZATION_PARAMS
    
    params = DEFAULT_VISUALIZATION_PARAMS["trapezoid"].copy()
    params["type"] = "trapezoid"
    
    # Ищем размеры трапеции с более точными регулярными выражениями
    base_patterns = [
        r'основания\s*равнобедренной\s*трапеции\s*равны\s*(\d+(?:[,.]\d+)?)\s*и\s*(\d+(?:[,.]\d+)?)',
        r'трапеци[а-я]*\s*с\s*основаниями\s*(\d+(?:[,.]\d+)?)\s*и\s*(\d+(?:[,.]\d+)?)',
        r'основани[а-я]\s*(?:трапеции\s*)?равны\s*(\d+(?:[,.]\d+)?)\s*и\s*(\d+(?:[,.]\d+)?)',
        r'основани[а-я][\s:]*(\d+(?:[,.]\d+)?)\s*и\s*(\d+(?:[,.]\d+)?)',
        r'основани[а-я]\s*[=:]\s*(\d+(?:[,.]\d+)?)[^,;.]*(?:втор[а-я]+|друг[а-я]+)\s*основани[а-я]\s*[=:]\s*(\d+(?:[,.]\d+)?)'
    ]
    
    for pattern in base_patterns:
        base_matches = re.findall(pattern, task_text, re.IGNORECASE)
        if base_matches:
            try:
                # Преобразуем строки в числа, заменяя запятые на точки
                val1 = float(base_matches[0][0].replace(',', '.'))
                val2 = float(base_matches[0][1].replace(',', '.'))
                # Нижнее основание обычно больше верхнего
                bottom = max(val1, val2)
                top = min(val1, val2)
                params['bottom_width'] = bottom
                params['top_width'] = top
                print(f"Обнаружены основания трапеции: {bottom} и {top}")
                break
            except Exception as e:
                print(f"Ошибка при преобразовании оснований: {e}")
                pass
    
    # Специальная проверка для задачи с равнобедренной трапецией с основаниями 10 и 4
    special_pattern = r'основания\s*равнобедренной\s*трапеции\s*равны\s*10\s*и\s*4'
    if re.search(special_pattern, task_text, re.IGNORECASE):
        params['bottom_width'] = 10
        params['top_width'] = 4
        print("Обнаружена специальная задача с основаниями 10 и 4")
    
    # Распознаем равнобедренную трапецию
    is_isosceles = False
    if re.search(r'равнобедренн[а-я]*\s*трапеци', task_text, re.IGNORECASE):
        is_isosceles = True
        params['is_isosceles'] = True
        
    # Ищем высоту трапеции
    height_pattern = r'высота\s*[=:]\s*(\d+(?:[,.]\d+)?)'
    height_matches = re.findall(height_pattern, task_text, re.IGNORECASE)
    
    if height_matches:
        params['height'] = float(height_matches[0].replace(',', '.'))
    
    # Ищем информацию об описанной окружности
    radius_pattern = r'радиус\s*описанной\s*окружности\s*[=:]\s*(\d+(?:[,.]\d+)?)'
    radius_matches = re.findall(radius_pattern, task_text, re.IGNORECASE)
    
    if radius_matches:
        # Для визуализации трапеции с описанной окружностью
        # можем сохранить эту информацию, но не будем отображать саму окружность
        params['radius'] = float(radius_matches[0].replace(',', '.'))
    
    # Определяем, какие параметры надо показывать
    params['show_lengths'] = "основани" in task_text.lower() or "сторон" in task_text.lower()
    
    # Для задачи с основаниями показываем только основания, а не все стороны
    if params['show_lengths'] and "основани" in task_text.lower():
        params['show_specific_sides'] = ["AB", "DC"]
    
    return params

def extract_coordinate_params(task_text):
    """
    Извлекает параметры для координатной плоскости из текста задачи.
    
    Args:
        task_text: Текст задачи
        
    Returns:
        dict: Словарь с параметрами для визуализации координатной плоскости
    """
    import re
    
    params = {"type": "coordinate"}
    
    # Ищем точки на координатной плоскости
    point_pattern = r'([A-Z])\s*\((-?\d+(?:[,.]\d+)?)\s*[;,]\s*(-?\d+(?:[,.]\d+)?)\)'
    point_matches = re.findall(point_pattern, task_text)
    
    if point_matches:
        points = []
        for match in point_matches:
            label, x, y = match
            points.append((float(x.replace(',', '.')), float(y.replace(',', '.')), label))
        params['points'] = points
    
    # Ищем функции
    function_patterns = [
        r'y\s*=\s*([-+0-9x\^\*\/\(\)\s]+)(?=[,.:;)]|\s*$|\n)',
        r'f\s*\(\s*x\s*\)\s*=\s*([-+0-9x\^\*\/\(\)\s]+)(?=[,.:;)]|\s*$|\n)'
    ]
    
    functions = []
    for pattern in function_patterns:
        function_matches = re.findall(pattern, task_text)
        for match in function_matches:
            function_expr = match.strip()
            # Заменяем ^ на ** для Python
            function_expr = function_expr.replace('^', '**')
            functions.append((function_expr, 'blue'))
    
    if functions:
        params['functions'] = functions
    
    # Ищем векторы
    vector_pattern = r'вектор\s*([A-Z]{2})'
    vector_matches = re.findall(vector_pattern, task_text, re.IGNORECASE)
    
    if vector_matches and 'points' in params:
        vectors = []
        point_dict = {p[2]: (p[0], p[1]) for p in params['points']}
        
        for match in vector_matches:
            vector_name = match
            if len(vector_name) == 2 and vector_name[0] in point_dict and vector_name[1] in point_dict:
                start_point = point_dict[vector_name[0]]
                end_point = point_dict[vector_name[1]]
                vectors.append((start_point[0], start_point[1], end_point[0], end_point[1], vector_name))
        
        if vectors:
            params['vectors'] = vectors
    
    return params

def create_image_from_params(params):
    """
    Создает изображение на основе параметров визуализации.
    
    Args:
        params: Словарь с параметрами для визуализации
        
    Returns:
        str: Путь к созданному изображению или None, если не удалось создать
    """
    try:
        if not params:
            return None
        
        viz_type = params.get("type")
        if not viz_type:
            return None
            
        if viz_type == "graph":
            # Для графика нужна функция и диапазон X
            if "function" in params:
                # Одиночная функция
                function_expr = params["function"]
                x_range = params.get("x_range", (-10, 10))
                y_range = params.get("y_range")
                
                # Проверяем, есть ли параметр functions, и если есть, используем его вместо одиночной функции
                if "functions" in params:
                    import logging
                    logging.info(f"Обнаружен список функций, используем его вместо одиночной функции: {params['functions']}")
                    functions = params["functions"]
                    special_points = params.get("special_points")
                    return generate_multi_function_graph(functions, x_range, y_range, special_points)
                
                # Если нет списка функций, используем одиночную функцию
                import logging
                logging.info(f"Используем одиночную функцию: {function_expr}")
                return generate_graph_image(function_expr, x_range, y_range)
                
            elif "functions" in params:
                # Множественные функции (например, для нахождения точек пересечения)
                functions = params["functions"]
                import logging
                logging.info(f"Используем список функций: {functions}")
                x_range = params.get("x_range", (-10, 10))
                y_range = params.get("y_range")
                special_points = params.get("special_points")
                return generate_multi_function_graph(functions, x_range, y_range, special_points)
            else:
                import logging
                logging.warning("Не найдено ни одной функции в параметрах")
                return None
        
        elif viz_type == "triangle":
            return generate_geometric_figure("triangle", params)
            
        elif viz_type == "rectangle":
            return generate_geometric_figure("rectangle", params)
            
        elif viz_type == "parallelogram":
            return generate_geometric_figure("parallelogram", params)
            
        elif viz_type == "trapezoid":
            return generate_geometric_figure("trapezoid", params)
            
        elif viz_type == "circle":
            return generate_geometric_figure("circle", params)
            
        elif viz_type == "coordinate":
            points = params.get("points")
            functions = params.get("functions")
            vectors = params.get("vectors")
            return generate_coordinate_system(points, functions, vectors)
            
        return None
        
    except Exception as e:
        logging.error(f"Ошибка при создании изображения: {e}")
        traceback.print_exc()
        return None

def generate_complete_task(category, subcategory="", difficulty_level=3, is_basic_level=False):
    """
    Генерирует полную задачу, решение и подсказки с использованием YandexGPT.
    
    Args:
        category: Категория задачи
        subcategory: Подкатегория задачи (опционально)
        difficulty_level: Уровень сложности подсказок (1-5)
        is_basic_level: Выбор между базовым (True) и профильным (False) уровнем ЕГЭ
        
    Returns:
        dict: Словарь с задачей, решением, подсказками и другими данными
    """
    try:
        # Выбираем случайную задачу из каталога с учетом выбранного уровня
        data = select_file(category, subcategory, is_basic_level)

        logging.info(f"Выбранная задача: {data}")
        
        if not data:
            return {"error": f"Не удалось найти задачи в категории '{category}' и подкатегории '{subcategory}'"}
        
        # Извлекаем задачу из данных
        html_task = None
        text_task = None
        images = []
        
        # Проверяем новую структуру JSON (с вложенным condition)
        if "condition" in data and isinstance(data["condition"], dict):
            html_task = data["condition"].get("html")
            logging.info(f"Найден html в condition.html: {bool(html_task)}")
            
            # Проверяем наличие текста в condition
            text_task = data["condition"].get("text")
            logging.info(f"Найден text в condition.text: {bool(text_task)}")
            
            # Проверяем наличие изображений в condition
            images = data["condition"].get("images", [])
            if images:
                logging.info(f"Найдены изображения в condition.images: {len(images)} шт.")
        
        # Для обратной совместимости
        if not html_task:
            html_task = data.get("html")
            logging.info(f"Найден html напрямую в данных: {bool(html_task)}")
        
        if not text_task:
            text_task = data.get("task", "")
            logging.info(f"Найден task напрямую в данных: {bool(text_task)}")
        
        # Определяем окончательное содержимое задачи
        if html_task:
            original_task = html_task
            logging.info("Используем HTML-версию задачи")
        elif text_task:
            original_task = text_task
            logging.info("Используем текстовую версию задачи")
            # # Если нет html и есть изображения, пытаемся распознать текст с изображений
            # ocr_text = ""
            # if images and not html_task:
            #     logging.info(f"Пытаемся распознать текст с {len(images)} изображений")
            #     for img_url in images:
            #         img_text = ocr_math_image(img_url)
            #         if img_text:
            #             ocr_text += img_text + " "
            #     if ocr_text:
            #         logging.info(f"Распознан текст с изображений: {ocr_text}")
            #         # Добавляем распознанный текст к основному тексту задачи
            #         text_task = f"{text_task} {ocr_text}"
            #     else:
            #         # Если OCR не сработал, используем текст с пометкой о содержимом на изображениях
            #         logging.info("OCR не смог распознать текст с изображений")
            #         if images:
            #             text_task = f"{text_task} *... продолжение задачи*"

            # try:
            #     # Если нет HTML, но есть текст, добавляем формулу из шаблона
            #     from app.formula_templates import generate_task_with_formula
                
            #     # Добавляем шаблон формулы в текст задачи
            #     category_number = str(category)
            #     subcategory_number = str(subcategory).split('/')[-1] if subcategory else None
                
            #     logging.info(f"Пытаемся добавить формулу из шаблона для категории {category_number}, подкатегории {subcategory_number}")
            #     text_task = generate_task_with_formula(text_task, category_number, subcategory_number)
            #     logging.info(f"Текст задачи с шаблоном формулы: {text_task}")
            # except Exception as e:
            #     logging.error(f"Ошибка при добавлении формулы из шаблона: {e}")
            #     # В случае ошибки просто используем оригинальный текст задачи

            original_task = text_task
            logging.info("Используем текстовую версию задачи")
        else:
            original_task = ""
            logging.error("Не удалось найти содержимое задачи")
            
        if not original_task:
            logging.error(f"Задача пуста. Структура данных: {json.dumps(data, ensure_ascii=False)[:300]}")
        
        logging.info(f"Выбрана исходная задача: {original_task[:100]}...")
        
        # Определяем, нужна ли визуализация
        add_visualization = needs_visualization(original_task, category, subcategory, is_basic_level)
        
        # Принудительно включаем визуализацию для планиметрии и геометрии
        if category.lower() == "планиметрия" or "геометр" in category.lower():
            add_visualization = True
            logging.info(f"Принудительное включение визуализации для категории {category}")
        
        logging.info(f"Визуализация для категории '{category}' и подкатегории '{subcategory}': {add_visualization}")
        
        # Генерируем промпт для создания полного материала
        prompt = create_complete_task_prompt(category, subcategory, original_task, difficulty_level, add_visualization)
        
        # Сохраняем информацию о запросе
        task_info = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "category": category,
            "subcategory": subcategory,
            "difficulty_level": difficulty_level,
            "is_basic_level": is_basic_level,
            "original_task": original_task
        }
        
        # Сохраняем информацию о запросе в файл
        debug_task_info_file = os.path.join(DEBUG_FILES_DIR, "debug_task_info.json")
        with open(debug_task_info_file, 'w', encoding='utf-8') as f:
            json.dump(task_info, f, ensure_ascii=False, indent=2)
        print(f"Информация о задаче сохранена в файл: {debug_task_info_file}")
        
        generated_text = yandex_gpt_generate(prompt, temperature=0.6, is_basic_level=is_basic_level)
        
        if not generated_text:
            return {"error": "Не удалось получить ответ от YandexGPT API"}
        
        # Сохраняем сгенерированный текст
        save_to_file(generated_text)

        #generated_text = generated_text.replace('\\[', '$$').replace('\\]', '$$') # заменяем квадратные скобки на двойные доллары для LaTeX
        
        # Извлекаем части из сгенерированного текста

        def extract_block(text, block_name):
            pattern = f"---{block_name}---\\s*(.*?)(?=\\s*---[A-ZА-Я _]+---|\s*$)"
            match = re.search(pattern, text, re.DOTALL)
            return match.group(1).strip() if match else ""
            
        # Используем функцию extract_block для извлечения всех необходимых блоков
        task = extract_block(generated_text, "ЗАДАЧА")
        solution = extract_block(generated_text, "РЕШЕНИЕ")
        hints_string = extract_block(generated_text, "ПОДСКАЗКИ")
        visualization_params = extract_block(generated_text, "ПАРАМЕТРЫ ДЛЯ ВИЗУАЛИЗАЦИИ")
        
        if not task:
            task = "Не удалось извлечь задачу"
        if not solution:
            solution = "Не удалось извлечь решение"
        
        # Парсим подсказки
        hints = parse_hints(hints_string)
        
        # Извлекаем ответ из решения
        answer = extract_answer_with_latex(solution)
        
        image_path = None
        visualization_type = None
        
        if visualization_params:
            # Обрабатываем параметры визуализации
            logging.info(f"Извлечены параметры для визуализации, первые 200 символов: {visualization_params[:200]}")
            
            # Проверяем, содержит ли текст параметров строку "Визуализация не требуется"
            if re.search(r'визуализация\s+не\s+требуется', visualization_params.lower()):
                logging.info("В параметрах содержится 'Визуализация не требуется'")
            else:
                # Ищем тип визуализации
                type_match = re.search(r'Тип:?\s*([^\n]+)', visualization_params)
                if type_match:
                    extracted_type = type_match.group(1).strip().lower()
                    logging.info(f"Извлеченный тип визуализации: {extracted_type}")
                else:
                    logging.warning("Тип визуализации не найден в параметрах")
            
            image_path, visualization_type = process_visualization_params(visualization_params)
            
            logging.info(f"Результат обработки параметров: путь к изображению = {image_path}, тип = {visualization_type}")
        else:
            logging.info("Раздел с параметрами для визуализации не найден в ответе ИИ")
        
        # Формируем результат
        result = {
            "task": convert_markdown_to_html(task),
            "solution": convert_markdown_to_html(solution),
            "hints": [convert_markdown_to_html(hint) for hint in hints],
            "answer": answer,
            "difficulty_level": difficulty_level,
            "category": category,
            "subcategory": subcategory,
            "is_basic_level": is_basic_level
        }
        
        # Добавляем информацию об изображении и параметрах визуализации, если они есть
        if image_path:
            # Проверяем, существует ли файл изображения
            if os.path.exists(image_path):
                logging.info(f"Файл изображения существует: {image_path}")
                result["image_path"] = image_path
                result["visualization_params"] = visualization_params
                result["visualization_type"] = visualization_type
                
                # Для удобного отображения на веб-странице
                image_url = f"/static/images/generated/{os.path.basename(image_path)}"
                result["image_url"] = image_url
            else:
                logging.error(f"Файл изображения не существует: {image_path}")
        else:
            logging.info("Изображение не было создано")
        
        # Сохраняем полный результат для использования в приложении
        save_to_file(result, "last_generated_task.txt")
        
        return result
    except Exception as e:
        logging.error(f"Ошибка при генерации задачи: {e}")
        return {"error": f"Произошла ошибка при генерации задачи: {str(e)}"}

def parse_sections(raw_text):
    """
    Разбирает сырой текст ответа AI на разделы.
    
    Args:
        raw_text: Сырой текст ответа
        
    Returns:
        dict: Словарь с разделами (задача, решение, подсказки и т.д.)
    """
    # Инициализируем словарь с пустыми секциями
    sections = {
        "task": "",
        "solution": "",
        "answer": "",
        "hints": [],
        "visualization_params": ""
    }
    
    if not raw_text:
        return sections
    
    # Функция для извлечения содержимого между маркерами разделов
    def extract_block(text, block_name):
        pattern = f"---{block_name}---\\s*(.*?)(?=\\s*---[A-ZА-Я _]+---|\s*$)"
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else ""
    
    # Извлекаем каждый раздел из текста
    sections["task"] = extract_block(raw_text, "ЗАДАЧА")
    sections["solution"] = extract_block(raw_text, "РЕШЕНИЕ")
    sections["visualization_params"] = extract_block(raw_text, "ПАРАМЕТРЫ ДЛЯ ВИЗУАЛИЗАЦИИ")
    
    # Извлекаем ответ из решения, если решение есть
    if sections["solution"]:
        sections["answer"] = extract_answer_with_latex(sections["solution"])
    
    # Извлекаем подсказки и разбиваем их на список
    hints_text = extract_block(raw_text, "ПОДСКАЗКИ")
    sections["hints"] = parse_hints(hints_text)
    
    return sections

def convert_markdown_to_html(text):
    if not text:
        return ""
    
    # Сначала обработаем блоки формул с двойными долларами
    # Сохраняем блоки формул во временном хранилище
    formula_blocks = []
    
    def replace_formula_block(match):
        formula_blocks.append(match.group(1))
        return f"FORMULA_BLOCK_{len(formula_blocks)-1}"
    
    # Находим и заменяем блоки формул временными маркерами
    text = re.sub(r'\$\$(.*?)\$\$', replace_formula_block, text, flags=re.DOTALL)
    
    # Обработка жирного и курсива
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'\*(?!\s)(.+?)(?<!\s)\*', r'<i>\1</i>', text)
    
    lines = text.split('\n')
    result = []
    
    in_list = False
    in_bullet_list = False
    i = 0
    
    while i < len(lines):
        line = lines[i].strip()
        
        # Пропускаем пустые строки
        if not line:
            # Если были в списке, закрываем его перед пустой строкой
            if in_list:
                result.append('</ol>')
                in_list = False
            if in_bullet_list:
                result.append('</ul>')
                in_bullet_list = False
            
            i += 1
            continue
        
        # Проверяем, является ли строка временным маркером формулы
        formula_match = re.match(r'^FORMULA_BLOCK_(\d+)$', line)
        if formula_match:
            index = int(formula_match.group(1))
            formula_content = formula_blocks[index]
            result.append(f'<div class="math-block">$${ formula_content }$$</div>')
            i += 1
            continue
        
        # Нумерованный список
        list_match = re.match(r'^(\d+)\.\s+(.+)$', line)
        if list_match:
            number = list_match.group(1)
            content = list_match.group(2)
            
            if not in_list:
                result.append('<ol>')
                in_list = True
                
            result.append(f'<li value="{number}">{content}</li>')
        # Маркированный список (начинается с дефиса или звездочки)
        elif line.startswith('- ') or line.startswith('* '):
            content = line[2:]  # Убираем дефис/звездочку и пробел
            
            if not in_bullet_list:
                result.append('<ul class="bullet-list">')
                in_bullet_list = True
                
            result.append(f'<li>{content}</li>')
        else:
            # Закрываем список, если не в нем
            if in_list and not line.startswith('   '):
                result.append('</ol>')
                in_list = False
            if in_bullet_list and not line.startswith('   '):
                result.append('</ul>')
                in_bullet_list = False
            
            # Специально обрабатываем строки с дефисом в начале (не в списке)
            if not in_bullet_list and (line.startswith('- ') or line.startswith('* ')):
                dashed_line = line.replace('- ', '<span class="dash-marker">- </span>', 1)
                dashed_line = dashed_line.replace('* ', '<span class="dash-marker">• </span>', 1)
                result.append(f'<p>{dashed_line}</p>')
            else:
                # Обрабатываем обычные строки или отступы
                result.append(f'<p>{line}</p>')
        
        i += 1
    
    # Закрываем незакрытый список
    if in_list:
        result.append('</ol>')
    if in_bullet_list:
        result.append('</ul>')
    
    html_text = '\n'.join(result)
    
    # Обработка переносов внутри параграфов
    html_text = re.sub(r'<p>(.*?)</p>', 
                  lambda m: '<p>' + m.group(1).replace('\n', '<br>') + '</p>', 
                  html_text, flags=re.DOTALL)
    
    # Восстанавливаем формулы внутри параграфов (встроенные формулы с одинарными $)
    html_text = re.sub(r'\$(.*?)\$', r'$\1$', html_text)
    
    return html_text

def process_rectangle_visualization(params_text, extract_param):
    """Обрабатывает параметры для прямоугольника и создает визуализацию"""
    # Извлекаем параметры для прямоугольника
    dimensions_str = extract_param(REGEX_PATTERNS["rectangle"]["dimensions"], params_text)
    coords_str = extract_param(REGEX_PATTERNS["rectangle"]["coords"], params_text)
    vertex_labels_str = extract_param(REGEX_PATTERNS["rectangle"]["vertex_labels"], params_text)
    show_labels = extract_param(REGEX_PATTERNS["rectangle"]["show_labels"], params_text, "нет").lower() in ["да", "yes", "true"]
    show_angles = extract_param(REGEX_PATTERNS["rectangle"]["show_angles"], params_text, "нет").lower() in ["да", "yes", "true"]
    show_lengths = extract_param(REGEX_PATTERNS["rectangle"]["show_lengths"], params_text, "нет").lower() in ["да", "yes", "true"]
    side_lengths_str = extract_param(REGEX_PATTERNS["rectangle"]["side_lengths"], params_text)
    show_angle_arcs = extract_param(REGEX_PATTERNS["rectangle"]["show_angle_arcs"], params_text, "нет").lower() in ["да", "yes", "true"]
    
    # Параметры для отображения прямоугольника
    params = DEFAULT_VISUALIZATION_PARAMS["rectangle"].copy()
    
    # Парсинг координат
    if coords_str:
        try:
            # Для координат ожидаем формат (x,y) - левый нижний угол
            coords = coords_str.strip()
            match = re.search(r'\(([^,]+),([^)]+)\)', coords)
            if match:
                x = float(match.group(1))
                y = float(match.group(2))
                params['x'] = x
                params['y'] = y
        except Exception as e:
            logging.warning(f"Ошибка при разборе координат прямоугольника: {e}")
    
    # Парсинг размеров
    if dimensions_str:
        try:
            # Проверяем, если dimensions_str уже является списком
            if isinstance(dimensions_str, list):
                if len(dimensions_str) >= 2:
                    width = float(dimensions_str[0])
                    height = float(dimensions_str[1])
                    params['width'] = width
                    params['height'] = height
            else:
                # Ожидаем два числа через запятую: ширина, высота
                dims = dimensions_str.split(',')
                if len(dims) >= 2:
                    width = float(dims[0].strip())
                    height = float(dims[1].strip())
                    params['width'] = width
                    params['height'] = height
        except Exception as e:
            logging.warning(f"Ошибка при разборе размеров прямоугольника: {e}")
    
    # Парсинг меток вершин
    if vertex_labels_str:
        try:
            if isinstance(vertex_labels_str, list):
                labels = vertex_labels_str
            else:
                labels = [label.strip() for label in vertex_labels_str.split(',')]
            
            # Проверяем, что для прямоугольника у нас не более 4 вершин
            if len(labels) > 4:
                logging.warning(f"Указано слишком много вершин для прямоугольника ({len(labels)}). Использую только первые 4 вершины.")
                labels = labels[:4]
            
            params['vertex_labels'] = labels
            params['show_labels'] = True
        except Exception as e:
            logging.warning(f"Ошибка при разборе меток вершин прямоугольника: {e}")
    
    # Парсинг длин сторон (обычно не требуется для прямоугольника, но оставляем для совместимости)
    if side_lengths_str:
        try:
            side_lengths = []
            for length_str in side_lengths_str.split(','):
                length_str = length_str.strip()
                if length_str.lower() in ["нет", "no", "none", "-"]:
                    side_lengths.append(None)
                else:
                    side_lengths.append(float(length_str))
            params['side_lengths'] = side_lengths
        except Exception as e:
            logging.warning(f"Ошибка при разборе длин сторон прямоугольника: {e}")
    
    # Автоматически определяем, показывать ли стороны, в зависимости от наличия размеров или длин сторон
    if dimensions_str or side_lengths_str:
        params['show_lengths'] = True
    else:
        params['show_lengths'] = False
        if show_lengths:
            logging.warning("Автоматически отключено отображение длин сторон прямоугольника, т.к. не заданы ни размеры, ни длины сторон.")
    
    # Если side_lengths не был задан явно, но есть параметры размеров, заполняем side_lengths
    if 'width' in params and 'height' in params and 'side_lengths' not in params:
        width = params['width']
        height = params['height']
        # Заполняем длины сторон [нижняя, правая, верхняя, левая]
        params['side_lengths'] = [width, height, width, height]
    
    # Добавляем основные параметры
    params['show_angles'] = show_angles
    params['show_lengths'] = show_lengths
    params['show_angle_arcs'] = show_angle_arcs
    
    # Обработка параметров конкретных углов для отображения
    show_specific_angles = extract_param(REGEX_PATTERNS["rectangle"]["show_specific_angles"], params_text)
    if show_specific_angles:
        # Проверяем, если это уже список, то используем как есть
        if isinstance(show_specific_angles, list):
            params["show_specific_angles"] = show_specific_angles
        else:
            try:
                # Иначе пытаемся разделить строку
                params["show_specific_angles"] = [angle.strip() for angle in show_specific_angles.split(',')]
            except Exception as e:
                logging.warning(f"Ошибка при разборе конкретных углов прямоугольника: {e}")
    
    # Обработка параметров конкретных сторон для отображения
    show_specific_sides = extract_param(REGEX_PATTERNS["rectangle"]["show_specific_sides"], params_text)
    if show_specific_sides:
        # Проверяем, если это уже список, то используем как есть
        if isinstance(show_specific_sides, list):
            params["show_specific_sides"] = show_specific_sides
        else:
            try:
                # Иначе пытаемся разделить строку
                params["show_specific_sides"] = [side.strip() for side in show_specific_sides.split(',')]
            except Exception as e:
                logging.warning(f"Ошибка при разборе конкретных сторон прямоугольника: {e}")
    
    # Создаем директорию для сохранения изображения, если она не существует
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static/images/generated")
    os.makedirs(output_dir, exist_ok=True)
    
    # Генерируем имя файла с уникальным идентификатором
    filename = f'rectangle_{uuid.uuid4().hex[:8]}.png'
    output_path = os.path.join(output_dir, filename)
    
    try:
        # Используем новую ООП-структуру для создания и отрисовки прямоугольника
        from app.geometry import Rectangle
        from app.visualization import GeometryRenderer
        
        # Создаем прямоугольник
        rectangle = Rectangle(params)
        
        # Отрисовываем и сохраняем изображение
        GeometryRenderer.render_figure(rectangle, output_path)
        
        return output_path
    except ImportError:
        # Если новые модули недоступны, используем старый подход
        logging.warning("Не удалось импортировать новые модули для ООП-визуализации, используем старый подход")
    
    # Генерируем изображение старым методом
    generate_geometric_figure('rectangle', params, output_path)
    return output_path

def process_parallelogram_visualization(params_text, extract_param):
    """Обрабатывает параметры для параллелограмма и создает визуализацию"""
    # Извлекаем параметры для параллелограмма
    dimensions_str = extract_param(REGEX_PATTERNS["parallelogram"]["dimensions"], params_text)
    coords_str = extract_param(REGEX_PATTERNS["parallelogram"]["coords"], params_text)
    vertex_labels_str = extract_param(REGEX_PATTERNS["parallelogram"]["vertex_labels"], params_text)
    show_labels = extract_param(REGEX_PATTERNS["parallelogram"]["show_labels"], params_text, "нет").lower() in ["да", "yes", "true"]
    show_angles = extract_param(REGEX_PATTERNS["parallelogram"]["show_angles"], params_text, "нет").lower() in ["да", "yes", "true"]
    show_lengths = extract_param(REGEX_PATTERNS["parallelogram"]["show_lengths"], params_text, "нет").lower() in ["да", "yes", "true"]
    side_lengths_str = extract_param(REGEX_PATTERNS["parallelogram"]["side_lengths"], params_text)
    skew_str = extract_param(REGEX_PATTERNS["parallelogram"]["skew_angle"], params_text)
    show_angle_arcs = extract_param(REGEX_PATTERNS["parallelogram"]["show_angle_arcs"], params_text, "нет").lower() in ["да", "yes", "true"]
    
    # Параметры для отображения параллелограмма
    params = DEFAULT_VISUALIZATION_PARAMS["parallelogram"].copy()
    
    # Парсинг координат
    if coords_str:
        try:
            # Для координат ожидаем формат (x,y) - левый нижний угол
            coords = coords_str.strip()
            match = re.search(r'\(([^,]+),([^)]+)\)', coords)
            if match:
                x = float(match.group(1))
                y = float(match.group(2))
                params['x'] = x
                params['y'] = y
        except Exception as e:
            logging.warning(f"Ошибка при разборе координат параллелограмма: {e}")
    
    # Парсинг размеров
    if dimensions_str:
        try:
            # Проверяем, если dimensions_str уже является списком
            if isinstance(dimensions_str, list):
                if len(dimensions_str) >= 2:
                    width = float(dimensions_str[0])
                    height = float(dimensions_str[1])
                    params['width'] = width
                    params['height'] = height
            else:
                # Ожидаем два числа через запятую: ширина, высота
                dims = dimensions_str.split(',')
                if len(dims) >= 2:
                    width = float(dims[0].strip())
                    height = float(dims[1].strip())
                    params['width'] = width
                    params['height'] = height
        except Exception as e:
            logging.warning(f"Ошибка при разборе размеров параллелограмма: {e}")
    
    # Парсинг угла наклона
    if skew_str:
        try:
            skew = float(skew_str.strip())
            params['skew'] = skew
        except Exception as e:
            logging.warning(f"Ошибка при разборе угла наклона параллелограмма: {e}")
    
    # Парсинг меток вершин
    if vertex_labels_str:
        try:
            if isinstance(vertex_labels_str, list):
                labels = vertex_labels_str
            else:
                labels = [label.strip() for label in vertex_labels_str.split(',')]
            
            # Проверяем, что для параллелограмма у нас не более 4 вершин
            if len(labels) > 4:
                logging.warning(f"Указано слишком много вершин для параллелограмма ({len(labels)}). Использую только первые 4 вершины.")
                labels = labels[:4]
            
            params['vertex_labels'] = labels
            params['show_labels'] = True
        except Exception as e:
            logging.warning(f"Ошибка при разборе меток вершин параллелограмма: {e}")
    
    # Парсинг длин сторон
    if side_lengths_str:
        try:
            side_lengths = []
            for length_str in side_lengths_str.split(','):
                length_str = length_str.strip()
                if length_str.lower() in ["нет", "no", "none", "-"]:
                    side_lengths.append(None)
                else:
                    side_lengths.append(float(length_str))
            params['side_lengths'] = side_lengths
        except Exception as e:
            logging.warning(f"Ошибка при разборе длин сторон параллелограмма: {e}")
    
    # Автоматически определяем, показывать ли стороны, в зависимости от наличия размеров или длин сторон
    if dimensions_str or side_lengths_str:
        params['show_lengths'] = True
    else:
        params['show_lengths'] = False
        if show_lengths:
            logging.warning("Автоматически отключено отображение длин сторон параллелограмма, т.к. не заданы ни размеры, ни длины сторон.")
    
    # Если side_lengths не был задан явно, но есть параметры размеров, заполняем side_lengths
    if 'width' in params and 'height' in params and 'side_lengths' not in params:
        width = params['width']
        height = params['height']
        # Заполняем длины сторон [нижняя, правая, верхняя, левая]
        params['side_lengths'] = [width, height, width, height]
    
    # Добавляем основные параметры
    params['show_angles'] = show_angles
    params['show_lengths'] = show_lengths
    params['show_angle_arcs'] = show_angle_arcs
    
    # Обработка параметров конкретных углов для отображения
    show_specific_angles = extract_param(REGEX_PATTERNS["parallelogram"]["show_specific_angles"], params_text)
    if show_specific_angles:
        # Проверяем, если это уже список, то используем как есть
        if isinstance(show_specific_angles, list):
            params["show_specific_angles"] = show_specific_angles
        else:
            try:
                # Иначе пытаемся разделить строку
                params["show_specific_angles"] = [angle.strip() for angle in show_specific_angles.split(',')]
            except Exception as e:
                logging.warning(f"Ошибка при разборе конкретных углов параллелограмма: {e}")
    
    # Обработка параметров конкретных сторон для отображения
    show_specific_sides = extract_param(REGEX_PATTERNS["parallelogram"]["show_specific_sides"], params_text)
    if show_specific_sides:
        # Проверяем, если это уже список, то используем как есть
        if isinstance(show_specific_sides, list):
            params["show_specific_sides"] = show_specific_sides
        else:
            try:
                # Иначе пытаемся разделить строку
                params["show_specific_sides"] = [side.strip() for side in show_specific_sides.split(',')]
            except Exception as e:
                logging.warning(f"Ошибка при разборе конкретных сторон параллелограмма: {e}")
    
    # Создаем директорию для сохранения изображения, если она не существует
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static/images/generated")
    os.makedirs(output_dir, exist_ok=True)
    
    # Генерируем имя файла с уникальным идентификатором
    filename = f'parallelogram_{uuid.uuid4().hex[:8]}.png'
    output_path = os.path.join(output_dir, filename)
    
    try:
        # Используем новую ООП-структуру для создания и отрисовки параллелограмма
        from app.geometry import Parallelogram
        from app.visualization import GeometryRenderer
        
        # Создаем параллелограмм
        parallelogram = Parallelogram(params)
        
        # Отрисовываем и сохраняем изображение
        GeometryRenderer.render_figure(parallelogram, output_path)
        
        return output_path
    except ImportError:
        # Если новые модули недоступны, используем старый подход
        logging.warning("Не удалось импортировать новые модули для ООП-визуализации, используем старый подход")
    
    # Генерируем изображение старым методом
    generate_geometric_figure('parallelogram', params, output_path)
    return output_path

def process_trapezoid_visualization(params_text, extract_param):
    """Обрабатывает параметры для трапеции и создает визуализацию с помощью OOP-классов"""
    # Извлекаем параметры для трапеции
    dimensions_str = extract_param(REGEX_PATTERNS["trapezoid"]["dimensions"], params_text)
    coords_str = extract_param(REGEX_PATTERNS["trapezoid"]["coords"], params_text)
    vertex_labels_str = extract_param(REGEX_PATTERNS["trapezoid"]["vertex_labels"], params_text)
    show_dimensions = extract_param(REGEX_PATTERNS["trapezoid"]["show_dimensions"], params_text, "нет").lower() in ["да", "yes", "true"]
    show_labels = extract_param(REGEX_PATTERNS["trapezoid"]["show_labels"], params_text, "нет").lower() in ["да", "yes", "true"]
    show_angles = extract_param(REGEX_PATTERNS["trapezoid"]["show_angles"], params_text, "нет").lower() in ["да", "yes", "true"]
    show_lengths = extract_param(REGEX_PATTERNS["trapezoid"]["show_lengths"], params_text, "нет").lower() in ["да", "yes", "true"]
    side_lengths_str = extract_param(REGEX_PATTERNS["trapezoid"]["side_lengths"], params_text)
    show_angle_arcs = extract_param(REGEX_PATTERNS["trapezoid"]["show_angle_arcs"], params_text, "нет").lower() in ["да", "yes", "true"]
    
    # Параметры для отображения трапеции
    params = DEFAULT_VISUALIZATION_PARAMS["trapezoid"].copy()
    
    # Парсинг координат
    if coords_str:
        try:
            # Для координат ожидаем формат (x,y) - левый нижний угол
            coords = coords_str.strip()
            match = re.search(r'\(([^,]+),([^)]+)\)', coords)
            if match:
                x = float(match.group(1))
                y = float(match.group(2))
                params['x'] = x
                params['y'] = y
        except Exception as e:
            logging.warning(f"Ошибка при разборе координат трапеции: {e}")
    
    # Парсинг размеров
    if dimensions_str:
        try:
            # Проверяем, если dimensions_str уже является списком
            if isinstance(dimensions_str, list):
                if len(dimensions_str) >= 3:
                    bottom_width = float(dimensions_str[0])
                    top_width = float(dimensions_str[1])
                    height = float(dimensions_str[2])
                    params['bottom_width'] = bottom_width
                    params['top_width'] = top_width
                    params['height'] = height
            else:
                # Ожидаем два или три числа через запятую: ширина_основания, ширина_верха, [высота]
                dims = dimensions_str.split(',')
                if len(dims) >= 2:
                    bottom_width = float(dims[0].strip())
                    top_width = float(dims[1].strip())
                    params['bottom_width'] = bottom_width
                    params['top_width'] = top_width
                    
                    if len(dims) >= 3:
                        height = float(dims[2].strip())
                        params['height'] = height
        except Exception as e:
            logging.warning(f"Ошибка при разборе размеров трапеции: {e}")
    
    # Парсинг меток вершин
    if vertex_labels_str:
        try:
            if isinstance(vertex_labels_str, list):
                labels = vertex_labels_str
            else:
                labels = [label.strip() for label in vertex_labels_str.split(',')]
            
            # Проверяем, что для трапеции у нас не более 4 вершин
            if len(labels) > 4:
                logging.warning(f"Указано слишком много вершин для трапеции ({len(labels)}). Использую только первые 4 вершины.")
                labels = labels[:4]
            
            params['vertex_labels'] = labels
            params['show_labels'] = True
        except Exception as e:
            logging.warning(f"Ошибка при разборе меток вершин трапеции: {e}")
    
    # Парсинг длин сторон
    if side_lengths_str:
        try:
            side_lengths = []
            for length_str in side_lengths_str.split(','):
                length_str = length_str.strip()
                if length_str.lower() in ["нет", "no", "none", "-"]:
                    side_lengths.append(None)
                else:
                    side_lengths.append(float(length_str))
            params['side_lengths'] = side_lengths
        except Exception as e:
            logging.warning(f"Ошибка при разборе длин сторон трапеции: {e}")
    # Если side_lengths не был задан явно, но есть параметры размеров, заполняем side_lengths
    elif 'bottom_width' in params and 'top_width' in params:
        # Вычисляем боковые стороны трапеции
        bottom_width = params['bottom_width']
        top_width = params['top_width']
        height = params.get('height', 3)
        dx = (bottom_width - top_width) / 2
        side_length = np.sqrt(dx**2 + height**2)
        # Заполняем длины сторон [нижнее_основание, правая_сторона, верхнее_основание, левая_сторона]
        params['side_lengths'] = [bottom_width, side_length, top_width, side_length]
    
    # Автоматически определяем, показывать ли стороны, в зависимости от наличия размеров или длин сторон
    if dimensions_str or side_lengths_str:
        params['show_lengths'] = True
    else:
        params['show_lengths'] = False
        if show_lengths:
            logging.warning("Автоматически отключено отображение длин сторон трапеции, т.к. не заданы ни размеры, ни длины сторон.")
    
    # Добавляем основные параметры
    params['show_dimensions'] = show_dimensions
    params['show_angles'] = show_angles
    params['show_lengths'] = show_lengths
    params['show_angle_arcs'] = show_angle_arcs
    
    # Обработка параметров конкретных углов для отображения
    show_specific_angles = extract_param(REGEX_PATTERNS["trapezoid"]["show_specific_angles"], params_text)
    if show_specific_angles:
        # Проверяем, если это уже список, то используем как есть
        if isinstance(show_specific_angles, list):
            params["show_specific_angles"] = show_specific_angles
        else:
            try:
                # Иначе пытаемся разделить строку
                params["show_specific_angles"] = [angle.strip() for angle in show_specific_angles.split(',')]
            except Exception as e:
                logging.warning(f"Ошибка при разборе конкретных углов трапеции: {e}")
    
    # Обработка параметров конкретных сторон для отображения
    show_specific_sides = extract_param(REGEX_PATTERNS["trapezoid"]["show_specific_sides"], params_text)
    if show_specific_sides:
        # Проверяем, если это уже список, то используем как есть
        if isinstance(show_specific_sides, list):
            params["show_specific_sides"] = show_specific_sides
        else:
            try:
                # Иначе пытаемся разделить строку
                params["show_specific_sides"] = [side.strip() for side in show_specific_sides.split(',')]
            except Exception as e:
                logging.warning(f"Ошибка при разборе конкретных сторон трапеции: {e}")
    
    # Создаем директорию для сохранения изображения, если она не существует
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static/images/generated")
    os.makedirs(output_dir, exist_ok=True)
    
    # Генерируем имя файла с уникальным идентификатором
    filename = f'trapezoid_{uuid.uuid4().hex[:8]}.png'
    output_path = os.path.join(output_dir, filename)
    
    try:
        # Используем новую ООП-структуру для создания и отрисовки трапеции
        from app.geometry import Trapezoid
        from app.visualization import GeometryRenderer
        
        # Создаем трапецию
        trapezoid = Trapezoid(params)
        
        # Отрисовываем и сохраняем изображение
        GeometryRenderer.render_figure(trapezoid, output_path)
        
        return output_path
    except ImportError:
        # Если новые модули недоступны, используем старый подход
        logging.warning("Не удалось импортировать новые модули для ООП-визуализации, используем старый подход")
    
    # Генерируем изображение старым методом
    generate_geometric_figure('trapezoid', params, output_path)
    return output_path

def generate_markdown_task(category, subcategory="", difficulty_level=3, is_basic_level=False):
    """
    Генерирует полный пакет задачи, решения и подсказок в формате Markdown.
    
    Args:
        category: Категория задачи
        subcategory: Подкатегория задачи (опционально)
        difficulty_level: Уровень сложности подсказок (1-5)
        is_basic_level: Выбор между базовым (True) и профильным (False) уровнем ЕГЭ, 
                        определяет только директорию, откуда берутся задачи
        
    Returns:
        dict: Словарь с задачей, решением и подсказками в Markdown
    """
    try:
        # Используем существующую функцию для генерации задачи
        result = generate_complete_task(category, subcategory, difficulty_level, is_basic_level)
        
        # Проверяем на ошибки
        if "error" in result:
            return result
        
        # Преобразуем HTML в Markdown
        from app.utils.converters import convert_html_to_markdown as html_to_markdown
        md_task = html_to_markdown(result.get("task", ""))
        md_solution = html_to_markdown(result.get("solution", ""))
        md_hints = [html_to_markdown(hint) for hint in result.get("hints", [])]
        
        # Формируем результат
        markdown_result = {
            "problem": md_task,
            "problem_picture": "",  # Пока пустое
            "solution": md_solution,
            "hint1": md_hints[0] if len(md_hints) > 0 else "",
            "hint2": md_hints[1] if len(md_hints) > 1 else "",
            "hint3": md_hints[2] if len(md_hints) > 2 else "",
            "difficulty_level": result.get("difficulty_level", difficulty_level),
            "is_basic_level": is_basic_level
        }
        
        # Если есть изображение, добавляем его URL
        if "image_url" in result:
            markdown_result["problem_picture"] = result["image_url"]
        elif "image_path" in result:
            image_path = result["image_path"]
            image_filename = os.path.basename(image_path)
            image_url = f"/static/images/generated/{image_filename}"
            markdown_result["problem_picture"] = image_url
        
        return markdown_result
    except Exception as e:
        logging.error(f"Ошибка при генерации задачи в формате Markdown: {e}")
        logging.error(traceback.format_exc())
        return {"error": f"Ошибка при генерации задачи в формате Markdown: {str(e)}"}

def generate_json_task(category, subcategory="", difficulty_level=3, is_basic_level=False):
    """
    Генерирует полный пакет задачи, решения и подсказок в формате JSON.
    
    Args:
        category: Категория задачи
        subcategory: Подкатегория задачи (опционально)
        difficulty_level: Уровень сложности подсказок (1-5)
        is_basic_level: Выбор между базовым (True) и профильным (False) уровнем ЕГЭ, 
                        определяет только директорию, откуда берутся задачи
        
    Returns:
        dict: Словарь с задачей, решением и подсказками в формате JSON
    """
    # Используем существующую функцию для генерации задачи
    result = generate_complete_task(category, subcategory, difficulty_level, is_basic_level)
    
    # Проверяем на ошибки
    if "error" in result:
        return result
    
    # Извлекаем ответ из решения для отдельного поля
    solution = result.get("solution", "")
    answer = result.get("answer", "")
    
    # Если ответ не был успешно извлечен, пробуем найти его снова
    if not answer or answer == "См. решение":
        answer_match = re.search(r"(?:Ответ|ОТВЕТ|ответ)\s*:(.+?)(?=$|\.|\.\s*$|\n\s*\n|\<\/p\>)", solution, re.IGNORECASE | re.DOTALL)
        if answer_match:
            answer = answer_match.group(1).strip()
    
    # Подготавливаем изображения
    task_images = []
    solution_images = []
    
    # Если есть изображение в результате, добавляем его
    if "image_path" in result:
        image_path = result["image_path"]
        image_filename = os.path.basename(image_path)
        image_url = f"/static/images/generated/{image_filename}"
        
        # Добавляем изображение к задаче
        task_images.append({
            "url": image_url,
            "alt": "Изображение к задаче"
        })
    elif "image_url" in result:
        # Если есть прямая ссылка на изображение
        image_url = result["image_url"]
        task_images.append({
            "url": image_url,
            "alt": "Изображение к задаче"
        })
    
    # Формируем JSON-результат
    json_result = {
        "task": {
            "text": result.get("task", ""),
            "images": task_images
        },
        "solution": {
            "text": solution,
            "images": solution_images
        },
        "answer": answer,
        "hints": result.get("hints", []),
        "difficulty_level": result.get("difficulty_level", difficulty_level),
        "category": category,
        "subcategory": subcategory,
        "is_basic_level": is_basic_level,
        "format": "html"  # Указываем формат данных
    }
    
    return json_result

def generate_json_markdown_task(category, subcategory="", difficulty_level=3, is_basic_level=False):
    """
    Генерирует полный пакет задачи, решения и подсказок в формате JSON с Markdown для текста.
    
    Args:
        category: Категория задачи
        subcategory: Подкатегория задачи (опционально)
        difficulty_level: Уровень сложности подсказок (1-5)
        is_basic_level: Выбор между базовым (True) и профильным (False) уровнем ЕГЭ, 
                        определяет только директорию, откуда берутся задачи
        
    Returns:
        dict: Словарь с задачей, решением и подсказками в формате JSON с Markdown
    """
    try:
        # Используем существующую функцию для генерации задачи
        result = generate_complete_task(category, subcategory, difficulty_level, is_basic_level)
        
        # Проверяем на ошибки
        if "error" in result:
            return result
        
        # Извлекаем ответ из решения для отдельного поля
        task_html = result.get("task", "")
        solution_html = result.get("solution", "")
        answer = result.get("answer", "")
        
        # Если ответ не был успешно извлечен, пробуем найти его снова
        if not answer or answer == "См. решение":
            answer_match = re.search(r"(?:Ответ|ОТВЕТ|ответ)\s*:(.+?)(?=$|\.|\.\s*$|\n\s*\n|\<\/p\>)", solution_html, re.IGNORECASE | re.DOTALL)
            if answer_match:
                answer = answer_match.group(1).strip()
        
        # Преобразуем HTML в Markdown
        from app.utils.converters import convert_html_to_markdown as html_to_markdown
        task_md = html_to_markdown(task_html)
        solution_md = html_to_markdown(solution_html)
        hints_md = [html_to_markdown(hint) for hint in result.get("hints", [])]
        
        # Подготавливаем изображения
        task_images = []
        solution_images = []
        
        # Если есть изображение в результате, добавляем его
        if "image_path" in result:
            image_path = result["image_path"]
            image_filename = os.path.basename(image_path)
            image_url = f"/static/images/generated/{image_filename}"
            
            # Добавляем изображение к задаче
            task_images.append({
                "url": image_url,
                "alt": "Изображение к задаче"
            })
            
            # В Markdown добавляем ссылку на изображение в текст задачи
            task_md = f"![Изображение к задаче]({image_url})\n\n{task_md}"
        elif "image_url" in result:
            # Если есть прямая ссылка на изображение
            image_url = result["image_url"]
            task_images.append({
                "url": image_url,
                "alt": "Изображение к задаче"
            })
            
            # В Markdown добавляем ссылку на изображение в текст задачи
            task_md = f"![Изображение к задаче]({image_url})\n\n{task_md}"
        
        # Формируем JSON-результат с Markdown
        json_result = {
            "task": {
                "text": task_md,
                "images": task_images
            },
            "solution": {
                "text": solution_md,
                "images": solution_images
            },
            "answer": answer,
            "hints": hints_md,
            "difficulty_level": result.get("difficulty_level", difficulty_level),
            "category": category,
            "subcategory": subcategory,
            "is_basic_level": is_basic_level,
            "format": "markdown"  # Указываем формат данных
        }
        
        return json_result
    except Exception as e:
        logging.error(f"Ошибка при генерации JSON с Markdown: {e}")
        logging.error(traceback.format_exc())
        return {"error": f"Ошибка при генерации задачи с Markdown: {str(e)}"}