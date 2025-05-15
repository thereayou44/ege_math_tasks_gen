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

try:
    from app.prompts import HINT_PROMPTS, SYSTEM_PROMPT, create_complete_task_prompt, REGEX_PATTERNS, DEFAULT_VISUALIZATION_PARAMS
    from app.visualization import process_bar_chart, process_pie_chart, process_chart_visualization
    from app.visualization.chart_utils import normalize_function_expression
    from app.visualization.renderer import GeometryRenderer

except ImportError:
    from prompts import HINT_PROMPTS, SYSTEM_PROMPT, create_complete_task_prompt, REGEX_PATTERNS, DEFAULT_VISUALIZATION_PARAMS
    from visualization import process_multiple_function_plots, process_bar_chart, process_pie_chart, process_chart_visualization
    from visualization.renderer import GeometryRenderer
import traceback
import matplotlib.patches as patches
# Импортируем utils.converters так, чтобы работало как из корня проекта, так и из папки app
try:
    from app.utils.converters import convert_html_to_markdown as html_to_markdown
except ImportError:
    from utils.converters import convert_html_to_markdown as html_to_markdown

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Загрузка переменных окружения из .env файла
load_dotenv()

# Инициализация Yandex API
YANDEX_API_KEY = os.getenv("YANDEX_API_KEY")
YANDEX_FOLDER_ID = os.getenv("YANDEX_FOLDER_ID")

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

def yandex_gpt_generate(prompt, temperature=0.3, max_tokens=8000, is_basic_level=None):
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
    
    # Проверяем, есть ли ответ в кэше
    if cache_key in _task_cache:
        print("Результат найден в кэше!")
        return _task_cache[cache_key]
        
    try:
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
    special_pattern = r'[-]{2,}(?:ОТВЕТ|Ответ|ответ)[-]{2,}\s*([^-\n].*?)(?:$|\n\s*-|\n\s*\n)'
    special_match = re.search(special_pattern, solution, re.IGNORECASE | re.DOTALL)
    
    if special_match:
        answer = special_match.group(1).strip()
        logging.info(f"Найден ответ в формате ---ОТВЕТ---: {answer}")
        # Применяем форматирование LaTeX
        return format_latex_answer(answer)
        
    # Ищем "Ответ:" или "Ответ :"
    answer_pattern = r"(?:Ответ|ОТВЕТ|ответ)\s*:(.+?)(?:$|\.|\n|\<\/p\>)"
    answer_match = re.search(answer_pattern, solution, re.IGNORECASE | re.DOTALL)
    
    if answer_match:
        answer = answer_match.group(1).strip()
        logging.info(f"Найден ответ: {answer}")
        return format_latex_answer(answer)
    
    # Если не нашли по первому паттерну, ищем альтернативы
    alternative_patterns = [
        r"(?:Итоговый ответ|итоговый ответ|Итого|итого)\s*:(.+?)(?:$|\.|\n|\<\/p\>)",
        r"(?:Таким образом|Итак|Окончательный ответ|окончательный ответ)\s*:(.+?)(?:$|\.|\n|\<\/p\>)",
        r"(?:В ответе получаем|в ответе получим|В ответе|в ответе)\s*:(.+?)(?:$|\.|\n|\<\/p\>)"
    ]
    
    for pattern in alternative_patterns:
        alt_match = re.search(pattern, solution, re.IGNORECASE | re.DOTALL)
        if alt_match:
            answer = alt_match.group(1).strip()
            logging.info(f"Найден ответ: {answer}")
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
    # Если ответ уже содержит $, заменяем их на $$
    if '$' in answer:
        # Заменяем одинарные $ на двойные $$
        answer = answer.replace('$', '$$')
    else:
        # Проверяем, есть ли в ответе формулы LaTeX и корректируем их
        # Ищем выражения без окружения $ и оборачиваем их
        formula_pattern = r'(\\frac|\\sqrt|\\sum|\\prod|\\int|\\lim|\\sin|\\cos|\\tan|\\log|\\ln)'
        answer = re.sub(formula_pattern, r'$$\1', answer)
        
        # Если мы добавили открывающий символ $$, но нет закрывающего, добавляем его
        open_count = answer.count('$$')
        if open_count % 2 != 0:
            answer += '$$'
        
    # Экранируем угловые скобки, если они не являются частью HTML-тега
    if '<' in answer and not re.search(r'<[a-z/]', answer):
        answer = answer.replace('<', '&lt;').replace('>', '&gt;')
        
    return answer

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
    Генерирует координатную плоскость с точками, векторами или графиками функций.
    
    Args:
        points: Список точек в формате [(x1,y1,label1), (x2,y2,label2), ...]
        functions: Список функций в формате [('x**2', 'blue'), ('2*x+1', 'red'), ...]
        vectors: Список векторов в формате [(x1,y1,x2,y2,label), ...]
        grid: Отображать ли сетку
        filename: Имя файла для сохранения
        
    Returns:
        str: Путь к сохраненному изображению
    """
    try:
        # Создаем новую фигуру
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Настраиваем оси
        ax.spines['left'].set_position('zero')
        ax.spines['bottom'].set_position('zero')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
        # Определяем пределы осей
        x_min, x_max = -10, 10
        y_min, y_max = -10, 10
        
        # Если заданы точки, корректируем пределы
        if points:
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            if x_coords:
                x_min = min(x_min, min(x_coords) - 1)
                x_max = max(x_max, max(x_coords) + 1)
            if y_coords:
                y_min = min(y_min, min(y_coords) - 1)
                y_max = max(y_max, max(y_coords) + 1)
        
        # Устанавливаем пределы осей
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        
        # Добавляем сетку, если нужно
        if grid:
            ax.grid(True, alpha=0.3)
        
        # Рисуем стрелки на осях
        arrow_props = dict(arrowstyle='->', linewidth=1.5)
        ax.annotate('', xy=(1, 0), xytext=(0, 0), arrowprops=arrow_props)
        ax.annotate('', xy=(0, 1), xytext=(0, 0), arrowprops=arrow_props)
        
        # Подписываем оси
        ax.text(x_max - 0.5, 0.5, 'x', fontsize=12)
        ax.text(0.5, y_max - 0.5, 'y', fontsize=12)
        
        # Отображаем точки
        if points:
            for point in points:
                x, y = point[0], point[1]
                # Подписываем точку только если есть метка
                if len(point) > 2 and point[2]:
                    label = point[2]
                    ax.plot(x, y, 'o', markersize=6)
                    ax.text(x + 0.2, y + 0.2, label, fontsize=10)
                else:
                    # Если нет метки, просто рисуем точку без подписи
                    ax.plot(x, y, 'o', markersize=6)
        
        # Отображаем функции без подписей на графике
        if functions:
            x = np.linspace(x_min, x_max, 1000)
            for func_expr, color in functions:
                # Безопасное вычисление функции
                expr = func_expr.replace('^', '**')
                for func_name in ['sin', 'cos', 'tan', 'exp', 'log', 'sqrt']:
                    expr = expr.replace(func_name, f'np.{func_name}')
                
                try:
                    y = eval(expr)
                    # Убираем метку функции, рисуем только график
                    ax.plot(x, y, color=color, linewidth=2)
                except Exception as e:
                    logging.error(f"Ошибка при вычислении функции '{func_expr}': {e}")
        
        # Отображаем векторы
        if vectors:
            for vector in vectors:
                x1, y1, x2, y2 = vector[:4]
                # Рисуем вектор
                ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                            arrowprops=dict(arrowstyle='->', linewidth=1.5, color='blue'))
                
                # Подписываем вектор только если есть метка
                if len(vector) > 4 and vector[4]:
                    label = vector[4]
                    mid_x = (x1 + x2) / 2
                    mid_y = (y1 + y2) / 2
                    ax.text(mid_x, mid_y, label, fontsize=10)
        
        # Создаем директорию для изображений, если она не существует
        images_dir = 'static/images/generated'
        os.makedirs(images_dir, exist_ok=True)
        
        # Генерируем имя файла, если не указано
        if not filename:
            filename = f"coord_{uuid.uuid4().hex[:8]}.png"
            
        filepath = os.path.join(images_dir, filename)
        
        # Сохраняем изображение
        plt.savefig(filepath, dpi=100, bbox_inches='tight')
        plt.close()  # Закрываем фигуру, чтобы очистить память
        
        return filepath
    except Exception as e:
        logging.error(f"Ошибка при генерации координатной плоскости: {e}")
        return None

def generate_graph_image(function_expr, x_range=(-10, 10), y_range=None, filename=None):
    """
    Генерирует изображение графика функции.
    
    Args:
        function_expr: Строка с выражением функции (например, "x**2 - 3*x + 2")
        x_range: Диапазон значений x (min, max)
        y_range: Диапазон значений y (min, max)
        filename: Имя файла для сохранения
        
    Returns:
        str: Путь к сохраненному изображению
    """
    try:
        # Создаем новую фигуру
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Настраиваем оси
        ax.spines['left'].set_position('zero')
        ax.spines['bottom'].set_position('zero')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
        # Создаем массив значений x
        x = np.linspace(x_range[0], x_range[1], 1000)
        
        # Безопасное вычисление функции
        expr = function_expr.replace('^', '**')
        for func_name in ['sin', 'cos', 'tan', 'exp', 'log', 'sqrt']:
            expr = expr.replace(func_name, f'np.{func_name}')
        
        try:
            # Вычисляем значения функции
            y = eval(expr)
            
            # Строим график
            ax.plot(x, y, 'b-', linewidth=2)
            
            # Устанавливаем пределы осей
            ax.set_xlim(x_range)
            if y_range:
                ax.set_ylim(y_range)
            
            # Добавляем сетку
            ax.grid(True, linestyle='--', alpha=0.6)
            
            # Убираем лишние подписи функции
            ax.set_title("")
            
            # Только подписи осей
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            
            # Создаем директорию для изображений, если она не существует
            images_dir = 'static/images/generated'
            os.makedirs(images_dir, exist_ok=True)
            
            # Генерируем имя файла, если не указано
            if not filename:
                filename = f"graph_{uuid.uuid4().hex[:8]}.png"
                
            filepath = os.path.join(images_dir, filename)
            
            # Сохраняем изображение
            plt.savefig(filepath, dpi=100, bbox_inches='tight')
            plt.close()  # Закрываем фигуру, чтобы очистить память
            
            return filepath
        except Exception as e:
            logging.error(f"Ошибка при вычислении функции '{function_expr}': {e}")
            return None
            
    except Exception as e:
        logging.error(f"Ошибка при генерации графика: {e}")
        return None

import numpy as np
import matplotlib.pyplot as plt
import os, uuid, logging

def draw_polygon(ax, pts, edgecolor='blue', lw=2):
    """Рисует неподписанный полигон по списку точек."""
    xs, ys = zip(*pts)
    pts_closed = np.vstack([pts, pts[0]])
    ax.plot(pts_closed[:,0], pts_closed[:,1], color=edgecolor, linewidth=lw)
    return xs, ys

def compute_parallelogram(base, height, skew_deg):
    skew = np.radians(skew_deg)
    dx = height / np.tan(skew)
    return [(0,0), (base,0), (base+dx,height), (dx,height)]

def compute_trapezoid(bottom, top, height, is_isosceles=False):
    """
    Вычисляет координаты вершин трапеции.
    
    Args:
        bottom: Длина нижнего основания
        top: Длина верхнего основания
        height: Высота трапеции
        is_isosceles: Является ли трапеция равнобедренной
        
    Returns:
        list: Список координат вершин [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
    """
    if is_isosceles:
        # Для равнобедренной трапеции боковые стороны равны
        dx = (bottom - top) / 2
        return [(0, 0), (bottom, 0), (bottom - dx, height), (dx, height)]
    else:
        # Для произвольной трапеции (по умолчанию смещение верхнего основания слева)
        dx = (bottom - top)/2
        return [(0, 0), (bottom, 0), (bottom - dx, height), (dx, height)]

def add_vertex_labels(params, figure_type, pts):
    if 'vertex_labels' not in params:
        if figure_type == 'triangle':
            labels = ['A', 'B', 'C']
        else:
            labels = ['A', 'B', 'C', 'D']
        params['vertex_labels'] = labels

def generate_geometric_figure(figure_type, params, filename=None):
    print(f"generate_geometric_figure: {figure_type}, {params}, {filename}")
    """
    Универсальный генератор геометрических фигур, использующий matplotlib.patches.
    Поддерживает отображение длин сторон в соответствии с параметрами.
    """
    try:
        # Увеличиваем размер фигуры для лучшего качества
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_aspect('equal')
        ax.axis('off')

        # Создаем фигуру и точки в зависимости от типа
        pts = None
        patch = None

        if figure_type == 'circle':
            cx, cy = params.get('center', (0,0))
            r = params.get('radius', 3)
            patch = plt.Circle((cx, cy), r, fill=False, edgecolor='blue', linewidth=2)
            ax.add_patch(patch)
            xs, ys = [cx-r, cx+r], [cy-r, cy+r]
            
            # Отображение центра
            if params.get('show_center', True):
                ax.text(cx, cy, params.get('center_label', 'O'),
                       ha='center', va='center', fontsize=14,  # Увеличен размер шрифта
                       bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
            
            # Отображение радиуса, если задан
            radius_value = params.get('radius_value', None)
            if params.get('show_radius', False) or radius_value is not None:
                ax.plot([cx, cx+r], [cy, cy], 'r-', lw=1.5)  # Увеличена толщина
                displayed_radius = radius_value if radius_value is not None else r
                ax.text(cx+r/2, cy+0.3, f"r={displayed_radius}", ha='center', fontsize=12)  # Увеличен размер шрифта
                
            # Отображение диаметра, если задан
            diameter_value = params.get('diameter_value', None)
            if params.get('show_diameter', False) or diameter_value is not None:
                ax.plot([cx-r, cx+r], [cy, cy], 'g-', lw=1.5)  # Увеличена толщина
                displayed_diameter = diameter_value if diameter_value is not None else 2*r
                ax.text(cx, cy-0.3, f"d={displayed_diameter}", ha='center', fontsize=12)  # Увеличен размер шрифта
                
            # Отображение хорды, если задано значение (исправлено)
            chord_value = params.get('chord_value', None)
            if params.get('show_chord', False) or chord_value is not None:
                if chord_value is not None:
                    # Проверяем, что хорда не больше диаметра
                    chord_value = min(chord_value, 2*r)
                    
                    # Вычисляем положение хорды
                    half_chord = chord_value / 2
                    
                    # Расстояние от центра до хорды (по теореме Пифагора)
                    if half_chord < r:  # Защита от ошибок вычисления
                        h = np.sqrt(r**2 - half_chord**2)
                    else:
                        h = 0
                    
                    # Рисуем хорду горизонтально ниже центра
                    chord_y = cy - h
                    chord_start_x = cx - half_chord
                    chord_end_x = cx + half_chord
                    
                    # Рисуем хорду
                    ax.plot([chord_start_x, chord_end_x], [chord_y, chord_y], 'b-', lw=1.5)
                    
                    # Подпись расположена четко под хордой
                    ax.text((chord_start_x + chord_end_x)/2, chord_y-0.4, 
                            f"{chord_value}", ha='center', fontsize=12)
                    
        else:  # Многоугольники
            if 'points' in params:
                pts = params['points']
            else:
                if figure_type == 'triangle':
                    pts = params.get('points', [(0,0),(1,0),(0.5,0.86)])
                    
                    # Для прямоугольного треугольника
                    if params.get('is_right', False):
                        pts = [(0,0), (0,3), (4,0)]  # Прямоугольный треугольник
                        
                elif figure_type == 'rectangle':
                    x, y = params.get('x',0), params.get('y',0)
                    w, h = params.get('width',4), params.get('height',3)
                    pts = [(x,y), (x+w,y), (x+w,y+h), (x,y+h)]
                    
                elif figure_type == 'parallelogram':
                    base = params.get('width',4)
                    height = params.get('height',3)
                    skew = params.get('skew',60)
                    raw = compute_parallelogram(base, height, skew)
                    x, y = params.get('x',0), params.get('y',0)
                    pts = [(px+x, py+y) for px,py in raw]
                    
                elif figure_type == 'trapezoid':
                    bottom = params.get('bottom_width',6)
                    top = params.get('top_width',3)
                    height = params.get('height',3)
                    is_isosceles = params.get('is_isosceles', False)
                    raw = compute_trapezoid(bottom, top, height, is_isosceles)
                    x, y = params.get('x',0), params.get('y',0)
                    pts = [(px+x, py+y) for px,py in raw]
                else:
                    raise ValueError(f"Неизвестный тип: {figure_type}")

            # Создаем патч для многоугольника
            if pts:
                xs, ys = zip(*pts)
                patch = plt.Polygon(pts, fill=False, edgecolor='blue', linewidth=2)
                ax.add_patch(patch)
                
                # Подписи вершин
                if params.get('show_labels', True):
                    # Если есть конкретные метки, которые нужно показать
                    show_specific_labels = params.get('show_specific_labels', None)
                    labels = params.get('vertex_labels')
                    if not labels:
                        labels = [chr(65+i) for i in range(len(pts))]
                    
                    for i, ((x0,y0), lab) in enumerate(zip(pts, labels)):
                        # Проверяем, нужно ли отображать эту конкретную метку
                        if show_specific_labels is None or lab in show_specific_labels:
                            ax.text(x0, y0, lab, ha='center', va='center', fontsize=14,
                               bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
                
                # Длины сторон
                side_lengths = params.get('side_lengths', None)
                show_lengths = params.get('show_lengths', False)
                show_specific_sides = params.get('show_specific_sides', None)
                
                if side_lengths or show_lengths:
                    for i in range(len(pts)):
                        x0, y0 = pts[i]
                        x1, y1 = pts[(i+1)%len(pts)]
                        mx, my = (x0+x1)/2, (y0+y1)/2
                        
                        # Рассчитываем длину стороны
                        L = np.hypot(x1-x0, y1-y0)
                        
                        # Получаем обозначения вершин для этой стороны
                        v1 = params.get('vertex_labels', [chr(65+j) for j in range(len(pts))])[i]
                        v2 = params.get('vertex_labels', [chr(65+j) for j in range(len(pts))])[(i+1)%len(pts)]
                        side_name = f"{v1}{v2}"
                        side_name_rev = f"{v2}{v1}"  # Обратный порядок для проверки
                        
                        # Проверяем, нужно ли отображать эту конкретную сторону
                        should_show = show_specific_sides is None or side_name in show_specific_sides or side_name_rev in show_specific_sides
                        
                        if should_show:
                            # Вектор нормали к стороне для размещения текста (увеличен вынос подписи)
                            nx, ny = -(y1-y0)/L, (x1-x0)/L
                            offset = 0.35  # Увеличен отступ для лучшей видимости
                        
                            # Особая обработка для трапеции
                            if figure_type == 'trapezoid':
                                # Нижнее основание (AB)
                                if i == 0 and ('bottom_width' in params):
                                    ax.text(mx+nx*offset, my+ny*offset, f"{params['bottom_width']}", 
                                        ha='center', fontsize=12,
                                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
                                    continue
                                # Верхнее основание (DC)
                                elif i == 2 and ('top_width' in params):
                                    ax.text(mx+nx*offset, my+ny*offset, f"{params['top_width']}", 
                                        ha='center', fontsize=12,
                                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
                                    continue
                                # Для боковых сторон трапеции не отображаем значения, если указаны конкретные стороны
                                # и текущая сторона не является верхним или нижним основанием
                                elif show_specific_sides is not None and side_name not in ["AB", "BA", "DC", "CD"]:
                                    continue
                            
                        # Выбираем значение для отображения
                        if side_lengths and i < len(side_lengths) and side_lengths[i] is not None:
                            # Если указано конкретное значение
                            ax.text(mx+nx*offset, my+ny*offset, f"{side_lengths[i]}", 
                                   ha='center', fontsize=12,  # Увеличен размер шрифта
                                   bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
                        elif show_lengths:
                            # Показываем фактическую длину
                            ax.text(mx+nx*offset, my+ny*offset, f"{L:.2f}", 
                                   ha='center', fontsize=12,  # Увеличен размер шрифта
                                   bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
                
                # Углы (для любого многоугольника)
                if params.get('show_angles', False):
                    angle_values = params.get('angle_values', None)
                    show_angle_arcs = params.get('show_angle_arcs', False) or params.get('rисовать_дуги_углов', False)
                    show_specific_angles = params.get('show_specific_angles', None)
                    
                    for i in range(len(pts)):
                        # Получаем обозначение вершины для проверки
                        vertex_label = params.get('vertex_labels', [chr(65+j) for j in range(len(pts))])[i]
                        
                        # Проверяем, нужно ли отображать этот угол
                        if show_specific_angles is not None and vertex_label not in show_specific_angles:
                            continue
                        
                        # Получаем три последовательные точки для вычисления угла
                        A, B, C = np.array(pts[(i-1)%len(pts)]), np.array(pts[i]), np.array(pts[(i+1)%len(pts)])
                        
                        # Вычисляем векторы от вершины к соседним точкам
                        v1, v2 = A-B, C-B
                        
                        # Вычисляем угол в градусах
                        if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:  # Проверка на нулевой вектор
                            cos_angle = v1.dot(v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
                            # Ограничиваем значение в диапазоне [-1, 1] для arccos
                            cos_angle = max(-1, min(1, cos_angle))
                            ang = np.degrees(np.arccos(cos_angle))
                            
                            # Проверяем внутренний или внешний угол
                            # Вычисляем векторное произведение для определения типа угла
                            cross_product = np.cross(np.append(v1, 0), np.append(v2, 0))[2]
                            
                            # Определяем, внутренний или внешний угол
                            # Для многоугольников всегда хотим показывать внутренний угол (< 180°)
                            if cross_product > 0:  # Если угол внешний
                                    ang = 360 - ang
                            
                            # Используем значение угла из параметров, если оно предоставлено
                            if angle_values and i < len(angle_values) and angle_values[i] is not None:
                                displayed_angle = angle_values[i]
                            else:
                                displayed_angle = round(ang, 1)
                            
                            # Размещение текста - усреднение направлений
                            uv = (v1/np.linalg.norm(v1) + v2/np.linalg.norm(v2))
                            if np.linalg.norm(uv) > 0:  # Проверка на нулевой вектор
                                uv = uv/np.linalg.norm(uv) * 0.4  # Увеличено для лучшей видимости
                                
                                # Отображение значения угла
                                ax.text(*(B+uv), f"{displayed_angle}°", ha='center', fontsize=12,
                                       bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
                                
                                # Рисуем дугу угла только если это явно указано
                                if show_angle_arcs:
                                    theta1 = np.arctan2(v1[1], v1[0]) * 180 / np.pi
                                    theta2 = np.arctan2(v2[1], v2[0]) * 180 / np.pi
                                
                                # Обеспечиваем правильное направление дуги
                                if theta2 < theta1:
                                    theta2 += 360
                                
                                # Регулируем радиус дуги в зависимости от размера фигуры
                                radius = min(np.linalg.norm(v1), np.linalg.norm(v2)) * 0.2
                                arc = patches.Arc(B, 2*radius, 2*radius, 
                                                 theta1=theta1, theta2=theta2, 
                                                 color='red', linewidth=1.5)
                                ax.add_patch(arc)
                
                # Специальные элементы для треугольника
                if figure_type == 'triangle' and len(pts) == 3:
                    # Высоты
                    show_heights = params.get('show_heights', False)
                    show_specific_heights = params.get('show_specific_heights', None)
                    
                    if show_heights or show_specific_heights:
                        for i in range(3):
                            # Получаем вершину и противоположную сторону
                            A = np.array(pts[i])
                            B = np.array(pts[(i+1)%3])
                            C = np.array(pts[(i+2)%3])
                            
                            # Получаем обозначение вершины для проверки
                            vertex_label = params.get('vertex_labels', ['A', 'B', 'C'])[i]
                            
                            # Проверяем, нужно ли отображать эту высоту
                            should_show = show_specific_heights is None or vertex_label in show_specific_heights
                            
                            if should_show:
                                # Вычисляем вектор стороны BC
                                BC = C - B
                                
                                # Вычисляем нормаль к стороне BC
                                n = np.array([-BC[1], BC[0]])
                                n = n / np.linalg.norm(n)
                                
                                # Проекция вектора BA на нормаль
                                BA = A - B
                                h = np.abs(np.dot(BA, n))
                                
                                # Вычисляем направление от вершины к стороне
                                D = B + np.dot(BA, BC) / np.dot(BC, BC) * BC
                                
                                # Рисуем высоту
                                ax.plot([A[0], D[0]], [A[1], D[1]], 'g-', lw=1.2)
                                
                                # Подписываем высоту
                                ax.text((A[0] + D[0])/2 + 0.1, (A[1] + D[1])/2 + 0.1, 
                                        f"h{vertex_label}={h:.2f}" if h != int(h) else f"h{vertex_label}={int(h)}", 
                                        ha='center', fontsize=10,
                                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
                    
                    # Медианы
                    show_medians = params.get('show_medians', False)
                    show_specific_medians = params.get('show_specific_medians', None)
                    
                    if show_medians or show_specific_medians:
                        for i in range(3):
                            # Получаем вершину и противоположную сторону
                            A = np.array(pts[i])
                            B = np.array(pts[(i+1)%3])
                            C = np.array(pts[(i+2)%3])
                            
                            # Получаем обозначение вершины для проверки
                            vertex_label = params.get('vertex_labels', ['A', 'B', 'C'])[i]
                            
                            # Проверяем, нужно ли отображать эту медиану
                            should_show = show_specific_medians is None or vertex_label in show_specific_medians
                            
                            if should_show:
                                # Вычисляем середину противоположной стороны
                                M = (B + C) / 2
                                
                                # Длина медианы
                                median_length = np.linalg.norm(A - M)
                                
                                # Рисуем медиану
                                ax.plot([A[0], M[0]], [A[1], M[1]], 'm-', lw=1.2)
                                
                                # Подписываем медиану
                                ax.text((A[0] + M[0])/2 + 0.1, (A[1] + M[1])/2 + 0.1, 
                                        f"m{vertex_label}={median_length:.2f}" if median_length != int(median_length) else f"m{vertex_label}={int(median_length)}", 
                                        ha='center', fontsize=10,
                                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
                    
                    # Средние линии
                    show_midlines = params.get('show_midlines', False)
                    show_specific_midlines = params.get('show_specific_midlines', None)
                    
                    if show_midlines or show_specific_midlines:
                        # Для треугольника есть 3 средние линии
                        midlines = [
                            ((pts[1][0] + pts[2][0])/2, (pts[1][1] + pts[2][1])/2, (pts[0][0] + pts[1][0])/2, (pts[0][1] + pts[1][1])/2),
                            ((pts[0][0] + pts[2][0])/2, (pts[0][1] + pts[2][1])/2, (pts[1][0] + pts[2][0])/2, (pts[1][1] + pts[2][1])/2),
                            ((pts[0][0] + pts[1][0])/2, (pts[0][1] + pts[1][1])/2, (pts[0][0] + pts[2][0])/2, (pts[0][1] + pts[2][1])/2)
                        ]
                        
                        midline_labels = ['BC', 'AC', 'AB']
                        
                        for i, (x1, y1, x2, y2) in enumerate(midlines):
                            # Проверяем, нужно ли отображать эту среднюю линию
                            midline_name = midline_labels[i]
                            should_show = show_specific_midlines is None or midline_name in show_specific_midlines
                            
                            if should_show:
                                # Длина средней линии
                                midline_length = np.hypot(x2-x1, y2-y1)
                                
                                # Рисуем среднюю линию
                                ax.plot([x1, x2], [y1, y2], 'c-', lw=1.2)
                                
                                # Подписываем среднюю линию
                                ax.text((x1 + x2)/2 + 0.1, (y1 + y2)/2 + 0.1, 
                                        f"m{midline_name}={midline_length:.2f}" if midline_length != int(midline_length) else f"m{midline_name}={int(midline_length)}", 
                                        ha='center', fontsize=10,
                                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
                
                # Прямые углы для четырехугольников
                if figure_type in ['rectangle', 'parallelogram', 'trapezoid'] and params.get('is_right', False):
                    # Обработка прямых углов в четырехугольниках
                    for i in range(len(pts)):
                        A, B, C = np.array(pts[(i-1)%len(pts)]), np.array(pts[i]), np.array(pts[(i+1)%len(pts)])
                        v1, v2 = A-B, C-B
                        
                        if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                            cos_angle = v1.dot(v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
                            cos_angle = max(-1, min(1, cos_angle))
                            ang = np.degrees(np.arccos(cos_angle))
                            
                            if abs(ang - 90) < 5:  # Допуск 5 градусов для прямого угла
                                # Рисуем символ прямого угла
                                v1 = v1 / np.linalg.norm(v1) * 0.4  # Увеличен размер символа прямого угла
                                v2 = v2 / np.linalg.norm(v2) * 0.4
                                
                                p1 = np.array(B)
                                p2 = p1 + v1
                                p3 = p1 + v1 + v2
                                p4 = p1 + v2
                                
                                ax.plot([p1[0], p2[0], p3[0], p4[0], p1[0]], 
                                       [p1[1], p2[1], p3[1], p4[1], p1[1]], 'r-', linewidth=1.5)

        # Авто-лимиты для осей
        if 'xs' in locals() and 'ys' in locals():
            m = 1.5  # Увеличен отступ от краев
            ax.set_xlim(min(xs)-m, max(xs)+m)
            ax.set_ylim(min(ys)-m, max(ys)+m)

        # Сохранение изображения
        if filename and os.path.dirname(filename):
            # Если filename содержит путь, используем его напрямую
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            plt.savefig(filename, bbox_inches='tight', dpi=150)
            plt.close()
            return filename
        else:
            # Если путь не указан, используем директорию по умолчанию
            output_dir = os.path.join('static', 'images', 'generated')
            os.makedirs(output_dir, exist_ok=True)
            
            if not filename:
                filename = f"{figure_type}_{uuid.uuid4().hex[:8]}.png"
            
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, bbox_inches='tight', dpi=150)
            plt.close()
            return filepath
    except Exception as e:
        import traceback
        print(f"Ошибка при создании геометрической фигуры: {e}")
        print(traceback.format_exc())
        return None


def get_image_base64(image_path):
    """
    Преобразует изображение в строку base64 для встраивания в HTML.
    
    Args:
        image_path: Путь к изображению
        
    Returns:
        str: Строка в формате base64
    """
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except Exception as e:
        logging.error(f"Ошибка при конвертации изображения в base64: {e}")
        return None

def remove_latex_markup(text):
    """
    Удаляет LaTeX-разметку и символы из текста.
    
    Args:
        text: Исходный текст с LaTeX-разметкой
        
    Returns:
        str: Очищенный текст
    """
    if text is None:
        return None
        
    if isinstance(text, list):
        # Если text это список, обрабатываем каждый элемент
        return [remove_latex_markup(item) for item in text]
        
    if not isinstance(text, str):
        return text
        
    # Удаляем символы $ и другие LaTeX-разметки
    text = text.replace('$', '')
    
    # Заменяем LaTeX-операторы на Python
    text = text.replace('^', '**')  # Степень
    text = text.replace('\\cdot', '*')  # Умножение
    text = text.replace('\\times', '*')  # Умножение
    text = text.replace('\\frac{', '(')  # Начало дроби
    text = text.replace('}{', ')/(')     # Середина дроби
    text = text.replace('}', ')')        # Конец дроби
    
    # Для корней
    text = text.replace('\\sqrt{', 'sqrt(')
    
    # Тригонометрические функции
    text = text.replace('\\sin', 'sin')
    text = text.replace('\\cos', 'cos')
    text = text.replace('\\tan', 'tan')
    text = text.replace('\\tg', 'tan')
    text = text.replace('\\ctg', '1/tan')
    
    # Логарифмы
    text = text.replace('\\ln', 'log')
    text = text.replace('\\log', 'log10')
    
    return text.strip()

def process_visualization_params(params_text):
    """
    Обрабатывает параметры для визуализации из текста.
    
    Args:
        params_text: Текст с параметрами визуализации
        
    Returns:
        tuple: (изображение, тип визуализации) или (None, None) в случае ошибки
    """
    try:
        if not params_text:
            return None, None
        
        lines = params_text.strip().split('\n')
        viz_type = None
        
        # Ищем тип визуализации
        for line in lines:
            if line.lower().startswith('тип:'):
                viz_type = line.split(':', 1)[1].strip().lower()
                break
            if line.lower().startswith('тип фигуры:'):  
                viz_type = line.split(':', 1)[1].strip().lower()
                break
            if re.search(r'фигура\s*:', line.lower()):
                viz_type = re.split(r'фигура\s*:', line.lower(), 1)[1].strip()
                break
        
        if not viz_type:
            logging.warning("Не удалось определить тип визуализации")
            return None, None
        
        # Функция для извлечения параметра по шаблону
        def extract_param(pattern, text, default=None):
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                try:
                    value = match.group(1).strip()
                    # Очищаем от LaTeX разметки
                    value = remove_latex_markup(value)
                    return value
                except Exception as e:
                    logging.error(f"Ошибка при извлечении параметра: {e}")
            return default
        
        if "график" in viz_type:
            try:
                # Используем общую функцию parse_graph_params для парсинга параметров
                funcs_to_plot, x_range, y_range, special_points = parse_graph_params(params_text)
                
                # Создаем директорию для сохранения изображения, если она не существует
                output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static/images/generated")
                os.makedirs(output_dir, exist_ok=True)
                
                # Генерируем имя файла с уникальным идентификатором
                filename = f'multifunction_{uuid.uuid4().hex[:8]}.png'
                output_path = os.path.join(output_dir, filename)
                
                # Логируем для отладки
                x_min, x_max = x_range
                y_min, y_max = (None, None) if y_range is None else y_range
                logging.info(f"Обработка функций с параметрами: x: [{x_min}, {x_max}], " +
                             f"y: [{y_min if y_min is not None else 'auto'}, {y_max if y_max is not None else 'auto'}]")
                
                # Отрисовываем график
                filepath = generate_multi_function_graph(funcs_to_plot, x_range=x_range, y_range=y_range, special_points=special_points)
                return filepath, "graph"
            except Exception as e:
                logging.error(f"Ошибка при создании графика функций: {e}")
                logging.error(traceback.format_exc())
                return None, None
                
        elif "треугольник" in viz_type:
            image_path = process_triangle_visualization(params_text, extract_param)
            return image_path, "triangle"
        elif "окружность" in viz_type:
            image_path = process_circle_visualization(params_text, extract_param)
            return image_path, "circle"
        elif "прямоугольник" in viz_type:
            image_path = process_rectangle_visualization(params_text, extract_param)
            return image_path, "rectangle"
        elif "параллелограмм" in viz_type:
            image_path = process_parallelogram_visualization(params_text, extract_param)
            return image_path, "parallelogram"
        elif "трапеция" in viz_type:
            image_path = process_trapezoid_visualization(params_text, extract_param)
            return image_path, "trapezoid"
        elif "координатная плоскость" in viz_type:
            image_path = process_coordinate_visualization(params_text, extract_param)
            return image_path, "coordinate"
        else:
            logging.warning(f"Неизвестный тип визуализации: {viz_type}")
            return None, None
            
    except Exception as e:
        logging.error(f"Ошибка при создании графика функции: {e}")
        logging.error(traceback.format_exc())
        return None, None

def process_coordinate_visualization(params_text, extract_param):
    """Обрабатывает параметры для координатной плоскости"""
    # Извлекаем параметры координатной плоскости
    points_str = extract_param(REGEX_PATTERNS["coordinate"]["points"], params_text)
    vectors_str = extract_param(REGEX_PATTERNS["coordinate"]["vectors"], params_text)
    functions_str = extract_param(REGEX_PATTERNS["coordinate"]["functions"], params_text)
    
    points = []
    vectors = []
    functions = []
    
    # Парсинг точек
    if points_str:
        try:
            # Формат A(1,2),B(3,4),C(5,6) или (1,2),(3,4),(5,6)
            for match in re.finditer(r'([A-Za-z]?)\s*\(([^,]+),([^)]+)\)', points_str):
                label, x, y = match.groups()
                x, y = float(x.strip()), float(y.strip())
                if label:
                    points.append((x, y, label))
                else:
                    points.append((x, y))
        except Exception as e:
            logging.warning(f"Ошибка при разборе точек: {e}")
    
    # Парсинг векторов
    if vectors_str:
        try:
            # Формат AB, CD или (1,2,3,4), (5,6,7,8)
            for match in re.finditer(r'([A-Za-z]{2})|(?:\(([^,]+),([^,]+),([^,]+),([^)]+)\))', vectors_str):
                if match.group(1):  # Формат AB
                    vector_name = match.group(1)
                    # Нужно найти точки A и B среди уже введенных
                    start_point = None
                    end_point = None
                    for point in points:
                        if len(point) > 2:  # Есть метка
                            if point[2] == vector_name[0]:
                                start_point = (point[0], point[1])
                            elif point[2] == vector_name[1]:
                                end_point = (point[0], point[1])
                    
                    if start_point and end_point:
                        vectors.append((start_point[0], start_point[1], end_point[0], end_point[1], vector_name))
                else:  # Формат (1,2,3,4)
                    x1, y1, x2, y2 = map(float, match.groups()[1:5])
                    vectors.append((x1, y1, x2, y2))
        except Exception as e:
            logging.warning(f"Ошибка при разборе векторов: {e}")
    
    # Парсинг функций
    if functions_str:
        try:
            # Формат x**2, 2*x+1
            for func in functions_str.split(','):
                func = func.strip()
                if func:
                    functions.append((func, 'blue'))  # Цвет по умолчанию - синий
        except Exception as e:
            logging.warning(f"Ошибка при разборе функций: {e}")
    
    # Создаем директорию для сохранения изображения, если она не существует
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static/images/generated")
    os.makedirs(output_dir, exist_ok=True)
    
    # Генерируем имя файла с уникальным идентификатором
    filename = f'coordinate_{uuid.uuid4().hex[:8]}.png'
    output_path = os.path.join(output_dir, filename)
    
    # Генерируем координатную плоскость
    generate_coordinate_system(points, functions, vectors, filename=output_path)
    
    return output_path

# Пример использования
if __name__ == "__main__":
    # Создаем .env файл с API ключом, если его нет
    if not os.path.exists(".env"):
        with open(".env", "w") as f:
            f.write("YANDEX_API_KEY=your_api_key_here\n")
            f.write("YANDEX_FOLDER_ID=your_folder_id_here")
        print("Создан файл .env. Пожалуйста, заполните в нем свой API ключ Яндекса и ID каталога.")
    
    # Тестируем генерацию задачи при наличии API ключа
    if YANDEX_API_KEY and YANDEX_FOLDER_ID:
        category = "Простейшие уравнения"
        result = generate_complete_task(category)
        
        print("\n=== СГЕНЕРИРОВАННАЯ ЗАДАЧА ===")
        print(result.get("task", "Ошибка генерации"))
        
        print("\n=== РЕШЕНИЕ ===")
        print(result.get("solution", "Решение недоступно"))
        
        print("\n=== ОТВЕТ ===")
        print(result.get("answer", "Ответ не найден"))
        
        print("\n=== ПОДСКАЗКИ ===")
        for i, hint in enumerate(result.get("hints", []), 1):
            print(f"Подсказка {i}: {hint}")
            
        print(f"\nПолный текст задачи сохранен в файл 'last_generated_task.txt'")
    else:
        print("API ключ Яндекса или ID каталога не найдены в .env файле.")
        print("Пожалуйста, заполните эти данные для работы с YandexGPT API.")

def needs_visualization(task_text, category, subcategory, is_basic_level=False):
    """
    Определяет, требуется ли визуализация для задачи.
    Делегирует вызов в check_visualization_requirement из prompts.
    Учитывает также уровень сложности задачи (базовый или профильный).
    
    Args:
        task_text: Текст задачи
        category: Категория задачи
        subcategory: Подкатегория задачи
        is_basic_level: Флаг базового уровня, но не влияет на решение о необходимости визуализации
        
    Returns:
        bool: True, если требуется визуализация (определяется по категории/тексту задачи)
    """
    from app.prompts import check_visualization_requirement
    # Независимо от уровня сложности (базовый или профильный), 
    # запрашиваем визуализацию если она требуется
    return check_visualization_requirement(category, subcategory, task_text)

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

def determine_visualization_type(task_text, category, subcategory):
    """
    Определяет тип визуализации на основе текста задачи и категории.
    
    Args:
        task_text: Текст задачи
        category: Категория задачи
        subcategory: Подкатегория задачи
        
    Returns:
        str: Тип визуализации или None, если не удалось определить
    """
    # Ключевые слова для различных типов визуализации
    type_keywords = {
        "graph": ["график", "функц", "парабол", "гипербол", "синусоид", "косинусоид", 
                 "экспонент", "y =", "f(x) =", "ось абсцисс", "ось ординат"],
        "triangle": ["треугольник", "равнобедренн", "равносторонн", "прямоугольный треугольник", 
                    "катет", "гипотенуз", "медиан", "биссектрис", "высот"],
        "circle": ["окружность", "круг", "радиус", "диаметр", "хорд", "сектор", "сегмент",
                  "касательн", "центр окружности"],
        "rectangle": ["прямоугольник", "квадрат", "ромб", "периметр прямоугольника", 
                     "площадь прямоугольника", "сторона прямоугольника"],
        "parallelogram": ["параллелограмм", "ромб", "периметр параллелограмма", 
                         "площадь параллелограмма", "сторона параллелограмма"],
        "trapezoid": ["трапеци", "основание трапеции", "боковая сторона трапеции", 
                     "высота трапеции", "средняя линия трапеции"],
        "coordinate": ["координатн", "декартов", "система координат", "коорд.", "точка", 
                      "точки", "вектор", "прямая", "отрезок", "вектор"]
    }
    
    # Категории, связанные с типами визуализации
    category_types = {
        "graph": ["Функции", "Графики функций", "Производная", "Интеграл"],
        "triangle": ["Треугольники", "Планиметрия", "Геометрия"],
        "circle": ["Окружности", "Планиметрия", "Геометрия"],
        "rectangle": ["Многоугольники", "Планиметрия", "Геометрия"],
        "parallelogram": ["Многоугольники", "Планиметрия", "Геометрия"],
        "trapezoid": ["Многоугольники", "Планиметрия", "Геометрия"],
        "coordinate": ["Координатная плоскость", "Векторы", "Геометрия"]
    }
    
    # Проверяем категорию и подкатегорию
    for viz_type, cats in category_types.items():
        if any(cat.lower() in category.lower() for cat in cats) or \
           any(cat.lower() in (subcategory or "").lower() for cat in cats):
            # Дополнительно проверяем текст задачи на ключевые слова для этого типа
            if any(keyword.lower() in task_text.lower() for keyword in type_keywords[viz_type]):
                return viz_type
    
    # Если не определили по категории, определяем только по тексту задачи
    type_scores = {}
    for viz_type, keywords in type_keywords.items():
        score = 0
        for keyword in keywords:
            if keyword.lower() in task_text.lower():
                score += 1
        if score > 0:
            type_scores[viz_type] = score
    
    # Возвращаем тип с наибольшим количеством совпадений
    if type_scores:
        return max(type_scores.items(), key=lambda x: x[1])[0]
    
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
            x_range_pattern = r'Диапазон X:\s*\[(.*?)\]'
            x_range_match = re.search(x_range_pattern, param_section)
            
            if x_range_match:
                try:
                    x_range_str = x_range_match.group(1)
                    x_min, x_max = map(float, x_range_str.split(','))
                    params['x_range'] = (x_min, x_max)
                except Exception as e:
                    logging.warning(f"Ошибка при разборе диапазона X: {e}")
            
            y_range_pattern = r'Диапазон Y:\s*\[(.*?)\]'
            y_range_match = re.search(y_range_pattern, param_section)
            
            if y_range_match:
                try:
                    y_range_str = y_range_match.group(1)
                    if y_range_str.lower() != 'автоматический':
                        y_min, y_max = map(float, y_range_str.split(','))
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

def generate_multi_function_graph(functions, x_range=(-10, 10), y_range=None, special_points=None):
    """
    Генерирует график с несколькими функциями.
    
    Args:
        functions: Список функций в формате [(функция1, цвет1, имя1), (функция2, цвет2, имя2), ...]
        x_range: Диапазон значений x для отображения (кортеж из двух значений)
        y_range: Диапазон значений y для отображения (кортеж из двух значений или None для авто)
        special_points: Список особых точек в формате [(x1, y1, метка1), (x2, y2, метка2), ...]
        
    Returns:
        str: Путь к сохраненному изображению
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    import uuid
    import math  # Импортируем модуль math полностью
    import traceback
    from math import sqrt, sin, cos, tan, exp, log, pi  # Импортируем все нужные математические функции
    
    # Создаем директорию для сохранения изображения, если она не существует
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static/images/generated")
    os.makedirs(output_dir, exist_ok=True)
    
    # Генерируем уникальное имя файла
    filename = f"multifunction_{uuid.uuid4().hex[:8]}.png"
    filepath = os.path.join(output_dir, filename)
    
    # Создаем график
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    
    # Настройка осей
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    
    # Стрелки на осях
    ax.plot((1), (0), ls="", marker=">", ms=10, color="k", transform=ax.get_yaxis_transform(), clip_on=False)
    ax.plot((0), (1), ls="", marker="^", ms=10, color="k", transform=ax.get_xaxis_transform(), clip_on=False)
    
    # Добавляем подписи осей
    plt.text(0.98, 0.02, "x", transform=plt.gca().transAxes, ha='right')
    plt.text(0.02, 0.98, "y", transform=plt.gca().transAxes, va='top')
    
    # Определяем ограничения для x в зависимости от типов функций
    has_sqrt = any('sqrt' in func[0] or '**0.5' in func[0] for func in functions)
    has_log = any('log' in func[0] or 'ln' in func[0] for func in functions)
    has_division = any('/' in func[0] for func in functions)
    
    # Генерируем значения x
    x_min, x_max = x_range
    
    # Для функций с корнем, корректируем диапазон x
    if has_sqrt and x_min < 0:
        # Проверяем, есть ли функции, которые требуют только положительных x
        only_positive_funcs = []
        for func in functions:
            if 'sqrt' in func[0] or '**0.5' in func[0]:
                only_positive_funcs.append(func)
        
        if len(only_positive_funcs) < len(functions):
            # Если есть и другие функции, сохраняем оригинальный диапазон
            pass
        else:
            # Если только функции с корнем, ограничиваем x неотрицательными значениями
            x_min = 0
    
    # Для функций с логарифмом, ограничиваем x положительными значениями
    if has_log and x_min <= 0:
        # Проверяем, есть ли функции, которые требуют только положительных x
        only_positive_funcs = []
        for func in functions:
            if 'log' in func[0] or 'ln' in func[0]:
                only_positive_funcs.append(func)
        
        if len(only_positive_funcs) < len(functions):
            # Если есть и другие функции, сохраняем оригинальный диапазон
            pass
        else:
            # Если только логарифмические функции, ограничиваем x положительными значениями
            x_min = 0.01
    
    # Генерируем массив значений x с большим количеством точек для более гладких графиков
    x = np.linspace(x_min, x_max, 3000)  # Увеличиваем количество точек до 3000 для более гладких графиков
    
    # Функция для безопасного вычисления значений функции
    def safe_eval(func_expr, x_val):
        """
        Безопасно вычисляет значение функции для заданного значения x.
        Обрабатывает распространенные ошибки, такие как деление на ноль,
        корень из отрицательного числа, логарифм от неположительного числа и т.д.
        
        Args:
            func_expr: Строковое выражение функции
            x_val: Значение x для вычисления
            
        Returns:
            float: Результат вычисления или NaN при ошибке
        """
        try:
            # Предварительно заменяем функции и конвертируем выражения
            # Заменяем ^, sqrt и другие математические обозначения
            func_expr = func_expr.replace('^', '**').replace('math.sqrt', 'sqrt')
            
            # Предварительные проверки для предотвращения ошибок при вычислениях
            
            # 1. Проверка на корень из отрицательного числа
            if any(sub in func_expr for sub in ['sqrt(', '**0.5']):
                # Более тщательная проверка для вложенных выражений со sqrt
                # Например, sqrt(x-5) при x < 5
                sqrt_check = False
                
                if 'sqrt(' in func_expr:
                    # Извлекаем аргументы sqrt
                    try:
                        import re
                        sqrt_args = re.findall(r'sqrt\(([^)]+)\)', func_expr)
                        for arg in sqrt_args:
                            # Создаем выражение для проверки аргумента
                            arg_expr = arg.replace('x', f'({x_val})')
                            try:
                                arg_val = eval(arg_expr, {"__builtins__": {}}, 
                                          {"math": math, "np": np, "sqrt": math.sqrt, 
                                           "sin": math.sin, "cos": math.cos, "tan": math.tan,
                                           "exp": math.exp, "log": math.log, "log10": math.log10,
                                           "abs": abs, "pi": math.pi})
                                if arg_val < 0:
                                    return float('nan')
                            except:
                                # Если не можем вычислить, продолжаем с осторожностью
                                pass
                    except:
                        # При ошибке разбора регулярного выражения продолжаем
                        pass
                
                # Дополнительная проверка для выражений с **0.5
                if '**0.5' in func_expr or '**0,5' in func_expr:
                    try:
                        # Находим выражения вида (выражение)**0.5
                        import re
                        pow_args = re.findall(r'([^*]+)\*\*(0\.5|0,5)', func_expr)
                        for arg, _ in pow_args:
                            # Создаем выражение для проверки аргумента
                            arg_expr = arg.replace('x', f'({x_val})')
                            try:
                                arg_val = eval(arg_expr, {"__builtins__": {}}, 
                                          {"math": math, "np": np, "sqrt": math.sqrt, 
                                           "sin": math.sin, "cos": math.cos, "tan": math.tan,
                                           "exp": math.exp, "log": math.log, "log10": math.log10,
                                           "abs": abs, "pi": math.pi})
                                if arg_val < 0:
                                    return float('nan')
                            except:
                                # Если не можем вычислить, продолжаем
                                pass
                    except:
                        # При ошибке разбора регулярного выражения продолжаем
                        pass
            
            # 2. Проверка на логарифм от неположительного числа
            if 'log(' in func_expr or 'log10(' in func_expr or 'log(' in func_expr:
                # Извлекаем аргументы логарифма
                try:
                    import re
                    log_args = re.findall(r'log\(([^)]+)\)', func_expr)
                    log_args.extend(re.findall(r'log10\(([^)]+)\)', func_expr))
                    log_args.extend(re.findall(r'ln\(([^)]+)\)', func_expr))
                    
                    for arg in log_args:
                        # Создаем выражение для проверки аргумента
                        arg_expr = arg.replace('x', f'({x_val})')
                        try:
                            arg_val = eval(arg_expr, {"__builtins__": {}}, 
                                      {"math": math, "np": np, "sqrt": math.sqrt, 
                                       "sin": math.sin, "cos": math.cos, "tan": math.tan,
                                       "exp": math.exp, "log": math.log, "log10": math.log10,
                                       "abs": abs, "pi": math.pi})
                            if arg_val <= 0:
                                return float('nan')
                        except:
                            # Если не можем вычислить, продолжаем с осторожностью
                            pass
                except:
                    # При ошибке разбора регулярного выражения продолжаем
                    pass
            
            # 3. Проверка на деление на ноль
            if '/' in func_expr:
                try:
                    import re
                    # Ищем выражения вида (числитель)/(знаменатель)
                    div_exprs = re.findall(r'(.+)\/(.+)', func_expr)
                    for _, denominator in div_exprs:
                        # Создаем выражение для проверки знаменателя
                        denom_expr = denominator.replace('x', f'({x_val})')
                        try:
                            denom_val = eval(denom_expr, {"__builtins__": {}}, 
                                        {"math": math, "np": np, "sqrt": math.sqrt, 
                                         "sin": math.sin, "cos": math.cos, "tan": math.tan,
                                         "exp": math.exp, "log": math.log, "log10": math.log10,
                                         "abs": abs, "pi": math.pi})
                            if abs(denom_val) < 1e-10:  # Очень близко к нулю
                                return float('nan')
                        except:
                            # Если не можем вычислить, продолжаем с осторожностью
                            pass
                except:
                    # При ошибке разбора регулярного выражения продолжаем
                    pass
            
            # Если все проверки пройдены, выполняем вычисление
            result = eval(func_expr.replace('x', f'({x_val})'), {"__builtins__": {}}, 
                      {"math": math, "np": np, "sqrt": math.sqrt, 
                       "sin": math.sin, "cos": math.cos, "tan": math.tan,
                       "exp": math.exp, "log": math.log, "log10": math.log10, 
                       "abs": abs, "pi": math.pi})
            
            # Проверяем на бесконечность или NaN
            if math.isnan(result) or math.isinf(result):
                return float('nan')
                
            return result
        except Exception as e:
            # Если произошла ошибка при вычислении, возвращаем NaN
            return float('nan')
    
    # Создаем функцию для построения графика
    def create_func(expr):
        def func(x_val):
            return safe_eval(expr, x_val)
        return func
    
    # Вычисляем значения функции для каждой точки x и строим график
    for func_expr, color, name in functions:
        try:
            # Предварительно конвертируем функцию
            func_expr = func_expr.replace('^', '**')
            if 'sqrt(' in func_expr and 'math.sqrt(' not in func_expr and 'np.sqrt(' not in func_expr:
                func_expr = func_expr.replace('sqrt(', 'math.sqrt(')
            
            # Конвертируем русские названия цветов в английские
            color_map = {
                'синий': 'blue',
                'красный': 'red',
                'зеленый': 'green',
                'зелёный': 'green',
                'желтый': 'yellow',
                'жёлтый': 'yellow',
                'черный': 'black',
                'чёрный': 'black',
                'белый': 'white',
                'серый': 'gray',
                'оранжевый': 'orange',
                'фиолетовый': 'purple',
                'розовый': 'pink'
            }
            if isinstance(color, str) and color.lower() in color_map:
                color = color_map[color.lower()]
                
            # Удаляем LaTeX-разметку из метки функции
            if isinstance(name, str):
                name = remove_latex_markup(name)
            
            # Удаляем LaTeX-разметку, если она осталась
            func_expr = remove_latex_markup(func_expr)
            
            # Логируем функцию для отладки
            import logging
            logging.info(f"Строим график функции: {func_expr}, цвет: {color}, метка: {name}")
            
            # Вычисляем значения y для каждого x
            y = np.array([safe_eval(func_expr, xi) for xi in x])
            
            # Фильтруем значения NaN, бесконечности и очень большие/маленькие числа
            valid_indices = np.isfinite(y) & (np.abs(y) < 1e10)
            if np.any(valid_indices):
                x_valid = x[valid_indices]
                y_valid = y[valid_indices]
                
                # Строим график для этой функции с соответствующим цветом и названием
                plt.plot(x_valid, y_valid, color=color, label=name, linewidth=2)
            else:
                import logging
                logging.warning(f"Не удалось построить график функции {func_expr}: нет валидных точек")
            
        except Exception as e:
            import logging
            logging.error(f"Ошибка при построении графика для функции {func_expr}: {e}")
            traceback.print_exc()
    
    # Если есть особые точки (например, точки пересечения), добавляем их на график
    if special_points:
        for point in special_points:
            try:
                x_val, y_val, label = point
                
                # Массив цветов для точек
                point_colors = ['red', 'green', 'blue', 'purple', 'orange', 'brown', 'pink', 'gray', 'cyan', 'magenta']
                # Выбираем цвет по индексу (используем хеш от метки для стабильности цветов между запусками)
                color_idx = hash(label) % len(point_colors)
                point_color = point_colors[color_idx]
                
                # Очищаем метку от LaTeX-разметки, если необходимо
                if isinstance(label, str):
                    label = remove_latex_markup(label)
                
                # Отрисовываем точку
                plt.plot(x_val, y_val, 'o', color=point_color, markersize=6)
                
                # Определяем положение метки в зависимости от значения y
                if y_val == 0:  # Если точка на оси X
                    y_offset = -0.5
                    x_offset = 0
                    va = 'top'
                elif y_val < 0:  # Если точка ниже оси X
                    y_offset = -0.5
                    x_offset = 0.2
                    va = 'top'
                else:  # Если точка выше оси X
                    y_offset = 0.5
                    x_offset = 0.2
                    va = 'bottom'
                
                # Добавляем метку с учетом позиционирования
                plt.annotate(label, (x_val, y_val), 
                           xytext=(x_val + x_offset, y_val + y_offset), 
                           textcoords='data', ha='center', va=va,
                           fontsize=12, bbox=dict(boxstyle='round,pad=0.3', 
                                                 fc='white', ec=point_color, alpha=0.8))
            except Exception as e:
                logging.error(f"Ошибка при добавлении особой точки {point}: {e}")
                traceback.print_exc()
    
    # Настраиваем оси и диапазоны
    if y_range is not None:
        plt.ylim(y_range)
    
    plt.xlim(x_range)
    
    # Добавляем сетку
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Добавляем легенду, если есть хотя бы одна функция
    if functions:
        plt.legend(loc='best', framealpha=0.5)
    
    # Добавляем заголовок
    plt.title("График функций", fontsize=14)
    
    # Добавляем подписи осей
    plt.xlabel("x", fontsize=12)
    plt.ylabel("y", fontsize=12)
    
    # Сохраняем график
    plt.tight_layout()
    try:
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        return filepath
    except Exception as e:
        logging.error(f"Ошибка при сохранении графика: {e}")
        traceback.print_exc()
        plt.close()
        return None

def test_visualization_functions():
    """
    Тестовая функция для проверки визуализации различных типов математических функций.
    
    Эта функция генерирует графики для различных типов функций, включая те, 
    которые могут вызывать проблемы (корни, логарифмы, деление на ноль и т.д.).
    
    Возвращает список путей к созданным изображениям.
    """
    import os
    
    # Список функций для тестирования
    test_functions = [
        # Простые функции
        ("x**2", "blue", "f1"),
        ("2*x+1", "red", "f2"),
        
        # Функции с разрывами
        ("1/x", "green", "f3"),
        ("1/(x-2)", "purple", "f4"),
        
        # Корни и логарифмы
        ("sqrt(x)", "orange", "f5"),
        ("sqrt(x-4)", "blue", "f6"),
        ("log(x)", "red", "f7"),
        ("log(abs(x-5))", "green", "f8"),
        
        # Тригонометрические функции
        ("sin(x)", "purple", "f9"),
        ("tan(x)", "orange", "f10"),
        
        # Модули и кусочные функции
        ("abs(x-3)", "blue", "f11"),
        ("x if x>0 else -x", "red", "f12"),
        
        # Комбинированные функции
        ("sqrt(abs(x))*sin(x)", "green", "f13"),
        ("log(abs(x)+1)*cos(x)", "purple", "f14")
    ]
    
    # Создаем разные комбинации функций для тестирования
    test_sets = [
        # Одиночные функции
        [test_functions[0]],  # Квадратичная функция
        [test_functions[2]],  # Функция с разрывом
        [test_functions[4]],  # Функция с корнем
        [test_functions[6]],  # Логарифмическая функция
        
        # Пары функций
        [test_functions[0], test_functions[1]],  # Квадратичная и линейная
        [test_functions[4], test_functions[6]],  # Корень и логарифм
        [test_functions[8], test_functions[9]],  # sin и tan
        
        # Более сложные комбинации
        [test_functions[0], test_functions[2], test_functions[8]],  # Квадратичная, разрыв, sin
        [test_functions[4], test_functions[6], test_functions[10]],  # Корень, логарифм, модуль
        [test_functions[12], test_functions[13]]  # Две комбинированные функции
    ]
    
    results = []
    
    # Генерируем графики с разными диапазонами
    ranges = [
        (-10, 10),
        (-5, 5),
        (0, 10),
        (-2, 8)
    ]
    
    for i, func_set in enumerate(test_sets):
        for j, x_range in enumerate(ranges):
            try:
                filepath = generate_multi_function_graph(
                    func_set, 
                    x_range=x_range,
                    y_range=None
                )
                print(f"Успешно сгенерирован график для набора {i+1}, диапазон {x_range}: {filepath}")
                results.append(filepath)
            except Exception as e:
                import traceback
                print(f"Ошибка при генерации графика для набора {i+1}, диапазон {x_range}: {e}")
                traceback.print_exc()
    
    # Также тестируем функцию normalize_function_expression
    test_expressions = [
        "x^2 + 3*x - 5",
        "√(x+1)",
        "log_2(x)",
        "|x-3|",
        "sin(x) + cos(x)",
        "tg(x) / ctg(x)",
        "(x+1)/(x-2)"
    ]
    
    print("\nРезультаты нормализации выражений:")
    for expr in test_expressions:
        try:
            normalized = normalize_function_expression(expr)
            print(f"'{expr}' -> '{normalized}'")
        except Exception as e:
            print(f"Ошибка при нормализации '{expr}': {e}")
    
    return results

# Эта функция перемещена в app/visualization/chart_utils.py и импортируется оттуда

def process_circle_visualization(params_text, extract_param):
    """Обрабатывает параметры для окружности и создает визуализацию"""
    # Извлекаем параметры для окружности
    center_str = extract_param(REGEX_PATTERNS["circle"]["center"], params_text)
    radius_str = extract_param(REGEX_PATTERNS["circle"]["radius"], params_text)
    center_label = extract_param(REGEX_PATTERNS["circle"]["center_label"], params_text, "O")
    show_radius = extract_param(REGEX_PATTERNS["circle"]["show_radius"], params_text, "нет").lower() in ["да", "yes", "true"]
    show_diameter = extract_param(REGEX_PATTERNS["circle"]["show_diameter"], params_text, "нет").lower() in ["да", "yes", "true"]
    show_chord = extract_param(REGEX_PATTERNS["circle"]["show_chord"], params_text, "нет").lower() in ["да", "yes", "true"]
    radius_value_str = extract_param(REGEX_PATTERNS["circle"]["radius_value"], params_text)
    diameter_value_str = extract_param(REGEX_PATTERNS["circle"]["diameter_value"], params_text)
    chord_value_str = extract_param(REGEX_PATTERNS["circle"]["chord_value"], params_text)
    
    # Параметры для отображения окружности
    params = DEFAULT_VISUALIZATION_PARAMS["circle"].copy()
    
    # Парсинг центра
    if center_str:
        try:
            center = center_str.strip()
            match = re.search(r'\(([^,]+),([^)]+)\)', center)
            if match:
                cx, cy = float(match.group(1)), float(match.group(2))
                params['center'] = (cx, cy)
        except Exception as e:
            logging.warning(f"Ошибка при разборе центра окружности: {e}")
    
    # Парсинг радиуса
    if radius_str:
        try:
            radius = float(radius_str.strip())
            params['radius'] = radius
        except Exception as e:
            logging.warning(f"Ошибка при разборе радиуса окружности: {e}")
    
    # Добавляем метку центра
    if center_label and center_label.lower() not in ["нет", "no", "none", "-"]:
        params['center_label'] = center_label
        params['show_center'] = True
    else:
        params['show_center'] = False
    
    # Добавляем значение радиуса для отображения
    if radius_value_str and radius_value_str.lower() not in ["нет", "no", "none", "-"]:
        try:
            params['radius_value'] = float(radius_value_str.strip())
        except Exception as e:
            logging.warning(f"Ошибка при разборе значения радиуса для отображения: {e}")
    
    # Добавляем значение диаметра для отображения
    if diameter_value_str and diameter_value_str.lower() not in ["нет", "no", "none", "-"]:
        try:
            params['diameter_value'] = float(diameter_value_str.strip())
        except Exception as e:
            logging.warning(f"Ошибка при разборе значения диаметра для отображения: {e}")
    
    # Добавляем значение хорды для отображения
    if chord_value_str and chord_value_str.lower() not in ["нет", "no", "none", "-"]:
        try:
            params['chord_value'] = float(chord_value_str.strip())
        except Exception as e:
            logging.warning(f"Ошибка при разборе значения хорды для отображения: {e}")
    
    # Добавляем флаги отображения
    params['show_radius'] = show_radius
    params['show_diameter'] = show_diameter
    params['show_chord'] = show_chord
    
    # Обработка параметров конкретных углов для отображения
    show_specific_angles = extract_param(REGEX_PATTERNS["circle"]["show_specific_angles"], params_text)
    if show_specific_angles:
        # Проверяем, если это уже список, то используем как есть
        if isinstance(show_specific_angles, list):
            params["show_specific_angles"] = show_specific_angles
        else:
            try:
                # Иначе пытаемся разделить строку
                params["show_specific_angles"] = [angle.strip() for angle in show_specific_angles.split(',')]
            except Exception as e:
                logging.warning(f"Ошибка при разборе конкретных углов окружности: {e}")
    
    # Обработка параметров конкретных сторон для отображения
    show_specific_sides = extract_param(REGEX_PATTERNS["circle"]["show_specific_sides"], params_text)
    if show_specific_sides:
        # Проверяем, если это уже список, то используем как есть
        if isinstance(show_specific_sides, list):
            params["show_specific_sides"] = show_specific_sides
        else:
            try:
                # Иначе пытаемся разделить строку
                params["show_specific_sides"] = [side.strip() for side in show_specific_sides.split(',')]
            except Exception as e:
                logging.warning(f"Ошибка при разборе конкретных сторон окружности: {e}")
    
    # Создаем директорию для сохранения изображения, если она не существует
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static/images/generated")
    os.makedirs(output_dir, exist_ok=True)
    
    # Генерируем имя файла с уникальным идентификатором
    filename = f'circle_{uuid.uuid4().hex[:8]}.png'
    output_path = os.path.join(output_dir, filename)
    
    try:
        # Используем новую ООП-структуру для создания и отрисовки окружности
        from app.geometry import Circle
        from app.visualization import GeometryRenderer
        
        # Создаем окружность
        circle = Circle(params)
        
        # Отрисовываем и сохраняем изображение
        GeometryRenderer.render_figure(circle, output_path)
        
        return output_path
    except ImportError:
        # Если новые модули недоступны, используем старый подход
        logging.warning("Не удалось импортировать новые модули для ООП-визуализации, используем старый подход")
        
    # Генерируем изображение старым методом
    generate_geometric_figure('circle', params, output_path)
    return output_path

def process_triangle_visualization(params_text, extract_param):
    """Обрабатывает параметры для треугольника и создает визуализацию"""
    # Извлекаем параметры для треугольника
    coords_str = extract_param(REGEX_PATTERNS["triangle"]["coords"], params_text)
    show_angles = extract_param(REGEX_PATTERNS["triangle"]["angles"], params_text, "нет").lower() in ["да", "yes", "true"]
    show_sides = extract_param(REGEX_PATTERNS["triangle"]["lengths"], params_text, "нет").lower() in ["да", "yes", "true"]
    vertex_labels_str = extract_param(REGEX_PATTERNS["triangle"]["vertex_labels"], params_text)
    is_right = extract_param(REGEX_PATTERNS["triangle"]["is_right"], params_text, "нет").lower() in ["да", "yes", "true"]
    side_lengths_str = extract_param(REGEX_PATTERNS["triangle"]["side_lengths"], params_text)
    show_angle_arcs = extract_param(REGEX_PATTERNS["triangle"]["show_angle_arcs"], params_text, "нет").lower() in ["да", "yes", "true"]
    
    # Параметры для отображения треугольника
    params = DEFAULT_VISUALIZATION_PARAMS["triangle"].copy()
    
    # Парсинг координат
    if coords_str:
        try:
            # Для координат ожидаем формат [(x1,y1), (x2,y2), (x3,y3)]
            coords = coords_str.strip()
            # Удаляем внешние скобки, если они есть
            if coords.startswith("[") and coords.endswith("]"):
                coords = coords[1:-1]
            
            # Разбиваем на отдельные точки
            points = []
            pattern = r'\(([^,]+),([^)]+)\)'
            matches = re.findall(pattern, coords)
            
            if len(matches) >= 3:
                for i in range(3):
                    x = float(matches[i][0])
                    y = float(matches[i][1])
                    points.append((x, y))
                
                params['points'] = points
        except Exception as e:
            logging.warning(f"Ошибка при парсинге координат треугольника: {e}")
    
    # Парсинг меток вершин
    if vertex_labels_str:
        try:
            # Для меток вершин ожидаем формат [A, B, C] или A, B, C
            labels = vertex_labels_str.strip()
            # Удаляем внешние скобки, если они есть
            if labels.startswith("[") and labels.endswith("]"):
                labels = labels[1:-1]
            
            # Разбиваем на отдельные метки
            vertex_labels = [label.strip() for label in re.split(r'[,\s]+', labels) if label.strip()]
            
            if len(vertex_labels) >= 3:
                params['vertex_labels'] = vertex_labels[:3]
        except Exception as e:
            logging.warning(f"Ошибка при парсинге меток вершин треугольника: {e}")
    
    # Дополнительные параметры
    params['show_angles'] = show_angles
    params['show_sides'] = show_sides
    params['is_right'] = is_right
    params['show_angle_arcs'] = show_angle_arcs
    
    # Парсинг длин сторон
    if side_lengths_str:
        try:
            # Для длин сторон ожидаем формат [side1, side2, side3] или side1, side2, side3
            lengths = side_lengths_str.strip()
            # Удаляем внешние скобки, если они есть
            if lengths.startswith("[") and lengths.endswith("]"):
                lengths = lengths[1:-1]
            
            # Разбиваем на отдельные значения
            side_lengths = []
            for length in re.split(r'[,\s]+', lengths):
                if length.strip():
                    try:
                        # Пробуем преобразовать в число
                        side_lengths.append(float(length.strip()))
                    except ValueError:
                        # Если не число, просто добавляем строку
                        side_lengths.append(length.strip())
            
            if len(side_lengths) >= 3:
                params['side_lengths'] = side_lengths[:3]
        except Exception as e:
            logging.warning(f"Ошибка при парсинге длин сторон треугольника: {e}")
    
    # Парсинг значений углов
    angle_values_str = extract_param(REGEX_PATTERNS["triangle"]["angle_values"], params_text)
    if angle_values_str:
        try:
            # Для значений углов ожидаем формат A=30, B=60, C=90 или аналогичный
            parts = re.split(r'[,\s]+', angle_values_str)
            angle_values = {}
            
            for part in parts:
                if '=' in part:
                    angle_key, angle_val = part.split('=', 1)
                    angle_values[angle_key.strip()] = angle_val.strip()
            
            if angle_values:
                params['angle_values'] = angle_values
        except Exception as e:
            logging.warning(f"Ошибка при разборе значений углов треугольника: {e}")
    
    # Создаем директорию для сохранения изображения, если она не существует
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static/images/generated")
    os.makedirs(output_dir, exist_ok=True)
    
    # Генерируем имя файла с уникальным идентификатором
    filename = f'triangle_{uuid.uuid4().hex[:8]}.png'
    output_path = os.path.join(output_dir, filename)
    
    try:
        # Используем новую ООП-структуру для создания и отрисовки треугольника
        from app.geometry import Triangle
        from app.visualization import GeometryRenderer
        
        # Создаем треугольник
        triangle = Triangle(params)
        
        # Отрисовываем и сохраняем изображение
        GeometryRenderer.render_figure(triangle, output_path)
        
        return output_path
    except ImportError as e:
        # Если новые модули недоступны, используем старый подход
        logging.warning(f"Не удалось импортировать новые модули для ООП-визуализации, используем старый подход: {e}")
    except Exception as e:
        # Обрабатываем любые другие ошибки
        logging.error(f"Ошибка при создании треугольника: {e}")
        logging.error(traceback.format_exc())
        return None
    
    # Генерируем изображение старым методом
    try:
        generate_geometric_figure('triangle', params, output_path)
        return output_path
    except Exception as e:
        logging.error(f"Ошибка при создании треугольника старым методом: {e}")
        logging.error(traceback.format_exc())
        return None

def generate_complete_task(category, subcategory="", difficulty_level=3, is_basic_level=False):
    """
    Генерирует полный пакет: задачу, решение и подсказки.
    
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
        
        if not data:
            return {"error": f"Не удалось найти задачи в категории '{category}' и подкатегории '{subcategory}'"}
        
        # Извлекаем задачу и решение из данных
        html_task = data.get("html")
        original_task = extract_text_and_formulas(html_task) if html_task else data.get("task", "")
        original_solution = data.get("solution", "")
        
        logging.info(f"Выбрана исходная задача: {original_task[:100]}...")
        
        # Генерируем промпт для создания полного материала
        prompt = create_complete_task_prompt(category, subcategory, original_task, original_solution, difficulty_level)
        
        # Генерируем текст с помощью YandexGPT с повышенной температурой для разнообразия
        generated_text = yandex_gpt_generate(prompt, temperature=0.6, is_basic_level=is_basic_level)
        
        if not generated_text:
            return {"error": "Не удалось получить ответ от YandexGPT API"}
        
        # Сохраняем сгенерированный текст
        save_to_file(generated_text)
        
        # Извлекаем части из сгенерированного текста
        task_match = re.search(r'---ЗАДАЧА---\s*(.*?)(?=\s*---РЕШЕНИЕ---|\s*$)', generated_text, re.DOTALL)
        solution_match = re.search(r'---РЕШЕНИЕ---\s*(.*?)(?=\s*---ПОДСКАЗКИ---|\s*$)', generated_text, re.DOTALL)
        hints_match = re.search(r'---ПОДСКАЗКИ---\s*(.*?)(?=\s*---ПАРАМЕТРЫ ДЛЯ ВИЗУАЛИЗАЦИИ---|\s*$)', generated_text, re.DOTALL)
        
        task = task_match.group(1).strip() if task_match else "Не удалось извлечь задачу"
        solution = solution_match.group(1).strip() if solution_match else "Не удалось извлечь решение"
        hints_string = hints_match.group(1).strip() if hints_match else ""
        
        # Парсим подсказки
        hints = parse_hints(hints_string)
        
        # Извлекаем ответ из решения
        answer = extract_answer_with_latex(solution)
        
        # Проверяем наличие параметров для визуализации
        visualization_match = re.search(r'---ПАРАМЕТРЫ ДЛЯ ВИЗУАЛИЗАЦИИ---\s*(.*?)(?=\s*$)', generated_text, re.DOTALL)
        
        image_path = None
        visualization_params = None
        visualization_type = None
        
        if visualization_match:
            # Обрабатываем параметры визуализации
            params_text = visualization_match.group(1).strip()
            image_path, visualization_type = process_visualization_params(params_text)
            visualization_params = params_text
        
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
            result["image_path"] = image_path
            result["visualization_params"] = visualization_params
            result["visualization_type"] = visualization_type
            
            # Для удобного отображения на веб-странице
            image_url = f"/static/images/generated/{os.path.basename(image_path)}"
            result["image_url"] = image_url
        
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
    def extract_section(text, section_name):
        pattern = rf"---{section_name}---\s*(.*?)(?=---[A-ZА-Я _]+---|$)"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else ""
    
    # Извлекаем каждый раздел из текста
    sections["task"] = extract_section(raw_text, "ЗАДАЧА")
    sections["solution"] = extract_section(raw_text, "РЕШЕНИЕ")
    sections["answer"] = extract_answer_with_latex(sections["solution"])
    
    # Извлекаем подсказки и разбиваем их на список
    hints_text = extract_section(raw_text, "ПОДСКАЗКИ")
    sections["hints"] = parse_hints(hints_text)
    
    # Извлекаем параметры для визуализации, если они есть
    viz_params = extract_section(raw_text, "ПАРАМЕТРЫ ДЛЯ ВИЗУАЛИЗАЦИИ")
    sections["visualization_params"] = viz_params
    
    return sections

def convert_markdown_to_html(markdown_text):
    """
    Конвертирует текст в формате Markdown в HTML.
    
    Args:
        markdown_text: Текст в формате Markdown
        
    Returns:
        str: HTML-разметка
    """
    if not markdown_text:
        return ""
    
    # Простая конвертация основных элементов Markdown
    html = markdown_text
    
    # Заголовки
    html = re.sub(r'^# (.+)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)
    html = re.sub(r'^## (.+)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
    html = re.sub(r'^### (.+)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
    
    # Выделение жирным
    html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)
    
    # Выделение курсивом
    html = re.sub(r'\*(.+?)\*', r'<em>\1</em>', html)
    
    # Маркированный список
    html = re.sub(r'^- (.+)$', r'<li>\1</li>', html, flags=re.MULTILINE)
    html = re.sub(r'(<li>.+</li>\s*)+', r'<ul>\g<0></ul>', html)
    
    # Нумерованный список
    html = re.sub(r'^(\d+)\. (.+)$', r'<li>\2</li>', html, flags=re.MULTILINE)
    html = re.sub(r'(<li>.+</li>\s*)+', r'<ol>\g<0></ol>', html)
    
    # Абзацы (любой текст, который не является другим элементом)
    lines = html.split('\n')
    in_list = False
    in_html = False
    for i in range(len(lines)):
        line = lines[i]
        if not line.strip():  # Пустая строка
            continue
        if re.match(r'^<(h\d|p|ul|ol|li|hr|blockquote|table|tr|td|th)', line):
            # Уже является HTML-элементом
            in_html = True
            continue
        if re.match(r'^- ', line) or re.match(r'^\d+\. ', line):
            # Элемент списка
            in_list = True
            continue
        if not in_html and not in_list:
            # Обычный текст, не являющийся другим элементом
            lines[i] = f'<p>{line}</p>'
        in_list = False
        in_html = False
    html = '\n'.join(lines)
    
    # Возвращаем HTML
    return html

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

def parse_graph_params(params_text):
    """
    Универсальная функция для извлечения параметров графиков функций.
    Возвращает список функций, диапазоны осей и особые точки.
    
    Args:
        params_text: Текст с параметрами
        
    Returns:
        tuple: (functions, x_range, y_range, special_points)
            functions - список кортежей (функция, цвет, метка)
            x_range - диапазон оси X (x_min, x_max)
            y_range - диапазон оси Y (y_min, y_max) или None
            special_points - список особых точек [(x1, y1, метка1), ...]
    """
    import re
    import logging
    import traceback
    
    # Список функций для отображения
    funcs_to_plot = []
    
    # Диапазоны осей
    x_range = (-10, 10)  # По умолчанию [-10, 10]
    y_range = None  # По умолчанию автоматический
    
    # Особые точки
    special_points = []
    
    # Функция для извлечения параметра по шаблону
    def extract_param(pattern, text, default=None):
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            try:
                value = match.group(1).strip()
                # Очищаем от LaTeX разметки
                return remove_latex_markup(value)
            except Exception as e:
                logging.error(f"Ошибка при извлечении параметра: {e}")
        return default
    
    try:
        # МЕТОД 1: Поиск функций, цветов и меток в параметрах визуализации
        # Для извлечения используем несколько подходов, начиная с наиболее структурированных
        
        # 1. Сначала ищем "Функция 1: x^2" и т.д.
        func_pattern = r'Функция\s+(\d+):\s*(.*?)(?=Функция\s+\d+:|Цвет|Название|Диапазон|Особые|$)'
        func_matches = re.findall(func_pattern, params_text, re.IGNORECASE | re.DOTALL)
        
        # Создаем словари для цветов и названий
        color_dict = {}
        name_dict = {}
        
        # Извлекаем цвета в формате "Цвет 1: красный"
        color_pattern = r'Цвет\s+(\d+):\s*(.*?)(?=Функция|Цвет|Название|Диапазон|Особые|$)'
        color_matches = re.findall(color_pattern, params_text, re.IGNORECASE | re.DOTALL)
        for num_str, color in color_matches:
            try:
                num = int(num_str.strip())
                color_dict[num] = color.strip()
            except ValueError:
                continue
                
        # Извлекаем названия в формате "Название 1: f(x)"
        name_pattern = r'Название\s+(\d+):\s*(.*?)(?=Функция|Цвет|Название|Диапазон|Особые|$)'
        name_matches = re.findall(name_pattern, params_text, re.IGNORECASE | re.DOTALL)
        for num_str, name in name_matches:
            try:
                num = int(num_str.strip())
                name_dict[num] = name.strip()
            except ValueError:
                continue
        
        # Стандартные цвета, если не указаны
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        # Конвертер русских названий цветов в английские
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
        
        # Обрабатываем найденные функции
        functions = []
        for match in func_matches:
            try:
                num = int(match[0].strip())
                func_expr = match[1].strip()
                
                # Очищаем от LaTeX-разметки и приводим к Python-синтаксису
                func_expr = remove_latex_markup(func_expr)
                
                # Определяем цвет и метку для функции
                color = color_dict.get(num, colors[min(num-1, len(colors)-1)])
                
                # Преобразуем русское название цвета в английское
                if color.lower() in color_mapping:
                    color = color_mapping[color.lower()]
                
                name = name_dict.get(num, f"f_{num}(x)")
                
                # Добавляем функцию в список
                functions.append((func_expr, color, name))
                logging.info(f"Найдена функция {num}: {func_expr}, цвет: {color}, метка: {name}")
            except Exception as e:
                logging.warning(f"Ошибка при обработке функции {match[0]}: {e}")
        
        # 2. Если функции не найдены методом выше, ищем по блокам "График 1: ..."
        if not functions:
            for i in range(1, 6):  # Поиск до 5 графиков
                graph_block_pattern = rf'График\s*{i}:.*?(?=График\s*{i+1}:|Общие параметры:|$)'
                graph_block_match = re.search(graph_block_pattern, params_text, re.DOTALL | re.IGNORECASE)
                
                if graph_block_match:
                    graph_block = graph_block_match.group(0)
                    
                    # Извлекаем функцию из блока
                    func_pattern = rf'Функция:?\s*([^\n]+)'
                    func_match = re.search(func_pattern, graph_block, re.IGNORECASE)
                    
                    if func_match:
                        func_str = func_match.group(1).strip()
                        func_str = remove_latex_markup(func_str)
                        
                        # Ищем цвет
                        color_pattern = r'Цвет:?\s*([^\n]+)'
                        color_match = re.search(color_pattern, graph_block, re.IGNORECASE)
                        color = color_match.group(1).strip() if color_match else colors[min(i-1, len(colors)-1)]
                        
                        # Преобразуем русское название цвета в английское
                        if color.lower() in color_mapping:
                            color = color_mapping[color.lower()]
                        
                        # Ищем метку
                        label_pattern = r'(Название|Метка|Подпись):?\s*([^\n]+)'
                        label_match = re.search(label_pattern, graph_block, re.IGNORECASE)
                        label = label_match.group(2).strip() if label_match else f"f_{i}(x)"
                        
                        # Извлекаем диапазоны X и Y для этой функции, если указаны
                        x_range_pattern = r'Диапазон\s+X:?\s*\[?(.*?)\]?(?=Диапазон\s+Y|Цвет|Название|Метка|Особые|$)'
                        x_range_match = re.search(x_range_pattern, graph_block, re.IGNORECASE)
                        
                        if x_range_match:
                            x_range_str = x_range_match.group(1).strip()
                            try:
                                x_min, x_max = map(float, x_range_str.split(','))
                                x_range = (x_min, x_max)
                            except Exception as e:
                                logging.warning(f"Ошибка при разборе диапазона X для графика {i}: {e}")
                        
                        y_range_pattern = r'Диапазон\s+Y:?\s*\[?(.*?)\]?(?=Диапазон\s+X|Цвет|Название|Метка|Особые|$)'
                        y_range_match = re.search(y_range_pattern, graph_block, re.IGNORECASE)
                        
                        if y_range_match:
                            y_range_str = y_range_match.group(1).strip()
                            if y_range_str.lower() != 'автоматический':
                                try:
                                    y_min, y_max = map(float, y_range_str.split(','))
                                    y_range = (y_min, y_max)
                                except Exception as e:
                                    logging.warning(f"Ошибка при разборе диапазона Y для графика {i}: {e}")
                        
                        # Нормализуем выражение функции для Python
                        func_str = func_str.replace('^', '**')
                        
                        # Добавляем функцию в список
                        functions.append((func_str, color, label))
                        logging.info(f"Найдена функция для графика {i}: {func_str}, цвет: {color}, метка: {label}")
        
        # 3. Поиск функций напрямую в тексте задачи
        if not functions:
            # Ищем функции напрямую в формате "Функция 1: x^2"
            for i in range(1, 6):  # Поиск до 5 функций
                func_pattern = rf'Функция\s*{i}:?\s*([^\n]+)'
                func_match = re.search(func_pattern, params_text, re.IGNORECASE | re.MULTILINE)
                
                if func_match:
                    func_str = func_match.group(1).strip()
                    # Очищаем от LaTeX-разметки полностью
                    func_str = remove_latex_markup(func_str)
                    
                    # Определяем цвет и метку для функции
                    color = color_dict.get(i, colors[min(i-1, len(colors)-1)])
                    
                    # Преобразуем русское название цвета в английское
                    if color.lower() in color_mapping:
                        color = color_mapping[color.lower()]
                    
                    name = name_dict.get(i, f"f_{i}(x)")
                    
                    # Добавляем функцию в список
                    functions.append((func_str, color, name))
                    logging.info(f"Напрямую найдена функция {i}: {func_str}, цвет: {color}, метка: {name}")
        
        # Обрабатываем общие параметры диапазонов осей
        x_range_pattern = r'Диапазон\s+X:?\s*\[?(.*?)\]?(?=Диапазон\s+Y|Функция|Цвет|Название|Особые|$)'
        x_range_match = re.search(x_range_pattern, params_text, re.IGNORECASE | re.MULTILINE)
        
        if x_range_match:
            x_range_str = x_range_match.group(1).strip()
            try:
                x_min, x_max = map(float, x_range_str.split(','))
                x_range = (x_min, x_max)
                logging.info(f"Найден диапазон X: [{x_min}, {x_max}]")
            except Exception as e:
                logging.warning(f"Ошибка при разборе диапазона X: {e}")
        
        y_range_pattern = r'Диапазон\s+Y:?\s*\[?(.*?)\]?(?=Диапазон\s+X|Функция|Цвет|Название|Особые|$)'
        y_range_match = re.search(y_range_pattern, params_text, re.IGNORECASE | re.MULTILINE)
        
        if y_range_match:
            y_range_str = y_range_match.group(1).strip()
            if y_range_str.lower() != 'автоматический':
                try:
                    y_min, y_max = map(float, y_range_str.split(','))
                    y_range = (y_min, y_max)
                    logging.info(f"Найден диапазон Y: [{y_min}, {y_max}]")
                except Exception as e:
                    logging.warning(f"Ошибка при разборе диапазона Y: {e}")
        
        # Ищем особые точки
        special_points_pattern = r'Особые точки:\s*(.*?)(?=Функция|Диапазон|$)'
        special_points_match = re.search(special_points_pattern, params_text, re.DOTALL | re.IGNORECASE)
        
        if special_points_match:
            try:
                special_points_str = special_points_match.group(1).strip()
                # Извлекаем точки в формате (x, y, метка)
                points_list = re.findall(r'\(([^)]+)\)', special_points_str)
                
                for point_str in points_list:
                    try:
                        # Разделяем по запятой на x, y, label
                        parts = point_str.split(',', 2)
                        
                        if len(parts) >= 2:
                            x_expr = parts[0].strip()
                            y_expr = parts[1].strip()
                            label = parts[2].strip() if len(parts) > 2 else ""
                            
                            # Вычисляем значения координат, поддерживая математические выражения
                            import math
                            
                            # Обрабатываем x_expr
                            x_expr = x_expr.replace('^', '**').replace('sqrt', 'math.sqrt')
                            if any(func in x_expr for func in ['math.', 'sqrt', 'sin', 'cos']):
                                x_val = eval(x_expr)
                            else:
                                x_val = float(x_expr)
                            
                            # Обрабатываем y_expr
                            if 'f(' in y_expr:
                                # Если y задан через значение функции
                                if functions:
                                    func_expr = functions[0][0]
                                    y_val = eval(func_expr.replace('x', f'({x_val})'))
                                else:
                                    logging.warning(f"Не удалось вычислить значение функции для точки ({x_expr}, {y_expr})")
                                    continue
                            else:
                                # Если y задан явно
                                y_expr = y_expr.replace('^', '**').replace('sqrt', 'math.sqrt')
                                if any(func in y_expr for func in ['math.', 'sqrt', 'sin', 'cos']):
                                    y_val = eval(y_expr)
                                else:
                                    y_val = float(y_expr)
                            
                            # Добавляем точку с вычисленными координатами
                            special_points.append((x_val, y_val, label))
                            logging.info(f"Добавлена особая точка: ({x_val}, {y_val}, {label})")
                    except Exception as e:
                        logging.warning(f"Ошибка при обработке точки '{point_str}': {e}")
            except Exception as e:
                logging.warning(f"Ошибка при обработке списка особых точек: {e}")
        
        # Если функции всё еще не найдены, попробуем найти в общем тексте задачи
        if not functions:
            # Поиск функций в формате y=... или f(x)=...
            general_func_patterns = [
                r'y\s*=\s*([-+0-9a-zA-Z\^\*\/\(\)\s\{\}\[\]\.,]+)(?=[,.:;)]|\s*$|\n)',
                r'f\s*\(\s*x\s*\)\s*=\s*([-+0-9a-zA-Z\^\*\/\(\)\s\{\}\[\]\.,]+)(?=[,.:;)]|\s*$|\n)',
                r'([a-zA-Z])\s*\(\s*x\s*\)\s*=\s*([-+0-9a-zA-Z\^\*\/\(\)\s\{\}\[\]\.,]+)(?=[,.:;)]|\s*$|\n|\$)',
                r'\$([a-zA-Z])\s*\(\s*x\s*\)\s*=\s*([-+0-9a-zA-Z\^\*\/\(\)\s\{\}\[\]\.,\\]+)\$'
            ]
            
            for i, pattern in enumerate(general_func_patterns):
                matches = re.findall(pattern, params_text)
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
                    
                    # Очищаем выражение
                    func_expr = func_expr.strip()
                    func_expr = func_expr.replace('^', '**')
                    
                    # Выбираем цвет
                    color_idx = j % len(colors)
                    color = colors[color_idx]
                    
                    # Добавляем функцию, если она уникальна
                    if not any(f[0] == func_expr for f in functions):
                        functions.append((func_expr, color, func_name))
                        logging.info(f"Найдена функция в общем тексте: {func_expr}, метка: {func_name}")
        
        # Если функции всё ещё не найдены, используем стандартную функцию
        if not functions:
            functions = [('x**2', 'blue', 'f(x)')]
            logging.warning("Не удалось найти ни одной функции, использую стандартный график параболы")
        
        # Возвращаем найденные параметры
        funcs_to_plot = functions
        
    except Exception as e:
        logging.error(f"Ошибка при разборе параметров графика: {e}")
        logging.error(traceback.format_exc())
        # В случае ошибки используем стандартные значения
        funcs_to_plot = [('x**2', 'blue', 'f(x)')]
    
    return funcs_to_plot, x_range, y_range, special_points