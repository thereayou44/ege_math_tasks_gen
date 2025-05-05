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
from prompts import HINT_PROMPTS, SYSTEM_PROMPT, create_complete_task_prompt, REGEX_PATTERNS, DEFAULT_VISUALIZATION_PARAMS
import traceback
import matplotlib.patches as patches

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
from prompts import HINT_PROMPTS, SYSTEM_PROMPT, REGEX_PATTERNS, DEFAULT_VISUALIZATION_PARAMS

def select_file(category, subcategory=""):
    """
    Выбирает случайный файл с задачей из указанной категории и подкатегории.
    
    Args:
        category: Название категории задач
        subcategory: Название подкатегории (опционально)
        
    Returns:
        dict: JSON-данные выбранной задачи
    """
    base_dir = "Data/math_catalog_subcategories"
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

def yandex_gpt_generate(prompt, temperature=0.3, max_tokens=5000):
    """
    Отправляет запрос к API YandexGPT и возвращает ответ.
    
    Args:
        prompt: Текст запроса
        temperature: Температура генерации (от 0 до 1)
        max_tokens: Максимальное количество токенов в ответе (увеличено до 5000)
        
    Returns:
        str: Сгенерированный текст ответа
    """
    # Создаем ключ кэша на основе параметров запроса
    cache_key = f"{prompt}_{temperature}_{max_tokens}"
    
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
    
    # Ищем "Ответ:" или "Ответ :"
    answer_pattern = r"(?:Ответ|ОТВЕТ|ответ)\s*:(.+?)(?:$|\.|\n|\<\/p\>)"
    answer_match = re.search(answer_pattern, solution, re.IGNORECASE | re.DOTALL)
    
    if answer_match:
        answer = answer_match.group(1).strip()
        logging.info(f"Найден ответ: {answer}")
        
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
        # (чтобы они не интерпретировались как теги)
        if '<' in answer and not re.search(r'<[a-z/]', answer):
            answer = answer.replace('<', '&lt;').replace('>', '&gt;')
            
        return answer
    
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
            
            # Если ответ уже содержит $, заменяем их на $$
            if '$' in answer:
                # Заменяем одинарные $ на двойные $$
                answer = answer.replace('$', '$$')
            else:
                # Применяем те же преобразования, что и выше
                formula_pattern = r'(\\frac|\\sqrt|\\sum|\\prod|\\int|\\lim|\\sin|\\cos|\\tan|\\log|\\ln)'
                answer = re.sub(formula_pattern, r'$$\1', answer)
                
                # Если мы добавили открывающий символ $$, но нет закрывающего, добавляем его
                open_count = answer.count('$$')
                if open_count % 2 != 0:
                    answer += '$$'
                
            if '<' in answer and not re.search(r'<[a-z/]', answer):
                answer = answer.replace('<', '&lt;').replace('>', '&gt;')
                
            return answer
    
    # Если ответ не найден
    logging.warning("Ответ не найден в решении")
    return "См. решение"

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
        hints.append(text.strip())
    
    # Если мы нашли меньше 3 подсказок, добавляем заглушки
    while len(hints) < 3:
        hints.append("Подсказка недоступна")
    
    # Берем только первые 3 подсказки
    return hints[:3]

def save_to_file(content, filename="last_generated_task.txt"):
    """
    Сохраняет полный сгенерированный текст в файл
    
    Args:
        content: Сгенерированный текст
        filename: Имя файла для сохранения
    """
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        logging.info(f"Сгенерированный текст сохранен в файл: {filename}")
    except Exception as e:
        logging.error(f"Ошибка при сохранении в файл {filename}: {e}")

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

def compute_trapezoid(bottom, top, height):
    dx = (bottom - top)/2
    return [(0,0), (bottom,0), (bottom-dx,height), (dx,height)]

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
                    raw = compute_trapezoid(bottom, top, height)
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
                    labels = params.get('vertex_labels')
                    if not labels:
                        labels = [chr(65+i) for i in range(len(pts))]
                    for (x0,y0), lab in zip(pts, labels):
                        ax.text(x0, y0, lab, ha='center', va='center', fontsize=14,  # Увеличен размер шрифта
                               bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
                
                # Длины сторон
                side_lengths = params.get('side_lengths', None)
                show_lengths = params.get('show_lengths', False)
                
                if side_lengths or show_lengths:
                    for i in range(len(pts)):
                        x0, y0 = pts[i]
                        x1, y1 = pts[(i+1)%len(pts)]
                        mx, my = (x0+x1)/2, (y0+y1)/2
                        
                        # Рассчитываем длину стороны
                        L = np.hypot(x1-x0, y1-y0)
                        
                        # Вектор нормали к стороне для размещения текста (увеличен вынос подписи)
                        nx, ny = -(y1-y0)/L, (x1-x0)/L
                        offset = 0.35  # Увеличен отступ для лучшей видимости
                        
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
                    
                    for i in range(len(pts)):
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
                            
                            # Проверяем внутренний или внешний угол (для выпуклых многоугольников)
                            # Определяем направление по часовой или против часовой стрелки
                            if len(pts) > 3:  # Для четырехугольников и более
                                # Вычисляем векторное произведение для определения выпуклости
                                cross_product = np.cross(np.append(v1, 0), np.append(v2, 0))[2]
                                if cross_product < 0:  # Если угол выпуклый (внешний)
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
                                
                                # Рисуем дугу угла
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
            path = filename
        else:
            # Иначе сохраняем в стандартную директорию
            out = 'static/images/generated'
            os.makedirs(out, exist_ok=True)
            if not filename:
                filename = f"{figure_type}_{uuid.uuid4().hex[:8]}.png"
            path = os.path.join(out, filename)
            
        plt.savefig(path, dpi=300, bbox_inches='tight')  # Увеличено разрешение (DPI)
        plt.close(fig)
        return path

    except Exception as e:
        logging.error(f"Ошибка при генерации фигуры: {e}")
        traceback.print_exc()  # Печатаем стек трейс для отладки
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

def process_visualization_params(params_text):
    """
    Обрабатывает параметры визуализации из текста и генерирует изображение.
    Возвращает путь к сгенерированному изображению.
    """
    try:
        # Функция извлечения параметров с помощью regex
        def extract_param(pattern, text, default=None):
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()
            return default
        
        # Определяем тип изображения
        image_type = extract_param(REGEX_PATTERNS["generic"]["type"], params_text)
        
        if not image_type:
            logging.error("Тип изображения не найден в параметрах визуализации")
            return None, None
        
        # Приводим тип к стандартному виду
        image_type = image_type.lower().strip()
        
        # Маппинг типов визуализации
        type_mapping = {
            "треугольник": "triangle",
            "прямоугольник": "rectangle",
            "квадрат": "rectangle", 
            "параллелограмм": "parallelogram",
            "трапеция": "trapezoid",
            "окружность": "circle",
            "круг": "circle",
            "график": "graph",
            "координатная_плоскость": "coordinate"
        }
        
        # Преобразуем русский тип в английский для внутреннего использования
        eng_type = type_mapping.get(image_type, image_type)
        
        # Обрабатываем параметры в зависимости от типа
        if eng_type == "triangle":
            image_path = process_triangle_visualization(params_text, extract_param)
        elif eng_type == "rectangle":
            image_path = process_rectangle_visualization(params_text, extract_param)
        elif eng_type == "parallelogram":
            image_path = process_parallelogram_visualization(params_text, extract_param)
        elif eng_type == "trapezoid":
            image_path = process_trapezoid_visualization(params_text, extract_param)
        elif eng_type == "circle":
            image_path = process_circle_visualization(params_text, extract_param)
        elif eng_type == "graph":
            func_str = extract_param(REGEX_PATTERNS["graph"]["function"], params_text)
            x_range_str = extract_param(REGEX_PATTERNS["graph"]["x_range"], params_text)
            y_range_str = extract_param(REGEX_PATTERNS["graph"]["y_range"], params_text)
            
            # Парсим диапазоны для осей
            try:
                x_min, x_max = map(float, x_range_str.split(','))
            except:
                x_min, x_max = -10, 10
                
            if y_range_str and y_range_str.lower() != "авто":
                try:
                    y_min, y_max = map(float, y_range_str.split(','))
                except:
                    y_min, y_max = None, None
            else:
                y_min, y_max = None, None
            
            image_path = process_function_plot(func_str, x_min, x_max, y_min, y_max)
        elif eng_type == "coordinate":
            image_path = process_coordinate_visualization(params_text, extract_param)
        else:
            logging.error(f"Неизвестный тип изображения: {image_type}")
            return None, None
        
        # Если изображение создано успешно, конвертируем его в base64
        if image_path:
            with open(image_path, "rb") as img_file:
                img_data = img_file.read()
                img_base64 = base64.b64encode(img_data).decode('utf-8')
                return image_path, img_base64
        
        return None, None
        
    except Exception as e:
        logging.error(f"Ошибка при обработке параметров визуализации: {e}")
        traceback.print_exc()
        return None, None

def process_graph_visualization(params_text, extract_param):
    """Обрабатывает параметры для графика функции"""
    # Извлекаем параметры для графика
    function_expr = extract_param(r'Функция[^:]*:\s*["\']?([^"\'\n]+)["\']?', params_text, "x**2")
    x_range_str = extract_param(r'Диапазон X[^:]*:\s*["\']?([^"\'\n]+)["\']?', params_text, "-10,10")
    y_range_str = extract_param(r'Диапазон Y[^:]*:\s*["\']?([^"\'\n]+)["\']?', params_text, "авто")
    
    # Преобразуем диапазоны в числа
    try:
        x_min, x_max = map(float, x_range_str.split(','))
        x_range = (x_min, x_max)
    except:
        x_range = (-10, 10)
    
    if y_range_str.lower() != "авто":
        try:
            y_min, y_max = map(float, y_range_str.split(','))
            y_range = (y_min, y_max)
        except:
            y_range = None
    else:
        y_range = None
        
    # Генерируем изображение графика
    return generate_graph_image(function_expr, x_range, y_range)

def process_triangle_visualization(params_text, extract_param):
    """Обрабатывает параметры для треугольника"""
    # Извлекаем параметры для треугольника
    coords_str = extract_param(REGEX_PATTERNS["triangle"]["coords"], params_text)
    show_angles = extract_param(REGEX_PATTERNS["triangle"]["angles"], params_text, "нет").lower() in ["да", "yes", "true"]
    show_lengths = extract_param(REGEX_PATTERNS["triangle"]["lengths"], params_text, "нет").lower() in ["да", "yes", "true"]
    vertex_labels_str = extract_param(REGEX_PATTERNS["triangle"]["vertex_labels"], params_text)
    is_right = extract_param(REGEX_PATTERNS["triangle"]["is_right"], params_text, "нет").lower() in ["да", "yes", "true"]
    side_lengths_str = extract_param(REGEX_PATTERNS["triangle"]["side_lengths"], params_text)
    
    # Параметры для отображения треугольника
    params = DEFAULT_VISUALIZATION_PARAMS["triangle"].copy()
    
    # Парсинг координат
    if coords_str:
        try:
            coords = coords_str.strip()
            # Преобразуем строку с координатами в список кортежей
            points = []
            
            # Разбираем строку типа "(x1,y1),(x2,y2),(x3,y3)"
            for match in re.finditer(r'\(([^)]+)\)', coords):
                coord_str = match.group(1)
                x, y = map(float, coord_str.split(','))
                points.append((x, y))
            
            if len(points) == 3:
                params['points'] = points
        except Exception as e:
            logging.warning(f"Ошибка при разборе координат треугольника: {e}")
    
    # Парсинг меток вершин
    if vertex_labels_str:
        try:
            labels = [label.strip() for label in vertex_labels_str.split(',')]
            if len(labels) >= 3:
                params['vertex_labels'] = labels[:3]
            params['show_labels'] = True
        except Exception as e:
            logging.warning(f"Ошибка при разборе меток вершин треугольника: {e}")
    
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
            logging.warning(f"Ошибка при разборе длин сторон треугольника: {e}")
    
    # Добавляем другие параметры
    params['show_angles'] = show_angles
    params['is_right'] = is_right
    
    # Генерируем треугольник
    output_image = generate_geometric_figure('triangle', params, f'triangle_{uuid.uuid4().hex[:8]}.png')
    return output_image

def process_rectangle_visualization(params_text, extract_param):
    """Обрабатывает параметры для прямоугольника"""
    # Извлекаем параметры для прямоугольника
    dimensions_str = extract_param(REGEX_PATTERNS["rectangle"]["dimensions"], params_text)
    coords_str = extract_param(REGEX_PATTERNS["rectangle"]["coords"], params_text)
    vertex_labels_str = extract_param(REGEX_PATTERNS["rectangle"]["vertex_labels"], params_text)
    show_dimensions = extract_param(REGEX_PATTERNS["rectangle"]["show_dimensions"], params_text, "нет").lower() in ["да", "yes", "true"]
    show_labels = extract_param(REGEX_PATTERNS["rectangle"]["show_labels"], params_text, "нет").lower() in ["да", "yes", "true"]
    side_lengths_str = extract_param(REGEX_PATTERNS["rectangle"]["side_lengths"], params_text)
    show_angles = extract_param(REGEX_PATTERNS["rectangle"]["show_angles"], params_text, "нет").lower() in ["да", "yes", "true"]
    angle_values_str = extract_param(REGEX_PATTERNS["rectangle"]["angle_values"], params_text)
    
    # Параметры для отображения прямоугольника
    params = DEFAULT_VISUALIZATION_PARAMS["rectangle"].copy()
    
    # Парсинг размеров
    if dimensions_str:
        try:
            dimensions = dimensions_str.strip()
            width, height = map(float, dimensions.split(','))
            params['width'] = width
            params['height'] = height
        except Exception as e:
            logging.warning(f"Ошибка при разборе размеров прямоугольника: {e}")
    
    # Парсинг координат
    if coords_str:
        try:
            coords = coords_str.strip()
            match = re.search(r'\(([^,]+),([^)]+)\)', coords)
            if match:
                x, y = float(match.group(1)), float(match.group(2))
                params['x'] = x
                params['y'] = y
        except Exception as e:
            logging.warning(f"Ошибка при разборе координат прямоугольника: {e}")
    
    # Парсинг меток вершин
    if vertex_labels_str:
        try:
            labels = [label.strip() for label in vertex_labels_str.split(',')]
            if len(labels) >= 4:
                params['vertex_labels'] = labels[:4]
            params['show_labels'] = True
        except Exception as e:
            logging.warning(f"Ошибка при разборе меток вершин прямоугольника: {e}")
    
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
            logging.warning(f"Ошибка при разборе длин сторон прямоугольника: {e}")
    
    # Парсинг значений углов
    if angle_values_str:
        try:
            angle_values = []
            for angle_str in angle_values_str.split(','):
                angle_str = angle_str.strip()
                if angle_str.lower() in ["нет", "no", "none", "-"]:
                    angle_values.append(None)
                else:
                    angle_values.append(float(angle_str))
            params['angle_values'] = angle_values
        except Exception as e:
            logging.warning(f"Ошибка при разборе значений углов прямоугольника: {e}")
    
    # Добавляем другие параметры
    params['show_labels'] = show_labels
    params['show_lengths'] = show_dimensions
    params['show_angles'] = show_angles
    
    # Генерируем прямоугольник
    output_image = generate_geometric_figure('rectangle', params, f'rectangle_{uuid.uuid4().hex[:8]}.png')
    return output_image

def process_parallelogram_visualization(params_text, extract_param):
    """Обрабатывает параметры для параллелограмма"""
    # Извлекаем параметры для параллелограмма
    dimensions_str = extract_param(REGEX_PATTERNS["parallelogram"]["dimensions"], params_text)
    coords_str = extract_param(REGEX_PATTERNS["parallelogram"]["coords"], params_text)
    vertex_labels_str = extract_param(REGEX_PATTERNS["parallelogram"]["vertex_labels"], params_text)
    show_dimensions = extract_param(REGEX_PATTERNS["parallelogram"]["show_dimensions"], params_text, "нет").lower() in ["да", "yes", "true"]
    show_labels = extract_param(REGEX_PATTERNS["parallelogram"]["show_labels"], params_text, "нет").lower() in ["да", "yes", "true"]
    skew_str = extract_param(REGEX_PATTERNS["parallelogram"]["skew"], params_text)
    side_lengths_str = extract_param(REGEX_PATTERNS["parallelogram"]["side_lengths"], params_text)
    show_angles = extract_param(REGEX_PATTERNS["parallelogram"]["show_angles"], params_text, "нет").lower() in ["да", "yes", "true"]
    angle_values_str = extract_param(REGEX_PATTERNS["parallelogram"]["angle_values"], params_text)
    
    # Параметры для отображения параллелограмма
    params = DEFAULT_VISUALIZATION_PARAMS["parallelogram"].copy()
    
    # Парсинг размеров
    if dimensions_str:
        try:
            dimensions = dimensions_str.strip()
            width, height = map(float, dimensions.split(','))
            params['width'] = width
            params['height'] = height
        except Exception as e:
            logging.warning(f"Ошибка при разборе размеров параллелограмма: {e}")
    
    # Парсинг координат
    if coords_str:
        try:
            coords = coords_str.strip()
            match = re.search(r'\(([^,]+),([^)]+)\)', coords)
            if match:
                x, y = float(match.group(1)), float(match.group(2))
                params['x'] = x
                params['y'] = y
        except Exception as e:
            logging.warning(f"Ошибка при разборе координат параллелограмма: {e}")
    
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
            labels = [label.strip() for label in vertex_labels_str.split(',')]
            if len(labels) >= 4:
                params['vertex_labels'] = labels[:4]
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
    
    # Парсинг значений углов
    if angle_values_str:
        try:
            angle_values = []
            for angle_str in angle_values_str.split(','):
                angle_str = angle_str.strip()
                if angle_str.lower() in ["нет", "no", "none", "-"]:
                    angle_values.append(None)
                else:
                    angle_values.append(float(angle_str))
            params['angle_values'] = angle_values
        except Exception as e:
            logging.warning(f"Ошибка при разборе значений углов параллелограмма: {e}")
    
    # Добавляем другие параметры
    params['show_labels'] = show_labels
    params['show_lengths'] = show_dimensions
    params['show_angles'] = show_angles
    
    # Генерируем параллелограмм
    output_image = generate_geometric_figure('parallelogram', params, f'parallelogram_{uuid.uuid4().hex[:8]}.png')
    return output_image

def process_trapezoid_visualization(params_text, extract_param):
    """Обрабатывает параметры для трапеции"""
    # Извлекаем параметры для трапеции
    dimensions_str = extract_param(REGEX_PATTERNS["trapezoid"]["dimensions"], params_text)
    coords_str = extract_param(REGEX_PATTERNS["trapezoid"]["coords"], params_text)
    vertex_labels_str = extract_param(REGEX_PATTERNS["trapezoid"]["vertex_labels"], params_text)
    show_dimensions = extract_param(REGEX_PATTERNS["trapezoid"]["show_dimensions"], params_text, "нет").lower() in ["да", "yes", "true"]
    show_labels = extract_param(REGEX_PATTERNS["trapezoid"]["show_labels"], params_text, "нет").lower() in ["да", "yes", "true"]
    top_width_str = extract_param(REGEX_PATTERNS["trapezoid"]["top_width"], params_text)
    side_lengths_str = extract_param(REGEX_PATTERNS["trapezoid"]["side_lengths"], params_text)
    show_angles = extract_param(REGEX_PATTERNS["trapezoid"]["show_angles"], params_text, "нет").lower() in ["да", "yes", "true"]
    angle_values_str = extract_param(REGEX_PATTERNS["trapezoid"]["angle_values"], params_text)
    
    # Параметры для отображения трапеции
    params = DEFAULT_VISUALIZATION_PARAMS["trapezoid"].copy()
    
    # Парсинг размеров
    if dimensions_str:
        try:
            dimensions = dimensions_str.strip()
            bottom_width, height = map(float, dimensions.split(','))
            params['bottom_width'] = bottom_width
            params['height'] = height
        except Exception as e:
            logging.warning(f"Ошибка при разборе размеров трапеции: {e}")
    
    # Парсинг верхнего основания
    if top_width_str:
        try:
            top_width = float(top_width_str.strip())
            params['top_width'] = top_width
        except Exception as e:
            logging.warning(f"Ошибка при разборе верхнего основания трапеции: {e}")
    
    # Парсинг координат
    if coords_str:
        try:
            coords = coords_str.strip()
            match = re.search(r'\(([^,]+),([^)]+)\)', coords)
            if match:
                x, y = float(match.group(1)), float(match.group(2))
                params['x'] = x
                params['y'] = y
        except Exception as e:
            logging.warning(f"Ошибка при разборе координат трапеции: {e}")
    
    # Парсинг меток вершин
    if vertex_labels_str:
        try:
            labels = [label.strip() for label in vertex_labels_str.split(',')]
            if len(labels) >= 4:
                params['vertex_labels'] = labels[:4]
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
    
    # Парсинг значений углов
    if angle_values_str:
        try:
            angle_values = []
            for angle_str in angle_values_str.split(','):
                angle_str = angle_str.strip()
                if angle_str.lower() in ["нет", "no", "none", "-"]:
                    angle_values.append(None)
                else:
                    angle_values.append(float(angle_str))
            params['angle_values'] = angle_values
        except Exception as e:
            logging.warning(f"Ошибка при разборе значений углов трапеции: {e}")
    
    # Добавляем другие параметры
    params['show_labels'] = show_labels
    params['show_lengths'] = show_dimensions
    params['show_angles'] = show_angles
    
    # Генерируем трапецию
    output_image = generate_geometric_figure('trapezoid', params, f'trapezoid_{uuid.uuid4().hex[:8]}.png')
    return output_image

def process_circle_visualization(params_text, extract_param):
    """Обрабатывает параметры для окружности"""
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
    
    # Генерируем окружность
    output_image = generate_geometric_figure('circle', params, f'circle_{uuid.uuid4().hex[:8]}.png')
    return output_image

def convert_markdown_to_html(text):
    """
    Конвертирует Markdown-разметку в HTML-разметку
    
    Args:
        text: Текст с возможной Markdown-разметкой
        
    Returns:
        str: Текст с HTML-разметкой
    """
    if not text:
        return ""
    
    # Заменяем жирный текст с ** на <b>
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    
    # Заменяем курсив с * на <i>
    text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
    
    # Заменяем одиночные переносы строк на <br>
    text = re.sub(r'(?<!\n)\n(?!\n)', '<br>\n', text)
    
    # Заменяем двойные переносы строк на новые параграфы
    text = re.sub(r'\n\n+', '</p><p>', text)
    
    # Если текст начинается с тега </p>, удаляем его
    text = re.sub(r'^</p>', '', text)
    
    # Если текст не заканчивается на </p>, добавляем его
    if not text.endswith('</p>') and '</p>' in text:
        text += '</p>'
    
    # Если в тексте есть тег </p>, но нет открывающего <p> в начале, добавляем его
    if '</p>' in text and not text.startswith('<p>'):
        text = '<p>' + text
    
    return text

def fix_html_tags(html_text):
    """
    Проверяет и исправляет HTML-теги в тексте
    
    Args:
        html_text: Текст с HTML-тегами
        
    Returns:
        str: Исправленный текст с корректными HTML-тегами
    """
    if not html_text:
        return ""
    
    # Обрабатываем случай, когда в тексте нет HTML
    if "<" not in html_text and ">" not in html_text:
        # Оборачиваем каждый абзац в теги <p>
        paragraphs = html_text.split("\n\n")
        wrapped_paragraphs = [f"<p>{p}</p>" for p in paragraphs if p.strip()]
        return "\n".join(wrapped_paragraphs)
    
    # Проверяем наличие незакрытых тегов p
    open_p_count = html_text.count("<p>")
    close_p_count = html_text.count("</p>")
    
    if open_p_count > close_p_count:
        # Добавляем недостающие закрывающие теги
        html_text += "</p>" * (open_p_count - close_p_count)
    
    # Проверяем теги <b> и </b>
    open_b_count = html_text.count("<b>")
    close_b_count = html_text.count("</b>")
    
    if open_b_count > close_b_count:
        # Добавляем недостающие закрывающие теги
        html_text += "</b>" * (open_b_count - close_b_count)
    
    # Проверяем теги <i> и </i>
    open_i_count = html_text.count("<i>")
    close_i_count = html_text.count("</i>")
    
    if open_i_count > close_i_count:
        # Добавляем недостающие закрывающие теги
        html_text += "</i>" * (open_i_count - close_i_count)
    
    # Если текст не начинается с <p> и не заканчивается </p>, оборачиваем его
    if not html_text.startswith("<p>") and not html_text.endswith("</p>"):
        html_text = f"<p>{html_text}</p>"
    elif not html_text.endswith("</p>"):
        # Если текст не заканчивается </p>, но содержит HTML-теги, добавляем закрывающий тег
        html_text += "</p>"
    
    # Дополнительная обработка для случая с несколькими абзацами
    lines = html_text.split('\n')
    result = []
    
    for line in lines:
        # Если строка не пустая и не начинается с HTML-тега, оборачиваем её в <p>
        if line.strip() and not line.strip().startswith('<'):
            if not line.strip().endswith('</p>'):
                line = f"<p>{line}</p>"
        result.append(line)
    
    return '\n'.join(result)

def generate_markdown_task(category, subcategory="", difficulty_level=3):
    """
    Генерирует полный пакет задачи, решения и подсказок в формате Markdown.
    
    Args:
        category: Категория задачи
        subcategory: Подкатегория задачи (опционально)
        difficulty_level: Уровень сложности подсказок (1-5)
        
    Returns:
        dict: Словарь с задачей, решением и подсказками в Markdown
    """
    # Используем существующую функцию для генерации задачи
    result = generate_complete_task(category, subcategory, difficulty_level)
    
    # Проверяем на ошибки
    if "error" in result:
        return result
    
    # Преобразуем HTML в Markdown
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
        "difficulty_level": result.get("difficulty_level", difficulty_level)
    }
    
    return markdown_result

def html_to_markdown(html_text):
    """
    Преобразует HTML в Markdown
    
    Args:
        html_text: Текст с HTML-тегами
        
    Returns:
        str: Текст в формате Markdown
    """
    if not html_text:
        return ""
    
    # Сначала обработаем LaTeX-формулы, чтобы не потерять их
    # Сохраняем формулы в словаре и заменяем их плейсхолдерами
    math_placeholders = {}
    math_counter = 0
    
    # Находим все LaTeX-формулы (внутри $ $ или $$ $$)
    def save_math(match):
        nonlocal math_counter
        placeholder = f"__MATH_PLACEHOLDER_{math_counter}__"
        math_placeholders[placeholder] = match.group(0)
        math_counter += 1
        return placeholder
    
    # Сохраняем одинарные $ формулы
    html_text = re.sub(r'\$(.*?)\$', save_math, html_text)
    # Сохраняем двойные $$ формулы
    html_text = re.sub(r'\$\$(.*?)\$\$', save_math, html_text)
    
    # Замена HTML тегов на Markdown
    # Используем BeautifulSoup для корректного парсинга HTML
    try:
        soup = BeautifulSoup(html_text, 'html.parser')
        
        # Обработка параграфов
        for p in soup.find_all('p'):
            p.insert_after('\n\n')
            p.unwrap()
        
        # Обработка жирного текста
        for b in soup.find_all(['b', 'strong']):
            b.insert_before('**')
            b.insert_after('**')
            b.unwrap()
        
        # Обработка курсива
        for i in soup.find_all(['i', 'em']):
            i.insert_before('*')
            i.insert_after('*')
            i.unwrap()
        
        # Обработка переносов строк
        for br in soup.find_all('br'):
            br.replace_with('\n')
        
        # Получаем текст без HTML-тегов
        text = str(soup)
        
        # Убираем оставшиеся HTML-теги
        text = re.sub(r'<[^>]+>', '', text)
        
    except ImportError:
        # Если BeautifulSoup не установлен, используем регулярные выражения
        # Параграфы
        text = re.sub(r'<p>(.*?)</p>', r'\1\n\n', html_text, flags=re.DOTALL)
        
        # Жирный текст
        text = re.sub(r'<b>(.*?)</b>', r'**\1**', text, flags=re.DOTALL)
        text = re.sub(r'<strong>(.*?)</strong>', r'**\1**', text, flags=re.DOTALL)
        
        # Курсив
        text = re.sub(r'<i>(.*?)</i>', r'*\1*', text, flags=re.DOTALL)
        text = re.sub(r'<em>(.*?)</em>', r'*\1*', text, flags=re.DOTALL)
        
        # Переносы строк
        text = re.sub(r'<br\s*/?>', r'\n', text)
        
        # Убираем оставшиеся HTML-теги
        text = re.sub(r'<[^>]+>', '', text)
    
    # Удаляем лишние пробелы и переносы строк
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()
    
    # Возвращаем формулы обратно
    for placeholder, formula in math_placeholders.items():
        text = text.replace(placeholder, formula)
    
    return text

def generate_json_task(category, subcategory="", difficulty_level=3):
    """
    Генерирует полный пакет задачи, решения и подсказок в формате JSON.
    
    Args:
        category: Категория задачи
        subcategory: Подкатегория задачи (опционально)
        difficulty_level: Уровень сложности подсказок (1-5)
        
    Returns:
        dict: Словарь с задачей, решением и подсказками в формате JSON
    """
    # Используем существующую функцию для генерации задачи
    result = generate_complete_task(category, subcategory, difficulty_level)
    
    # Проверяем на ошибки
    if "error" in result:
        return result
    
    # Извлекаем ответ из решения для отдельного поля
    solution = result.get("solution", "")
    answer = result.get("answer", "")
    
    # Если ответ не был успешно извлечен, пробуем найти его снова
    if not answer or answer == "См. решение":
        answer_match = re.search(r'(?:Ответ|ОТВЕТ|ответ)\s*:(.+?)(?=$|\.|\.\s*$|\n\s*\n|\<\/p\>)', solution, re.IGNORECASE | re.DOTALL)
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
        "format": "html"  # Указываем формат данных
    }
    
    return json_result

def generate_json_markdown_task(category, subcategory="", difficulty_level=3):
    """
    Генерирует полный пакет задачи, решения и подсказок в формате JSON с Markdown-форматированием.
    
    Args:
        category: Категория задачи
        subcategory: Подкатегория задачи (опционально)
        difficulty_level: Уровень сложности подсказок (1-5)
        
    Returns:
        dict: Словарь с задачей, решением и подсказками в формате JSON (Markdown)
    """
    # Используем существующую функцию для генерации задачи в HTML
    result = generate_complete_task(category, subcategory, difficulty_level)
    
    # Проверяем на ошибки
    if "error" in result:
        return result
    
    # Преобразуем HTML в Markdown
    task_md = html_to_markdown(result.get("task", ""))
    solution_md = html_to_markdown(result.get("solution", ""))
    hints_md = [html_to_markdown(hint) for hint in result.get("hints", [])]
    
    # Извлекаем ответ из решения
    solution = result.get("solution", "")
    answer = result.get("answer", "")
    
    # Если ответ не был успешно извлечен, пробуем найти его снова
    if not answer or answer == "См. решение":
        answer_match = re.search(r'(?:Ответ|ОТВЕТ|ответ)\s*:(.+?)(?=$|\.|\.\s*$|\n\s*\n|\<\/p\>)', solution, re.IGNORECASE | re.DOTALL)
        if answer_match:
            answer = answer_match.group(1).strip()
            # Преобразуем ответ в Markdown
            answer = html_to_markdown(answer)
    
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
        "format": "markdown"  # Указываем формат данных
    }
    
    return json_result

def generate_complete_task(category, subcategory="", difficulty_level=3):
    """
    Генерирует полный пакет: задачу, решение и подсказки.
    
    Args:
        category: Категория задачи
        subcategory: Подкатегория задачи (опционально)
        difficulty_level: Уровень сложности подсказок (1-5)
        
    Returns:
        dict: Словарь с задачей, решением, подсказками и другими данными
    """
    try:
        # Выбираем случайную задачу из каталога
        data = select_file(category, subcategory)
        
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
        generated_text = yandex_gpt_generate(prompt, temperature=0.6)
        
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
        image_base64 = None
        
        if visualization_match:
            # Обрабатываем параметры визуализации
            params_text = visualization_match.group(1).strip()
            image_path, image_base64 = process_visualization_params(params_text)
        
        # Формируем результат
        result = {
            "task": convert_markdown_to_html(task),
            "solution": convert_markdown_to_html(solution),
            "hints": [convert_markdown_to_html(hint) for hint in hints],
            "answer": answer,
            "difficulty_level": difficulty_level,
            "category": category,
            "subcategory": subcategory
        }
        
        # Добавляем информацию об изображении, если оно есть
        if image_path:
            result["image_path"] = image_path
            result["image_base64"] = image_base64
        
        return result
    except Exception as e:
        logging.error(f"Ошибка при генерации задачи: {e}")
        return {"error": f"Произошла ошибка при генерации задачи: {str(e)}"}

def process_function_plot(func_expr, x_min, x_max, y_min=None, y_max=None):
    """
    Обрабатывает параметры для графика функции и генерирует изображение.
    
    Args:
        func_expr: Выражение функции
        x_min, x_max: Диапазон по оси X
        y_min, y_max: Диапазон по оси Y (опционально)
        
    Returns:
        str: Путь к сгенерированному изображению
    """
    y_range = None
    if y_min is not None and y_max is not None:
        y_range = (y_min, y_max)
    
    return generate_graph_image(func_expr, (x_min, x_max), y_range)

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
    
    # Генерируем координатную плоскость
    return generate_coordinate_system(points, functions, vectors)

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