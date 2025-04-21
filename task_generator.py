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
from prompts import HINT_PROMPTS, SYSTEM_PROMPT, create_complete_task_prompt

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Загрузка переменных окружения из .env файла
load_dotenv()

# Инициализация Yandex API
YANDEX_API_KEY = os.getenv("YANDEX_API_KEY")
YANDEX_FOLDER_ID = os.getenv("YANDEX_FOLDER_ID")

# Кэш для хранения результатов запросов API
_task_cache = {}

# Словарь с готовыми инструкциями для подсказок разных уровней
hint_prompts = {
    0: """
---ПОДСКАЗКИ---
1. Для данного уровня подсказки не предусмотрены.
2. Для данного уровня подсказки не предусмотрены.
3. Для данного уровня подсказки не предусмотрены.
""",
    1: """
---ПОДСКАЗКИ---
1. [Краткое указание на конкретный метод/теорему без объяснений (например: "Используй теорему Виета для решения квадратного уравнения")]
2. [Направляющий вопрос или утверждение (например: "Какие тригонометрические функции можно использовать для этого треугольника?")]
3. [Конкретное указание первого шага решения (например: "Начни с составления уравнения равномерного движения: S = v·t")]
""",
    2: """
---ПОДСКАЗКИ---
1. [Конкретная подсказка о методе с аналогией (например: "Эта задача решается аналогично нахождению площади криволинейной трапеции с помощью интеграла")]
2. [Ключевая формула или уравнение (например: "Используй формулу площади сферы S = 4πr²")]
3. [Детальный первый шаг с примером (например: "Составь систему уравнений: { x + y = 10, xy = 21 } и примени теорему Виета")]
""",
    3: """
---ПОДСКАЗКИ---
1. [Подробное объяснение метода решения с формулами (например: "Для нахождения наибольшего значения функции f(x) = x³ - 3x² + 3 найди производную f'(x) и приравняй её к нулю")]
2. [Преобразования уравнения с объяснением (например: "После дифференцирования получаем: f'(x) = 3x² - 6x = 0; вынесем общий множитель: 3x(x - 2) = 0, откуда x = 0 или x = 2")]
3. [Почти полное решение (например: "Проверим точки x = 0 и x = 2: f(0) = 3, f(2) = -1. Также проверим граничные точки интервала: x = -1 → f(-1) = -1 и x = 3 → f(3) = 3. Максимальное значение равно...")]
""",
    4: """
---ПОДСКАЗКИ---
1. [Начало решения с формулами и преобразованиями (например: "Запиши уравнение движения тела, брошенного под углом к горизонту: x = v₀·t·cos α, y = v₀·t·sin α - gt²/2")]
2. [Продолжение решения с подстановками (например: "Подставь v₀ = 20 м/с, α = 30°, g = 10 м/с². Получаем: x = 20t·cos 30° = 20t·√3/2 = 10√3·t, y = 20t·sin 30° - 5t² = 10t - 5t²")]
3. [Почти готовый ответ (например: "Максимальная высота достигается при t = 1 с: yₘₐₓ = 10·1 - 5·1² = ... м")]
""",
    5: """
---ПОДСКАЗКИ---
1. [Первая часть полного решения (например: "Для логарифмического уравнения log₃(x+4) + log₃(x-1) = 2 запишем: log₃((x+4)(x-1)) = 2. По определению: (x+4)(x-1) = 3²")]
2. [Вторая часть с вычислениями (например: "(x+4)(x-1) = 9; x² + 3x - 4 = 9; x² + 3x - 13 = 0. По формуле дискриминанта: D = 3² + 4·13 = 9 + 52 = 61; x = (-3 ± √61)/2")]
3. [Финальная часть решения (например: "x₁ = (-3 + √61)/2 ≈ 3.4, x₂ = (-3 - √61)/2 ≈ -6.4. Проверяем x₁: log₃(3.4+4) + log₃(3.4-1) = log₃(7.4) + log₃(2.4) = log₃(17.76) = 2. Ответ: x = ...")]
"""
}

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

def yandex_gpt_generate(prompt, temperature=0.3, max_tokens=3000):
    """
    Отправляет запрос к API YandexGPT и возвращает ответ.
    
    Args:
        prompt: Текст запроса
        temperature: Температура генерации (от 0 до 1)
        max_tokens: Максимальное количество токенов в ответе
        
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
        
        # Проверяем, есть ли в ответе формулы LaTeX и корректируем их
        # Ищем выражения без окружения $ и оборачиваем их
        formula_pattern = r'(\\frac|\\sqrt|\\sum|\\prod|\\int|\\lim|\\sin|\\cos|\\tan|\\log|\\ln)'
        answer = re.sub(formula_pattern, r'$\1', answer)
        
        # Если мы добавили открывающий символ $, но нет закрывающего, добавляем его
        open_count = answer.count('$')
        if open_count % 2 != 0:
            answer += '$'
            
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
            logging.info(f"Найден альтернативный ответ: {answer}")
            
            # Применяем те же преобразования, что и выше
            formula_pattern = r'(\\frac|\\sqrt|\\sum|\\prod|\\int|\\lim|\\sin|\\cos|\\tan|\\log|\\ln)'
            answer = re.sub(formula_pattern, r'$\1', answer)
            
            open_count = answer.count('$')
            if open_count % 2 != 0:
                answer += '$'
                
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
    Универсальный генератор:
     - Для любых ftype != 'circle': сначала ищем params['points'], иначе — строим своими compute_*.
     - Потом единым draw+labels+lengths+angles.
    """
    try:
        fig, ax = plt.subplots(figsize=(6,6))
        ax.set_aspect('equal')
        ax.axis('off')

        # 1) Вершины для многоугольников
        pts = None
        if figure_type == 'circle':
            # circle рисуем отдельно ниже
            pass
        else:
            if 'points' in params:
                pts = params['points']
            else:
                if figure_type == 'triangle':
                    pts = params.get('points', [(0,0),(1,0),(0.5,0.86)])
                elif figure_type == 'rectangle':
                    x, y = params.get('x',0), params.get('y',0)
                    w, h = params.get('width',4), params.get('height',3)
                    pts = [(x,y), (x+w,y), (x+w,y+h), (x,y+h)]
                elif figure_type == 'parallelogram':
                    base   = params.get('width',4)
                    height = params.get('height',3)
                    skew   = params.get('skew',60)
                    raw    = compute_parallelogram(base, height, skew)
                    x, y   = params.get('x',0), params.get('y',0)
                    pts    = [(px+x, py+y) for px,py in raw]
                elif figure_type == 'trapezoid':
                    bottom = params.get('bottom_width',6)
                    top    = params.get('top_width',3)
                    height = params.get('height',3)
                    raw    = compute_trapezoid(bottom, top, height)
                    x, y   = params.get('x',0), params.get('y',0)
                    pts    = [(px+x, py+y) for px,py in raw]
                else:
                    raise ValueError(f"Неизвестный тип: {figure_type}")

            # рисуем полигон
            xs, ys = zip(*pts)
            pts_closed = np.vstack([pts, pts[0]])
            ax.plot(pts_closed[:,0], pts_closed[:,1], 'b-', lw=2)

            # подписи вершин
            if params.get('show_labels', True):
                labels = params.get('vertex_labels')
                if not labels:
                    labels = [chr(65+i) for i in range(len(pts))]
                for (x0,y0), lab in zip(pts, labels):
                    ax.text(x0, y0, lab, ha='center', va='center', fontsize=12,
                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

            # длины сторон
            if params.get('show_lengths', False):
                for i in range(len(pts)):
                    x0,y0 = pts[i]
                    x1,y1 = pts[(i+1)%len(pts)]
                    L = np.hypot(x1-x0, y1-y0)
                    mx,my = (x0+x1)/2, (y0+y1)/2
                    nx,ny = -(y1-y0)/L, (x1-x0)/L
                    ax.text(mx+nx*0.2, my+ny*0.2, f"{L:.2f}", ha='center', fontsize=10)

            # углы (только для треугольника)
            if figure_type=='triangle' and params.get('show_angles', False):
                for i in range(3):
                    A,B,C = np.array(pts[(i-1)%3]), np.array(pts[i]), np.array(pts[(i+1)%3])
                    v1,v2 = A-B, C-B
                    ang = np.degrees(np.arccos(v1.dot(v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))))
                    uv = (v1/np.linalg.norm(v1)+v2/np.linalg.norm(v2))
                    uv = uv/np.linalg.norm(uv)*0.3
                    ax.text(*(B+uv), f"{ang:.1f}°", ha='center', fontsize=10)

        # 2) Окружность
        if figure_type == 'circle':
            cx, cy = params.get('center',(0,0))
            r      = params.get('radius',3)
            circ   = plt.Circle((cx,cy), r, fill=False, edgecolor='blue', linewidth=2)
            ax.add_patch(circ)
            xs, ys = [cx-r, cx+r], [cy-r, cy+r]
            if params.get('show_center', True):
                ax.text(cx, cy, params.get('center_label','O'),
                        ha='center', va='center', fontsize=12,
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
            if params.get('show_radius', False):
                ax.plot([cx, cx+r], [cy, cy], 'r-', lw=1)
                ax.text(cx+r/2, cy+0.2, f"r={r}", ha='center', fontsize=10)
            if params.get('show_diameter', False):
                ax.plot([cx-r, cx+r], [cy, cy], 'g-', lw=1)
                ax.text(cx, cy-0.2, f"d={2*r}", ha='center', fontsize=10)

        # 3) Авто-лимиты
        if 'xs' in locals() and 'ys' in locals():
            m = 1
            ax.set_xlim(min(xs)-m, max(xs)+m)
            ax.set_ylim(min(ys)-m, max(ys)+m)

        # 4) Сохранение
        out = 'static/images/generated'
        os.makedirs(out, exist_ok=True)
        if not filename:
            filename = f"{figure_type}_{uuid.uuid4().hex[:8]}.png"
        path = os.path.join(out, filename)
        plt.savefig(path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        return path

    except Exception as e:
        logging.error(f"Ошибка при генерации фигуры: {e}")
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

def process_visualization_params(visualization_params_text):
    """
    Обрабатывает параметры визуализации и генерирует соответствующее изображение.
    
    Args:
        visualization_params_text: Текст с параметрами визуализации
        
    Returns:
        tuple: (путь к изображению, base64 представление изображения)
    """
    image_path = None
    image_base64 = None
    
    # Функция для извлечения параметра по шаблону
    def extract_param(pattern, text, default=None):
        match = re.search(pattern, text, re.IGNORECASE)
        return match.group(1).strip() if match else default
    
    try:
        # Определяем тип изображения
        image_type_match = re.search(r'Тип изображения[^:]*:\s*["\']?([^"\'\n]+)["\']?', visualization_params_text, re.IGNORECASE)

        print(f"image_type_match: {image_type_match}")
        if image_type_match:
            image_type = image_type_match.group(1).strip().lower()
            
            # Параметры в зависимости от типа изображения
            if image_type == "график":
                image_path = process_graph_visualization(visualization_params_text, extract_param)
                
            elif image_type in ["треугольник", "triangle"]:
                image_path = process_triangle_visualization(visualization_params_text, extract_param)
                
            elif image_type in ["четырехугольник", "прямоугольник", "rectangle"]:
                # Определяем конкретный тип четырехугольника
                shape_type = extract_param(r'Тип[^:]*:\s*["\']?([^"\'\n]+)["\']?', visualization_params_text, "прямоугольник").lower()
                
                if shape_type in ["трапеция", "trapezoid"]:
                    image_path = process_trapezoid_visualization(visualization_params_text, extract_param)
                elif shape_type in ["параллелограмм", "parallelogram"]:
                    image_path = process_parallelogram_visualization(visualization_params_text, extract_param)
                else:  # По умолчанию - прямоугольник
                    image_path = process_rectangle_visualization(visualization_params_text, extract_param)
            
            elif image_type in ["трапеция", "trapezoid"]:
                image_path = process_trapezoid_visualization(visualization_params_text, extract_param)
                
            elif image_type in ["параллелограмм", "parallelogram"]:
                image_path = process_parallelogram_visualization(visualization_params_text, extract_param)
                
            elif image_type in ["окружность", "круг", "circle"]:
                image_path = process_circle_visualization(visualization_params_text, extract_param)
                
            elif image_type in ["координатная_плоскость", "coordinate"]:
                image_path = process_coordinate_visualization(visualization_params_text, extract_param)
    
        # Если изображение было сгенерировано, конвертируем его в base64
        if image_path:
            image_base64 = get_image_base64(image_path)
    
    except Exception as e:
        logging.error(f"Ошибка при обработке параметров визуализации: {e}")
    
    return image_path, image_base64

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
    coords_str = extract_param(r'Координаты вершин[^:]*:\s*["\']?([^"\'\n]+)["\']?', params_text)
    show_angles = extract_param(r'Показать углы[^:]*:\s*["\']?([^"\'\n]+)["\']?', params_text, "нет").lower() in ["да", "yes", "true"]
    show_lengths = extract_param(r'Показать длины[^:]*:\s*["\']?([^"\'\n]+)["\']?', params_text, "нет").lower() in ["да", "yes", "true"]
    vertex_labels_str = extract_param(r'Подписи вершин[^:]*:\s*["\']?([^"\'\n]+)["\']?', params_text, "A,B,C")
    
    # Парсим метки вершин
    vertex_labels = [lbl.strip() for lbl in vertex_labels_str.split(',') if lbl.strip()]
    if len(vertex_labels) < 3:
        vertex_labels = ['A', 'B', 'C']
    
    # Парсим координаты вершин
    points = []
    if coords_str:
        coords_pattern = r'\((-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\)'
        coords_matches = re.findall(coords_pattern, coords_str)
        points = [(float(x), float(y)) for x, y in coords_matches]
    
    # Если не удалось извлечь координаты, используем значения по умолчанию
    if len(points) < 3:
        points = [(1, 1), (8, 2), (4, 8)]
        
    # Создаем параметры для треугольника
    triangle_params = {
        'points': points,
        'show_angles': show_angles,
        'show_lengths': show_lengths,
        'show_labels': True,  # Показывать подписи вершин
        'vertex_labels': vertex_labels
    }
    
    # Генерируем изображение треугольника
    return generate_geometric_figure("triangle", triangle_params)

def process_rectangle_visualization(params_text, extract_param):
    """Обрабатывает параметры для прямоугольника"""
    # Извлекаем параметры
    dimensions_str = extract_param(r'Размеры[^:]*:\s*["\']?([^"\'\n]+)["\']?', params_text, "6,4")
    coords_str = extract_param(r'Координаты[^:]*:\s*["\']?([^"\'\n]+)["\']?', params_text, "(2,2)")
    vertex_labels_str = extract_param(r'Подписи вершин[^:]*:\s*["\']?([^"\'\n]+)["\']?', params_text, "A,B,C,D")
    show_dimensions_str = extract_param(r'Показать размеры[^:]*:\s*["\']?([^"\'\n]+)["\']?', params_text, "нет").lower()
    show_dimensions = show_dimensions_str in ["да", "yes", "true"]
    show_labels_str = extract_param(r'Показать метки[^:]*:\s*["\']?([^"\'\n]+)["\']?', params_text, "да").lower()
    show_labels = show_labels_str in ["да", "yes", "true"]
    
    # Парсим метки вершин
    vertex_labels = [lbl.strip() for lbl in vertex_labels_str.split(',') if lbl.strip()]
    if len(vertex_labels) < 4:
        vertex_labels = ['A', 'B', 'C', 'D']
    
    # Парсим координаты
    try:
        coord_match = re.search(r'\((-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\)', coords_str)
        if coord_match:
            x, y = float(coord_match.group(1)), float(coord_match.group(2))
        else:
            x, y = 2, 2
    except:
        x, y = 2, 2
        
    # Парсим размеры
    try:
        if ',' in dimensions_str:
            width, height = map(float, dimensions_str.split(','))
        else:
            width = height = float(dimensions_str)
    except:
        width, height = 6, 4

    # Параметры для прямоугольника
    rect_params = {
        'x': x,
        'y': y,
        'width': width,
        'height': height,
        'show_dimensions': show_dimensions,
        'show_labels': show_labels,
        'vertex_labels': vertex_labels
    }
    
    # Генерируем изображение прямоугольника
    return generate_geometric_figure("rectangle", rect_params)

def process_parallelogram_visualization(params_text, extract_param):
    """Обрабатывает параметры для параллелограмма"""
    # Извлекаем параметры
    dimensions_str = extract_param(r'Размеры[^:]*:\s*["\']?([^"\'\n]+)["\']?', params_text, "6,4")
    coords_str = extract_param(r'Координаты[^:]*:\s*["\']?([^"\'\n]+)["\']?', params_text, "(2,2)")
    vertex_labels_str = extract_param(r'Подписи вершин[^:]*:\s*["\']?([^"\'\n]+)["\']?', params_text, "A,B,C,D")
    show_dimensions_str = extract_param(r'Показать размеры[^:]*:\s*["\']?([^"\'\n]+)["\']?', params_text, "нет").lower()
    show_dimensions = show_dimensions_str in ["да", "yes", "true"]
    show_labels_str = extract_param(r'Показать метки[^:]*:\s*["\']?([^"\'\n]+)["\']?', params_text, "да").lower()
    show_labels = show_labels_str in ["да", "yes", "true"]
    skew_str = extract_param(r'Наклон[^:]*:\s*["\']?([^"\'\n]+)["\']?', params_text, "30")
    
    # Парсим метки вершин
    vertex_labels = [lbl.strip() for lbl in vertex_labels_str.split(',') if lbl.strip()]
    if len(vertex_labels) < 4:
        vertex_labels = ['A', 'B', 'C', 'D']
    
    # Парсим координаты
    try:
        coord_match = re.search(r'\((-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\)', coords_str)
        if coord_match:
            x, y = float(coord_match.group(1)), float(coord_match.group(2))
        else:
            x, y = 2, 2
    except:
        x, y = 2, 2
        
    # Парсим размеры
    try:
        if ',' in dimensions_str:
            width, height = map(float, dimensions_str.split(','))
        else:
            width = height = float(dimensions_str)
    except:
        width, height = 6, 4
    
    # Парсим наклон
    try:
        skew = float(skew_str)
    except:
        skew = 30
        
    # Параметры для параллелограмма
    parallelogram_params = {
        'x': x,
        'y': y,
        'width': width,
        'height': height,
        'skew': skew,
        'show_dimensions': show_dimensions,
        'show_labels': show_labels,
        'vertex_labels': vertex_labels
    }
    
    # Генерируем изображение параллелограмма
    return generate_geometric_figure("parallelogram", parallelogram_params)

def process_trapezoid_visualization(params_text, extract_param):
    """Обрабатывает параметры для трапеции"""
    # Извлекаем параметры
    dimensions_str = extract_param(r'Размеры[^:]*:\s*["\']?([^"\'\n]+)["\']?', params_text, "6,4")
    coords_str = extract_param(r'Координаты[^:]*:\s*["\']?([^"\'\n]+)["\']?', params_text, "(2,2)")
    vertex_labels_str = extract_param(r'Подписи вершин[^:]*:\s*["\']?([^"\'\n]+)["\']?', params_text, "A,B,C,D")
    show_dimensions_str = extract_param(r'Показать размеры[^:]*:\s*["\']?([^"\'\n]+)["\']?', params_text, "нет").lower()
    show_dimensions = show_dimensions_str in ["да", "yes", "true"]
    show_labels_str = extract_param(r'Показать метки[^:]*:\s*["\']?([^"\'\n]+)["\']?', params_text, "да").lower()
    show_labels = show_labels_str in ["да", "yes", "true"]
    top_width_str = extract_param(r'Верхняя база[^:]*:\s*["\']?([^"\'\n]+)["\']?', params_text, "3")
    
    # Парсим метки вершин
    vertex_labels = [lbl.strip() for lbl in vertex_labels_str.split(',') if lbl.strip()]
    if len(vertex_labels) < 4:
        vertex_labels = ['A', 'B', 'C', 'D']
    
    # Парсим координаты
    try:
        coord_match = re.search(r'\((-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\)', coords_str)
        if coord_match:
            x, y = float(coord_match.group(1)), float(coord_match.group(2))
        else:
            x, y = 2, 2
    except:
        x, y = 2, 2
        
    # Парсим размеры
    try:
        if ',' in dimensions_str:
            width, height = map(float, dimensions_str.split(','))
        else:
            width = height = float(dimensions_str)
    except:
        width, height = 6, 4
    
    # Парсим верхнюю базу
    try:
        top_width = float(top_width_str)
    except:
        top_width = width / 2
        
    # Параметры для трапеции
    trapezoid_params = {
        'x': x,
        'y': y,
        'bottom_width': width,
        'top_width': top_width,
        'height': height,
        'show_dimensions': show_dimensions,
        'show_labels': show_labels,
        'vertex_labels': vertex_labels
    }
    
    # Генерируем изображение трапеции
    return generate_geometric_figure("trapezoid", trapezoid_params)

def process_circle_visualization(params_text, extract_param):
    """Обрабатывает параметры для окружности"""
    # Извлекаем параметры для окружности
    center_str = extract_param(r'Центр[^:]*:\s*["\']?([^"\'\n]+)["\']?', params_text, "(0,0)")
    radius_str = extract_param(r'Радиус[^:]*:\s*["\']?([^"\'\n]+)["\']?', params_text, "3")
    show_diameter = extract_param(r'Показать диаметр[^:]*:\s*["\']?([^"\'\n]+)["\']?', params_text, "нет").lower() in ["да", "yes", "true"]
    center_label = extract_param(r'Метка центра[^:]*:\s*["\']?([^"\'\n]+)["\']?', params_text, "O")
    show_center = center_label.lower() != "нет"
    show_radius = extract_param(r'Показать радиус[^:]*:\s*["\']?([^"\'\n]+)["\']?', params_text, "нет").lower() in ["да", "yes", "true"]
    
    # Парсим центр
    try:
        center_match = re.search(r'\((-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\)', center_str)
        if center_match:
            cx, cy = float(center_match.group(1)), float(center_match.group(2))
        else:
            cx, cy = 0, 0
    except:
        cx, cy = 0, 0
        
    # Парсим радиус
    try:
        radius = float(radius_str)
    except:
        radius = 3
        
    # Создаем параметры для окружности
    circle_params = {
        'center': (cx, cy),
        'radius': radius,
        'show_radius': show_radius,
        'show_diameter': show_diameter,
        'show_center': show_center,
        'center_label': center_label
    }
    
    # Генерируем изображение окружности
    return generate_geometric_figure("circle", circle_params)

def process_coordinate_visualization(params_text, extract_param):
    """Обрабатывает параметры для координатной плоскости"""
    # Извлекаем параметры для координатной плоскости
    points_str = extract_param(r'Точки[^:]*:\s*["\']?([^"\'\n]+)["\']?', params_text, "")
    vectors_str = extract_param(r'Векторы[^:]*:\s*["\']?([^"\'\n]+)["\']?', params_text, "")
    functions_str = extract_param(r'Функции[^:]*:\s*["\']?([^"\'\n]+)["\']?', params_text, "")
    
    # Парсим точки
    points = []
    if points_str:
        point_pattern = r'([A-Z])\s*\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)'
        point_matches = re.findall(point_pattern, points_str)
        points = [(float(x), float(y), label) for label, x, y in point_matches]
        
        # Если точки заданы без меток, парсим просто координаты
        if not points:
            coords_pattern = r'\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)'
            coords_matches = re.findall(coords_pattern, points_str)
            # Не добавляем метки, только координаты
            points = [(float(x), float(y)) for x, y in coords_matches]
        
    # Парсим векторы
    vectors = []
    if vectors_str:
        # Проверяем формат векторов
        if re.search(r'[A-Z]{2}', vectors_str):  # Формат AB, BC, ...
            vector_labels = re.findall(r'([A-Z]{2})', vectors_str)
            # Находим точки для каждого вектора
            for label in vector_labels:
                start_label, end_label = label[0], label[1]
                start_point = next((p for p in points if len(p) > 2 and p[2] == start_label), None)
                end_point = next((p for p in points if len(p) > 2 and p[2] == end_label), None)
                if start_point and end_point:
                    vectors.append((start_point[0], start_point[1], end_point[0], end_point[1], label))
        else:  # Предполагаем числовой формат
            # Ищем обозначения векторов в формате "(x1,y1,x2,y2[,label])"
            vector_pattern_with_label = r'\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*([A-Za-z0-9]+)\s*\)'
            vector_matches_with_label = re.findall(vector_pattern_with_label, vectors_str)
            if vector_matches_with_label:
                vectors = [(float(x1), float(y1), float(x2), float(y2), label) for x1, y1, x2, y2, label in vector_matches_with_label]
            else:
                # Ищем векторы без меток
                vector_pattern = r'\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)'
                vector_matches = re.findall(vector_pattern, vectors_str)
                # Не добавляем метки к векторам
                vectors = [(float(x1), float(y1), float(x2), float(y2)) for x1, y1, x2, y2 in vector_matches]
    
    # Парсим функции
    functions = []
    if functions_str:
        function_parts = [p.strip() for p in functions_str.split(',')]
        colors = ['blue', 'red', 'green', 'purple', 'orange']
        functions = [(f, colors[i % len(colors)]) for i, f in enumerate(function_parts) if f]
    
    # Генерируем координатную плоскость с точками, векторами и функциями
    return generate_coordinate_system(points, functions, vectors)

def generate_complete_task(category, subcategory="", difficulty_level=3):
    """
    Генерирует полный пакет: задачу, решение и подсказки за один запрос к API.
    """
    logging.info(f"Начало генерации задачи для категории: {category}, подкатегории: {subcategory}")
    try:
        difficulty_level = int(difficulty_level)
        original_data = select_file(category, subcategory)
        if not original_data:
            logging.error("Не удалось выбрать задачу из базы.")
            return {"error": "Не удалось выбрать задачу из базы"}
        
        original_task = extract_text_and_formulas(original_data.get("condition", {}).get("html", ""))
        original_solution = original_data.get("solution", {}).get("text", "")
        
        if not original_task or not original_solution:
            logging.error("Недостаточно данных в исходной задаче.")
            return {"error": "Недостаточно данных в исходной задаче"}
        
        # Проверяем кэш перед генерацией новой задачи
        cache_key = f"{category}_{subcategory}_{original_task}"
        if cache_key in _task_cache and isinstance(_task_cache[cache_key], dict):
            logging.info(f"Задача найдена в кэше: {cache_key}")
            cached_result = _task_cache[cache_key]
            
            # Формируем полный текст задачи из кэша для сохранения в файл
            full_text = f"""===ЗАДАЧА===
{cached_result.get('task', '')}

===РЕШЕНИЕ===
{cached_result.get('solution', '')}

===ПОДСКАЗКИ===
1. {cached_result.get('hints', [''])[0] if len(cached_result.get('hints', [])) > 0 else ''}
2. {cached_result.get('hints', ['', ''])[1] if len(cached_result.get('hints', [])) > 1 else ''}
3. {cached_result.get('hints', ['', '', ''])[2] if len(cached_result.get('hints', [])) > 2 else ''}
"""
            save_to_file(full_text)
            return cached_result
        
        # Создаем промпт для генерации полного материала
        prompt = create_complete_task_prompt(
            category, 
            subcategory, 
            original_task, 
            original_solution, 
            difficulty_level
        )

        # Делаем запрос к API с увеличенным лимитом токенов и фиксированной температурой 0.3
        result_text = yandex_gpt_generate(prompt, temperature=0.3, max_tokens=3000)
        
        # Сохраняем полный ответ API в файл
        save_to_file(result_text)
        
        if not result_text:
            logging.error("Не удалось сгенерировать задачу.")
            return {"error": "Не удалось сгенерировать задачу"}
        
        # Парсим результат
        try:
            task_match = re.search(r'---ЗАДАЧА---\s*(.*?)(?=---РЕШЕНИЕ---)', result_text, re.DOTALL)
            solution_match = re.search(r'---РЕШЕНИЕ---\s*(.*?)(?=---ПОДСКАЗКИ---|---ПАРАМЕТРЫ ДЛЯ ВИЗУАЛИЗАЦИИ---)', result_text, re.DOTALL)
            hints_match = re.search(r'---ПОДСКАЗКИ---\s*(.*?)(?=---ПАРАМЕТРЫ ДЛЯ ВИЗУАЛИЗАЦИИ---|$)', result_text, re.DOTALL)
            visualization_match = re.search(r'---ПАРАМЕТРЫ ДЛЯ ВИЗУАЛИЗАЦИИ---\s*(.*?)$', result_text, re.DOTALL)
            
            task = task_match.group(1).strip() if task_match else "Не удалось извлечь текст задачи"
            solution = solution_match.group(1).strip() if solution_match else "Не удалось извлечь решение"
            
            # Если решение пустое или слишком короткое, выходим с ошибкой
            if len(solution.strip()) < 50:
                logging.error("Решение отсутствует или слишком короткое")
                return {"error": "Не удалось сгенерировать полное решение, попробуйте еще раз"}
            
            # Извлекаем подсказки
            hints_text = hints_match.group(1).strip() if hints_match else ""
            hints = parse_hints(hints_text)
            
            # Извлекаем ответ из решения
            answer = extract_answer_with_latex(solution)
            
            # Проверяем качество полученного материала
            if len(task) < 20 or len(solution) < 50:
                logging.error("Сгенерированный материал слишком короткий или некачественный.")
                return {"error": "Не удалось сгенерировать качественный материал"}
            
            # Заменяем Markdown-форматирование на HTML-форматирование, если оно осталось
            task = convert_markdown_to_html(task)
            solution = convert_markdown_to_html(solution)
            hints = [convert_markdown_to_html(hint) for hint in hints]
            
            # Проверяем наличие открывающих и закрывающих HTML-тегов в решении
            solution = fix_html_tags(solution)
            
            # Формируем структурированный полный текст для сохранения в файл
            formatted_text = f"""===ЗАДАЧА===
{task}

===РЕШЕНИЕ===
{solution}

===ПОДСКАЗКИ===
1. {hints[0] if len(hints) > 0 else ''}
2. {hints[1] if len(hints) > 1 else ''}
3. {hints[2] if len(hints) > 2 else ''}
"""
            # Перезаписываем файл с отформатированным текстом
            save_to_file(formatted_text)
            
            # Обрабатываем параметры для визуализации, если они есть
            image_path = None
            image_base64 = None
            
            if visualization_match:
                visualization_params_text = visualization_match.group(1).strip()
                image_path, image_base64 = process_visualization_params(visualization_params_text)
        
            # Сохраняем результат в кэше
            result = {
                "task": task,
                "solution": solution,
                "answer": answer,
                "hints": hints,
                "difficulty_level": difficulty_level
            }
            
            # Если сгенерировано изображение, добавляем его путь в результат
            if image_path:
                result["image_path"] = image_path
                if image_base64:
                    result["image_base64"] = image_base64
            
            _task_cache[cache_key] = result
            
            logging.info(f"Задача успешно сгенерирована за один запрос: {task[:30]}...")
            return result
            
        except Exception as e:
            logging.error(f"Ошибка при парсинге результатов: {e}")
            return {"error": f"Ошибка при обработке результатов: {str(e)}"}
        
    except Exception as e:
        logging.error(f"Ошибка при генерации полного пакета: {e}")
        return {"error": f"Произошла ошибка: {str(e)}"}

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
    
    return html_text

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
        
        # Если есть base64 изображения, добавляем его
        image_base64 = result.get("image_base64", "")
        
        # Добавляем изображение к задаче
        task_images.append({
            "url": image_url,
            "base64": image_base64,
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
        
        # Если есть base64 изображения, добавляем его
        image_base64 = result.get("image_base64", "")
        
        # Добавляем изображение к задаче
        task_images.append({
            "url": image_url,
            "base64": image_base64,
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