import requests
from PIL import Image
import io
import cairosvg
from pix2tex.cli import LatexOCR
import json
import os
import random
from bs4 import BeautifulSoup
import re

# Функция для получения LaTeX-кода из изображения
def get_latex_from_image(image_url):
    # Загружаем изображение по URL
    response = requests.get(image_url)
    if response.status_code != 200:
        raise Exception("Не удалось загрузить изображение.")
    svg_data = response.content

    # Конвертируем SVG в PNG с повышенным разрешением с помощью cairosvg
    png_data = cairosvg.svg2png(bytestring=svg_data, scale=3.0)

    # Загружаем PNG в объект PIL.Image
    img = Image.open(io.BytesIO(png_data))

    # Инициализация модели LatexOCR
    model = LatexOCR()

    # Распознавание формулы на изображении
    latex_code = model(img)

    return latex_code

# Функция для случайной замены числа (в пределах от 0.1x до 10x)
def random_replace_number(match):
    original = match.group(0)
    num = float(original)
    if num == 0:
        return original
    if '.' in original:
        # Если число с плавающей запятой, изменяем его в пределах 0.1x до 10x
        new_num = random.uniform(0.1 * num, 10 * num)
        return f"{new_num:.2f}"
    else:
        # Если целое число, меняем его в пределах от 0.1 до 10 умноженного на число
        new_num = random.randint(max(1, int(0.1 * num)), int(10 * num))
        return str(new_num)

# Функция для случайного изменения операторов с вероятностью 50%
def random_operator_swap(match):
    op = match.group(0)
    if op == '+':
        return '-' if random.random() < 0.5 else '+'
    elif op == '-':
        return '+' if random.random() < 0.5 else '-'
    elif op == '*':
        return '/' if random.random() < 0.5 else '*'
    elif op == '/':
        return '*' if random.random() < 0.5 else '/'
    elif op == '<=':
        return '>=' if random.random() < 0.5 else '<='
    elif op == '>=':
        return '<=' if random.random() < 0.5 else '>='
    elif op == '<':
        return '>' if random.random() < 0.5 else '<'
    elif op == '>':
        return '<' if random.random() < 0.5 else '>'
    else:
        return op

# Основная функция для замены чисел и операторов в LaTeX-выражениях
def modify_math_expression(text):
    # Заменяем все числа
    text = re.sub(r'\d+\.?\d*', random_replace_number, text)

    # Заменяем операторы
    pattern = r'(<=|>=|<|>|\+|\-|\*|\/)'
    text = re.sub(pattern, random_operator_swap, text)

    return text


def select_file(category, subcategory=""):
    base_dir = "/home/thereayou/dimploma/diploma/Data/math_catalog_subcategories"
    category_dir = os.path.join(base_dir, category)
    if not subcategory:
        try:
            subdirs = [d for d in os.listdir(category_dir) if os.path.isdir(os.path.join(category_dir, d))]
        except FileNotFoundError:
            print(f"Каталог {category_dir} не найден.")
            return
        if not subdirs:
            print(f"В каталоге {category_dir} нет подпапок.")
            return
        subcategory = random.choice(subdirs)
        print(f"Случайно выбрана подкатегория: {subcategory}")
    folder = os.path.join(category_dir, subcategory)
    try:
        files = [f for f in os.listdir(folder) if f.endswith(".json") and f.lower() != "subcategories.json"]
    except FileNotFoundError:
        print(f"Каталог {folder} не найден.")
        return
    if not files:
        print("Нет подходящих JSON файлов в каталоге.")
        return
    chosen_file = random.choice(files)
    filepath = os.path.join(folder, chosen_file)
    print(f"Выбран файл: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data

def extract_html(data):
    # Извлечение HTML кода из JSON
    html_code = data.get("condition", {}).get("html", "")
    if not html_code:
        print("HTML-код условия не найден.")
        return

    # Парсим HTML с помощью BeautifulSoup
    soup = BeautifulSoup(html_code, "html.parser")

    return soup

def change_formulas_inside(text):
    # Найдем все изображения и заменим их на измененные LaTeX-коды
    for img in text.find_all("img"):
        alt_text = img.get("alt", "").strip()
        if alt_text:
            latex_code = get_latex_from_image(img['src'])  # Получаем LaTeX из изображения
            # Изменяем LaTeX
            modified_latex = modify_math_expression(latex_code)  # Изменяем LaTeX-формулу
            # Заменяем img на соответствующий измененный LaTeX код
            img.insert_before(f"$$ {modified_latex} $$")  # Вставляем LaTeX код перед тегом <img>
            img.decompose()  # Удаляем тег <img>

    return str(text)



def unimplemented(text):
    return text

# Определяем обработчики по категориям в виде словаря
handlers = {
    "Неравенства": change_formulas_inside,
    "Уравнения": change_formulas_inside,
}


# Функция для обработки JSON и генерации HTML
def process_random_json(category, subcategory=""):
    data = select_file(category, subcategory)

    soup = extract_html(data)

    handler = handlers.get(category, unimplemented)

    # Call handler function to modify the soup
    modified_html = handler(soup)

    return modified_html