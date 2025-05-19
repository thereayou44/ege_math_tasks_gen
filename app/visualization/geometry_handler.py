import re
import os
import logging
import traceback
from app.geometry.trapezoid import Trapezoid
from app.geometry.triangle import Triangle
from app.geometry.rectangle import Rectangle
from app.geometry.circle import Circle
from app.prompts.prompts import REGEX_PATTERNS, DEFAULT_VISUALIZATION_PARAMS
from app.visualization.graphs.function_graphs import generate_multi_function_graph
from app.visualization.parsers.graph_parsers import parse_graph_params
import matplotlib.pyplot as plt

# Словарь, связывающий типы фигур с соответствующими классами
GEOMETRY_CLASSES = {
    'triangle': Triangle,
    'rectangle': Rectangle,
    'trapezoid': Trapezoid,
    'circle': Circle
}

def extract_visualization_params(text):
    """
    Извлекает блок параметров визуализации из текста ответа модели.
    
    Args:
        text (str): Текст ответа модели
        
    Returns:
        str: Блок параметров визуализации или None, если не найден
    """
    # Ищем блок параметров между маркерами
    viz_block_pattern = r'---ПАРАМЕТРЫ ДЛЯ ВИЗУАЛИЗАЦИИ---(.*?)(?=---|\Z)'
    viz_match = re.search(viz_block_pattern, text, re.DOTALL)
    
    if viz_match:
        viz_params = viz_match.group(1).strip()
        logging.info("Найдены параметры для визуализации")
        return viz_params
    
    logging.info("Блок параметров визуализации не найден")
    return None

def identify_figure_type(params_text):
    """
    Определяет тип фигуры из текста параметров.
    
    Args:
        params_text (str): Текст с параметрами визуализации
        
    Returns:
        str: Тип фигуры или None, если не определен
    """
    # Проверяем, нужна ли визуализация вообще
    if "Визуализация не требуется" in params_text:
        logging.info("Визуализация не требуется согласно параметрам")
        return None
    
    # Извлекаем тип фигуры
    type_pattern = r'Тип\s*:\s*([^\n]+)'
    type_match = re.search(type_pattern, params_text, re.IGNORECASE)
    
    if type_match:
        figure_type = type_match.group(1).strip().lower()
        logging.info(f"Определен тип фигуры: {figure_type}")
        return figure_type
    
    logging.warning("Тип фигуры не найден в параметрах")
    return None

def normalize_figure_type(figure_type):
    """
    Нормализует тип фигуры к стандартному названию.
    
    Args:
        figure_type (str): Исходный тип фигуры
        
    Returns:
        str: Нормализованный тип фигуры
    """
    if not figure_type:
        return None
        
    figure_type = figure_type.lower().strip()
    
    # Карта соответствий типов фигур
    type_map = {
        'треугольник': 'triangle',
        'triangle': 'triangle',
        
        'прямоугольник': 'rectangle',
        'rectangle': 'rectangle',
        'квадрат': 'rectangle',
        'square': 'rectangle',
        
        'трапеция': 'trapezoid',
        'trapezoid': 'trapezoid',
        
        'окружность': 'circle',
        'круг': 'circle',
        'circle': 'circle',
        
        'график': 'graph',
        'функция': 'graph',
        'function': 'graph',
        'graph': 'graph',
        
        'координатная_плоскость': 'coordinate',
        'coordinate': 'coordinate'
    }
    
    return type_map.get(figure_type, figure_type)

def create_visualization(params_text, output_dir="static/images/generated"):
    """
    Создает визуализацию по заданным параметрам.
    
    Args:
        params_text (str): Текст с параметрами визуализации
        output_dir (str): Директория для сохранения изображений
        
    Returns:
        str: Путь к сохраненному изображению или None в случае ошибки
    """
    if not params_text:
        logging.warning("Нет параметров для визуализации")
        return None
    
    # Создаем директорию для изображений, если её нет
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Определяем тип фигуры
        figure_type = identify_figure_type(params_text)
        if not figure_type:
            logging.warning("Тип фигуры не определен или визуализация не требуется")
            return None
        
        # Нормализуем тип фигуры
        norm_figure_type = normalize_figure_type(figure_type)
        logging.info(f"Нормализованный тип фигуры: {norm_figure_type}")
        
        # Обрабатываем различные типы фигур
        if norm_figure_type == 'triangle':
            return create_triangle(params_text, output_dir)
            
        elif norm_figure_type == 'rectangle':
            return create_rectangle(params_text, output_dir)
            
        elif norm_figure_type == 'trapezoid':
            return create_trapezoid(params_text, output_dir)
            
        elif norm_figure_type == 'circle':
            return create_circle(params_text, output_dir)
            
        elif norm_figure_type == 'graph':
            return create_function_graph(params_text, output_dir)
            
        elif norm_figure_type == 'coordinate':
            return create_coordinate_plane(params_text, output_dir)
            
        else:
            logging.warning(f"Неизвестный тип фигуры: {figure_type}")
            return None
            
    except Exception as e:
        logging.error(f"Ошибка при создании визуализации: {e}")
        logging.error(traceback.format_exc())
        return None

def create_triangle(params_text, output_dir):
    """
    Создает и сохраняет изображение треугольника.
    
    Args:
        params_text (str): Текст с параметрами треугольника
        output_dir (str): Директория для сохранения
        
    Returns:
        str: Путь к сохраненному изображению или None в случае ошибки
    """
    try:
        # Создаем объект треугольника из текста параметров
        triangle = Triangle.from_text(params_text)
        
        if triangle:
            # Определяем имя файла
            filename = f"{output_dir}/triangle_{hash(params_text) % 10000}.png"
            
            # Создаем фигуру и сохраняем
            fig, ax = plt.subplots(figsize=(10, 10))
            ax = triangle.draw(ax)
            fig.savefig(filename)
            plt.close(fig)
            
            logging.info(f"Сохранено изображение треугольника: {filename}")
            return filename
        else:
            logging.warning("Не удалось создать объект треугольника")
            return None
    except Exception as e:
        logging.error(f"Ошибка при создании треугольника: {e}")
        logging.error(traceback.format_exc())
        return None

def create_rectangle(params_text, output_dir):
    """
    Создает и сохраняет изображение прямоугольника или квадрата.
    
    Args:
        params_text (str): Текст с параметрами прямоугольника
        output_dir (str): Директория для сохранения
        
    Returns:
        str: Путь к сохраненному изображению или None в случае ошибки
    """
    try:
        # Проверяем, является ли прямоугольник квадратом
        is_square = False
        is_square_match = re.search(r'Квадрат\s*:\s*([^\n]+)', params_text, re.IGNORECASE)
        if is_square_match:
            is_square_value = is_square_match.group(1).strip().lower()
            is_square = is_square_value in ['true', 'да', 'yes', '+']
        
        # Создаем объект прямоугольника из текста параметров
        rectangle = Rectangle.from_text(params_text)
        
        if rectangle:
            # Определяем имя файла
            shape_name = "square" if is_square else "rectangle"
            filename = f"{output_dir}/{shape_name}_{hash(params_text) % 10000}.png"
            
            # Создаем фигуру и сохраняем
            fig, ax = plt.subplots(figsize=(10, 10))
            ax = rectangle.draw(ax)
            fig.savefig(filename)
            plt.close(fig)
            
            logging.info(f"Сохранено изображение {shape_name}: {filename}")
            return filename
        else:
            logging.warning("Не удалось создать объект прямоугольника")
            return None
    except Exception as e:
        logging.error(f"Ошибка при создании прямоугольника: {e}")
        logging.error(traceback.format_exc())
        return None

def create_trapezoid(params_text, output_dir):
    """
    Создает и сохраняет изображение трапеции.
    
    Args:
        params_text (str): Текст с параметрами трапеции
        output_dir (str): Директория для сохранения
        
    Returns:
        str: Путь к сохраненному изображению или None в случае ошибки
    """
    try:
        # Создаем объект трапеции из текста параметров
        trapezoid = Trapezoid.from_text(params_text)
        
        if trapezoid:
            # Определяем имя файла
            filename = f"{output_dir}/trapezoid_{hash(params_text) % 10000}.png"
            
            # Создаем фигуру и сохраняем
            fig, ax = plt.subplots(figsize=(10, 10))
            ax = trapezoid.draw(ax)
            fig.savefig(filename)
            plt.close(fig)
            
            logging.info(f"Сохранено изображение трапеции: {filename}")
            return filename
        else:
            logging.warning("Не удалось создать объект трапеции")
            return None
    except Exception as e:
        logging.error(f"Ошибка при создании трапеции: {e}")
        logging.error(traceback.format_exc())
        return None

def create_circle(params_text, output_dir):
    """
    Создает и сохраняет изображение окружности.
    
    Args:
        params_text (str): Текст с параметрами окружности
        output_dir (str): Директория для сохранения
        
    Returns:
        str: Путь к сохраненному изображению или None в случае ошибки
    """
    try:
        # Создаем объект окружности из текста параметров
        circle = Circle.from_text(params_text)
        
        if circle:
            # Определяем имя файла
            filename = f"{output_dir}/circle_{hash(params_text) % 10000}.png"
            
            # Создаем фигуру и сохраняем
            fig, ax = plt.subplots(figsize=(10, 10))
            ax = circle.draw(ax)
            fig.savefig(filename)
            plt.close(fig)
            
            logging.info(f"Сохранено изображение окружности: {filename}")
            return filename
        else:
            logging.warning("Не удалось создать объект окружности")
            return None
    except Exception as e:
        logging.error(f"Ошибка при создании окружности: {e}")
        logging.error(traceback.format_exc())
        return None

def create_function_graph(params_text, output_dir):
    """
    Создает и сохраняет изображение графика функции.
    
    Args:
        params_text (str): Текст с параметрами графика
        output_dir (str): Директория для сохранения
        
    Returns:
        str: Путь к сохраненному изображению или None в случае ошибки
    """
    try:
        # Используем существующие функции для парсинга параметров и генерации графика
        functions, x_range, y_range, special_points = parse_graph_params(params_text)
        
        if functions:
            # Генерируем и сохраняем график
            filename = f"{output_dir}/graph_{hash(params_text) % 10000}.png"
            
            # Создаем график
            fig, ax = plt.subplots(figsize=(10, 10))
            ax = generate_multi_function_graph(functions, x_range, y_range, special_points, ax=ax)
            fig.savefig(filename)
            plt.close(fig)
            
            logging.info(f"Сохранено изображение графика функции: {filename}")
            return filename
        else:
            logging.warning("Не удалось извлечь параметры функций для графика")
            return None
    except Exception as e:
        logging.error(f"Ошибка при создании графика функции: {e}")
        logging.error(traceback.format_exc())
        return None

def create_coordinate_plane(params_text, output_dir):
    """
    Создает и сохраняет изображение координатной плоскости с точками и/или векторами.
    
    Args:
        params_text (str): Текст с параметрами координатной плоскости
        output_dir (str): Директория для сохранения
        
    Returns:
        str: Путь к сохраненному изображению или None в случае ошибки
    """
    try:
        # Извлекаем параметры координатной плоскости
        points_match = re.search(REGEX_PATTERNS['coordinate']['points'], params_text, re.IGNORECASE)
        vectors_match = re.search(REGEX_PATTERNS['coordinate']['vectors'], params_text, re.IGNORECASE)
        functions_match = re.search(REGEX_PATTERNS['coordinate']['functions'], params_text, re.IGNORECASE)
        
        # Создаем координатную плоскость
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Настраиваем оси
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # Определяем границы отображения
        x_min, x_max, y_min, y_max = -10, 10, -10, 10
        
        # Добавляем точки, если они указаны
        if points_match:
            points_str = points_match.group(1).strip()
            try:
                # Извлекаем точки из строки (x,y,метка)
                point_pattern = r'\(([-+]?\d*\.?\d+),([-+]?\d*\.?\d+),([^)]*)\)'
                point_matches = re.finditer(point_pattern, points_str)
                
                for match in point_matches:
                    x = float(match.group(1))
                    y = float(match.group(2))
                    label = match.group(3).strip()
                    
                    # Отображаем точку
                    ax.plot(x, y, 'bo', markersize=6)
                    
                    # Добавляем метку
                    ax.text(x + 0.2, y + 0.2, label, fontsize=12)
                    
                    # Обновляем границы
                    x_min = min(x_min, x - 2)
                    x_max = max(x_max, x + 2)
                    y_min = min(y_min, y - 2)
                    y_max = max(y_max, y + 2)
            except Exception as e:
                logging.warning(f"Ошибка при обработке точек: {e}")
        
        # Добавляем векторы, если они указаны
        if vectors_match:
            vectors_str = vectors_match.group(1).strip()
            try:
                # Извлекаем векторы из строки (x1,y1,x2,y2,метка)
                vector_pattern = r'\(([-+]?\d*\.?\d+),([-+]?\d*\.?\d+),([-+]?\d*\.?\d+),([-+]?\d*\.?\d+),([^)]*)\)'
                vector_matches = re.finditer(vector_pattern, vectors_str)
                
                for match in vector_matches:
                    x1 = float(match.group(1))
                    y1 = float(match.group(2))
                    x2 = float(match.group(3))
                    y2 = float(match.group(4))
                    label = match.group(5).strip()
                    
                    # Отображаем вектор
                    ax.arrow(x1, y1, x2-x1, y2-y1, head_width=0.3, head_length=0.5, 
                              fc='blue', ec='blue', length_includes_head=True)
                    
                    # Добавляем метку
                    mid_x = (x1 + x2) / 2
                    mid_y = (y1 + y2) / 2
                    ax.text(mid_x + 0.2, mid_y + 0.2, label, fontsize=12)
                    
                    # Обновляем границы
                    x_min = min(x_min, x1 - 2, x2 - 2)
                    x_max = max(x_max, x1 + 2, x2 + 2)
                    y_min = min(y_min, y1 - 2, y2 - 2)
                    y_max = max(y_max, y1 + 2, y2 + 2)
            except Exception as e:
                logging.warning(f"Ошибка при обработке векторов: {e}")
        
        # Добавляем функции, если они указаны
        if functions_match:
            functions_str = functions_match.group(1).strip()
            try:
                # Здесь можно использовать функцию parse_graph_params
                functions, x_range, y_range, special_points = parse_graph_params(
                    f"Количество функций: {functions_str.count(',') + 1}\n" + 
                    "\n".join([f"Функция {i+1}: {func.strip()}" for i, func in enumerate(functions_str.split(','))])
                )
                
                if functions:
                    generate_multi_function_graph(functions, [x_min, x_max], None, special_points, ax=ax)
            except Exception as e:
                logging.warning(f"Ошибка при обработке функций: {e}")
        
        # Устанавливаем границы отображения
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        
        # Сохраняем изображение
        filename = f"{output_dir}/coordinate_plane_{hash(params_text) % 10000}.png"
        fig.savefig(filename)
        plt.close(fig)
        
        logging.info(f"Сохранено изображение координатной плоскости: {filename}")
        return filename
    except Exception as e:
        logging.error(f"Ошибка при создании координатной плоскости: {e}")
        logging.error(traceback.format_exc())
        return None

def process_visualization_from_text(text):
    """
    Обрабатывает визуализацию из полного текста ответа модели.
    
    Args:
        text (str): Полный текст ответа модели
        
    Returns:
        str: Путь к сохраненному изображению или None, если визуализация не требуется или ошибка
    """
    # Извлекаем параметры визуализации
    viz_params = extract_visualization_params(text)
    
    if not viz_params:
        logging.info("Параметры визуализации не найдены в тексте")
        return None
    
    # Создаем визуализацию
    image_path = create_visualization(viz_params)
    
    if image_path:
        logging.info(f"Визуализация успешно создана и сохранена: {image_path}")
    else:
        logging.warning("Не удалось создать визуализацию")
    
    return image_path 