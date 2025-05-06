import matplotlib.pyplot as plt
import numpy as np
import os
import uuid
import re
from pathlib import Path

def create_output_dir():
    """
    Создает директорию для сохранения изображений, если она не существует
    
    Returns:
        str: Путь к директории для сохранения изображений
    """
    output_dir = Path("static/images/generated")
    output_dir.mkdir(parents=True, exist_ok=True)
    return str(output_dir)

def plot_function(equation, x_range=(-10, 10), num_points=1000, title="График функции"):
    """
    Создает график функции и сохраняет его в виде изображения
    
    Args:
        equation: Уравнение функции в виде строки, например "x**2 + 2*x + 1"
        x_range: Диапазон значений x для отображения (кортеж из двух значений)
        num_points: Количество точек для построения графика
        title: Заголовок графика
        
    Returns:
        str: Путь к сохраненному изображению
    """
    # Заменяем стандартные математические операции на их Python-эквиваленты
    equation = equation.replace('^', '**')
    equation = equation.replace('sin', 'np.sin')
    equation = equation.replace('cos', 'np.cos')
    equation = equation.replace('tan', 'np.tan')
    equation = equation.replace('sqrt', 'np.sqrt')
    equation = equation.replace('log', 'np.log')
    equation = equation.replace('ln(', 'np.log(')
    equation = equation.replace('pi', 'np.pi')
    equation = equation.replace('e^', 'np.exp')
    equation = equation.replace('abs', 'np.abs')
    
    # Генерируем значения x
    x = np.linspace(x_range[0], x_range[1], num_points)
    
    # Создаем функцию для вычисления y
    def f(x_val):
        try:
            # Вычисляем значение y для заданного x
            return eval(equation)
        except Exception as e:
            print(f"Ошибка при вычислении значения функции: {e}")
            return np.nan
    
    # Вычисляем значения y
    y = np.array([f(x_val) for x_val in x])
    
    # Создаем график
    plt.figure(figsize=(10, 6))
    plt.plot(x, y)
    plt.grid(True)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    
    # Создаем директорию для сохранения изображения, если она не существует
    output_dir = create_output_dir()
    
    # Генерируем уникальное имя файла
    filename = f"function_{uuid.uuid4()}.png"
    filepath = os.path.join(output_dir, filename)
    
    # Сохраняем изображение
    plt.savefig(filepath)
    plt.close()
    
    return filepath

def plot_triangle(points, title="Треугольник"):
    """
    Создает изображение треугольника и сохраняет его
    
    Args:
        points: Список из трех точек, каждая точка - кортеж (x, y)
        title: Заголовок изображения
        
    Returns:
        str: Путь к сохраненному изображению
    """
    # Преобразуем точки в формат numpy
    points = np.array(points)
    
    # Создаем замкнутый многоугольник, добавляя первую точку в конец
    closed_points = np.vstack([points, points[0]])
    
    # Создаем график
    plt.figure(figsize=(10, 10))
    plt.plot(closed_points[:, 0], closed_points[:, 1], 'b-')
    
    # Добавляем метки для вершин
    for i, (x, y) in enumerate(points):
        plt.text(x, y, f"({x}, {y})", fontsize=12)
    
    # Настройка графика
    plt.grid(True)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    plt.title(title)
    
    # Устанавливаем одинаковый масштаб по осям
    plt.axis('equal')
    
    # Создаем директорию для сохранения изображения, если она не существует
    output_dir = create_output_dir()
    
    # Генерируем уникальное имя файла
    filename = f"triangle_{uuid.uuid4()}.png"
    filepath = os.path.join(output_dir, filename)
    
    # Сохраняем изображение
    plt.savefig(filepath)
    plt.close()
    
    return filepath

def plot_circle(center, radius, title="Окружность"):
    """
    Создает изображение окружности и сохраняет его
    
    Args:
        center: Координаты центра окружности (x, y)
        radius: Радиус окружности
        title: Заголовок изображения
        
    Returns:
        str: Путь к сохраненному изображению
    """
    # Создаем график
    plt.figure(figsize=(10, 10))
    
    # Добавляем окружность
    circle = plt.Circle(center, radius, fill=False, color='blue')
    ax = plt.gca()
    ax.add_patch(circle)
    
    # Добавляем метку для центра
    plt.plot(center[0], center[1], 'ro')
    plt.text(center[0], center[1], f"O({center[0]}, {center[1]})", fontsize=12)
    
    # Настройка графика
    plt.grid(True)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    plt.title(title)
    
    # Устанавливаем размеры графика, чтобы окружность полностью помещалась
    ax.set_xlim(center[0] - radius - 1, center[0] + radius + 1)
    ax.set_ylim(center[1] - radius - 1, center[1] + radius + 1)
    
    # Устанавливаем одинаковый масштаб по осям
    plt.axis('equal')
    
    # Создаем директорию для сохранения изображения, если она не существует
    output_dir = create_output_dir()
    
    # Генерируем уникальное имя файла
    filename = f"circle_{uuid.uuid4()}.png"
    filepath = os.path.join(output_dir, filename)
    
    # Сохраняем изображение
    plt.savefig(filepath)
    plt.close()
    
    return filepath

def parse_visualization_parameters(text):
    """
    Извлекает параметры для визуализации из текста задачи
    
    Args:
        text: Текст задачи или решения
        
    Returns:
        dict: Словарь с параметрами для визуализации
    """
    params = {}
    
    # Ищем функции вида y = f(x)
    function_matches = re.findall(r'y\s*=\s*(.*?)(?=[,.:;)]|\s*$|\n)', text)
    if function_matches:
        params['function'] = function_matches[0].strip()
    
    # Ищем координаты точек
    point_matches = re.findall(r'[A-Z]\s*\((-?\d+(?:[,.]\d+)?)\s*[;,]\s*(-?\d+(?:[,.]\d+)?)\)', text)
    if point_matches:
        params['points'] = [(float(x.replace(',', '.')), float(y.replace(',', '.'))) for x, y in point_matches]
    
    # Ищем радиус окружности
    radius_matches = re.findall(r'радиус[а-я\s]*[=:]\s*(\d+(?:[,.]\d+)?)', text, re.IGNORECASE)
    if radius_matches:
        params['radius'] = float(radius_matches[0].replace(',', '.'))
    
    # Ищем центр окружности
    center_matches = re.findall(r'центр\s*\((-?\d+(?:[,.]\d+)?)\s*[;,]\s*(-?\d+(?:[,.]\d+)?)\)', text, re.IGNORECASE)
    if center_matches:
        params['center'] = (float(center_matches[0][0].replace(',', '.')), 
                           float(center_matches[0][1].replace(',', '.')))
    
    return params

def create_visualization(task_text):
    """
    Создает визуализацию для задачи на основе ее текста
    
    Args:
        task_text: Текст задачи
        
    Returns:
        str: Путь к созданному изображению или None, если визуализация не требуется
    """
    # Извлекаем параметры для визуализации
    params = parse_visualization_parameters(task_text)
    
    # Если найдена функция, строим ее график
    if 'function' in params:
        return plot_function(params['function'], title="График функции y = " + params['function'])
    
    # Если найдены точки и их ровно 3, строим треугольник
    if 'points' in params and len(params['points']) == 3:
        return plot_triangle(params['points'])
    
    # Если найден центр и радиус, строим окружность
    if 'center' in params and 'radius' in params:
        return plot_circle(params['center'], params['radius'])
    
    # Если не нашли подходящих параметров для визуализации
    return None 