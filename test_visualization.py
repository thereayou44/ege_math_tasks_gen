#!/usr/bin/env python3
"""
Тестирование функций визуализации геометрических фигур
"""
import os
import logging
import sys

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),  # вывод в консоль
        logging.FileHandler('visualization_test.log')  # запись в файл
    ]
)

# Добавляем текущую директорию в путь поиска модулей
sys.path.append(os.path.abspath('.'))

# Импортируем нужные функции
from app.visualization.processors import process_visualization_params

def test_triangle_visualization():
    """Тестирует визуализацию треугольника"""
    params_text = """
Тип: треугольник
Метки вершин: A,B,C
Стороны: 3, 4, 5
Углы: 30, 60, 90
Показать высоты: да
"""
    logging.info("Тестирование визуализации треугольника")
    image_path, viz_type = process_visualization_params(params_text)
    logging.info(f"Результат: {image_path}, тип: {viz_type}")
    if image_path:
        logging.info(f"Файл существует: {os.path.exists(image_path)}")
    return image_path, viz_type

def test_rectangle_visualization():
    """Тестирует визуализацию прямоугольника"""
    params_text = """
Тип: прямоугольник
Размеры: 5, 3
Метки вершин: A,B,C,D
Показать диагонали: true
"""
    logging.info("Тестирование визуализации прямоугольника")
    image_path, viz_type = process_visualization_params(params_text)
    logging.info(f"Результат: {image_path}, тип: {viz_type}")
    if image_path:
        logging.info(f"Файл существует: {os.path.exists(image_path)}")
    return image_path, viz_type

def test_circle_visualization():
    """Тестирует визуализацию окружности"""
    params_text = """
Тип: окружность
Радиус: 5
Показать радиус: true
"""
    logging.info("Тестирование визуализации окружности")
    image_path, viz_type = process_visualization_params(params_text)
    logging.info(f"Результат: {image_path}, тип: {viz_type}")
    if image_path:
        logging.info(f"Файл существует: {os.path.exists(image_path)}")
    return image_path, viz_type

def test_trapezoid_visualization():
    """Тестирует визуализацию трапеции"""
    params_text = """
Тип: трапеция
Основания: 6, 3
Высота: 4
Метки вершин: A,B,C,D
Показать высоту: true
Показать среднюю линию: true
"""
    logging.info("Тестирование визуализации трапеции")
    image_path, viz_type = process_visualization_params(params_text)
    logging.info(f"Результат: {image_path}, тип: {viz_type}")
    if image_path:
        logging.info(f"Файл существует: {os.path.exists(image_path)}")
    return image_path, viz_type

def test_graph_visualization():
    """Тестирует визуализацию графика функции"""
    params_text = """
Тип: график
Количество функций: 2
Функция 1: x^2 - 3*x + 2
Цвет 1: blue
Название 1: f(x)
Функция 2: 2*x + 1
Цвет 2: red
Название 2: g(x)
Диапазон X: [-5, 5]
"""
    logging.info("Тестирование визуализации графика функции")
    image_path, viz_type = process_visualization_params(params_text)
    logging.info(f"Результат: {image_path}, тип: {viz_type}")
    if image_path:
        logging.info(f"Файл существует: {os.path.exists(image_path)}")
    return image_path, viz_type

def run_all_tests():
    """Запускает все тесты визуализации"""
    logging.info("=== Начало тестирования визуализации ===")
    
    test_triangle_visualization()
    test_rectangle_visualization()
    test_circle_visualization()
    test_trapezoid_visualization()
    test_graph_visualization()
    
    logging.info("=== Тестирование визуализации завершено ===")

if __name__ == "__main__":
    run_all_tests() 