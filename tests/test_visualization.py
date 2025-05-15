import os
import sys
import matplotlib.pyplot as plt
import numpy as np

# Добавляем путь к проекту в sys.path
project_path = os.path.abspath(os.path.dirname(__file__))
if project_path not in sys.path:
    sys.path.append(project_path)

# Импортируем нужные функции из проекта
from app.task_generator import generate_geometric_figure

# Создаем директорию для тестовых изображений
os.makedirs('test_images', exist_ok=True)

def test_triangle():
    """Тестирует отображение треугольника с избирательными параметрами"""
    print("Тестирование треугольника...")
    
    # Треугольник с отображением только некоторых углов и сторон
    params = {
        'points': [(0,0), (4,0), (2,3)],
        'vertex_labels': ['A', 'B', 'C'],
        'show_labels': True,
        'show_angles': True,
        'show_specific_angles': ['A', 'C'],
        'show_lengths': True,
        'show_specific_sides': ['AB', 'BC'],
        'show_angle_arcs': True
    }
    
    output_path = os.path.join('test_images', 'triangle_specific.png')
    generate_geometric_figure('triangle', params, output_path)
    print(f"Треугольник с избирательными углами и сторонами создан: {output_path}")
    
    # Треугольник с отображением только некоторых высот и медиан
    params = {
        'points': [(0,0), (4,0), (2,3)],
        'vertex_labels': ['A', 'B', 'C'],
        'show_labels': True,
        'show_heights': True,
        'show_specific_heights': ['A'],
        'show_medians': True,
        'show_specific_medians': ['B'],
        'show_midlines': True,
        'show_specific_midlines': ['BC']
    }
    
    output_path = os.path.join('test_images', 'triangle_elements.png')
    generate_geometric_figure('triangle', params, output_path)
    print(f"Треугольник с избирательными высотами и медианами создан: {output_path}")

def test_rectangle():
    """Тестирует отображение прямоугольника с избирательными параметрами"""
    print("Тестирование прямоугольника...")
    
    # Прямоугольник с отображением только некоторых углов и сторон
    params = {
        'x': 0,
        'y': 0,
        'width': 4,
        'height': 3,
        'vertex_labels': ['A', 'B', 'C', 'D'],
        'show_labels': True,
        'show_angles': True,
        'show_specific_angles': ['A', 'C'],
        'show_lengths': True,
        'show_specific_sides': ['AB', 'BC'],
        'show_angle_arcs': True
    }
    
    output_path = os.path.join('test_images', 'rectangle_specific.png')
    generate_geometric_figure('rectangle', params, output_path)
    print(f"Прямоугольник с избирательными углами и сторонами создан: {output_path}")

def test_parallelogram():
    """Тестирует отображение параллелограмма с избирательными параметрами"""
    print("Тестирование параллелограмма...")
    
    # Параллелограмм с отображением только некоторых углов и сторон
    params = {
        'x': 0,
        'y': 0,
        'width': 4,
        'height': 3,
        'skew': 60,
        'vertex_labels': ['A', 'B', 'C', 'D'],
        'show_labels': True,
        'show_angles': True,
        'show_specific_angles': ['A', 'C'],
        'show_lengths': True,
        'show_specific_sides': ['AB', 'BC'],
        'show_angle_arcs': True
    }
    
    output_path = os.path.join('test_images', 'parallelogram_specific.png')
    generate_geometric_figure('parallelogram', params, output_path)
    print(f"Параллелограмм с избирательными углами и сторонами создан: {output_path}")

def test_trapezoid():
    """Тестирует отображение трапеции с избирательными параметрами"""
    print("Тестирование трапеции...")
    
    # Трапеция с отображением только некоторых углов и сторон
    params = {
        'x': 0,
        'y': 0,
        'bottom_width': 6,
        'top_width': 3,
        'height': 3,
        'vertex_labels': ['A', 'B', 'C', 'D'],
        'show_labels': True,
        'show_angles': True,
        'show_specific_angles': ['A', 'C'],
        'show_lengths': True,
        'show_specific_sides': ['AB', 'BC'],
        'show_angle_arcs': True
    }
    
    output_path = os.path.join('test_images', 'trapezoid_specific.png')
    generate_geometric_figure('trapezoid', params, output_path)
    print(f"Трапеция с избирательными углами и сторонами создан: {output_path}")

if __name__ == "__main__":
    print("Начинаем тестирование избирательного отображения элементов геометрических фигур...")
    
    test_triangle()
    test_rectangle()
    test_parallelogram()
    test_trapezoid()
    
    print("Тестирование завершено. Проверьте изображения в папке 'test_images'.") 