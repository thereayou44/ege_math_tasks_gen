#!/usr/bin/env python3
"""
Тестовый скрипт для проверки выборочного отображения параметров геометрических фигур.
"""

import os
import sys
import matplotlib.pyplot as plt

# Получаем корневую директорию проекта
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.geometry import Triangle, Rectangle, Parallelogram, Trapezoid, Circle

def test_selective_params():
    """
    Создает фигуры с выборочными параметрами отображения сторон и углов.
    """
    # Создаем папку для тестовых изображений
    os.makedirs('test_images', exist_ok=True)
    
    # Тест 1: Треугольник с отображением только выбранных сторон
    triangle_selective_sides = Triangle({
        'points': [(0, 0), (4, 0), (2, 3)],
        'show_labels': True,
        'show_lengths': True,
        'show_specific_sides': ['AB', 'BC'],  # Отображаем только стороны AB и BC
        'vertex_labels': ['A', 'B', 'C']
    })
    
    # Сохраняем изображение
    _, ax = plt.subplots(figsize=(8, 8))
    triangle_selective_sides.draw(ax)
    ax.set_title('Треугольник с выборочными сторонами')
    plt.savefig('test_images/triangle_selective_sides.png', dpi=300)
    plt.close()
    
    # Тест 2: Треугольник с отображением только одного угла
    triangle_selective_angles = Triangle({
        'points': [(0, 0), (4, 0), (2, 3)],
        'show_labels': True,
        'show_angles': True,
        'show_specific_angles': ['B'],  # Отображаем только угол B
        'show_angle_arcs': True,
        'vertex_labels': ['A', 'B', 'C']
    })
    
    # Сохраняем изображение
    _, ax = plt.subplots(figsize=(8, 8))
    triangle_selective_angles.draw(ax)
    ax.set_title('Треугольник с выборочными углами')
    plt.savefig('test_images/triangle_selective_angles.png', dpi=300)
    plt.close()
    
    # Тест 3: Параллелограмм с выборочным отображением
    parallelogram_selective = Parallelogram({
        'width': 5,
        'height': 3,
        'skew': 60,
        'show_labels': True,
        'show_lengths': True,
        'show_angles': True,
        'show_specific_sides': ['AB', 'CD'],  # Отображаем только противоположные стороны
        'show_specific_angles': ['A', 'C'],   # Отображаем только противоположные углы
        'vertex_labels': ['A', 'B', 'C', 'D']
    })
    
    # Сохраняем изображение
    _, ax = plt.subplots(figsize=(8, 8))
    parallelogram_selective.draw(ax)
    ax.set_title('Параллелограмм с выборочными параметрами')
    plt.savefig('test_images/parallelogram_selective.png', dpi=300)
    plt.close()
    
    # Тест 4: Прямоугольник с выборочным отображением
    rectangle_selective = Rectangle({
        'width': 5,
        'height': 3,
        'show_labels': True,
        'show_lengths': True,
        'show_specific_sides': ['AB', 'BC'],  # Отображаем только соседние стороны
        'vertex_labels': ['A', 'B', 'C', 'D']
    })
    
    # Сохраняем изображение
    _, ax = plt.subplots(figsize=(8, 8))
    rectangle_selective.draw(ax)
    ax.set_title('Прямоугольник с выборочными сторонами')
    plt.savefig('test_images/rectangle_selective.png', dpi=300)
    plt.close()
    
    # Тест 5: Трапеция с выборочным отображением
    trapezoid_selective = Trapezoid({
        'bottom_width': 6,
        'top_width': 3,
        'height': 3,
        'show_labels': True,
        'show_lengths': True,
        'show_angles': True,
        'show_specific_sides': ['AB', 'CD'],  # Отображаем только основания
        'show_specific_angles': ['A', 'D'],   # Отображаем только два угла
        'vertex_labels': ['A', 'B', 'C', 'D']
    })
    
    # Сохраняем изображение
    _, ax = plt.subplots(figsize=(8, 8))
    trapezoid_selective.draw(ax)
    ax.set_title('Трапеция с выборочными параметрами')
    plt.savefig('test_images/trapezoid_selective.png', dpi=300)
    plt.close()
    
    # Тест 6: Окружность с выборочным отображением
    circle_selective = Circle({
        'center': (0, 0),
        'radius': 3,
        'center_label': 'O',
        'show_center': True,
        'show_radius': True,  # Показываем только радиус
        'show_diameter': False,
        'show_chord': False
    })
    
    # Сохраняем изображение
    _, ax = plt.subplots(figsize=(8, 8))
    circle_selective.draw(ax)
    ax.set_title('Окружность с выборочными параметрами')
    plt.savefig('test_images/circle_selective.png', dpi=300)
    plt.close()
    
    # Тест 7: Комбинированное изображение всех фигур с выборочными параметрами
    fig, axs = plt.subplots(3, 2, figsize=(16, 24))
    fig.suptitle('Фигуры с выборочными параметрами', fontsize=16)
    
    triangle_selective_sides.draw(axs[0, 0])
    axs[0, 0].set_title('Треугольник с выборочными сторонами')
    
    triangle_selective_angles.draw(axs[0, 1])
    axs[0, 1].set_title('Треугольник с выборочными углами')
    
    parallelogram_selective.draw(axs[1, 0])
    axs[1, 0].set_title('Параллелограмм с выборочными параметрами')
    
    rectangle_selective.draw(axs[1, 1])
    axs[1, 1].set_title('Прямоугольник с выборочными сторонами')
    
    trapezoid_selective.draw(axs[2, 0])
    axs[2, 0].set_title('Трапеция с выборочными параметрами')
    
    circle_selective.draw(axs[2, 1])
    axs[2, 1].set_title('Окружность с выборочными параметрами')
    
    # Выравниваем и настраиваем макет
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Сохраняем комбинированное изображение
    plt.savefig('test_images/all_selective_combined.png', dpi=300)
    plt.close()
    
    print("Тестовые изображения с выборочными параметрами сохранены в директории test_images/")
    return True

if __name__ == '__main__':
    test_selective_params() 