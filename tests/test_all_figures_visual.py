#!/usr/bin/env python3
"""
Скрипт для визуальной проверки отображения всех геометрических фигур в одном изображении.
"""

import os
import sys
import matplotlib.pyplot as plt
from app.geometry import Triangle, Rectangle, Parallelogram, Trapezoid, Circle

def create_combined_figure():
    """
    Создает комбинированное изображение со всеми типами фигур для визуального сравнения.
    """
    # Создаем новую фигуру с подграфиками
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Визуальное сравнение всех геометрических фигур', fontsize=16)
    
    # Создаем папку для изображений, если она не существует
    os.makedirs('test_images', exist_ok=True)
    
    # Тест трапеции
    params_trapezoid = {
        'bottom_width': 6,
        'top_width': 3,
        'height': 3,
        'show_labels': True,
        'show_lengths': True,
        'vertex_labels': ['A', 'B', 'C', 'D']
    }
    trapezoid = Trapezoid(params_trapezoid)
    trapezoid.draw(axs[0, 0])
    axs[0, 0].set_title('Трапеция')
    
    # Тест треугольника
    params_triangle = {
        'points': [(0, 0), (4, 0), (2, 3)],
        'show_labels': True,
        'show_lengths': True,
        'show_angles': True,
        'vertex_labels': ['A', 'B', 'C']
    }
    triangle = Triangle(params_triangle)
    triangle.draw(axs[0, 1])
    axs[0, 1].set_title('Треугольник')
    
    # Тест прямоугольника
    params_rectangle = {
        'width': 5,
        'height': 3,
        'show_labels': True,
        'show_lengths': True,
        'vertex_labels': ['A', 'B', 'C', 'D']
    }
    rectangle = Rectangle(params_rectangle)
    rectangle.draw(axs[0, 2])
    axs[0, 2].set_title('Прямоугольник')
    
    # Тест параллелограмма
    params_parallelogram = {
        'width': 5,
        'height': 3,
        'skew': 60,
        'show_labels': True,
        'show_lengths': True,
        'show_angles': True,
        'vertex_labels': ['A', 'B', 'C', 'D']
    }
    parallelogram = Parallelogram(params_parallelogram)
    parallelogram.draw(axs[1, 0])
    axs[1, 0].set_title('Параллелограмм')
    
    # Тест окружности
    params_circle = {
        'center': (0, 0),
        'radius': 3,
        'center_label': 'O',
        'show_center': True,
        'show_radius': True,
        'show_diameter': True,
        'show_chord': True,
        'chord_value': 4.5
    }
    circle = Circle(params_circle)
    circle.draw(axs[1, 1])
    axs[1, 1].set_title('Окружность')
    
    # Скрываем пустой график
    axs[1, 2].axis('off')
    
    # Выравниваем и настраиваем макет
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Сохраняем комбинированное изображение
    plt.savefig('test_images/all_figures_combined.png', dpi=300)
    print("Комбинированное изображение сохранено: test_images/all_figures_combined.png")
    
    return True

if __name__ == '__main__':
    create_combined_figure() 