import os
import sys
import matplotlib.pyplot as plt
from task_generator import (
    generate_geometric_figure,
    process_parallelogram_visualization,
    process_trapezoid_visualization,
    process_rectangle_visualization
)

def test_shapes():
    # Создаем параметры для каждой фигуры
    rectangle_params = {
        'x': 0, 
        'y': 0, 
        'width': 4, 
        'height': 2, 
        'show_dimensions': False,
        'show_labels': True
    }
    
    parallelogram_params = {
        'width': 4, 
        'height': 2, 
        'skew': 45, 
        'show_dimensions': False,
        'show_labels': True,
        'x': 0,
        'y': 0
    }
    
    trapezoid_params = {
        'bottom_width': 5, 
        'top_width': 2, 
        'height': 3, 
        'show_dimensions': False,
        'show_labels': True,
        'x': 0,
        'y': 0
    }
    
    # Генерируем изображения
    paths = []
    paths.append(('Rectangle', generate_geometric_figure('rectangle', rectangle_params, 'direct_rectangle.png')))
    paths.append(('Parallelogram', generate_geometric_figure('parallelogram', parallelogram_params, 'direct_parallelogram.png')))
    paths.append(('Trapezoid', generate_geometric_figure('trapezoid', trapezoid_params, 'direct_trapezoid.png')))
    
    # Тестируем через функции обработки (с extract_param)
    def extract_param(pattern, text, default=None):
        return default
    
    # Создаем текст параметров для каждой фигуры
    rectangle_text = """Размеры: 4,2
    Координаты: (0,0)
    Подписи вершин: A,B,C,D
    Показать размеры: нет
    Показать метки: да"""
    
    parallelogram_text = """Размеры: 4,2
    Координаты: (0,0)
    Наклон: 45
    Подписи вершин: A,B,C,D
    Показать размеры: нет
    Показать метки: да"""
    
    trapezoid_text = """Размеры: 5,3
    Верхняя база: 2
    Координаты: (0,0)
    Подписи вершин: A,B,C,D
    Показать размеры: нет
    Показать метки: да"""
    
    # Генерируем изображения через функции обработки
    paths.append(('Rectangle (processed)', process_rectangle_visualization(rectangle_text, extract_param)))
    paths.append(('Parallelogram (processed)', process_parallelogram_visualization(parallelogram_text, extract_param)))
    paths.append(('Trapezoid (processed)', process_trapezoid_visualization(trapezoid_text, extract_param)))
    
    # Отображаем результаты
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    axs = axs.flatten()
    
    for i, (title, path) in enumerate(paths):
        if path:
            img = plt.imread(path)
            axs[i].imshow(img)
            axs[i].set_title(title)
            axs[i].axis('off')
        else:
            axs[i].text(0.5, 0.5, "Failed to generate", ha='center', va='center')
            axs[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('static/images/generated/test_shapes_comparison.png')
    print("Test completed. Results saved to static/images/generated/test_shapes_comparison.png")

if __name__ == "__main__":
    test_shapes() 