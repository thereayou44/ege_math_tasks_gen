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

def test_all_shapes():
    """Генерирует тестовые изображения всех видов фигур с разными параметрами"""
    
    # Создаем директорию для тестовых изображений
    test_dir = 'test_images'
    os.makedirs(test_dir, exist_ok=True)
    
    # 1. Прямоугольный треугольник с заданными длинами
    triangle_right_params = {
        'points': [(0,0), (0,4), (3,0)],
        'show_labels': True,
        'vertex_labels': ['K', 'L', 'M'],
        'show_angles': True,
        'is_right': True,
        'side_lengths': [3, 4, 5]
    }
    img_path = generate_geometric_figure("triangle", triangle_right_params, f"{test_dir}/triangle_right.png")
    print(f"Создан прямоугольный треугольник: {img_path}")
    
    # 2. Обычный треугольник с частичным указанием длин (прочерк)
    triangle_params = {
        'points': [(1,1), (5,1), (3,4)],
        'show_labels': True,
        'vertex_labels': ['A', 'B', 'C'],
        'show_angles': True,
        'side_lengths': [None, 4.5, None]  # Только одна сторона указана
    }
    img_path = generate_geometric_figure("triangle", triangle_params, f"{test_dir}/triangle_regular.png")
    print(f"Создан треугольник с частичными длинами: {img_path}")
    
    # 3. Прямоугольник с указанием всех длин
    rectangle_params = {
        'x': 1,
        'y': 1,
        'width': 5,
        'height': 3,
        'show_labels': True,
        'vertex_labels': ['A', 'B', 'C', 'D'],
        'side_lengths': [5, 3, 5, 3]
    }
    img_path = generate_geometric_figure("rectangle", rectangle_params, f"{test_dir}/rectangle.png")
    print(f"Создан прямоугольник: {img_path}")
    
    # 4. Параллелограмм с заданным наклоном
    parallelogram_params = {
        'x': 1,
        'y': 1,
        'width': 5,
        'height': 3,
        'skew': 60,
        'show_labels': True,
        'side_lengths': [5, 3, 5, 3]
    }
    img_path = generate_geometric_figure("parallelogram", parallelogram_params, f"{test_dir}/parallelogram.png")
    print(f"Создан параллелограмм: {img_path}")
    
    # 5. Трапеция
    trapezoid_params = {
        'x': 1,
        'y': 1,
        'bottom_width': 6,
        'top_width': 3,
        'height': 3,
        'show_labels': True,
        'side_lengths': [6, None, 3, None]  # Указываем только основания
    }
    img_path = generate_geometric_figure("trapezoid", trapezoid_params, f"{test_dir}/trapezoid.png")
    print(f"Создана трапеция: {img_path}")
    
    # 6. Окружность с указанием радиуса
    circle_params = {
        'center': (3, 3),
        'radius': 2,
        'show_center': True,
        'center_label': 'O',
        'radius_value': 2.5,  # Значение радиуса для отображения отличается от фактического
        'show_radius': True
    }
    img_path = generate_geometric_figure("circle", circle_params, f"{test_dir}/circle_radius.png")
    print(f"Создана окружность с радиусом: {img_path}")
    
    # 7. Окружность с указанием диаметра и хорды
    circle_params = {
        'center': (3, 3),
        'radius': 2,
        'show_center': True,
        'center_label': 'O',
        'diameter_value': 5,
        'chord_value': 3,
        'show_diameter': True
    }
    img_path = generate_geometric_figure("circle", circle_params, f"{test_dir}/circle_diameter_chord.png")
    print(f"Создана окружность с диаметром и хордой: {img_path}")
    
    print("\nТестирование завершено. Все изображения сохранены в директории:", test_dir)

if __name__ == "__main__":
    test_all_shapes() 