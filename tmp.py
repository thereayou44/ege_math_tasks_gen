import os
import matplotlib.pyplot as plt
from matplotlib.image import imread
from app.task_generator import generate_geometric_figure  # ваш модуль

def main():
    # Список фигур и их параметров
    examples = [
        ('triangle',      {'points': [(0,0),(1,0),(0.3,0.8)], 'show_labels': True}),
        ('rectangle',     {'x': 0, 'y': 0, 'width': 4, 'height': 2, 'show_dimensions': False}),
        ('parallelogram', {'width': 4, 'height': 2, 'skew': 45, 'show_dimensions': False}),
        ('trapezoid',     {'bottom_width': 5, 'top_width': 2, 'height': 3, 'show_dimensions': False}),
        ('circle',        {'center': (0,0), 'radius': 2, 'show_radius': False, 'show_diameter': False}),
    ]

    # Генерируем все фигуры и сохраняем пути
    out_dir = 'static/images/generated'
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for fig_type, params in examples:
        filename = f"test_{fig_type}.png"
        path = generate_geometric_figure(fig_type, params, filename)
        paths.append((fig_type, path))

    # Рисуем сетку 2×3 (последняя ячейка пустая)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    for ax, (fig_type, path) in zip(axes, paths):
        img = imread(path)
        ax.imshow(img)
        ax.set_title(fig_type.capitalize(), fontsize=14)
        ax.axis('off')
    # Убрать лишнюю ось
    axes[-1].axis('off')
    plt.tight_layout()
    collage_path = os.path.join(out_dir, 'all_shapes.png')
    plt.savefig(collage_path)
    print(f"Коллаж сохранён в {collage_path}")

if __name__ == "__main__":
    main()
