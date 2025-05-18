import numpy as np
import matplotlib.pyplot as plt
import os
import uuid
import logging

def generate_coordinate_system(points=None, functions=None, vectors=None, grid=True, filename=None):
    """
    Генерирует координатную плоскость с точками, векторами или графиками функций.
    
    Args:
        points: Список точек в формате [(x1,y1,label1), (x2,y2,label2), ...]
        functions: Список функций в формате [('x**2', 'blue'), ('2*x+1', 'red'), ...]
        vectors: Список векторов в формате [(x1,y1,x2,y2,label), ...]
        grid: Отображать ли сетку
        filename: Имя файла для сохранения
        
    Returns:
        str: Путь к сохраненному изображению
    """
    try:
        # Создаем новую фигуру
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Настраиваем оси
        ax.spines['left'].set_position('zero')
        ax.spines['bottom'].set_position('zero')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
        # Определяем пределы осей
        x_min, x_max = -10, 10
        y_min, y_max = -10, 10
        
        # Если заданы точки, корректируем пределы
        if points:
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            if x_coords:
                x_min = min(x_min, min(x_coords) - 1)
                x_max = max(x_max, max(x_coords) + 1)
            if y_coords:
                y_min = min(y_min, min(y_coords) - 1)
                y_max = max(y_max, max(y_coords) + 1)
        
        # Устанавливаем пределы осей
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        
        # Добавляем сетку, если нужно
        if grid:
            ax.grid(True, alpha=0.3)
        
        # Рисуем стрелки на осях
        arrow_props = dict(arrowstyle='->', linewidth=1.5)
        ax.annotate('', xy=(1, 0), xytext=(0, 0), arrowprops=arrow_props)
        ax.annotate('', xy=(0, 1), xytext=(0, 0), arrowprops=arrow_props)
        
        # Подписываем оси
        ax.text(x_max - 0.5, 0.5, 'x', fontsize=12)
        ax.text(0.5, y_max - 0.5, 'y', fontsize=12)
        
        # Отображаем точки
        if points:
            for point in points:
                x, y = point[0], point[1]
                # Подписываем точку только если есть метка
                if len(point) > 2 and point[2]:
                    label = point[2]
                    ax.plot(x, y, 'o', markersize=6)
                    ax.text(x + 0.2, y + 0.2, label, fontsize=10)
                else:
                    # Если нет метки, просто рисуем точку без подписи
                    ax.plot(x, y, 'o', markersize=6)
        
        # Отображаем функции без подписей на графике
        if functions:
            x = np.linspace(x_min, x_max, 1000)
            for func_expr, color in functions:
                # Безопасное вычисление функции
                expr = func_expr.replace('^', '**')
                for func_name in ['sin', 'cos', 'tan', 'exp', 'log', 'sqrt']:
                    expr = expr.replace(func_name, f'np.{func_name}')
                
                try:
                    y = eval(expr)
                    # Убираем метку функции, рисуем только график
                    ax.plot(x, y, color=color, linewidth=2)
                except Exception as e:
                    logging.error(f"Ошибка при вычислении функции '{func_expr}': {e}")
        
        # Отображаем векторы
        if vectors:
            for vector in vectors:
                x1, y1, x2, y2 = vector[:4]
                # Рисуем вектор
                ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                            arrowprops=dict(arrowstyle='->', linewidth=1.5, color='blue'))
                
                # Подписываем вектор только если есть метка
                if len(vector) > 4 and vector[4]:
                    label = vector[4]
                    mid_x = (x1 + x2) / 2
                    mid_y = (y1 + y2) / 2
                    ax.text(mid_x, mid_y, label, fontsize=10)
        
        # Создаем директорию для изображений, если она не существует
        images_dir = 'static/images/generated'
        os.makedirs(images_dir, exist_ok=True)
        
        # Генерируем имя файла, если не указано
        if not filename:
            filename = f"coord_{uuid.uuid4().hex[:8]}.png"
            
        filepath = os.path.join(images_dir, filename)
        
        # Сохраняем изображение
        plt.savefig(filepath, dpi=100, bbox_inches='tight')
        plt.close()  # Закрываем фигуру, чтобы очистить память
        
        return filepath
    except Exception as e:
        logging.error(f"Ошибка при генерации координатной плоскости: {e}")
        return None 