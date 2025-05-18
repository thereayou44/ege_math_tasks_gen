import numpy as np
import matplotlib.pyplot as plt
import os
import uuid
import logging
import traceback
import re
import math

def generate_graph_image(function_expr, x_range=(-10, 10), y_range=None, filename=None):
    """
    Генерирует изображение графика функции.
    
    Args:
        function_expr: Строка с выражением функции (например, "x**2 - 3*x + 2")
        x_range: Диапазон значений x (min, max)
        y_range: Диапазон значений y (min, max)
        filename: Имя файла для сохранения
        
    Returns:
        str: Путь к сохраненному изображению
    """
    try:
        # Создаем новую фигуру
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Настраиваем оси
        ax.spines['left'].set_position('zero')
        ax.spines['bottom'].set_position('zero')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
        # Создаем массив значений x
        x = np.linspace(x_range[0], x_range[1], 1000)
        
        # Безопасное вычисление функции
        expr = function_expr.replace('^', '**')
        for func_name in ['sin', 'cos', 'tan', 'exp', 'log', 'sqrt']:
            expr = expr.replace(func_name, f'np.{func_name}')
        
        try:
            # Вычисляем значения функции
            y = eval(expr)
            
            # Строим график
            ax.plot(x, y, 'b-', linewidth=2)
            
            # Устанавливаем пределы осей
            ax.set_xlim(x_range)
            if y_range:
                ax.set_ylim(y_range)
            
            # Добавляем сетку
            ax.grid(True, linestyle='--', alpha=0.6)
            
            # Убираем лишние подписи функции
            ax.set_title("")
            
            # Только подписи осей
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            
            # Создаем директорию для изображений, если она не существует
            images_dir = 'static/images/generated'
            os.makedirs(images_dir, exist_ok=True)
            
            # Генерируем имя файла, если не указано
            if not filename:
                filename = f"graph_{uuid.uuid4().hex[:8]}.png"
                
            filepath = os.path.join(images_dir, filename)
            
            # Сохраняем изображение
            plt.savefig(filepath, dpi=100, bbox_inches='tight')
            plt.close()  # Закрываем фигуру, чтобы очистить память
            
            return filepath
        except Exception as e:
            logging.error(f"Ошибка при вычислении функции '{function_expr}': {e}")
            return None
            
    except Exception as e:
        logging.error(f"Ошибка при генерации графика: {e}")
        return None

def generate_multi_function_graph(functions, x_range=(-10, 10), y_range=None, special_points=None, filename=None):
    """
    Генерирует изображение графиков нескольких функций.
    
    Args:
        functions: Список функций в формате [(выражение, цвет, метка), ...]
        x_range: Диапазон значений x (min, max)
        y_range: Диапазон значений y (min, max) или None для автоматического
        special_points: Список особых точек [(x, y, метка), ...]
        filename: Имя файла для сохранения
        
    Returns:
        str: Путь к сохраненному изображению
    """
    try:
        # Создаем новую фигуру
        fig, ax = plt.subplots(figsize=(10, 7))
        
        # Настраиваем оси
        ax.spines['left'].set_position('zero')
        ax.spines['bottom'].set_position('zero')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
        # Словарь для перевода названий цветов с русского на английский
        color_mapping = {
            'красный': 'red',
            'синий': 'blue',
            'зеленый': 'green',
            'зелёный': 'green',
            'желтый': 'yellow',
            'жёлтый': 'yellow',
            'черный': 'black',
            'чёрный': 'black',
            'фиолетовый': 'purple',
            'оранжевый': 'orange',
            'коричневый': 'brown',
            'розовый': 'pink',
            'серый': 'gray',
            'голубой': 'cyan',
            'малиновый': 'magenta'
        }
        
        # Вычисляем минимальные и максимальные значения для y, если они не указаны
        y_values = []
        
        # Создаем массив значений x
        x = np.linspace(x_range[0], x_range[1], 1000)
        
        # Определяем функцию для безопасного вычисления выражений
        def safe_eval(func_expr, x_val):
            try:
                # Заменяем переменную x на конкретное значение
                expr = func_expr.replace('x', f'({x_val})')
                
                # Заменяем возведение в степень
                expr = expr.replace('^', '**')
                
                # Добавляем поддержку математических функций
                for func_name in ['sin', 'cos', 'tan', 'exp', 'log', 'sqrt']:
                    expr = expr.replace(func_name, f'math.{func_name}')
                
                # Вычисляем выражение
                return eval(expr)
            except Exception as e:
                logging.error(f"Ошибка при вычислении '{func_expr}' для x={x_val}: {e}")
                return None
        
        # Функция для построения графика
        def create_func(expr):
            def func(x_val):
                return safe_eval(expr, x_val)
            return func
        
        # Строим графики всех функций
        for func_data in functions:
            func_expr, color, label = func_data
            
            # Проверяем цвет, и если он на русском, переводим на английский
            if isinstance(color, str) and color.lower() in color_mapping:
                color = color_mapping[color.lower()]
            
            # Преобразуем математические выражения
            expr = func_expr.replace('^', '**')
            for func_name in ['sin', 'cos', 'tan', 'exp', 'log', 'sqrt']:
                expr = expr.replace(func_name, f'np.{func_name}')
            
            try:
                # Вычисляем значения функции
                y = eval(expr)
                
                # Добавляем значения y для вычисления диапазона
                y_values.extend(y)
                
                # Строим график
                ax.plot(x, y, color=color, linewidth=2, label=label)
                
            except Exception as e:
                logging.error(f"Ошибка при построении функции '{func_expr}': {e}")
        
        # Автоматически определяем диапазон y, если он не указан
        if y_range is None and y_values:
            # Убираем NaN и Inf значения
            valid_y = [y for y in y_values if not (np.isnan(y) or np.isinf(y))]
            
            if valid_y:
                y_min, y_max = min(valid_y), max(valid_y)
                y_margin = (y_max - y_min) * 0.1  # Запас 10%
                y_range = (y_min - y_margin, y_max + y_margin)
        
        # Устанавливаем пределы осей
        ax.set_xlim(x_range)
        if y_range:
            ax.set_ylim(y_range)
        
        # Добавляем сетку
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # Настраиваем положение осей
        ax.spines['left'].set_position(('data', 0))
        ax.spines['bottom'].set_position(('data', 0))
        
        # Добавляем подписи к осям
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12, rotation=0, labelpad=10)
        
        # Добавляем стрелки на концах осей
        ax.plot((1), (0), ls="", marker=">", ms=5, color="k",
                transform=ax.get_yaxis_transform(), clip_on=False)
        ax.plot((0), (1), ls="", marker="^", ms=5, color="k",
                transform=ax.get_xaxis_transform(), clip_on=False)
        
        # Отображаем особые точки
        if special_points:
            for point in special_points:
                x_val, y_val = point[0], point[1]
                label = point[2] if len(point) > 2 else ""
                
                # Проверяем, что точка находится в пределах осей
                if (x_range[0] <= x_val <= x_range[1]) and \
                   (y_range is None or (y_range[0] <= y_val <= y_range[1])):
                    # Отображаем точку
                    ax.plot(x_val, y_val, 'o', markersize=5, color='red')
                    
                    # Добавляем подпись к точке
                    if label:
                        ax.annotate(
                            label, 
                            (x_val, y_val),
                            xytext=(5, 5),  # Отступ текста от точки
                            textcoords='offset points',
                            fontsize=10,
                            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')
                        )
        
        # Добавляем легенду, если есть метки функций
        if any(label for _, _, label in functions):
            ax.legend(loc='best')
        
        # Создаем директорию для изображений, если она не существует
        images_dir = 'static/images/generated'
        os.makedirs(images_dir, exist_ok=True)
        
        # Генерируем имя файла, если не указано
        if not filename:
            filename = f"multi_graph_{uuid.uuid4().hex[:8]}.png"
            
        filepath = os.path.join(images_dir, filename)
        
        # Сохраняем изображение
        plt.savefig(filepath, dpi=100, bbox_inches='tight')
        plt.close()  # Закрываем фигуру, чтобы очистить память
        
        return filepath
        
    except Exception as e:
        logging.error(f"Ошибка при генерации графиков функций: {e}")
        logging.error(traceback.format_exc())
        return None 