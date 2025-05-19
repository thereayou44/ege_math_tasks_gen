import numpy as np
import matplotlib.pyplot as plt
import os
import uuid
import logging
import traceback
import re
import math
from matplotlib.ticker import MaxNLocator, MultipleLocator

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
        plt.rcParams.update({'font.size': 12, 'font.family': 'DejaVu Sans'})
        fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
        
        # Настраиваем оси
        ax.spines['left'].set_position('zero')
        ax.spines['bottom'].set_position('zero')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
        # Добавляем стрелки на концах осей
        ax.plot((1), (0), ls="", marker=">", ms=10, color="k",
                transform=ax.get_yaxis_transform(), clip_on=False)
        ax.plot((0), (1), ls="", marker="^", ms=10, color="k",
                transform=ax.get_xaxis_transform(), clip_on=False)
        
        # Подписи осей
        ax.set_xlabel('x', fontsize=14, labelpad=-24, x=1.03)
        ax.set_ylabel('y', fontsize=14, labelpad=-21, y=1.02, rotation=0)
        
        # Создаем массив значений x
        x = np.linspace(x_range[0], x_range[1], 2000)  # Увеличиваем количество точек для более гладкого графика
        
        # Безопасное вычисление функции
        expr = function_expr.replace('^', '**')
        for func_name in ['sin', 'cos', 'tan', 'exp', 'log', 'sqrt', 'abs']:
            expr = expr.replace(func_name, f'np.{func_name}')
        
        try:
            # Вычисляем значения функции
            with np.errstate(divide='ignore', invalid='ignore'):
                y = eval(expr)
            
            # Заменяем бесконечности на NaN для корректного отображения графика
            y = np.where(np.isfinite(y), y, np.nan)
            
            # Строим график с более толстой линией
            ax.plot(x, y, 'b-', linewidth=2.5)
            
            # Определение автоматических границ по y
            if y_range is None:
                mask = np.isfinite(y)  # Фильтруем только конечные значения
                if np.any(mask):
                    filtered_y = y[mask]
                    if len(filtered_y) > 0:
                        y_min, y_max = np.nanmin(filtered_y), np.nanmax(filtered_y)
                        y_padding = (y_max - y_min) * 0.1  # Отступ 10%
                        y_range = (y_min - y_padding, y_max + y_padding)
                else:
                    # Если нет валидных значений, используем диапазон по умолчанию
                    y_range = (-10, 10)
                    logging.info("Установлен диапазон Y по умолчанию для generate_graph_image: [-10, 10]")
            
            # Устанавливаем пределы осей
            ax.set_xlim(x_range)
            if y_range:
                ax.set_ylim(y_range)
            
            # Настраиваем сетку
            ax.grid(True, linestyle='--', alpha=0.6, color='gray', linewidth=0.5)
            
            # Настраиваем деления на осях
            x_range_size = x_range[1] - x_range[0]
            if x_range_size <= 20:
                ax.xaxis.set_major_locator(MultipleLocator(1))  # Основные деления через 1
            elif x_range_size <= 50:
                ax.xaxis.set_major_locator(MultipleLocator(5))  # Основные деления через 5
            else:
                ax.xaxis.set_major_locator(MultipleLocator(10))  # Основные деления через 10
            
            if y_range:
                y_range_size = y_range[1] - y_range[0]
                if y_range_size <= 20:
                    ax.yaxis.set_major_locator(MultipleLocator(1))
                elif y_range_size <= 50:
                    ax.yaxis.set_major_locator(MultipleLocator(5))
                else:
                    ax.yaxis.set_major_locator(MultipleLocator(10))
            
            # Добавляем подписи осей
            ax.set_title(f"График функции: {function_expr}", fontsize=14, pad=10)
            
            # Создаем директорию для изображений, если она не существует
            images_dir = 'static/images/generated'
            os.makedirs(images_dir, exist_ok=True)
            
            # Генерируем имя файла, если не указано
            if not filename:
                filename = f"graph_{uuid.uuid4().hex[:8]}.png"
                
            filepath = os.path.join(images_dir, filename)
            
            # Сохраняем изображение с высоким качеством
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close(fig)  # Закрываем фигуру, чтобы очистить память
            
            return filepath
        except Exception as e:
            logging.error(f"Ошибка при вычислении функции '{function_expr}': {e}")
            logging.error(traceback.format_exc())
            return None
            
    except Exception as e:
        logging.error(f"Ошибка при генерации графика: {e}")
        logging.error(traceback.format_exc())
        return None

def parse_special_points(special_points_str):
    """
    Парсит строку с описанием особых точек.
    
    Args:
        special_points_str: Строка с описанием точек в формате "(x1,y1,метка1), (x2,y2,метка2), ..."
        
    Returns:
        list: Список особых точек [(x1, y1, метка1), ...]
    """
    try:
        if not special_points_str:
            return []
            
        # Убираем LaTeX разметку и другие ненужные символы
        clean_str = special_points_str.replace('$', '')
        
        # Извлекаем координаты точек с помощью регулярного выражения
        pattern = r'\(([^,]+),([^,]+)(?:,([^)]+))?\)'
        matches = re.findall(pattern, clean_str)
        
        special_points = []
        for match in matches:
            x_str, y_str = match[0].strip(), match[1].strip()
            label = match[2].strip() if len(match) > 2 and match[2] else ""
            
            # Удаляем кавычки, скобки и другие лишние символы из меток
            label = label.strip('"\'')
            
            # Преобразуем координаты в числа
            try:
                x = float(x_str)
                y = float(y_str)
                special_points.append((x, y, label))
                logging.info(f"Обработана точка: ({x}, {y}, '{label}')")
            except ValueError:
                logging.warning(f"Не удалось преобразовать координаты точки: ({x_str}, {y_str})")
        
        return special_points
    except Exception as e:
        logging.error(f"Ошибка при парсинге особых точек: {e}")
        return []

def generate_multi_function_graph(functions, x_range=(-10, 10), y_range=None, special_points=None, filename=None, ax=None):
    """
    Генерирует изображение графиков нескольких функций.
    
    Args:
        functions: Список функций в формате [(выражение, цвет, метка), ...]
        x_range: Диапазон значений x (min, max)
        y_range: Диапазон значений y (min, max) или None для автоматического
        special_points: Список особых точек [(x, y, метка), ...]
        filename: Имя файла для сохранения
        ax: Существующие оси для рисования (если None, создаются новые)
        
    Returns:
        matplotlib.axes.Axes: Оси с отрисованным графиком или str: путь к сохраненному изображению
    """
    try:
        # Создаем новую фигуру с высоким качеством, если не переданы оси
        if ax is None:
            plt.rcParams.update({'font.size': 12, 'font.family': 'DejaVu Sans'})
            fig, ax = plt.subplots(figsize=(12, 9), dpi=100)
        else:
            fig = ax.figure
        
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
        
        # Цвета для функций по умолчанию, если цвет не указан
        default_colors = ['blue', 'red', 'green', 'purple', 'orange', 'cyan', 'magenta']
        
        # Создаем массив значений x с большим количеством точек
        x = np.linspace(x_range[0], x_range[1], 2000)
        
        # Вычисляем минимальные и максимальные значения для y
        valid_y_values = []
        
        # Строим графики всех функций
        for i, func_data in enumerate(functions):
            func_expr, color, label = func_data
            
            # Если цвет не указан, используем цвет по умолчанию
            if not color:
                color = default_colors[i % len(default_colors)]
            
            # Проверяем цвет, и если он на русском, переводим на английский
            if isinstance(color, str) and color.lower() in color_mapping:
                color = color_mapping[color.lower()]
            
            # Подготавливаем выражение функции
            try:
                # Безопасно вычисляем значения функции с обработкой ошибок
                with np.errstate(divide='ignore', invalid='ignore'):
                    # Подготовка безопасного контекста выполнения
                    safe_dict = {
                        "np": np, 
                        "x": x, 
                        "sin": np.sin, 
                        "cos": np.cos, 
                        "tan": np.tan, 
                        "sqrt": np.sqrt, 
                        "log": np.log, 
                        "exp": np.exp, 
                        "abs": np.abs,
                        "__builtins__": {}
                    }
                    
                    # Логируем выражение перед вычислением
                    logging.info(f"Вычисление функции: {func_expr}")
                    
                    # Проверяем наличие неявного умножения и фиксируем при необходимости
                    if re.search(r'(\d+)([a-zA-Z])', func_expr):
                        func_expr = re.sub(r'(\d+)([a-zA-Z])', r'\1*\2', func_expr)
                        logging.info(f"Исправлено неявное умножение: {func_expr}")
                    
                    # Вычисляем функцию
                    y = eval(func_expr, {"__builtins__": {}}, safe_dict)
                
                # Заменяем бесконечности на NaN для корректного отображения
                y = np.where(np.isfinite(y), y, np.nan)
                
                # Собираем валидные значения y для определения границ
                mask = np.isfinite(y)
                if np.any(mask):
                    valid_y_values.extend(y[mask])
                
                # Строим график с толстой линией
                line, = ax.plot(x, y, color=color, linewidth=2.5, label=label)
                logging.info(f"График функции {func_expr} успешно построен")
                
                # Для улучшения отображения разрывных функций
                if label and "разрыв" in label.lower():
                    line.set_linestyle('--')
                
            except Exception as e:
                logging.error(f"Ошибка при построении функции '{func_expr}': {e}")
                logging.error(traceback.format_exc())
        
        # Автоматически определяем диапазон y, если он не указан
        if y_range is None:
            if special_points:
                # Определяем диапазон по особым точкам + 2 единицы отступа
                y_values = [point[1] for point in special_points]
                if y_values:
                    y_min, y_max = min(y_values), max(y_values)
                    # Добавляем +/- 2 единицы к диапазону
                    y_min -= 2
                    y_max += 2
                    y_range = (y_min, y_max)
                    logging.info(f"Установлен диапазон Y по особым точкам: {y_range}")
            elif valid_y_values:
                # Если нет особых точек, но есть значения функции
                filtered_y = [y for y in valid_y_values if np.isfinite(y)]
                
                if filtered_y:
                    y_min, y_max = min(filtered_y), max(filtered_y)
                    y_padding = (y_max - y_min) * 0.1  # Отступ 10%
                    
                    # Если значения слишком близкие, добавляем немного "пространства"
                    if abs(y_max - y_min) < 1e-6:
                        y_min -= 1
                        y_max += 1
                    else:
                        y_min -= y_padding
                        y_max += y_padding
                    
                    y_range = (y_min, y_max)
            else:
                # Если нет ни особых точек, ни значений функции, используем диапазон по умолчанию
                y_range = (-10, 10)
                logging.info("Установлен диапазон Y по умолчанию: [-10, 10]")
        
        # Устанавливаем пределы осей
        ax.set_xlim(x_range)
        if y_range:
            ax.set_ylim(y_range)
        
        # Настраиваем сетку
        ax.grid(True, linestyle='--', alpha=0.6, color='gray', linewidth=0.5)
        
        # Настраиваем деления на осях
        x_range_size = x_range[1] - x_range[0]
        if x_range_size <= 20:
            ax.xaxis.set_major_locator(MultipleLocator(1))  # Основные деления через 1
        elif x_range_size <= 50:
            ax.xaxis.set_major_locator(MultipleLocator(5))  # Основные деления через 5
        else:
            ax.xaxis.set_major_locator(MultipleLocator(10))  # Основные деления через 10
        
        if y_range:
            y_range_size = y_range[1] - y_range[0]
            if y_range_size <= 20:
                ax.yaxis.set_major_locator(MultipleLocator(1))
            elif y_range_size <= 50:
                ax.yaxis.set_major_locator(MultipleLocator(5))
            else:
                ax.yaxis.set_major_locator(MultipleLocator(10))
        
        # Добавляем стрелки на концах осей
        ax.plot((1), (0), ls="", marker=">", ms=10, color="k",
                transform=ax.get_yaxis_transform(), clip_on=False)
        ax.plot((0), (1), ls="", marker="^", ms=10, color="k",
                transform=ax.get_xaxis_transform(), clip_on=False)
        
        # Подписи осей
        ax.set_xlabel('x', fontsize=14, labelpad=-24, x=1.03)
        ax.set_ylabel('y', fontsize=14, labelpad=-21, y=1.02, rotation=0)
        
        # Отображаем особые точки
        if special_points:
            # Определим, все ли точки лежат на оси X
            all_points_on_x_axis = all(abs(point[1]) < 1e-10 for point in special_points)
            
            for point in special_points:
                x_val, y_val = point[0], point[1]
                label = point[2] if len(point) > 2 else ""
                
                # Если все точки на оси X или эта конкретная точка близка к оси X, 
                # принудительно устанавливаем Y равным 0
                if all_points_on_x_axis or abs(y_val) < 1e-10:
                    y_val = 0
                
                # Проверяем, что точка находится в пределах осей
                x_in_range = x_range[0] <= x_val <= x_range[1]
                y_in_range = True if y_range is None else (y_range[0] <= y_val <= y_range[1])
                
                if x_in_range and y_in_range:
                    # Рисуем точку с контрастным ободком для лучшей видимости
                    if label.lower() in ['a', 'b', 'c', 'd', 'e'] and abs(y_val) < 1e-10:
                        # Рисуем точку на оси X
                        ax.plot(x_val, 0, 'o', markersize=8, markerfacecolor='blue', 
                                markeredgecolor='white', markeredgewidth=1.5, zorder=10)
                        
                        # Метка под осью X
                        ax.text(x_val, -0.04 * (y_range[1] - y_range[0]) if y_range else -0.5, 
                                label, fontsize=14, ha='center', va='top', color='blue', 
                                bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', boxstyle='round,pad=0.2'),
                                zorder=11)
                    else:
                        # Отображаем точку красного цвета
                        ax.plot(x_val, y_val, 'o', markersize=8, markerfacecolor='red', 
                                markeredgecolor='white', markeredgewidth=1.5, zorder=10)
                        
                        # Если есть метка, добавляем её
                        if label:
                            # Добавляем подпись с фоном для лучшей видимости
                            ax.text(x_val + 0.2, y_val, label, fontsize=12, ha='center', va='center',
                                   bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', 
                                            boxstyle='round,pad=0.3'), zorder=11)
        
        # Добавляем легенду, если есть метки функций
        if any(label for _, _, label in functions):
            legend = ax.legend(
                loc='best', 
                frameon=True, 
                fancybox=True, 
                framealpha=0.9, 
                shadow=True, 
                fontsize=12
            )
        
        # Если указан filename, сохраняем изображение и возвращаем путь
        if filename:
            # Создаем директорию для изображений, если она не существует
            images_dir = 'static/images/generated'
            os.makedirs(images_dir, exist_ok=True)
            
            filepath = os.path.join(images_dir, filename)
            
            # Сохраняем изображение с высоким качеством
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            
            return filepath
        else:
            # Иначе возвращаем оси для дальнейшей настройки
            return ax
        
    except Exception as e:
        logging.error(f"Ошибка при создании графика функций: {e}")
        logging.error(traceback.format_exc())
        return None 