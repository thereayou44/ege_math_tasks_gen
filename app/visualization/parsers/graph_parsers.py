import re
import logging
import traceback
import numpy as np

def remove_latex_markup(text):
    """
    Удаляет символы разметки LaTeX из строки.
    
    Args:
        text (str): Исходная строка с разметкой LaTeX
        
    Returns:
        str: Строка без разметки LaTeX
    """
    if not text:
        return text
        
    # Удаляем символы $, \ и другую разметку
    text = text.replace('$', '')
    text = text.replace('\\', '')
    
    # Удаляем команды LaTeX типа \sin, \cos, etc.
    text = re.sub(r'\\[a-zA-Z]+', '', text)
    
    return text.strip()

def parse_graph_params(params_text):
    """
    Парсит параметры для построения графика функции из текста.
    
    Args:
        params_text (str): Текст с параметрами
        
    Returns:
        tuple: (список функций, диапазон X, диапазон Y, особые точки)
    """
    try:
        logging.info("Парсинг параметров графика")
        
        # Список для хранения функций в формате [(выражение, цвет, метка), ...]
        functions_to_plot = []
        
        # Диапазоны значений по умолчанию
        x_range = (-10, 10)
        y_range = None
        special_points = []
        
        # Извлекаем количество функций
        num_functions_match = re.search(r'Количество функций\s*:\s*(\d+)', params_text, re.IGNORECASE)
        num_functions = 1
        if num_functions_match:
            try:
                num_functions = int(num_functions_match.group(1))
                logging.info(f"Найдено количество функций: {num_functions}")
            except ValueError:
                logging.warning(f"Не удалось преобразовать количество функций в число: {num_functions_match.group(1)}")
        
        # Ограничиваем количество функций от 1 до 4
        num_functions = max(1, min(num_functions, 4))
        
        # Извлекаем выражения функций, цвета и метки
        for i in range(1, num_functions + 1):
            # Ищем выражение функции
            func_pattern = rf'Функция {i}\s*:\s*([^\n]+)'
            func_match = re.search(func_pattern, params_text, re.IGNORECASE)
            if not func_match:
                logging.warning(f"Не найдено выражение для функции {i}")
                continue
                
            function_expr = func_match.group(1).strip()
            function_expr = remove_latex_markup(function_expr)
            
            # Ищем цвет функции
            color_pattern = rf'Цвет {i}\s*:\s*([^\n]+)'
            color_match = re.search(color_pattern, params_text, re.IGNORECASE)
            color = 'blue'  # Цвет по умолчанию
            if color_match:
                color = color_match.group(1).strip()
            
            # Ищем название функции
            name_pattern = rf'Название {i}\s*:\s*([^\n]+)'
            name_match = re.search(name_pattern, params_text, re.IGNORECASE)
            name = f"f_{i}(x)"  # Название по умолчанию
            if name_match:
                name = name_match.group(1).strip()
            
            # Добавляем функцию в список для построения
            functions_to_plot.append((function_expr, color, name))
            logging.info(f"Добавлена функция {i}: {function_expr}, цвет: {color}, название: {name}")
        
        # Извлекаем диапазон значений X
        x_range_pattern = r'Диапазон X\s*:\s*\[(.*?)\]'
        x_range_match = re.search(x_range_pattern, params_text, re.IGNORECASE)
        if x_range_match:
            x_range_str = x_range_match.group(1).strip()
            try:
                # Разбиваем строку на две части и преобразуем в числа
                x_min, x_max = map(float, re.split(r',\s*', x_range_str))
                x_range = (x_min, x_max)
                logging.info(f"Установлен диапазон X: {x_range}")
            except Exception as e:
                logging.warning(f"Не удалось преобразовать диапазон X: {x_range_str}, ошибка: {e}")
        
        # Извлекаем диапазон значений Y
        y_range_pattern = r'Диапазон Y\s*:\s*\[(.*?)\]'
        y_range_match = re.search(y_range_pattern, params_text, re.IGNORECASE)
        if y_range_match:
            y_range_str = y_range_match.group(1).strip()
            if y_range_str.lower() == 'автоматический':
                y_range = None
                logging.info("Установлен автоматический диапазон Y")
            else:
                try:
                    # Разбиваем строку на две части и преобразуем в числа
                    y_min, y_max = map(float, re.split(r',\s*', y_range_str))
                    y_range = (y_min, y_max)
                    logging.info(f"Установлен диапазон Y: {y_range}")
                except Exception as e:
                    logging.warning(f"Не удалось преобразовать диапазон Y: {y_range_str}, ошибка: {e}")
        
        # Извлекаем особые точки
        special_points_pattern = r'Особые точки\s*:\s*\[(.*?)\]'
        special_points_match = re.search(special_points_pattern, params_text, re.IGNORECASE)
        if special_points_match:
            try:
                special_points_str = special_points_match.group(1).strip()
                from app.visualization.graphs.function_graphs import parse_special_points
                special_points = parse_special_points(special_points_str)
                logging.info(f"Найдены особые точки: {special_points}")
            except Exception as e:
                logging.warning(f"Ошибка при обработке особых точек: {e}")
                logging.warning(traceback.format_exc())
        
        return functions_to_plot, x_range, y_range, special_points
        
    except Exception as e:
        logging.error(f"Ошибка при парсинге параметров графика: {e}")
        logging.error(traceback.format_exc())
        # Возвращаем базовые значения в случае ошибки
        return [("x**2", "blue", "f(x)")], (-10, 10), None, []

def parse_coordinate_params(params_text):
    """
    Парсит параметры для построения координатной плоскости.
    
    Args:
        params_text (str): Текст с параметрами
        
    Returns:
        tuple: (точки, функции, векторы)
    """
    try:
        logging.info("Парсинг параметров координатной плоскости")
        
        # Извлекаем точки
        points_pattern = r'Точки\s*:\s*\[(.*?)\]'
        points_match = re.search(points_pattern, params_text, re.IGNORECASE)
        points = []
        if points_match:
            try:
                points_str = points_match.group(1).strip()
                # Разбираем строку с точками
                from app.visualization.graphs.function_graphs import parse_special_points
                points = parse_special_points(points_str)
                logging.info(f"Найдены точки: {points}")
            except Exception as e:
                logging.warning(f"Ошибка при обработке точек: {e}")
        
        # Извлекаем функции (для отображения на координатной плоскости)
        functions_pattern = r'Функции\s*:\s*\[(.*?)\]'
        functions_match = re.search(functions_pattern, params_text, re.IGNORECASE)
        functions = []
        if functions_match:
            try:
                functions_str = functions_match.group(1).strip()
                # Если строка не пустая, разбиваем по запятым
                if functions_str:
                    for func_expr in re.split(r',\s*', functions_str):
                        func_expr = remove_latex_markup(func_expr)
                        functions.append((func_expr, "blue", ""))  # Цвет и метка по умолчанию
                        logging.info(f"Найдена функция: {func_expr}")
            except Exception as e:
                logging.warning(f"Ошибка при обработке функций: {e}")
        
        # Извлекаем векторы
        vectors_pattern = r'Векторы\s*:\s*\[(.*?)\]'
        vectors_match = re.search(vectors_pattern, params_text, re.IGNORECASE)
        vectors = []
        if vectors_match:
            try:
                vectors_str = vectors_match.group(1).strip()
                # Разбираем строку с векторами
                pattern = r'\(([^,]+),([^,]+),([^,]+),([^,]+)(?:,([^)]+))?\)'
                matches = re.findall(pattern, vectors_str)
                
                for match in matches:
                    x1_str, y1_str, x2_str, y2_str = match[0].strip(), match[1].strip(), match[2].strip(), match[3].strip()
                    label = match[4].strip() if len(match) > 4 and match[4] else ""
                    
                    # Преобразуем координаты в числа
                    try:
                        x1, y1, x2, y2 = float(x1_str), float(y1_str), float(x2_str), float(y2_str)
                        vectors.append((x1, y1, x2, y2, label))
                        logging.info(f"Найден вектор: ({x1}, {y1}) -> ({x2}, {y2}), метка: {label}")
                    except ValueError:
                        logging.warning(f"Не удалось преобразовать координаты вектора: ({x1_str}, {y1_str}, {x2_str}, {y2_str})")
            except Exception as e:
                logging.warning(f"Ошибка при обработке векторов: {e}")
        
        return points, functions, vectors
        
    except Exception as e:
        logging.error(f"Ошибка при парсинге параметров координатной плоскости: {e}")
        logging.error(traceback.format_exc())
        return [], [], [] 