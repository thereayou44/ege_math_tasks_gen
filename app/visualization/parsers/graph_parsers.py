import re
import logging
import traceback

def remove_latex_markup(text):
    """
    Удаляет разметку LaTeX из текста.
    
    Args:
        text: Текст с LaTeX разметкой
        
    Returns:
        str: Очищенный текст
    """
    if not text:
        return text
    
    # Удаляем команды LaTeX
    text = re.sub(r'\\[a-zA-Z]+', '', text)
    
    # Удаляем фигурные скобки от команд LaTeX
    text = re.sub(r'\{([^{}]*)\}', r'\1', text)
    
    # Заменяем LaTeX дроби на обычные дроби
    text = re.sub(r'\\frac\{([^{}]*)\}\{([^{}]*)\}', r'(\1)/(\2)', text)
    
    # Заменяем LaTeX степени на ** для Python
    text = re.sub(r'\^', '**', text)
    
    # Удаление команд матмода $...$ и $$...$$
    text = re.sub(r'\$\$(.*?)\$\$', r'\1', text)
    text = re.sub(r'\$(.*?)\$', r'\1', text)
    
    return text.strip()

def parse_graph_params(params_text):
    """
    Универсальная функция для извлечения параметров графиков функций.
    Возвращает список функций, диапазоны осей и особые точки.
    
    Args:
        params_text: Текст с параметрами
        
    Returns:
        tuple: (functions, x_range, y_range, special_points)
            functions - список кортежей (функция, цвет, метка)
            x_range - диапазон оси X (x_min, x_max)
            y_range - диапазон оси Y (y_min, y_max) или None
            special_points - список особых точек [(x1, y1, метка1), ...]
    """
    from app.prompts import REGEX_PATTERNS
    
    # Список функций для отображения
    funcs_to_plot = []
    
    # Диапазоны осей
    x_range = (-10, 10)  # По умолчанию [-10, 10]
    y_range = None  # По умолчанию автоматический
    
    # Особые точки
    special_points = []
    
    # Получаем паттерны для графиков
    graph_patterns = REGEX_PATTERNS.get("graph", {})
    
    # Функция для извлечения параметра по шаблону
    def extract_param(param_name, default=None):
        pattern = graph_patterns.get(param_name)
        if not pattern:
            return default
            
        match = re.search(pattern, params_text, re.IGNORECASE | re.MULTILINE)
        if match:
            try:
                value = match.group(1).strip()
                # Очищаем от LaTeX разметки
                return remove_latex_markup(value)
            except Exception as e:
                logging.error(f"Ошибка при извлечении параметра: {e}")
        return default
    
    try:
        # Конвертер русских названий цветов в английские
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
        
        # Стандартные цвета, если не указаны
        default_colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        # Определяем количество функций
        num_functions_str = extract_param("num_functions")
        num_functions = int(num_functions_str) if num_functions_str and num_functions_str.isdigit() else 0
        
        # Список для хранения функций
        functions = []
        
        # Если количество функций указано и больше 0
        if num_functions > 0:
            for i in range(1, min(num_functions + 1, 5)):  # Максимум 4 функции
                # Извлекаем функцию i
                func_expr = extract_param(f"function_{i}")
                if not func_expr:
                    continue
                
                # Очищаем выражение функции и конвертируем для Python
                func_expr = func_expr.replace('^', '**').replace('\\', '')
                
                # Извлекаем цвет для функции i
                color = extract_param(f"color_{i}")
                if not color or color.lower() not in color_mapping:
                    color = default_colors[min(i-1, len(default_colors)-1)]
                else:
                    color = color_mapping.get(color.lower(), color)
                
                # Извлекаем название/метку для функции i
                name = extract_param(f"name_{i}")
                if not name:
                    name = f"f_{i}(x)"
                
                # Добавляем функцию в список
                functions.append((func_expr, color, name))
                logging.info(f"Обработана функция {i}: {func_expr}, цвет: {color}, метка: {name}")
        
        # Если функции не найдены через параметры, пробуем извлечь из текста общими методами
        if not functions:
            # Поиск функций в формате y=... или f(x)=...
            general_func_patterns = [
                r'y\s*=\s*([-+0-9a-zA-Z\^\*\/\(\)\s\{\}\[\]\.,]+)(?=[,.:;)]|\s*$|\n)',
                r'f\s*\(\s*x\s*\)\s*=\s*([-+0-9a-zA-Z\^\*\/\(\)\s\{\}\[\]\.,]+)(?=[,.:;)]|\s*$|\n)',
                r'([a-zA-Z])\s*\(\s*x\s*\)\s*=\s*([-+0-9a-zA-Z\^\*\/\(\)\s\{\}\[\]\.,]+)(?=[,.:;)]|\s*$|\n|\$)',
                r'\$([a-zA-Z])\s*\(\s*x\s*\)\s*=\s*([-+0-9a-zA-Z\^\*\/\(\)\s\{\}\[\]\.,\\]+)\$'
            ]
            
            for i, pattern in enumerate(general_func_patterns):
                matches = re.findall(pattern, params_text)
                for j, match in enumerate(matches):
                    if i <= 1:  # Формат без имени функции (y= или f(x)=)
                        func_expr = match.strip()
                        if i == 0:  # y = expr
                            func_name = 'y'
                        else:  # f(x) = expr
                            func_name = 'f(x)'
                    else:  # Формат с именем функции
                        func_name, func_expr = match
                        func_name = f"{func_name}(x)"
                    
                    # Очищаем выражение
                    func_expr = func_expr.strip()
                    func_expr = func_expr.replace('^', '**')
                    
                    # Выбираем цвет
                    color_idx = j % len(default_colors)
                    color = default_colors[color_idx]
                    
                    # Добавляем функцию, если она уникальна
                    if not any(f[0] == func_expr for f in functions):
                        functions.append((func_expr, color, func_name))
                        logging.info(f"Найдена функция в общем тексте: {func_expr}, метка: {func_name}")
        
        # Извлекаем диапазоны X и Y
        x_range_str = extract_param("x_range")
        if x_range_str:
            try:
                # Извлекаем числа из строки
                x_range_values = re.findall(r'[-+]?\d*\.?\d+', x_range_str)
                if len(x_range_values) >= 2:
                    x_range = (float(x_range_values[0]), float(x_range_values[1]))
                    logging.info(f"Установлен диапазон X: {x_range}")
            except Exception as e:
                logging.warning(f"Ошибка при разборе диапазона X: {e}")
        
        y_range_str = extract_param("y_range")
        if y_range_str and y_range_str.lower() != 'автоматический':
            try:
                # Извлекаем числа из строки
                y_range_values = re.findall(r'[-+]?\d*\.?\d+', y_range_str)
                if len(y_range_values) >= 2:
                    y_range = (float(y_range_values[0]), float(y_range_values[1]))
                    logging.info(f"Установлен диапазон Y: {y_range}")
            except Exception as e:
                logging.warning(f"Ошибка при разборе диапазона Y: {e}")
        
        # Извлекаем особые точки
        special_points_str = extract_param("special_points")
        if special_points_str:
            try:
                # Извлекаем точки в формате (x, y, метка)
                points_list = re.findall(r'\(([^)]+)\)', special_points_str)
                
                for point_str in points_list:
                    try:
                        # Разделяем по запятой на x, y, label
                        parts = point_str.split(',', 2)
                        
                        if len(parts) >= 2:
                            x_expr = parts[0].strip()
                            y_expr = parts[1].strip()
                            label = parts[2].strip() if len(parts) > 2 else ""
                            
                            # Вычисляем значения координат, поддерживая математические выражения
                            import math
                            
                            # Обрабатываем x_expr
                            x_expr = x_expr.replace('^', '**').replace('sqrt', 'math.sqrt')
                            if any(func in x_expr for func in ['math.', 'sqrt', 'sin', 'cos']):
                                x_val = eval(x_expr)
                            else:
                                x_val = float(x_expr)
                                
                            # Обрабатываем y_expr
                            y_expr = y_expr.replace('^', '**').replace('sqrt', 'math.sqrt')
                            if any(func in y_expr for func in ['math.', 'sqrt', 'sin', 'cos']):
                                y_val = eval(y_expr)
                            else:
                                y_val = float(y_expr)
                                
                            # Добавляем точку в список
                            special_points.append((x_val, y_val, label))
                            logging.info(f"Добавлена особая точка: ({x_val}, {y_val}, '{label}')")
                    except Exception as e:
                        logging.warning(f"Ошибка при обработке особой точки '{point_str}': {e}")
            except Exception as e:
                logging.warning(f"Ошибка при разборе особых точек: {e}")
        
        # Возвращаем результаты
        return functions, x_range, y_range, special_points
        
    except Exception as e:
        logging.error(f"Общая ошибка при разборе параметров графиков: {e}")
        # Возвращаем пустые значения по умолчанию
        return [], (-10, 10), None, []
    
    return funcs_to_plot, x_range, y_range, special_points 