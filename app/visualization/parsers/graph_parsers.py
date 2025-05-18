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
    # Список функций для отображения
    funcs_to_plot = []
    
    # Диапазоны осей
    x_range = (-10, 10)  # По умолчанию [-10, 10]
    y_range = None  # По умолчанию автоматический
    
    # Особые точки
    special_points = []
    
    # Функция для извлечения параметра по шаблону
    def extract_param(pattern, text, default=None):
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            try:
                value = match.group(1).strip()
                # Очищаем от LaTeX разметки
                return remove_latex_markup(value)
            except Exception as e:
                logging.error(f"Ошибка при извлечении параметра: {e}")
        return default
    
    try:
        # МЕТОД 1: Поиск функций, цветов и меток в параметрах визуализации
        # Для извлечения используем несколько подходов, начиная с наиболее структурированных
        
        # 1. Сначала ищем "Функция 1: x^2" и т.д.
        func_pattern = r'Функция\s+(\d+):\s*(.*?)(?=Функция\s+\d+:|Цвет|Название|Диапазон|Особые|$)'
        func_matches = re.findall(func_pattern, params_text, re.IGNORECASE | re.DOTALL)
        
        # Создаем словари для цветов и названий
        color_dict = {}
        name_dict = {}
        
        # Извлекаем цвета в формате "Цвет 1: красный"
        color_pattern = r'Цвет\s+(\d+):\s*(.*?)(?=Функция|Цвет|Название|Диапазон|Особые|$)'
        color_matches = re.findall(color_pattern, params_text, re.IGNORECASE | re.DOTALL)
        for num_str, color in color_matches:
            try:
                num = int(num_str.strip())
                color_dict[num] = color.strip()
            except ValueError:
                continue
                
        # Извлекаем названия в формате "Название 1: f(x)"
        name_pattern = r'Название\s+(\d+):\s*(.*?)(?=Функция|Цвет|Название|Диапазон|Особые|$)'
        name_matches = re.findall(name_pattern, params_text, re.IGNORECASE | re.DOTALL)
        for num_str, name in name_matches:
            try:
                num = int(num_str.strip())
                name_dict[num] = name.strip()
            except ValueError:
                continue
        
        # Стандартные цвета, если не указаны
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
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
        
        # Обрабатываем найденные функции
        functions = []
        for match in func_matches:
            try:
                num = int(match[0].strip())
                func_expr = match[1].strip()
                
                # Очищаем от LaTeX-разметки и приводим к Python-синтаксису
                func_expr = remove_latex_markup(func_expr)
                
                # Определяем цвет и метку для функции
                color = color_dict.get(num, colors[min(num-1, len(colors)-1)])
                
                # Преобразуем русское название цвета в английское
                if color.lower() in color_mapping:
                    color = color_mapping[color.lower()]
                
                name = name_dict.get(num, f"f_{num}(x)")
                
                # Добавляем функцию в список
                functions.append((func_expr, color, name))
                logging.info(f"Найдена функция {num}: {func_expr}, цвет: {color}, метка: {name}")
            except Exception as e:
                logging.warning(f"Ошибка при обработке функции {match[0]}: {e}")
        
        # 2. Если функции не найдены методом выше, ищем по блокам "График 1: ..."
        if not functions:
            for i in range(1, 6):  # Поиск до 5 графиков
                graph_block_pattern = rf'График\s*{i}:.*?(?=График\s*{i+1}:|Общие параметры:|$)'
                graph_block_match = re.search(graph_block_pattern, params_text, re.DOTALL | re.IGNORECASE)
                
                if graph_block_match:
                    graph_block = graph_block_match.group(0)
                    
                    # Извлекаем функцию из блока
                    func_pattern = rf'Функция:?\s*([^\n]+)'
                    func_match = re.search(func_pattern, graph_block, re.IGNORECASE)
                    
                    if func_match:
                        func_str = func_match.group(1).strip()
                        func_str = remove_latex_markup(func_str)
                        
                        # Ищем цвет
                        color_pattern = r'Цвет:?\s*([^\n]+)'
                        color_match = re.search(color_pattern, graph_block, re.IGNORECASE)
                        color = color_match.group(1).strip() if color_match else colors[min(i-1, len(colors)-1)]
                        
                        # Преобразуем русское название цвета в английское
                        if color.lower() in color_mapping:
                            color = color_mapping[color.lower()]
                        
                        # Ищем метку
                        label_pattern = r'(Название|Метка|Подпись):?\s*([^\n]+)'
                        label_match = re.search(label_pattern, graph_block, re.IGNORECASE)
                        label = label_match.group(2).strip() if label_match else f"f_{i}(x)"
                        
                        # Извлекаем диапазоны X и Y для этой функции, если указаны
                        x_range_pattern = r'Диапазон\s+X:?\s*\[?(.*?)\]?(?=Диапазон\s+Y|Цвет|Название|Метка|Особые|$)'
                        x_range_match = re.search(x_range_pattern, graph_block, re.IGNORECASE)
                        
                        if x_range_match:
                            x_range_str = x_range_match.group(1).strip()
                            try:
                                x_min, x_max = map(float, x_range_str.split(','))
                                x_range = (x_min, x_max)
                            except Exception as e:
                                logging.warning(f"Ошибка при разборе диапазона X для графика {i}: {e}")
                        
                        y_range_pattern = r'Диапазон\s+Y:?\s*\[?(.*?)\]?(?=Диапазон\s+X|Цвет|Название|Метка|Особые|$)'
                        y_range_match = re.search(y_range_pattern, graph_block, re.IGNORECASE)
                        
                        if y_range_match:
                            y_range_str = y_range_match.group(1).strip()
                            if y_range_str.lower() != 'автоматический':
                                try:
                                    y_min, y_max = map(float, y_range_str.split(','))
                                    y_range = (y_min, y_max)
                                except Exception as e:
                                    logging.warning(f"Ошибка при разборе диапазона Y для графика {i}: {e}")
                        
                        # Нормализуем выражение функции для Python
                        func_str = func_str.replace('^', '**')
                        
                        # Добавляем функцию в список
                        functions.append((func_str, color, label))
                        logging.info(f"Найдена функция для графика {i}: {func_str}, цвет: {color}, метка: {label}")
        
        # 3. Поиск функций напрямую в тексте задачи
        if not functions:
            # Ищем функции напрямую в формате "Функция 1: x^2"
            for i in range(1, 6):  # Поиск до 5 функций
                func_pattern = rf'Функция\s*{i}:?\s*([^\n]+)'
                func_match = re.search(func_pattern, params_text, re.IGNORECASE | re.MULTILINE)
                
                if func_match:
                    func_str = func_match.group(1).strip()
                    # Очищаем от LaTeX-разметки полностью
                    func_str = remove_latex_markup(func_str)
                    
                    # Определяем цвет и метку для функции
                    color = color_dict.get(i, colors[min(i-1, len(colors)-1)])
                    
                    # Преобразуем русское название цвета в английское
                    if color.lower() in color_mapping:
                        color = color_mapping[color.lower()]
                    
                    name = name_dict.get(i, f"f_{i}(x)")
                    
                    # Добавляем функцию в список
                    functions.append((func_str, color, name))
                    logging.info(f"Напрямую найдена функция {i}: {func_str}, цвет: {color}, метка: {name}")
        
        # Обрабатываем общие параметры диапазонов осей
        x_range_pattern = r'Диапазон\s+X:?\s*\[?(.*?)\]?(?=Диапазон\s+Y|Функция|Цвет|Название|Особые|$)'
        x_range_match = re.search(x_range_pattern, params_text, re.IGNORECASE | re.MULTILINE)
        
        if x_range_match:
            x_range_str = x_range_match.group(1).strip()
            try:
                x_min, x_max = map(float, x_range_str.split(','))
                x_range = (x_min, x_max)
                logging.info(f"Найден диапазон X: [{x_min}, {x_max}]")
            except Exception as e:
                logging.warning(f"Ошибка при разборе диапазона X: {e}")
        
        y_range_pattern = r'Диапазон\s+Y:?\s*\[?(.*?)\]?(?=Диапазон\s+X|Функция|Цвет|Название|Особые|$)'
        y_range_match = re.search(y_range_pattern, params_text, re.IGNORECASE | re.MULTILINE)
        
        if y_range_match:
            y_range_str = y_range_match.group(1).strip()
            if y_range_str.lower() != 'автоматический':
                try:
                    y_min, y_max = map(float, y_range_str.split(','))
                    y_range = (y_min, y_max)
                    logging.info(f"Найден диапазон Y: [{y_min}, {y_max}]")
                except Exception as e:
                    logging.warning(f"Ошибка при разборе диапазона Y: {e}")
        
        # Ищем особые точки
        special_points_pattern = r'Особые точки:\s*(.*?)(?=Функция|Диапазон|$)'
        special_points_match = re.search(special_points_pattern, params_text, re.DOTALL | re.IGNORECASE)
        
        if special_points_match:
            try:
                special_points_str = special_points_match.group(1).strip()
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
                            if 'f(' in y_expr:
                                # Если y задан через значение функции
                                if functions:
                                    func_expr = functions[0][0]
                                    y_val = eval(func_expr.replace('x', f'({x_val})'))
                                else:
                                    logging.warning(f"Не удалось вычислить значение функции для точки ({x_expr}, {y_expr})")
                                    continue
                            else:
                                # Если y задан явно
                                y_expr = y_expr.replace('^', '**').replace('sqrt', 'math.sqrt')
                                if any(func in y_expr for func in ['math.', 'sqrt', 'sin', 'cos']):
                                    y_val = eval(y_expr)
                                else:
                                    y_val = float(y_expr)
                            
                            # Добавляем точку с вычисленными координатами
                            special_points.append((x_val, y_val, label))
                            logging.info(f"Добавлена особая точка: ({x_val}, {y_val}, {label})")
                    except Exception as e:
                        logging.warning(f"Ошибка при обработке точки '{point_str}': {e}")
            except Exception as e:
                logging.warning(f"Ошибка при обработке списка особых точек: {e}")
        
        # Если функции всё еще не найдены, попробуем найти в общем тексте задачи
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
                    color_idx = j % len(colors)
                    color = colors[color_idx]
                    
                    # Добавляем функцию, если она уникальна
                    if not any(f[0] == func_expr for f in functions):
                        functions.append((func_expr, color, func_name))
                        logging.info(f"Найдена функция в общем тексте: {func_expr}, метка: {func_name}")
        
        # Если функции всё ещё не найдены, используем стандартную функцию
        if not functions:
            functions = [('x**2', 'blue', 'f(x)')]
            logging.warning("Не удалось найти ни одной функции, использую стандартный график параболы")
        
        # Возвращаем найденные параметры
        funcs_to_plot = functions
        
    except Exception as e:
        logging.error(f"Ошибка при разборе параметров графика: {e}")
        logging.error(traceback.format_exc())
        # В случае ошибки используем стандартные значения
        funcs_to_plot = [('x**2', 'blue', 'f(x)')]
    
    return funcs_to_plot, x_range, y_range, special_points 