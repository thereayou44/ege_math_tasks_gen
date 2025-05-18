import re
import logging

def process_coordinate_params(params_text):
    """Обрабатывает параметры для координатной плоскости"""
    # Регулярные выражения для поиска параметров
    REGEX_PATTERNS = {
        "coordinate": {
            "points": r'Точки:?\s*(.*?)(?=Функции|Векторы|$)',
            "vectors": r'Векторы:?\s*(.*?)(?=Точки|Функции|$)',
            "functions": r'Функции:?\s*(.*?)(?=Точки|Векторы|$)'
        }
    }
    
    # Функция для извлечения параметра по шаблону
    def extract_param(pattern, text, default=None):
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            try:
                value = match.group(1).strip()
                return value
            except Exception as e:
                logging.error(f"Ошибка при извлечении параметра: {e}")
        return default
    
    # Извлекаем параметры координатной плоскости
    points_str = extract_param(REGEX_PATTERNS["coordinate"]["points"], params_text)
    vectors_str = extract_param(REGEX_PATTERNS["coordinate"]["vectors"], params_text)
    functions_str = extract_param(REGEX_PATTERNS["coordinate"]["functions"], params_text)
    
    points = []
    vectors = []
    functions = []
    
    # Парсинг точек
    if points_str:
        try:
            # Формат A(1,2),B(3,4),C(5,6) или (1,2),(3,4),(5,6)
            for match in re.finditer(r'([A-Za-z]?)\s*\(([^,]+),([^)]+)\)', points_str):
                label, x, y = match.groups()
                x, y = float(x.strip()), float(y.strip())
                if label:
                    points.append((x, y, label))
                else:
                    points.append((x, y))
        except Exception as e:
            logging.warning(f"Ошибка при разборе точек: {e}")
    
    # Парсинг векторов
    if vectors_str:
        try:
            # Формат AB, CD или (1,2,3,4), (5,6,7,8)
            for match in re.finditer(r'([A-Za-z]{2})|(?:\(([^,]+),([^,]+),([^,]+),([^)]+)\))', vectors_str):
                if match.group(1):  # Формат AB
                    vector_name = match.group(1)
                    # Нужно найти точки A и B среди уже введенных
                    start_point = None
                    end_point = None
                    for point in points:
                        if len(point) > 2:  # Есть метка
                            if point[2] == vector_name[0]:
                                start_point = (point[0], point[1])
                            elif point[2] == vector_name[1]:
                                end_point = (point[0], point[1])
                    
                    if start_point and end_point:
                        vectors.append((start_point[0], start_point[1], end_point[0], end_point[1], vector_name))
                else:  # Формат (1,2,3,4)
                    x1, y1, x2, y2 = map(float, match.groups()[1:5])
                    vectors.append((x1, y1, x2, y2))
        except Exception as e:
            logging.warning(f"Ошибка при разборе векторов: {e}")
    
    # Парсинг функций
    if functions_str:
        try:
            # Формат x**2, 2*x+1
            for func in functions_str.split(','):
                func = func.strip()
                if func:
                    functions.append((func, 'blue'))  # Цвет по умолчанию - синий
        except Exception as e:
            logging.warning(f"Ошибка при разборе функций: {e}")
    
    return points, functions, vectors 