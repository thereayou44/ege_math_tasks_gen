"""
Модуль для определения необходимости и типа визуализации на основе текста задачи.
"""

import logging

def needs_visualization(task_text, category, subcategory, is_basic_level=False):
    """
    Определяет, нужна ли визуализация для задачи на основе её текста и категории.
    
    Args:
        task_text: Текст задачи
        category: Категория задачи
        subcategory: Подкатегория задачи
        is_basic_level: Флаг базового уровня ЕГЭ
        
    Returns:
        bool: True, если нужна визуализация, иначе False
    """
    # Проверяем геометрические категории, где часто требуется визуализация
    geometric_categories = [
        "Планиметрия", "Стереометрия", "Геометрия", "Треугольники", "Окружности", 
        "Многоугольники", "Векторы", "Координатная плоскость"
    ]
    
    if any(geo_cat.lower() in category.lower() for geo_cat in geometric_categories) or \
       any(geo_cat.lower() in (subcategory or "").lower() for geo_cat in geometric_categories):
        return True
    
    # Проверяем, есть ли в тексте задачи ключевые слова, указывающие на необходимость визуализации
    visualization_keywords = [
        "график", "функц", "треугольник", "окружность", "круг", "прямоугольник", 
        "параллелограмм", "трапеци", "координат", "вектор", "точк", "прямая", "отрезок",
        "квадрат", "ромб", "парабол", "гипербол", "синусоид", "косинусоид", "экспонент"
    ]
    
    if any(keyword in task_text.lower() for keyword in visualization_keywords):
        return True
    
    # Определяем тип визуализации на основе содержания задачи
    visualization_type = determine_visualization_type(task_text, category, subcategory)
    
    # Если удалось определить тип визуализации, значит она нужна
    return visualization_type is not None

def determine_visualization_type(task_text, category, subcategory):
    """
    Определяет тип визуализации на основе текста задачи и категории.
    
    Args:
        task_text: Текст задачи
        category: Категория задачи
        subcategory: Подкатегория задачи
        
    Returns:
        str: Тип визуализации или None, если не удалось определить
    """
    # Ключевые слова для различных типов визуализации
    type_keywords = {
        "graph": ["график", "функц", "парабол", "гипербол", "синусоид", "косинусоид", 
                 "экспонент", "y =", "f(x) =", "ось абсцисс", "ось ординат"],
        "triangle": ["треугольник", "равнобедренн", "равносторонн", "прямоугольный треугольник", 
                    "катет", "гипотенуз", "медиан", "биссектрис", "высот"],
        "circle": ["окружность", "круг", "радиус", "диаметр", "хорд", "сектор", "сегмент",
                  "касательн", "центр окружности"],
        "rectangle": ["прямоугольник", "квадрат", "ромб", "периметр прямоугольника", 
                     "площадь прямоугольника", "сторона прямоугольника"],
        "parallelogram": ["параллелограмм", "ромб", "периметр параллелограмма", 
                         "площадь параллелограмма", "сторона параллелограмма"],
        "trapezoid": ["трапеци", "основание трапеции", "боковая сторона трапеции", 
                     "высота трапеции", "средняя линия трапеции"],
        "coordinate": ["координатн", "декартов", "система координат", "коорд.", "точка", 
                      "точки", "вектор", "прямая", "отрезок", "вектор"]
    }
    
    # Категории, связанные с типами визуализации
    category_types = {
        "graph": ["Функции", "Графики функций", "Производная", "Интеграл"],
        "triangle": ["Треугольники", "Планиметрия", "Геометрия"],
        "circle": ["Окружности", "Планиметрия", "Геометрия"],
        "rectangle": ["Многоугольники", "Планиметрия", "Геометрия"],
        "parallelogram": ["Многоугольники", "Планиметрия", "Геометрия"],
        "trapezoid": ["Многоугольники", "Планиметрия", "Геометрия"],
        "coordinate": ["Координатная плоскость", "Векторы", "Геометрия"]
    }
    
    # Проверяем категорию и подкатегорию
    for viz_type, cats in category_types.items():
        if any(cat.lower() in category.lower() for cat in cats) or \
           any(cat.lower() in (subcategory or "").lower() for cat in cats):
            # Дополнительно проверяем текст задачи на ключевые слова для этого типа
            if any(keyword.lower() in task_text.lower() for keyword in type_keywords[viz_type]):
                return viz_type
    
    # Если не определили по категории, определяем только по тексту задачи
    type_scores = {}
    for viz_type, keywords in type_keywords.items():
        score = 0
        for keyword in keywords:
            if keyword.lower() in task_text.lower():
                score += 1
        if score > 0:
            type_scores[viz_type] = score
    
    # Возвращаем тип с наибольшим количеством совпадений
    if type_scores:
        return max(type_scores.items(), key=lambda x: x[1])[0]
    
    return None 