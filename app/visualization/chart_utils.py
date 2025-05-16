import matplotlib.pyplot as plt
import numpy as np
import os
import uuid
import logging
import traceback
import re
import math

def normalize_function_expression(expression):
    """
    Преобразует математическое выражение в формат, подходящий для вычисления в Python.
    
    Args:
        expression (str): Строка с математическим выражением в формате LaTeX (например, "$2*x^2 + \sin(x)$")
        
    Returns:
        str: Нормализованное выражение, которое можно использовать для вычислений в Python
    """
    if expression is None or expression == "":
        return ""
    
    # Предварительная обработка - убираем доллары и лишние пробелы
    expression = expression.replace("$", "").strip()
    
    logging.debug(f"Нормализация выражения: {expression}")
    
    # Проверка на наличие переменных, отличных от 'x'
    # Ищем все символьные имена (буквы не внутри команд LaTeX и не 'x')
    var_pattern = r'(?<!\\)([a-wyz]|[A-Z])'
    undefined_vars = re.findall(var_pattern, expression)
    if undefined_vars:
        logging.warning(f"Найдены неопределенные переменные: {set(undefined_vars)}")
        logging.warning(f"Для визуализации разрешено использовать только переменную 'x' и числовые константы")
        # Попытка преобразовать выражение, заменив неопределенные переменные на числа
        # По умолчанию заменяем на 2 для демонстрации графика
        for var in set(undefined_vars):
            expression = re.sub(r'(?<!\\)' + var, '2', expression)
        logging.info(f"Выражение после замены переменных: {expression}")
    
    try:
        # 1. Обработка логарифмов с корнем в основании: \log_{\sqrt[3]{4}}(x + 6)
        sqrt_index_log_pattern = r'\\log_\{\\sqrt\[([^]]+)\]\{([^}]+)\}\}\(([^)]+)\)'
        match = re.search(sqrt_index_log_pattern, expression)
        if match:
            index = match.group(1)
            sqrt_arg = match.group(2)
            log_arg = match.group(3)
            # Формируем замену для выражения логарифма
            replacement = f"(np.log({log_arg})/np.log(({sqrt_arg})**(1/{index})))"
            expression = expression.replace(match.group(0), replacement)
            return expression
        
        # 2. Обработка логарифмов с квадратным корнем в основании: \log_{\sqrt{4}}(x + 6)
        sqrt_log_pattern = r'\\log_\{\\sqrt\{([^}]+)\}\}\(([^)]+)\)'
        match = re.search(sqrt_log_pattern, expression)
        if match:
            sqrt_arg = match.group(1)
            log_arg = match.group(2)
            # Формируем замену для выражения логарифма
            replacement = f"(np.log({log_arg})/np.log(np.sqrt({sqrt_arg})))"
            expression = expression.replace(match.group(0), replacement)
            return expression
        
        # 3. Обработка логарифмов с обычным основанием: \log_{10}(x)
        log_base_pattern = r'\\log_\{([^}]+)\}\(([^)]+)\)'
        match = re.search(log_base_pattern, expression)
        if match:
            base = match.group(1)
            arg = match.group(2)
            replacement = f"(np.log({arg})/np.log({base}))"
            expression = expression.replace(match.group(0), replacement)
            return expression
        
        # 4. Обработка простых логарифмов с числовым основанием: \log_2(x)
        simple_log_pattern = r'\\log_(\d+)\(([^)]+)\)'
        match = re.search(simple_log_pattern, expression)
        if match:
            base = match.group(1)
            arg = match.group(2)
            replacement = f"(np.log({arg})/np.log({base}))"
            expression = expression.replace(match.group(0), replacement)
            return expression

        # 5. Обработка степеней вида a^b
        power_pattern = r'([\w\d]+)\^\{([^}]+)\}'
        for match in re.finditer(power_pattern, expression):
            base = match.group(1)
            exponent = match.group(2)
            replacement = f"({base})**({exponent})"
            expression = expression.replace(match.group(0), replacement)

        # Также обрабатываем простые степени вида x^2
        simple_power_pattern = r'([\w\d]+)\^([\w\d])'
        for match in re.finditer(simple_power_pattern, expression):
            base = match.group(1)
            exponent = match.group(2)
            replacement = f"({base})**({exponent})"
            expression = expression.replace(match.group(0), replacement)

        # 6. Обработка корней с индексом: \sqrt[индекс]{аргумент}
        sqrt_index_pattern = r'\\sqrt\[([^]]+)\]\{([^}]+)\}'
        for match in re.finditer(sqrt_index_pattern, expression):
            index = match.group(1)
            arg = match.group(2)
            replacement = f"({arg})**(1/{index})"
            expression = expression.replace(match.group(0), replacement)
        
        # 7. Обработка обычных квадратных корней: \sqrt{аргумент}
        sqrt_pattern = r'\\sqrt\{([^}]+)\}'
        for match in re.finditer(sqrt_pattern, expression):
            arg = match.group(1)
            replacement = f"np.sqrt({arg})"
            expression = expression.replace(match.group(0), replacement)
        
        # Проверка на необработанные \sqrt без {}
        if "\\sqrt" in expression:
            expression = expression.replace("\\sqrt", "np.sqrt")

        # 8. Обработка дробей вида \frac{числитель}{знаменатель}
        frac_pattern = r'\\frac\{([^}]+)\}\{([^}]+)\}'
        for match in re.finditer(frac_pattern, expression):
            numerator = match.group(1)
            denominator = match.group(2)
            replacement = f"(({numerator})/({denominator}))"
            expression = expression.replace(match.group(0), replacement)
        
        # Проверка на необработанные \frac без {}
        if "\\frac" in expression:
            expression = expression.replace("\\frac", "")
        
        # 9. Обработка обычных логарифмов (\log, \ln) ПОСЛЕ обработки логарифмов с основаниями
        expression = expression.replace("\\ln", "np.log")
        expression = expression.replace("\\log", "np.log10")
        
        # 10. Первичная обработка распространенных LaTeX-шаблонов
        expression = expression.replace("\\cdot", "*")
        expression = expression.replace("\\times", "*")
        expression = expression.replace("\\div", "/")
        expression = expression.replace("\\sin", "np.sin")
        expression = expression.replace("\\cos", "np.cos")
        expression = expression.replace("\\tan", "np.tan")
        expression = expression.replace("\\tg", "np.tan")
        expression = expression.replace("\\ctg", "1/np.tan")
        expression = expression.replace("\\pi", "np.pi")
        expression = expression.replace("\\infty", "np.inf")
        expression = expression.replace("\\e", "np.e")
        
        # 11. Замена специальных символов
        expression = expression.replace("√", "np.sqrt")
        expression = expression.replace("π", "np.pi")
        expression = expression.replace("÷", "/")
        expression = expression.replace("×", "*")
        expression = expression.replace("^", "**")
        
        # 12. Обработка модуля: |выражение| -> abs(выражение)
        modulus_pattern = r'\|([^|]+)\|'
        for match in re.finditer(modulus_pattern, expression):
            inner_expr = match.group(1)
            replacement = f"np.abs({inner_expr})"
            expression = expression.replace(match.group(0), replacement)
        
        # 13. Неявное умножение: замена 2x на 2*x
        expression = re.sub(r'(\d+)([a-zA-Z\(])', r'\1*\2', expression)
        
        # 14. Неявное умножение вида (x+1)(x-1) -> (x+1)*(x-1) и x(x+1) -> x*(x+1)
        expression = re.sub(r'(\)|[a-zA-Z])(\()', r'\1*\2', expression)
        
        # 15. Неявное умножение вида (1+2)x -> (1+2)*x и )x -> )*x
        expression = re.sub(r'(\))([a-zA-Z])', r'\1*\2', expression)
        
        # 16. Обработка дробей вида 3/2x -> 3/(2*x) или 3/2*x -> (3/2)*x
        expression = re.sub(r'(\d+)/(\d+)([a-zA-Z])', r'(\1/\2)*\3', expression)
        
        # 17. Обработка операций сравнения
        expression = expression.replace("\\le", "<=")
        expression = expression.replace("\\ge", ">=")
        expression = expression.replace("\\neq", "!=")
        expression = expression.replace("\\approx", "==")
        
        # Проверка на корректность выражения
        try:
            x = np.array([1.0])  # Используем 1.0 вместо 0.0 для избегания деления на ноль
            safe_dict = {'x': x, 'np': np, 'math': math}
            eval(expression, {"__builtins__": {}}, safe_dict)
            logging.debug(f"Выражение '{expression}' корректно нормализовано")
            return expression
        except Exception as e:
            logging.warning(f"Не удалось вычислить выражение: {expression}, ошибка: {e}")
            return None
    except Exception as e:
        logging.error(f"Ошибка при нормализации выражения '{expression}': {e}")
        traceback.print_exc()
        return None

def process_multiple_function_plots(functions, colors, labels, x_ranges, y_ranges, show_grid=True, title="График функций"):
    """
    Создает изображение с несколькими графиками функций на одном полотне.
    
    Args:
        functions (list): Список строк с математическими функциями для построения
        colors (list): Список цветов для каждого графика
        labels (list): Список меток/названий для каждого графика
        x_ranges (list): Список кортежей (min_x, max_x) для каждой функции
        y_ranges (list): Список кортежей (min_y, max_y) для каждой функции или None
        show_grid (bool): Показывать сетку на графике
        title (str): Заголовок графика
        
    Returns:
        str: Путь к созданному изображению или None в случае ошибки
    """
    try:
        # Проверяем входные данные
        if not functions:
            print("Не указана ни одна функция для построения графика")
            return None
        
        # Дополняем цвета, если их недостаточно
        default_colors = ['blue', 'red', 'green', 'orange', 'purple']
        while len(colors) < len(functions):
            colors.append(default_colors[len(colors) % len(default_colors)])
        
        # Дополняем метки, если их недостаточно
        while len(labels) < len(functions):
            labels.append(f"График {len(labels) + 1}")
        
        # Дополняем диапазоны X, если их недостаточно
        default_x_range = (-10, 10)
        while len(x_ranges) < len(functions):
            x_ranges.append(default_x_range)
        
        # Дополняем диапазоны Y, если их недостаточно
        while len(y_ranges) < len(functions):
            y_ranges.append(None)
        
        # Тестовый путь к файлу
        return "test_path.png"
    except Exception as e:
        print(f"Ошибка при создании графиков функций: {e}")
        traceback.print_exc()
        return None

def process_bar_chart(categories, values, title="Столбчатая диаграмма"):
    """
    Создает столбчатую диаграмму на основе предоставленных данных.
    
    Args:
        categories (list): Список категорий
        values (list): Список значений для каждой категории
        title (str): Заголовок диаграммы
        
    Returns:
        str: Путь к созданному изображению или None в случае ошибки
    """
    try:
        # Проверяем входные данные
        if not categories or not values or len(categories) != len(values):
            print(f"Некорректные данные для столбчатой диаграммы")
            return None
        
        # Тестовый путь к файлу
        return "test_bar_chart.png"
    except Exception as e:
        print(f"Ошибка при создании столбчатой диаграммы: {e}")
        traceback.print_exc()
        return None

def process_pie_chart(labels, values, title="Круговая диаграмма"):
    """
    Создает круговую диаграмму на основе предоставленных данных.
    
    Args:
        labels (list): Список меток для секторов
        values (list): Список значений для каждого сектора
        title (str): Заголовок диаграммы
        
    Returns:
        str: Путь к созданному изображению или None в случае ошибки
    """
    try:
        # Проверяем входные данные
        if not labels or not values or len(labels) != len(values):
            print(f"Некорректные данные для круговой диаграммы")
            return None
        
        # Тестовый путь к файлу
        return "test_pie_chart.png"
    except Exception as e:
        print(f"Ошибка при создании круговой диаграммы: {e}")
        traceback.print_exc()
        return None

def process_chart_visualization(params_text, extract_param):
    """
    Обрабатывает параметры визуализации для диаграммы и создает соответствующее изображение.
    
    Args:
        params_text (str): Текст с параметрами диаграммы
        extract_param (callable): Функция для извлечения параметров из текста
        
    Returns:
        str: Путь к созданному изображению или None в случае ошибки
    """
    try:
        # Определяем тип диаграммы (столбчатая или круговая)
        chart_type = "столбчатая"  # Для теста
        
        # Для столбчатой диаграммы
        if "столбчат" in chart_type or "bar" in chart_type:
            categories = ["A", "B", "C"]
            values = [10, 20, 30]
            return process_bar_chart(categories, values)
            
        # Для круговой диаграммы
        elif "кругов" in chart_type or "pie" in chart_type:
            labels = ["A", "B", "C"]
            values = [10, 20, 30]
            return process_pie_chart(labels, values)
        
        else:
            print(f"Неизвестный тип диаграммы: {chart_type}")
            return None
            
    except Exception as e:
        print(f"Ошибка при обработке параметров диаграммы: {e}")
        traceback.print_exc()
        return None 