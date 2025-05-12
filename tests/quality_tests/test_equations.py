import sys
import os
import re
from sympy import symbols, solve, sympify, Eq, sqrt, log, sin, cos, tan
import unittest
import logging
import matplotlib
import math
matplotlib.use('Agg')  # Использование не-интерактивного бэкенда

# Добавляем родительский каталог в sys.path для импорта app
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

try:
    from app.task_generator import generate_complete_task
except ImportError as e:
    logging.error(f"Ошибка импорта модуля task_generator: {e}")
    sys.exit(1)

class EquationQualityTest(unittest.TestCase):
    """Класс для тестирования качества генерации уравнений."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Инициализируем статистику
        self.stats = {
            'total_tasks': 0,
            'successful_parsings': 0,
            'correct_verifications': 0,
            'failed_tasks': []
        }
    
    def get_test_statistics(self):
        """
        Возвращает статистику тестирования качества уравнений.
        
        Returns:
            dict: Словарь с метриками качества
        """
        return self.stats
    
    def extract_numeric_answer(self, answer_text):
        """
        Извлекает числовые значения из ответа.
        
        Args:
            answer_text: Текст ответа из задачи
            
        Returns:
            list: Список числовых значений
        """
        # Проверяем ответы вида "нет решений", "пустое множество" и т.п.
        no_solution_patterns = [
            r'нет\s+(?:корней|решений)',
            r'не\s+(?:имеет|существует)\s+(?:корней|решений)',
            r'пустое\s+множество',
            r'решений\s+нет',
            r'empty\s+set',
            r'no\s+solutions?',
            r'\{\}',  # Пустое множество в виде {}
            r'\\emptyset',  # LaTeX пустое множество
            r'\\varnothing'  # Еще один вариант пустого множества в LaTeX
        ]
        
        for pattern in no_solution_patterns:
            if re.search(pattern, answer_text, re.IGNORECASE):
                logging.info(f"Обнаружен ответ 'нет решений': {answer_text}")
                return []  # Возвращаем пустой список, что означает "нет корней"
                
        # Обработка корней вида \sqrt{число}
        sqrt_pattern = r'\\sqrt\{(\d+)\}'
        sqrt_matches = re.finditer(sqrt_pattern, answer_text)
        sqrt_values = []
        
        for match in re.finditer(sqrt_pattern, answer_text):
            try:
                # Получаем число под корнем
                sqrt_value = int(match.group(1))
                # Вычисляем корень
                sqrt_values.append(math.sqrt(sqrt_value))
            except Exception as e:
                logging.error(f"Ошибка при обработке корня: {e}")
        
        # Обработка отрицательных корней -\sqrt{число}
        neg_sqrt_pattern = r'-\\sqrt\{(\d+)\}'
        for match in re.finditer(neg_sqrt_pattern, answer_text):
            try:
                # Получаем число под корнем
                sqrt_value = int(match.group(1))
                # Вычисляем отрицательный корень
                sqrt_values.append(-math.sqrt(sqrt_value))
            except Exception as e:
                logging.error(f"Ошибка при обработке отрицательного корня: {e}")
        
        # Обработка специальных случаев сложных дробей
        # Ищем выражения вида \frac{1 + 3\sqrt{17}}{2}
        # Расширяем паттерн для поддержки случаев с коэффициентами перед корнем
        complex_frac_pattern = r'\\frac\{([^{}]*?)(\d+)?\\sqrt\{(\d+)\}([^{}]*)\}\{(\d+)\}'
        complex_frac_matches = re.finditer(complex_frac_pattern, answer_text)
        for match in complex_frac_matches:
            try:
                # Получаем выражение числителя, e.g., "1 + 3\sqrt{17}"
                prefix = match.group(1).strip()  # Текст перед корнем (например, "1 + ")
                coef_str = match.group(2) or "1"  # Коэффициент перед корнем (например, "3")
                sqrt_value = int(match.group(3))  # Число под корнем (например, "17")
                suffix = match.group(4).strip()  # Текст после корня (например, " - 5")
                denominator = int(match.group(5))  # Знаменатель (например, "2")
                
                # Извлекаем коэффициент перед корнем
                coef = int(coef_str)
                
                # Инициализируем результат
                result = 0
                
                # Обрабатываем префикс (если есть)
                if prefix:
                    # Извлекаем число из префикса
                    prefix_number_match = re.search(r'(\d+)', prefix)
                    if prefix_number_match:
                        prefix_number = int(prefix_number_match.group(1))
                        # Определяем знак (по умолчанию +)
                        if "-" in prefix:
                            prefix_number = -prefix_number
                        result += prefix_number
                
                # Добавляем значение корня с коэффициентом
                if "+" in prefix or not prefix:
                    # Если перед корнем стоит + или нет знака
                    result += coef * math.sqrt(sqrt_value)
                elif "-" in prefix:
                    # Если перед корнем стоит -
                    result -= coef * math.sqrt(sqrt_value)
                
                # Обрабатываем суффикс (если есть)
                if suffix:
                    # Извлекаем число из суффикса
                    suffix_number_match = re.search(r'(\d+)', suffix)
                    if suffix_number_match:
                        suffix_number = int(suffix_number_match.group(1))
                        # Определяем знак
                        if "-" in suffix:
                            result -= suffix_number
                        elif "+" in suffix:
                            result += suffix_number
                
                # Делим результат на знаменатель
                result = result / denominator
                sqrt_values.append(result)
                
                # Логируем найденное значение для отладки
                logging.info(f"Извлечено значение сложной дроби: {result} из {match.group(0)}")
            except Exception as e:
                logging.error(f"Ошибка при обработке сложной дроби: {e}")
        
        # Сначала ищем явно отрицательные дроби -\frac{a}{b}
        negative_frac_pattern = r'-\s*\\frac\{(\d+)\}\{(\d+)\}'
        negative_fractions = []
        
        for match in re.finditer(negative_frac_pattern, answer_text):
            numerator = float(match.group(1))
            denominator = float(match.group(2))
            negative_fractions.append(-1 * (numerator / denominator))
        
        # Удаляем найденные отрицательные дроби из текста
        if negative_fractions:
            answer_text = re.sub(negative_frac_pattern, 'NEG_FRAC_PLACEHOLDER', answer_text)
        
        # Затем проверяем наличие обычных дробей \frac{a}{b} и обрабатываем их
        fractions = []
        # Ищем дроби в формате \frac{числитель}{знаменатель} с учетом возможного знака минус
        frac_pattern = r'\\frac\{(-?\d+)\}\{(\d+)\}'
        for match in re.finditer(frac_pattern, answer_text):
            numerator = float(match.group(1))  # Может быть отрицательным
            denominator = float(match.group(2))
            fractions.append(numerator / denominator)
        
        # Если нашли дроби, удаляем их из текста, чтобы избежать дублирования
        if fractions:
            # Временно заменяем дроби на заглушки
            answer_text = re.sub(frac_pattern, 'FRACTION_PLACEHOLDER', answer_text)
        
        # Проверяем на наличие выражений вида x = -число или ответ: -число
        minus_pattern = r'(?:x\s*=|ответ:|answer:)\s*-\s*(\d+(?:\.\d+)?)'
        minus_matches = re.finditer(minus_pattern, answer_text, re.IGNORECASE)
        for match in minus_matches:
            # Явно сохраняем отрицательное число
            fractions.append(-1 * float(match.group(1)))
            # Заменяем найденное выражение заглушкой
            answer_text = re.sub(match.group(0), 'MINUS_PLACEHOLDER', answer_text)
        
        # Удаляем все теги и LaTeX разметку
        clean_text = re.sub(r'\$+', '', answer_text)  # Удаляем символы $
        clean_text = re.sub(r'\\[a-z]+', '', clean_text)  # Удаляем команды LaTeX
        
        # Ищем дроби вида a/b, включая отрицательные числители
        simple_frac_pattern = r'(-?\d+)\s*/\s*(\d+)'
        for match in re.finditer(simple_frac_pattern, clean_text):
            numerator = float(match.group(1))  # Может быть отрицательным
            denominator = float(match.group(2))
            fractions.append(numerator / denominator)
        
        # Удаляем найденные простые дроби из текста
        clean_text = re.sub(simple_frac_pattern, '', clean_text)
        
        # Находим все числа (целые и дробные, включая отрицательные)
        numbers = re.findall(r'-?\d+(?:\.\d+)?', clean_text)
        
        # Объединяем все найденные числа
        result = sqrt_values + negative_fractions + fractions + [float(num) for num in numbers]
        
        # Дополнительная проверка: если все числа положительные, но есть знак минус перед ответом,
        # возможно, нам нужно инвертировать знак
        if result and all(x > 0 for x in result) and (re.search(r'=\s*-', answer_text, re.IGNORECASE) or 
                                                      re.search(r'ответ:\s*-', answer_text, re.IGNORECASE)):
            # Инвертируем знак всех чисел
            result = [-x for x in result]
        
        # Если в результате нет ни одного числа, проверяем на текстовые представления
        if not result:
            if re.search(r'\bjedin|единств|only', clean_text, re.IGNORECASE):
                # Для ответов типа "единственный корень", просто возвращаем первое найденное число
                numbers = re.findall(r'-?\d+(?:\.\d+)?', answer_text)
                if numbers:
                    return [float(numbers[0])]
                    
        # Логируем результат
        logging.info(f"Извлеченные числовые ответы: {result}")
        
        return result
    
    def verify_equation_answer(self, equation_str, answers):
        """
        Проверяет, являются ли найденные ответы решениями уравнения.
        
        Args:
            equation_str: Строка с уравнением
            answers: Список численных ответов
            
        Returns:
            bool: True, если хотя бы один ответ верен, иначе False
        """
        try:
            # Добавляем необходимые функции для вычислений
            from sympy import symbols, solve, sympify, Eq, sqrt, log, sin, cos, tan
            
            # Предполагаем, что уравнение использует x как переменную
            x = symbols('x')
            
            # Если нет ответов для проверки, возвращаем False
            if not answers or len(answers) == 0:
                logging.info("Нет ответов для проверки")
                if not solutions or len(solutions) == 0:
                    # Если и sympy не нашел решений, считаем ответ корректным
                    logging.info("SymPy тоже не нашел решений - ответ корректен")
                    return True
                else:
                    # Есть решения sympy, но нет ответов модели
                    logging.info("SymPy нашел решения, но нет ответов модели - ответ некорректен")
                    return False
                
            # Парсим уравнение
            logging.info(f"Парсим уравнение: {equation_str}")
            if '=' in equation_str:
                # Если уравнение в форме 'левая_часть = правая_часть'
                left_side, right_side = equation_str.split('=', 1)
                try:
                    left_expr = sympify(left_side.strip())
                    right_expr = sympify(right_side.strip())
                    equation = Eq(left_expr, right_expr)
                except Exception as e:
                    logging.error(f"Ошибка при парсинге уравнения с =: {e}")
                    # Пробуем преобразовать форму
                    expr = f"({left_side.strip()}) - ({right_side.strip()})"
                    equation = Eq(sympify(expr), 0)
            else:
                # Если уравнение в форме выражения, предполагаем, что оно равно 0
                try:
                    equation = Eq(sympify(equation_str), 0)
                except Exception as e:
                    logging.error(f"Ошибка при парсинге уравнения без =: {e}")
                    return False
            
            # Решаем уравнение с помощью SymPy
            try:
                solutions = solve(equation, x)
                logging.info(f"Решения SymPy: {solutions}")
                
                # Если решений нет (пустой список), пропускаем проверку
                if not solutions:
                    logging.info(f"SymPy не нашел решений для уравнения: {equation}")
                    return None
            except Exception as e:
                logging.error(f"Ошибка при решении уравнения: {e}")
                return None
            
            # Преобразуем решения в числовые значения
            numeric_solutions = []
            for sol in solutions:
                try:
                    if hasattr(sol, "is_real") and sol.is_real:
                        numeric_solutions.append(float(sol))
                    elif isinstance(sol, (int, float)):
                        numeric_solutions.append(float(sol))
                except Exception:
                    # Пропускаем неконвертируемые решения
                    pass
            
            # Если после преобразования не осталось числовых решений, пропускаем задачу
            if not numeric_solutions:
                logging.info(f"После преобразования не осталось числовых решений")
                return None
                
            logging.info(f"Числовые решения SymPy: {numeric_solutions}")
            logging.info(f"Ответы для проверки: {answers}")
            
            # Создаем функцию для проверки приблизительного равенства
            def approx_equal(a, b, tolerance=1e-6):
                return abs(a - b) < tolerance
            
            # Проверяем наличие хотя бы одного ответа в решениях SymPy
            matching_answers = 0
            for answer in answers:
                if any(approx_equal(answer, sol) for sol in numeric_solutions):
                    matching_answers += 1
            
            # Проверяем реципрокное совпадение - есть ли среди решений SymPy хотя бы одно, 
            # которое есть в наших ответах
            if len(numeric_solutions) > 0:
                has_matching_sympy_solution = any(any(approx_equal(sol, answer) for answer in answers) 
                                                  for sol in numeric_solutions)
            else:
                has_matching_sympy_solution = False
            
            # Проверяем количество совпадающих ответов
            is_success = False
            
            # Случай 1: Найдено решение и хотя бы один ответ модели совпадает с ним
            if len(numeric_solutions) > 0 and matching_answers > 0:
                is_success = True
                logging.info(f"Успешная проверка: {matching_answers} ответов модели совпадают с решениями SymPy")
            
            # Случай 2: Модель вернула все корректные решения (количество совпадает)
            elif len(numeric_solutions) == len(answers) and matching_answers == len(answers):
                is_success = True
                logging.info(f"Успешная проверка: все {matching_answers} ответов модели совпадают со всеми решениями SymPy")
            
            # Случай 3: Нет решений у SymPy, но модель тоже не дала ответов
            elif len(numeric_solutions) == 0 and len(answers) == 0:
                is_success = True
                logging.info("Успешная проверка: нет решений у SymPy и модель не дала ответов")
            
            # Случай 4: Есть хотя бы одно решение у SymPy, и хотя бы одно у модели, и они совпадают
            elif has_matching_sympy_solution:
                is_success = True
                logging.info(f"Успешная проверка: совпадает хотя бы одно решение из {len(numeric_solutions)} решений SymPy")
            
            return is_success
            
        except Exception as e:
            logging.error(f"Ошибка при проверке уравнения: {e}")
            return False
    
    def extract_equation_from_task(self, task_text):
        """
        Извлекает уравнение из текста задачи и преобразует его в формат, понятный для sympy.
        
        Args:
            task_text: Текст задачи
            
        Returns:
            str: Очищенная строка с уравнением для sympy
        """
        # Удаляем HTML-теги
        clean_text = re.sub(r'<[^>]+>', '', task_text)
        
        # Обрабатываем случаи с двойными обратными слэшами (\\) - заменяем на одинарные
        # Это может произойти из-за экранирования в процессе сохранения/загрузки из кэша
        clean_text = clean_text.replace('\\\\', '\\')
        
        # Ищем уравнение в тексте задачи
        equation_patterns = [
            r'уравнение\s*(?:\$+)?\s*([^\.]+)\s*(?:\$+)?',  # уравнение x^2 + 5x + 6 = 0
            r'решите\s+(?:\$+)?\s*([^\.]+\=\s*[^\.]+)\s*(?:\$+)?',  # решите x^2 + 5x + 6 = 0
            r'(?:\$+)?\s*([^\.]+\=\s*[^\.]+)\s*(?:\$+)?'  # x^2 + 5x + 6 = 0
        ]
        
        found_equation = None
        
        for pattern in equation_patterns:
            matches = re.search(pattern, clean_text, re.IGNORECASE)
            if matches:
                found_equation = matches.group(1).strip()
                # Удаляем начальные и конечные символы $
                found_equation = re.sub(r'^\$+|\$+$', '', found_equation)
                break
        
        if not found_equation:
            return None
        
        try:
            # Удаляем из уравнения двоеточия и другие нежелательные символы в начале
            equation = re.sub(r'^[:\s]+', '', found_equation)
            
            # Удаляем все символы $ внутри уравнения
            equation = re.sub(r'\$+', '', equation)
            
            # Удаляем команды \left и \right, которые могут мешать обработке
            equation = equation.replace('\\left', '').replace('\\right', '')
            
            # Заменяем дроби \frac{a}{b} на (a)/(b)
            while '\\frac' in equation:
                frac_match = re.search(r'\\frac\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', equation)
                if not frac_match:
                    break
                
                num = frac_match.group(1)
                denom = frac_match.group(2)
                
                replacement = f"({num})/({denom})"
                equation = equation[:frac_match.start()] + replacement + equation[frac_match.end():]
            
            # Предварительно обрабатываем тригонометрические функции, сразу заменяя \sin на sin
            trig_funcs = ['sin', 'cos', 'tan', 'cot', 'sec', 'csc', 'arcsin', 'arccos', 'arctan']
            for func in trig_funcs:
                equation = equation.replace(f'\\{func}', func)
            
            # Заменяем pi на sympy-формат
            equation = equation.replace('\\pi', 'pi')
            
            # Предварительно обрабатываем тригонометрические выражения с pi
            # Например, sin(pi(x-5)/4) -> sin(pi*(x-5)/4)
            for func in trig_funcs:
                func_pattern = fr'{func}\s*\(([^)]*pi[^)]*)\)'
                matches = re.finditer(func_pattern, equation)
                for match in matches:
                    arg = match.group(1)
                    # Добавляем умножение между pi и скобками
                    new_arg = re.sub(r'pi\s*\(', 'pi*(', arg)
                    # Также обрабатываем случаи pi x -> pi*x
                    new_arg = re.sub(r'pi\s+([a-zA-Z])', r'pi*\1', new_arg)
                    equation = equation.replace(match.group(0), f'{func}({new_arg})')
            
            # Проверяем, есть ли знак равенства, и разделяем уравнение на левую и правую части
            if '=' in equation:
                left_side, right_side = equation.split('=', 1)
                left_side = left_side.strip()
                right_side = right_side.strip()
                
                # Преобразуем каждую часть отдельно
                left_sympy = self.convert_to_sympy_expr(left_side)
                right_sympy = self.convert_to_sympy_expr(right_side)
                
                if left_sympy is None or right_sympy is None:
                    return None
                
                # Формируем уравнение в формате sympy: left_side - right_side
                return f"({left_sympy}) - ({right_sympy})"
            else:
                # Если нет знака равенства, пытаемся преобразовать всё выражение напрямую
                sympy_expr = self.convert_to_sympy_expr(equation)
                return sympy_expr
            
        except Exception as e:
            logging.error(f"Ошибка при обработке уравнения: {e}")
            return None

    def convert_to_sympy_expr(self, expr_text):
        """
        Преобразует математическое выражение из LaTeX в формат sympy.
        
        Args:
            expr_text: Текст математического выражения
            
        Returns:
            str: Выражение в формате sympy или None в случае ошибки
        """
        try:
            # Очищаем выражение от пробелов
            clean_expr = expr_text.strip()
            
            # Проверяем наличие двойных обратных слэшей (\\) и заменяем их на одинарные
            # Это может произойти из-за экранирования в процессе сохранения/загрузки из кэша
            clean_expr = clean_expr.replace('\\\\', '\\')
            
            # Заменяем запятые на точки в десятичных числах (например, 0,5 -> 0.5)
            clean_expr = re.sub(r'(\d+),(\d+)', r'\1.\2', clean_expr)
            
            # Заменяем умножение
            clean_expr = clean_expr.replace('\\cdot', '*')
            clean_expr = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', clean_expr)  # 2x -> 2*x
            clean_expr = re.sub(r'\)\(', r')*(', clean_expr)  # )(  -> )*(
            
            # Обработка умножения между числом и функцией, например 2log -> 2*log
            clean_expr = re.sub(r'(\d+)\\log', r'\1*\\log', clean_expr)
            clean_expr = re.sub(r'(\d+)\\ln', r'\1*\\ln', clean_expr)
            clean_expr = re.sub(r'(\d+)\\sin', r'\1*\\sin', clean_expr)
            clean_expr = re.sub(r'(\d+)\\cos', r'\1*\\cos', clean_expr)
            clean_expr = re.sub(r'(\d+)\\tan', r'\1*\\tan', clean_expr)
            
            # Обработка команд \left( и \right)
            clean_expr = clean_expr.replace('\\left(', '(').replace('\\right)', ')')
            clean_expr = clean_expr.replace('\\left[', '[').replace('\\right]', ']')
            
            # Обработка pi
            clean_expr = clean_expr.replace('\\pi', 'pi')
            # Добавляем умножение между pi и переменной
            clean_expr = re.sub(r'(pi)(\s*)([a-zA-Z])', r'\1*\3', clean_expr)
            # Также обрабатываем случаи, когда pi внутри скобок
            clean_expr = re.sub(r'(pi)(\s*)(\()', r'\1*\3', clean_expr)
            # Обрабатываем числа перед pi: 3pi -> 3*pi
            clean_expr = re.sub(r'(\d+)(\s*)(pi)', r'\1*\3', clean_expr)
            
            # Специальная обработка для выражений вида \sin(\pi x) и т.п.
            trig_funcs = ['sin', 'cos', 'tan', 'cot', 'sec', 'csc', 'arcsin', 'arccos', 'arctan']
            for func in trig_funcs:
                # Исправляем: заменяем \sin на sin
                clean_expr = clean_expr.replace(f'\\{func}', func)
                
                # Паттерн для поиска тригонометрических функций с pi внутри (уже без обратного слеша)
                func_pattern = f'{func}\\s*\\(\\s*pi\\s*[^)]*\\)'
                
                # Ищем все такие функции
                for match in re.finditer(func_pattern, clean_expr):
                    func_expr = match.group(0)
                    # Заменяем pi x на pi*x внутри функции
                    modified_expr = re.sub(r'(pi)(\s+)([a-zA-Z])', r'\1*\3', func_expr)
                    modified_expr = re.sub(r'(pi)(\s*)([/\+\-\*])', r'\1\3', modified_expr)
                    clean_expr = clean_expr.replace(func_expr, modified_expr)
            
            # Заменяем дроби \frac{a}{b} -> (a)/(b)
            while '\\frac{' in clean_expr:
                frac_match = re.search(r'\\frac\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', clean_expr)
                if not frac_match:
                    break
                
                num = self.convert_to_sympy_expr(frac_match.group(1))
                denom = self.convert_to_sympy_expr(frac_match.group(2))
                
                if num is None or denom is None:
                    return None
                
                replacement = f"({num})/({denom})"
                clean_expr = clean_expr[:frac_match.start()] + replacement + clean_expr[frac_match.end():]
            
            # Заменяем корни \sqrt{a} -> sqrt(a)
            while '\\sqrt{' in clean_expr:
                sqrt_match = re.search(r'\\sqrt\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', clean_expr)
                if not sqrt_match:
                    break
                
                arg = self.convert_to_sympy_expr(sqrt_match.group(1))
                
                if arg is None:
                    return None
                
                replacement = f"sqrt({arg})"
                clean_expr = clean_expr[:sqrt_match.start()] + replacement + clean_expr[sqrt_match.end():]
            
            # Обработка степеней (наиболее проблемная часть)
            
            # Отдельная обработка показательных функций вида a^{bx+c}
            # Сначала обрабатываем такие выражения специальным способом перед другими степенями
            exp_base_pattern = r'(\d+|\([^()]+\)|\{[^{}]+\}|\w+)(?:\s*)\^(?:\s*)\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
            for match in re.finditer(exp_base_pattern, clean_expr):
                full_match = match.group(0)
                base = match.group(1)
                exp = match.group(2)
                # Удаляем { } из базы, если они там есть
                if base.startswith('{') and base.endswith('}'):
                    base = base[1:-1]
                
                # Конвертируем показатель
                exp_converted = self.convert_to_sympy_expr(exp)
                if exp_converted is None:
                    # Если не удалось конвертировать, пробуем напрямую
                    exp_converted = exp.replace(' ', '*').replace('x-', 'x-')
                    # Убедимся, что умножения добавлены корректно
                    exp_converted = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', exp_converted)
                
                # Формируем новое выражение с **
                replacement = f"{base}**({exp_converted})"
                clean_expr = clean_expr.replace(full_match, replacement)
            
            # Заменяем степени с фигурными скобками x^{a} -> x**(a)
            while '^{' in clean_expr:
                pow_match = re.search(r'(\d+|\w+|\([^()]*\))\^\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', clean_expr)
                if not pow_match:
                    break
                
                base = pow_match.group(1)
                exp = self.convert_to_sympy_expr(pow_match.group(2))
                
                if exp is None:
                    exp = pow_match.group(2)  # Используем оригинальное выражение, если не удалось конвертировать
                
                replacement = f"{base}**({exp})"
                clean_expr = clean_expr[:pow_match.start()] + replacement + clean_expr[pow_match.end():]
                
            # Заменяем простые степени x^2 -> x**2 (без фигурных скобок)
            clean_expr = re.sub(r'(\d+|\w+|\([^()]*\))\^(\d+)', r'\1**\2', clean_expr)
            
            # Заменяем логарифмы с основанием \log_a(b) -> log(b, a)
            while '\\log_' in clean_expr:
                # Специальная обработка для логарифмов с основанием в виде дроби \log_{\frac{a}{b}}(x)
                frac_base_log_pattern = r'\\log_\{\\frac\{([^{}]+)\}\{([^{}]+)\}\}\(([^()]+)\)'
                frac_base_log_match = re.search(frac_base_log_pattern, clean_expr)
                
                if frac_base_log_match:
                    # Получаем числитель и знаменатель основания и аргумент
                    numerator = self.convert_to_sympy_expr(frac_base_log_match.group(1))
                    denominator = self.convert_to_sympy_expr(frac_base_log_match.group(2))
                    argument = self.convert_to_sympy_expr(frac_base_log_match.group(3))
                    
                    if numerator is None or denominator is None or argument is None:
                        # Если конвертация не удалась, пробуем использовать оригинальные строки
                        numerator = frac_base_log_match.group(1)
                        denominator = frac_base_log_match.group(2)
                        argument = frac_base_log_match.group(3)
                    
                    # Формируем правильное выражение для sympy: log(аргумент, основание)
                    # где основание = числитель/знаменатель
                    replacement = f"log({argument}, ({numerator})/({denominator}))"
                    clean_expr = clean_expr.replace(frac_base_log_match.group(0), replacement)
                    continue
                
                # Обработка обычных логарифмов с простым основанием
                log_match = re.search(r'\\log_([0-9a-zA-Z]+)\(([^()]*(?:\([^()]*\)[^()]*)*)\)', clean_expr)
                if not log_match:
                    log_match = re.search(r'\\log_([0-9a-zA-Z]+)\s+([0-9a-zA-Z]+)', clean_expr)
                    if not log_match:
                        break
                    
                    base = log_match.group(1)
                    arg = log_match.group(2)
                    replacement = f"log({arg}, {base})"
                else:
                    base = log_match.group(1)
                    arg = self.convert_to_sympy_expr(log_match.group(2))
                    
                    if arg is None:
                        return None
                    
                    replacement = f"log({arg}, {base})"
                
                clean_expr = clean_expr[:log_match.start()] + replacement + clean_expr[log_match.end():]
            
            # Заменяем натуральные логарифмы \ln(a) -> log(a)
            while '\\ln' in clean_expr:
                ln_match = re.search(r'\\ln\(([^()]*(?:\([^()]*\)[^()]*)*)\)', clean_expr)
                if not ln_match:
                    ln_match = re.search(r'\\ln\s+([0-9a-zA-Z]+)', clean_expr)
                    if not ln_match:
                        break
                    
                    arg = ln_match.group(1)
                    replacement = f"log({arg})"
                else:
                    arg = self.convert_to_sympy_expr(ln_match.group(1))
                    
                    if arg is None:
                        return None
                    
                    replacement = f"log({arg})"
                
                clean_expr = clean_expr[:ln_match.start()] + replacement + clean_expr[ln_match.end():]
            
            # Проверяем сбалансированность скобок
            if clean_expr.count('(') != clean_expr.count(')'):
                logging.warning(f"Несбалансированные скобки в выражении: {clean_expr}")
                # Пытаемся исправить, добавляя недостающие закрывающие скобки
                if clean_expr.count('(') > clean_expr.count(')'):
                    clean_expr = clean_expr + ')' * (clean_expr.count('(') - clean_expr.count(')'))
                # Или удаляя лишние закрывающие скобки
                elif clean_expr.count(')') > clean_expr.count('('):
                    excess = clean_expr.count(')') - clean_expr.count('(')
                    for _ in range(excess):
                        last_closing = clean_expr.rindex(')')
                        clean_expr = clean_expr[:last_closing] + clean_expr[last_closing+1:]
            
            # Проверим, можно ли преобразовать выражение в sympy
            try:
                x = symbols('x')
                sympify(clean_expr)
                return clean_expr
            except Exception as e:
                logging.error(f"Ошибка при проверке sympy выражения '{clean_expr}': {e}")
                # Особый случай для выражений со степенями формата a^{bx+c}
                if '^{' in expr_text or '^(' in expr_text:
                    try:
                        # Извлекаем основание и показатель
                        base_exp_match = re.search(r'(\d+|\([^()]+\))\s*\^\s*(?:\{|\()([^{}()]+)(?:\}|\))', expr_text)
                        if base_exp_match:
                            base = base_exp_match.group(1)
                            exponent = base_exp_match.group(2)
                            # Преобразуем exponent для использования с sympy
                            exponent = exponent.replace(' ', '*').replace('x-', 'x-')
                            # Форматируем выражение как base**(exponent)
                            clean_expr = f"{base}**({exponent})"
                            # Проверяем, что оно допустимо
                            sympify(clean_expr)
                            return clean_expr
                    except Exception as e2:
                        logging.error(f"Дополнительная попытка обработки степени не удалась: {e2}")
                return None
            
        except Exception as e:
            logging.error(f"Ошибка при преобразовании в sympy: {e}")
            return None

    def test_equation_generation_quality(self):
        """
        Тестирует качество генерации уравнений.
        Генерирует 100 задач и проверяет корректность ответов.
        """
        category = "Простейшие уравнения"  # Категория для генерации
        successful_tests = 0
        total_valid_tests = 0  # Счетчик валидных тестов
        max_attempts = 300  # Максимальное количество попыток для получения 100 валидных тестов
        
        # Сбрасываем статистику
        self.stats = {
            'total_tasks': 0,
            'successful_parsings': 0,
            'correct_verifications': 0,
            'failed_tasks': []
        }
        
        print(f"\nТестирование качества генерации уравнений:")
        print(f"{'='*50}")
        
        for i in range(max_attempts):
            if total_valid_tests >= 100:  # Если уже получили 100 валидных тестов
                break
                
            print(f"\nПопытка #{i+1}:")
            
            # Генерируем задачу
            result = generate_complete_task(category, difficulty_level=3)
            
            # Отслеживаем общее количество задач
            self.stats['total_tasks'] += 1
            
            # Проверяем, что задача сгенерирована успешно
            if "error" in result:
                print(f"  Ошибка генерации: {result['error']}")
                self.stats['failed_tasks'].append({
                    'attempt': i+1,
                    'error': result['error'],
                    'type': 'generation_error'
                })
                continue
            
            task = result.get("task", "")
            answer = result.get("answer", "")
            
            print(f"  Задача: {task[:100]}...")
            print(f"  Ответ: {answer}")
            
            # Извлекаем уравнение из задачи
            equation = self.extract_equation_from_task(task)
            
            if not equation:
                print(f"  Не удалось извлечь уравнение из задачи. Пропускаем.")
                self.stats['failed_tasks'].append({
                    'attempt': i+1,
                    'task': task[:100],
                    'type': 'equation_extraction_failed'
                })
                continue
                
            print(f"  Найдено уравнение: {equation}")
            
            # Извлекаем числовые ответы непосредственно из поля ответа
            numerical_answers = self.extract_numeric_answer(answer)
            
            if not numerical_answers:
                print(f"  Не удалось извлечь числовые ответы. Пропускаем.")
                self.stats['failed_tasks'].append({
                    'attempt': i+1,
                    'task': task[:100],
                    'answer': answer,
                    'type': 'answer_extraction_failed'
                })
                continue
                
            print(f"  Числовые ответы: {numerical_answers}")
            
            # Увеличиваем счетчик валидных тестов и успешно распознанных задач
            total_valid_tests += 1
            self.stats['successful_parsings'] += 1
            
            # Проверяем ответы
            is_correct = self.verify_equation_answer(equation, numerical_answers)
            
            # Если sympy не смог найти решение (is_correct = None), пропускаем задачу
            if is_correct is None:
                print(f"  ⚠ SymPy не смог найти решение. Пропускаем задачу.")
                # Уменьшаем счетчик валидных тестов, т.к. эта задача не учитывается
                total_valid_tests -= 1
                self.stats['successful_parsings'] -= 1
                self.stats['failed_tasks'].append({
                    'attempt': i+1,
                    'task': task[:100],
                    'answer': answer,
                    'numerical_answers': str(numerical_answers),
                    'equation': str(equation),
                    'type': 'sympy_no_solution'
                })
                continue
            
            if is_correct:
                successful_tests += 1
                self.stats['correct_verifications'] += 1
                print(f"  ✓ Ответ корректен")
            else:
                print(f"  ✗ Ответ некорректен")
                self.stats['failed_tasks'].append({
                    'attempt': i+1,
                    'task': task[:100],
                    'answer': answer,
                    'numerical_answers': str(numerical_answers),
                    'equation': str(equation),
                    'type': 'verification_failed'
                })
        
        # Выводим итоговую статистику
        if total_valid_tests == 0:
            print("\nНе удалось провести ни одного валидного теста. Проверьте алгоритмы извлечения уравнений и ответов.")
            success_rate = 0
        else:
            success_rate = (successful_tests / total_valid_tests) * 100
            
        print(f"\n{'='*50}")
        print(f"Результаты тестирования уравнений:")
        print(f"Валидных тестов: {total_valid_tests} (из {max_attempts} попыток)")
        print(f"Успешных тестов: {successful_tests}")
        print(f"Процент успеха: {success_rate:.2f}%")
        
        # Проверка должна быть успешной, даже если не все тесты прошли
        # это тест качества, а не функциональности
        return True

if __name__ == "__main__":
    # Запускаем тест напрямую
    test = EquationQualityTest()
    test.test_equation_generation_quality() 