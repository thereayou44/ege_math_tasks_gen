# Примеры кода для ключевых улучшений в системе тестирования

## 1. Обработка отрицательных чисел и дробей

```python
# Обработка отрицательных дробей -\frac{a}{b}
negative_frac_pattern = r'-\s*\\frac\{(\d+)\}\{(\d+)\}'
negative_fractions = []

for match in re.finditer(negative_frac_pattern, answer_text):
    numerator = float(match.group(1))
    denominator = float(match.group(2))
    negative_fractions.append(-1 * (numerator / denominator))
```

## 2. Обработка сложных дробей с корнями

```python
# Обработка сложных дробей вида \frac{1 + 3\sqrt{17}}{2}
complex_frac_pattern = r'\\frac\{([^{}]*?)(\d+)?\\sqrt\{(\d+)\}([^{}]*)\}\{(\d+)\}'
for match in re.finditer(complex_frac_pattern, answer_text):
    prefix = match.group(1).strip()  # Текст перед корнем (например, "1 + ")
    coef_str = match.group(2) or "1"  # Коэффициент перед корнем (например, "3")
    sqrt_value = int(match.group(3))  # Число под корнем (например, "17")
    suffix = match.group(4).strip()  # Текст после корня (например, " - 5")
    denominator = int(match.group(5))  # Знаменатель (например, "2")
    
    # Извлекаем коэффициент перед корнем
    coef = int(coef_str)
    
    # Инициализируем результат
    result = 0
    
    # Обрабатываем префикс и извлекаем число
    if prefix:
        prefix_number_match = re.search(r'(\d+)', prefix)
        if prefix_number_match:
            prefix_number = int(prefix_number_match.group(1))
            # Определяем знак (по умолчанию +)
            if "-" in prefix:
                prefix_number = -prefix_number
            result += prefix_number
    
    # Добавляем значение корня с коэффициентом
    if "+" in prefix or not prefix:
        result += coef * math.sqrt(sqrt_value)
    elif "-" in prefix:
        result -= coef * math.sqrt(sqrt_value)
    
    # Обрабатываем суффикс
    if suffix:
        suffix_number_match = re.search(r'(\d+)', suffix)
        if suffix_number_match:
            suffix_number = int(suffix_number_match.group(1))
            if "-" in suffix:
                result -= suffix_number
            elif "+" in suffix:
                result += suffix_number
    
    # Делим результат на знаменатель
    result = result / denominator
    sqrt_values.append(result)
```

## 3. Улучшенная обработка степеней в LaTeX формате

```python
# Обработка показательных функций вида a^{bx+c}
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
```

## 4. Обработка логарифмов с дробными основаниями

```python
# Обработка логарифмов с основанием в виде дроби \log_{\frac{a}{b}}(x)
frac_base_log_pattern = r'\\log_\{\\frac\{([^{}]+)\}\{([^{}]+)\}\}\(([^()]+)\)'
frac_base_log_match = re.search(frac_base_log_pattern, clean_expr)

if frac_base_log_match:
    # Получаем числитель и знаменатель основания и аргумент
    numerator = self.convert_to_sympy_expr(frac_base_log_match.group(1))
    denominator = self.convert_to_sympy_expr(frac_base_log_match.group(2))
    argument = self.convert_to_sympy_expr(frac_base_log_match.group(3))
    
    # Формируем правильное выражение для sympy: log(аргумент, основание)
    # где основание = числитель/знаменатель
    replacement = f"log({argument}, ({numerator})/({denominator}))"
    clean_expr = clean_expr.replace(frac_base_log_match.group(0), replacement)
```

## 5. Гибкая проверка ответов для уравнений

```python
# Гибкая проверка ответов с несколькими критериями успеха
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
```

## 6. Улучшенная проверка интервалов для неравенств

```python
# Проверка приблизительного равенства интервалов с допуском погрешности
def approx_equal(a, b, tolerance=1e-6):
    if a in (S.Infinity, S.NegativeInfinity) or b in (S.Infinity, S.NegativeInfinity):
        return a == b
    return abs(float(a) - float(b)) < tolerance

# Проверка на различных критериях
# Случай 1: Полное совпадение интервалов
if solution_set.is_subset(user_interval) and user_interval.is_subset(solution_set):
    logging.info(f"Интервалы полностью совпадают")
    return True

# Случай 2: Оценка по четырем параметрам (левая граница, правая граница, тип левой, тип правой)
left_close = approx_equal(solution_bounds[0], user_bounds[0]) or (
    solution_bounds[0] == float('-inf') and user_bounds[0] <= -1e6)
right_close = approx_equal(solution_bounds[1], user_bounds[1]) or (
    solution_bounds[1] == float('inf') and user_bounds[1] >= 1e6)

left_type_match = solution_set.left_open == (not is_min_inclusive)
right_type_match = solution_set.right_open == (not is_max_inclusive)

match_score = sum([left_close, right_close, left_type_match, right_type_match]) / 4.0
if match_score >= 0.75:  # Если совпадают хотя бы 3 из 4 параметров
    return True
```

## 7. Обработка тригонометрических выражений с pi

```python
# Обработка тригонометрических выражений с константой pi
trig_funcs = ['sin', 'cos', 'tan', 'cot', 'sec', 'csc', 'arcsin', 'arccos', 'arctan']
for func in trig_funcs:
    func_pattern = fr'\\{func}\s*\(([^)]*pi[^)]*)\)'
    matches = re.finditer(func_pattern, equation)
    for match in matches:
        arg = match.group(1)
        # Добавляем умножение между pi и скобками
        new_arg = re.sub(r'pi\s*\(', 'pi*(', arg)
        # Также обрабатываем случаи pi x -> pi*x
        new_arg = re.sub(r'pi\s+([a-zA-Z])', r'pi*\1', new_arg)
        equation = equation.replace(match.group(0), f'\\{func}({new_arg})')
```

## 8. Исправление несбалансированных скобок

```python
# Проверка и исправление несбалансированных скобок
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
``` 