## Форматирование формул для LaTeX-отображения в HTML

import re


def format_latex_answer(answer):
    """
    Форматирует ответ с правильным LaTeX-окружением.
    
    Args:
        answer: Исходный ответ
        
    Returns:
        str: Отформатированный ответ
    """
    if not answer or answer.strip() == "":
        return "Ответ не найден"
        
    # Удаляем возможные префиксы и суффиксы, такие как "Ответ: "
    answer = re.sub(r'^(?:Ответ|ОТВЕТ|ответ|Окончательный ответ|Итоговый ответ)\s*:\s*', '', answer)
    answer = answer.strip()
    
    # Если у нас многострочный ответ, обрабатываем каждую строку отдельно
    if '\n' in answer:
        lines = answer.split('\n')
        formatted_lines = []
        for line in lines:
            line = line.strip()
            if line:
                # Если строка уже содержит LaTeX, добавляем её как есть
                if '$' in line:
                    formatted_lines.append(line)
                else:
                    # Для каждой обычной строки проверяем, нужно ли её обернуть в $
                    formatted_line = format_latex_answer(line)
                    formatted_lines.append(formatted_line)
        
        return '\n'.join(formatted_lines)
    
    # Если ответ уже содержит LaTeX-окружение, сохраняем его как есть
    if answer.startswith('$') and answer.endswith('$'):
        # Проверяем, что $ встречается только в начале и конце ответа
        if answer.count('$') == 2:
            return answer
        
    # Если ответ уже содержит $$, заменяем их на $ для единообразия
    if '$$' in answer:
        # Удаляем все $$
        answer = answer.replace('$$', '')
        # Оборачиваем снова в $
        answer = f"${answer}$"
    
    # Обработка чисел 
    # Специальная обработка для отрицательных чисел
    neg_number_match = re.match(r'^-\s*(\d+[.,]?\d*)$', answer)
    if neg_number_match:
        num = neg_number_match.group(1).replace(',', '.')
        return f"${num}$"
    
    # Если у нас просто число, форматируем его как LaTeX
    number_match = re.match(r'^([+-]?\d*[.,]?\d+)$', answer)
    if number_match:
        # Заменяем запятые на точки для единообразия чисел
        num = number_match.group(1).replace(',', '.')
        return f"${num}$"
    
    # Если ответ уже содержит одинарные $, но внутри есть текст, очищаем текст
    if '$' in answer:
        # Извлекаем содержимое между долларами
        latex_parts = re.findall(r'\$(.*?)\$', answer)
        if latex_parts:
            # Берем первое LaTeX выражение
            cleaned_latex = latex_parts[0].strip()
            return f"${cleaned_latex}$"
    
    # Проверяем наличие дробей в форматах a/b или \frac{a}{b}
    frac_patterns = [
        r'\\frac\{([^{}]+)\}\{([^{}]+)\}',  # \frac{a}{b}
        r'(\d+)/(\d+)'                      # a/b
    ]
    
    for pattern in frac_patterns:
        frac_match = re.search(pattern, answer)
        if frac_match:
            if pattern.startswith(r'\\frac'):
                numerator, denominator = frac_match.groups()
                return f"$\\frac{{{numerator}}}{{{denominator}}}$"
            else:
                numerator, denominator = frac_match.groups()
                return f"$\\frac{{{numerator}}}{{{denominator}}}$"
    
    # Если в ответе есть специальные символы LaTeX, оборачиваем в $
    latex_symbols = ['\\', '^', '_', '{', '}', '\\sqrt', '\\pi', '\\cdot', '\\times', '\\div', '\\in', '\\subset', '\\cup', '\\cap']
    if any(symbol in answer for symbol in latex_symbols):
        return f"${answer}$"
    
    # Если в ответе есть математические множества или интервалы, оборачиваем в $
    if any(symbol in answer for symbol in ['(', ')', '[', ']', '∈', '⊂', '∪', '∩']):
        return f"${answer}$"
    
    # Для всех остальных случаев проверяем, является ли ответ числом или простым выражением
    answer_cleaned = answer.strip()
    
    # Если ответ выглядит как число (целое или с десятичной точкой)
    if re.match(r'^[+-]?\d*[.,]?\d+$', answer_cleaned):
        # Заменяем запятые на точки для единообразия
        answer_cleaned = answer_cleaned.replace(',', '.')
        return f"${answer_cleaned}$"
    
    # По умолчанию оборачиваем ответ в $, если он не пустой
    return f"${answer_cleaned}$" if answer_cleaned else "Ответ не найден"

def extract_answer_from_solution(solution):
    """
    Извлекает ответ из решения и корректирует отображение LaTeX.
    
    Args:
        solution: Полное решение задачи
        
    Returns:
        str: Правильно отформатированный ответ
    """
    if not solution or len(solution.strip()) < 10:
        return "Ответ не найден"
    
    # Проверяем особый формат с ---ОТВЕТ---
    special_pattern = r'[-]{2,}\s*(?:ОТВЕТ|Ответ|ответ)\s*[-]{2,}\s*(.*?)(?:$|\n\s*-|\n\s*\n)'
    special_match = re.search(special_pattern, solution, re.IGNORECASE | re.DOTALL)
    
    if special_match:
        answer = special_match.group(1).strip()
        # Ищем число в ответе, если это просто число
        number_pattern = r'-?\d+[.,]\d+|-?\d+'
        number_matches = re.findall(number_pattern, answer)
        if number_matches and len(number_matches) == 1:
            answer = number_matches[0]
            
        # Применяем форматирование LaTeX
        return format_latex_answer(answer)
        
    # Ищем "Ответ:" или "Ответ :" с различными вариантами
    answer_patterns = [
        '(?:Ответ|ОТВЕТ|ответ|Окончательный ответ|Итоговый ответ)\\s*:(.+?)(?=$|\\.|\\.\s*$|\\n\\s*\\n)',
        '\\n\\s*(?:Ответ|ОТВЕТ|ответ|Окончательный ответ|Итоговый ответ)\\s*:\\s*(.+?)(?=$|\\n\\s*\\n|\\n[0-9])',
        '\\n[0-9]+\\.\\s*(?:Ответ|ОТВЕТ|ответ|Окончательный ответ|Итоговый ответ)\\s*:\\s*(.+?)(?=$|\\n\\s*\\n|\\n[0-9])'
    ]
    
    for pattern in answer_patterns:
        answer_match = re.search(pattern, solution, re.IGNORECASE | re.DOTALL)
        if answer_match:
            answer = answer_match.group(1).strip()
            return format_latex_answer(answer)
    
    # Если ответ не найден, пробуем более широкий поиск для многострочных ответов с отступами
    multiline_patterns = [
        '\\n\\s*\\d+\\.\\s*(?:Ответ|ОТВЕТ|ответ|Окончательный ответ|Итоговый ответ)\\s*:\\s*\\n\\s*(.+?)(?=$|\\n\\s*\\n|\\n\\s*\\d+\\.)',
        '(?:Ответ|ОТВЕТ|ответ|Окончательный ответ|Итоговый ответ)\\s*:\\s*\\n\\s*(.+?)(?=$|\\n\\s*\\n)'
    ]
    
    for pattern in multiline_patterns:
        answer_match = re.search(pattern, solution, re.IGNORECASE | re.DOTALL)
        if answer_match:
            answer = answer_match.group(1).strip()
            return format_latex_answer(answer)
    
    # Если ответ не найден по предварительно заданным шаблонам
    return "См. решение"

def normalize_latex_in_text(text):
    """
    Нормализует LaTeX в тексте, заменяя различные форматы окружений на стандартный.
    
    Args:
        text: Текст с LaTeX-формулами
        
    Returns:
        str: Текст с нормализованными LaTeX-формулами
    """
    if not text:
        return ""
    
    # Заменяем \begin{equation} и \end{equation} на $$
    text = text.replace('\\begin{equation}', '$$')
    text = text.replace('\\end{equation}', '$$')
    
    # Нормализация LaTeX выражений с переносами строк
    # Заменяем \[...\] на $$...$$
    text = text.replace('\\[', '$$')
    text = text.replace('\\]', '$$')
    
    # Также заменяем \( и \) на $ если они есть
    text = text.replace('\\(', '$')
    text = text.replace('\\)', '$')
    
    # Исправляем команды \frac с переносами строк
    text = re.sub('\\\\frac\\s*\\n*\\s*\\{\\s*\\n*\\s*([^{}]+?)\\s*\\n*\\s*\\}\\s*\\n*\\s*\\{\\s*\\n*\\s*([^{}]+?)\\s*\\n*\\s*\\}', 
                 r'\\frac{\1}{\2}', text)
    
    # Исправляем другие фрагменты LaTeX с переносами строк
    text = re.sub('\\\\left\\s*\\n*\\s*([\\\\({[])', r'\\left\1', text)
    text = re.sub('\\\\right\\s*\\n*\\s*([\\\\)}\\]])', r'\\right\1', text)
    
    # Корректируем выравнивание в системах уравнений
    text = re.sub('\\\\begin\\s*\\n*\\s*\\{\\s*\\n*\\s*aligned\\s*\\n*\\s*\\}', r'\\begin{aligned}', text)
    text = re.sub('\\\\end\\s*\\n*\\s*\\{\\s*\\n*\\s*aligned\\s*\\n*\\s*\\}', r'\\end{aligned}', text)
    
    # Корректируем отображение систем уравнений
    text = re.sub('\\\\left\\s*\\\\\\s*\\{\\s*\\\\begin\\s*\\{\\s*aligned\\s*\\}', r'\\left\\{\\begin{aligned}', text)
    text = re.sub('\\\\end\\s*\\{\\s*aligned\\s*\\}\\s*\\\\right\\s*\\\\.', r'\\end{aligned}\\right\\.', text)
    
    # Исправляем команды \left и \right с переносами строк
    text = re.sub('\\\\left\\s*\\n*\\s*\\\\?\\s*\\{\\s*\\n*\\s*', r'\\left\\{', text)
    text = re.sub('\\s*\\n*\\s*\\\\right\\s*\\n*\\s*\\\\?\\s*\\.\\s*\\n*\\s*', r'\\right\\.', text)
    
    return text

def escape_html_in_text(text):
    """
    Экранирует HTML-специальные символы в тексте, но сохраняет LaTeX-формулы.
    
    Args:
        text: Текст с возможными HTML-символами и LaTeX-формулами
        
    Returns:
        str: Текст с экранированными HTML-символами вне LaTeX-формул
    """
    if not text:
        return ""
    
    # Заменяем символы < и > на HTML-сущности вне формул
    result = ""
    in_formula = False
    i = 0
    
    while i < len(text):
        if text[i:i+1] == "$":
            in_formula = not in_formula
            result += text[i]
            i += 1
        elif not in_formula and text[i] == "<":
            result += "&lt;"
            i += 1
        elif not in_formula and text[i] == ">":
            result += "&gt;"
            i += 1
        else:
            result += text[i]
            i += 1
    
    return result

def format_solution_punctuation(html_content):
    """
    Форматирует знаки препинания после математических формул.
    
    Args:
        html_content: HTML-содержимое с LaTeX-формулами
        
    Returns:
        str: HTML-содержимое с отформатированными знаками препинания
    """
    if not html_content:
        return ""
    
    # Обрабатываем знаки препинания после инлайновых формул
    html_content = re.sub('(\\$[^\\$]+\\$)([,.;:])', r'\1<span class="math-punctuation">\2</span>', html_content)
    
    # Обрабатываем знаки препинания после блочных формул
    html_content = re.sub('(\\$\$[^\\$]+\\$\\$)([,.;:])', r'\1<span class="math-punctuation">\2</span>', html_content)
    
    return html_content 