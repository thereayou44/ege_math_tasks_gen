## Форматирование формул для LaTeX-отображения в HTML

import re


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