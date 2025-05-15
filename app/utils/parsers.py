import re

def extract_answer_from_solution(solution_text):
    """
    Извлекает ответ из текста решения
    
    Args:
        solution_text: Текст решения
        
    Returns:
        str: Извлеченный ответ или пустую строку, если ответ не найден
    """
    if not solution_text:
        return ""
    
    # Ищем ответ в формате "Ответ: ..."
    answer_match = re.search(r'(?:Ответ|ОТВЕТ|ответ)\s*:(.+?)(?=$|\.|\.\s*$|\n\s*\n|\<\/p\>)', 
                            solution_text, re.IGNORECASE | re.DOTALL)
    
    if answer_match:
        return answer_match.group(1).strip()
    
    return ""

def extract_answer_with_latex(answer_text):
    """
    Проверяет и форматирует ответ, содержащий LaTeX-формулы
    
    Args:
        answer_text: Текст ответа
        
    Returns:
        str: Отформатированный ответ с корректно обернутыми LaTeX-формулами
    """
    if not answer_text or answer_text.strip() == "":
        return ""
    
    # Убираем \( и \) из ответа и заменяем на доллары, если они есть
    if '\\(' in answer_text and '\\)' in answer_text:
        answer_text = answer_text.replace('\\(', '$').replace('\\)', '$')
    
    # Если ответ уже содержит знаки доллара, считаем что LaTeX уже оформлен
    if '$' in answer_text:
        # Проверяем парность долларов
        open_count = answer_text.count('$')
        if open_count % 2 != 0:
            answer_text += '$'  # Добавляем закрывающий доллар если не хватает
        return answer_text
    
    # Расширенный список LaTeX-команд для проверки
    latex_patterns = [
        r'\\frac', r'\\sqrt', r'\\sum', r'\\prod', r'\\int', 
        r'\\lim', r'\\sin', r'\\cos', r'\\tan', r'\\log', r'\\ln',
        r'\\alpha', r'\\beta', r'\\gamma', r'\\delta', r'\\pi',
        r'\\le', r'\\ge', r'\\neq', r'\\approx', r'\\cdot',
        r'\\left', r'\\right', r'\\mathbb', r'\\mathcal', r'\\partial',
        r'\\begin\{.*?\}', r'\\end\{.*?\}', r'\\overline', r'\\underline',
        r'\\times', r'\\div', r'\\equiv', r'\\cup', r'\\cap', r'\\in', r'\\infty',
        r'\\tg', r'\\ctg', r'\\arctg'
    ]
    
    # Ищем любой паттерн LaTeX в ответе
    has_latex = any(re.search(pattern, answer_text) for pattern in latex_patterns)
    
    # Если ответ содержит LaTeX-команды, но не обернут в доллары
    if has_latex and '$' not in answer_text:
        return f"${answer_text}$"
    
    # Экранируем угловые скобки, если они не являются частью HTML-тега
    if '<' in answer_text and not re.search(r'<[a-z/]', answer_text):
        answer_text = answer_text.replace('<', '&lt;').replace('>', '&gt;')
    
    return answer_text

def extract_hints_from_text(text, num_hints=3):
    """
    Извлекает подсказки из текста
    
    Args:
        text: Текст, содержащий подсказки
        num_hints: Количество требуемых подсказок
        
    Returns:
        list: Список извлеченных подсказок
    """
    if not text:
        return [""] * num_hints
    
    # Ищем подсказки в формате "1. Подсказка"
    hints_match = re.findall(r'(?:\d+\.\s*|\*\s*)(.*?)(?=\n\d+\.\s*|\n\*\s*|\n\n|$)', 
                             text, re.DOTALL)
    
    # Если не нашли подсказки в пронумерованном формате, пробуем разделить по абзацам
    if not hints_match:
        hints_match = re.split(r'\n\s*\n', text)
    
    # Очищаем подсказки от лишних пробелов
    hints = [hint.strip() for hint in hints_match if hint.strip()]
    
    # Дополняем список подсказок до указанного количества, если нужно
    while len(hints) < num_hints:
        hints.append("Подсказка недоступна")
    
    # Если подсказок больше, чем нужно, берем только требуемое количество
    hints = hints[:num_hints]
    
    return hints

def extract_task_parameters(task_text):
    """
    Извлекает параметры из текста задачи (например, координаты точек, коэффициенты уравнений и т.д.)
    
    Args:
        task_text: Текст задачи
        
    Returns:
        dict: Словарь с извлеченными параметрами
    """
    params = {}
    
    # Ищем уравнения
    equations = re.findall(r'\\?[bfz]?\(([^)]+)\)|(y\s*=\s*[^,.:;]+)|(x\s*=\s*[^,.:;]+)', task_text)
    if equations:
        params['equations'] = [eq[0] or eq[1] or eq[2] for eq in equations if eq[0] or eq[1] or eq[2]]
    
    # Ищем координаты точек
    coordinates = re.findall(r'\((-?\d+(?:[,.]\d+)?)\s*[;,]\s*(-?\d+(?:[,.]\d+)?)\)', task_text)
    if coordinates:
        params['points'] = [{'x': float(x.replace(',', '.')), 'y': float(y.replace(',', '.'))} 
                           for x, y in coordinates]
    
    # Ищем числовые значения с единицами измерения
    measurements = re.findall(r'(\d+(?:[,.]\d+)?)\s*(см|м|км|кг|г|л|мл|градус(?:ов|а)?)', task_text)
    if measurements:
        params['measurements'] = [{'value': float(value.replace(',', '.')), 'unit': unit} 
                                 for value, unit in measurements]
    
    return params 