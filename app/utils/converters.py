import re
from bs4 import BeautifulSoup
import json

def convert_markdown_to_html(markdown_text):
    """
    Конвертирует текст из формата Markdown в HTML
    
    Args:
        markdown_text: Текст в формате Markdown
        
    Returns:
        str: Текст в формате HTML
    """
    if not markdown_text:
        return ""
    
    # Заменяем переносы строк на теги параграфов, исключая формулы
    paragraphs = re.split(r'\n\s*\n', markdown_text)
    
    for i in range(len(paragraphs)):
        # Обрабатываем только если параграф не содержит блок кода
        if not re.search(r'```', paragraphs[i]):
            # Заменяем одиночные переносы строк на <br>
            paragraphs[i] = paragraphs[i].replace('\n', '<br>')
            
            # Оборачиваем параграф в теги <p>, если он не начинается с тега
            if not re.search(r'^\s*<\w+', paragraphs[i]):
                paragraphs[i] = f"<p>{paragraphs[i]}</p>"
    
    html = '\n'.join(paragraphs)
    
    # Полужирный текст
    html = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', html)
    
    # Курсив
    html = re.sub(r'\*(.*?)\*', r'<em>\1</em>', html)
    
    # Подчеркивание
    html = re.sub(r'__(.*?)__', r'<u>\1</u>', html)
    
    # Маркированные списки
    html = re.sub(r'^\s*[-*]\s+(.*?)$', r'<li>\1</li>', html, flags=re.MULTILINE)
    html = re.sub(r'(<li>.*?</li>\s*)+', r'<ul>\g<0></ul>', html, flags=re.DOTALL)
    
    # Нумерованные списки
    html = re.sub(r'^\s*(\d+)\.\s+(.*?)$', r'<li>\2</li>', html, flags=re.MULTILINE)
    html = re.sub(r'(<li>.*?</li>\s*)+', r'<ol>\g<0></ol>', html, flags=re.DOTALL)
    
    # Блоки кода
    html = re.sub(r'```(.*?)```', r'<pre><code>\1</code></pre>', html, flags=re.DOTALL)
    
    # Исправляем вложенность списков
    html = fix_html_tags(html)
    
    return html

def fix_html_tags(html):
    """
    Исправляет проблемы с вложенностью HTML-тегов
    
    Args:
        html: HTML-текст с возможными проблемами
        
    Returns:
        str: Исправленный HTML
    """
    try:
        soup = BeautifulSoup(html, 'html.parser')
        return str(soup)
    except Exception as e:
        print(f"Ошибка при исправлении HTML: {e}")
        return html

def convert_html_to_markdown(html_text):
    """
    Конвертирует текст из формата HTML в Markdown
    
    Args:
        html_text: Текст в формате HTML
        
    Returns:
        str: Текст в формате Markdown
    """
    if not html_text:
        return ""
    
    soup = BeautifulSoup(html_text, 'html.parser')
    
    # Удаляем ненужные атрибуты
    for tag in soup.find_all(True):
        tag.attrs = {}
    
    # Заменяем теги на Markdown
    for strong in soup.find_all('strong'):
        strong.replace_with(f"**{strong.get_text()}**")
    
    for em in soup.find_all('em'):
        em.replace_with(f"*{em.get_text()}*")
    
    for u in soup.find_all('u'):
        u.replace_with(f"_{u.get_text()}_")
    
    # Обрабатываем списки
    for ul in soup.find_all('ul'):
        for li in ul.find_all('li'):
            li.replace_with(f"- {li.get_text()}\n")
    
    for ol in soup.find_all('ol'):
        for i, li in enumerate(ol.find_all('li')):
            li.replace_with(f"{i+1}. {li.get_text()}\n")
    
    # Заменяем <br> на переносы строк
    for br in soup.find_all('br'):
        br.replace_with('\n')
    
    # Параграфы
    for p in soup.find_all('p'):
        p.replace_with(f"{p.get_text()}\n\n")
    
    # Блоки кода
    for pre in soup.find_all('pre'):
        code = pre.find('code')
        if code:
            pre.replace_with(f"```\n{code.get_text()}\n```\n")
    
    markdown = soup.get_text()
    
    # Исправляем лишние переносы строк
    markdown = re.sub(r'\n{3,}', '\n\n', markdown)
    
    return markdown

def format_task_to_json(task, solution, hints, answer="", difficulty_level=3, is_basic_level=False, category="", subcategory=""):
    """
    Форматирует задачу и ее решение в формат JSON
    
    Args:
        task: Текст задачи
        solution: Решение задачи
        hints: Список подсказок
        answer: Ответ к задаче
        difficulty_level: Уровень сложности
        is_basic_level: Тип экзамена (базовый/профильный)
        category: Категория задачи
        subcategory: Подкатегория задачи
        
    Returns:
        str: Строка в формате JSON
    """
    data = {
        "task": task,
        "solution": solution,
        "hints": hints,
        "answer": answer,
        "difficulty_level": difficulty_level,
        "is_basic_level": is_basic_level,
        "category": category,
        "subcategory": subcategory
    }
    
    return json.dumps(data, ensure_ascii=False, indent=2) 