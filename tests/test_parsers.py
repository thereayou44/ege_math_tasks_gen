import unittest
import sys
import os
import re
from bs4 import BeautifulSoup

# Добавляем родительскую директорию в пути для импорта
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from task_generator import (
    extract_text_and_formulas,
    extract_answer_with_latex,
    parse_hints,
    convert_markdown_to_html,
    fix_html_tags,
    html_to_markdown
)

class TestParsers(unittest.TestCase):
    """Тесты для функций парсинга и обработки текста."""
    
    def test_extract_text_and_formulas(self):
        """Проверяет извлечение текста и формул из HTML."""
        # HTML с формулой в alt атрибуте изображения
        html = """
        <p>Решите уравнение: <img src="formula.png" alt="x^2 + 3x - 4 = 0"></p>
        """
        
        # Извлекаем текст и формулы
        result = extract_text_and_formulas(html)
        
        # Проверяем результат
        self.assertIn("Решите уравнение:", result)
        self.assertIn("$$ x^2 + 3x - 4 = 0 $$", result)
        
        # Пустой HTML
        self.assertEqual(extract_text_and_formulas(""), "")
    
    def test_extract_answer_with_latex(self):
        """Проверяет извлечение ответа из решения с формулами LaTeX."""
        # Простой ответ
        solution = "Проводим вычисления и получаем. Ответ: 42."
        self.assertEqual(extract_answer_with_latex(solution), "42")
        
        # Ответ с формулой LaTeX
        solution = "Ответ: \\frac{1}{2}."
        # Обратите внимание, что функция оборачивает LaTeX-выражения в двойные доллары $$
        self.assertEqual(extract_answer_with_latex(solution), "$$\\frac{1}{2}$$")
        
        # Ответ с альтернативным форматированием
        solution = "Решаем уравнение... Итоговый ответ: 3x + 2."
        self.assertEqual(extract_answer_with_latex(solution), "3x + 2")
        
        # Решение без ответа
        solution = "Здесь нет явного ответа."
        self.assertEqual(extract_answer_with_latex(solution), "См. решение")
    
    def test_parse_hints(self):
        """Проверяет разбор подсказок из текста."""
        # Нумерованные подсказки
        hints_text = "1. Первая подсказка\n2. Вторая подсказка\n3. Третья подсказка"
        hints = parse_hints(hints_text)
        self.assertEqual(len(hints), 3)
        self.assertEqual(hints[0], "Первая подсказка")
        self.assertEqual(hints[1], "Вторая подсказка")
        self.assertEqual(hints[2], "Третья подсказка")
        
        # Подсказки на отдельных строках без нумерации
        hints_text = "Первая подсказка\nВторая подсказка\nТретья подсказка"
        hints = parse_hints(hints_text)
        self.assertEqual(len(hints), 3)
        self.assertEqual(hints[0], "Первая подсказка")
        
        # Меньше 3 подсказок
        hints_text = "Единственная подсказка"
        hints = parse_hints(hints_text)
        self.assertEqual(len(hints), 3)
        self.assertEqual(hints[0], "Единственная подсказка")
        self.assertEqual(hints[1], "Подсказка недоступна")
        
        # Пустые подсказки
        hints = parse_hints("")
        self.assertEqual(hints, ["Подсказка недоступна", "Подсказка недоступна", "Подсказка недоступна"])
    
    def test_convert_markdown_to_html(self):
        """Проверяет конвертацию Markdown в HTML."""
        # Жирный и курсивный текст
        markdown = "Это **жирный** текст и *курсивный* текст."
        html = convert_markdown_to_html(markdown)
        self.assertIn("<b>жирный</b>", html)
        self.assertIn("<i>курсивный</i>", html)
        
        # Абзацы
        markdown = "Первый абзац.\n\nВторой абзац."
        html = convert_markdown_to_html(markdown)
        self.assertIn("</p><p>", html)
        
        # Переносы строк
        markdown = "Первая строка.\nВторая строка."
        html = convert_markdown_to_html(markdown)
        self.assertIn("<br>", html)
    
    def test_fix_html_tags(self):
        """Проверяет исправление HTML-тегов."""
        # Незакрытые теги
        html = "<p>Текст с <b>жирным шрифтом, но незакрытым тегом."
        fixed_html = fix_html_tags(html)
        self.assertIn("</b>", fixed_html)
        
        # Отсутствие тегов параграфа
        html = "Текст без тегов параграфа."
        fixed_html = fix_html_tags(html)
        self.assertTrue(fixed_html.startswith("<p>"))
        self.assertTrue(fixed_html.endswith("</p>"))
        
        # Смешанный текст с HTML и обычным текстом
        # Обратите внимание: функция fix_html_tags не оборачивает каждый абзац в <p>,
        # а только проверяет на нужность добавления тегов во всём тексте
        html = "<p>Первый параграф</p>\nВторой параграф без тегов."
        fixed_html = fix_html_tags(html)
        # Проверяем, что первый параграф сохранился
        self.assertIn("<p>Первый параграф</p>", fixed_html)
        # Проверяем, что весь текст завершается </p>
        self.assertTrue(fixed_html.endswith("</p>"), "Текст должен завершаться закрывающим тегом </p>")
        # Проверяем, что второй абзац тоже присутствует в исправленном тексте
        self.assertIn("Второй параграф без тегов", fixed_html)
    
    def test_html_to_markdown(self):
        """Проверяет конвертацию HTML в Markdown."""
        # Простой HTML
        html = "<p>Обычный текст</p>"
        markdown = html_to_markdown(html)
        self.assertEqual(markdown, "Обычный текст")
        
        # HTML с жирным и курсивным текстом
        html = "<p>Текст с <b>жирным</b> и <i>курсивным</i> форматированием.</p>"
        markdown = html_to_markdown(html)
        self.assertIn("**жирным**", markdown)
        self.assertIn("*курсивным*", markdown)
        
        # HTML с математическими формулами
        html = "<p>Формула: $\\frac{1}{2}$</p>"
        markdown = html_to_markdown(html)
        self.assertIn("$\\frac{1}{2}$", markdown)
        
        # Пустой HTML
        self.assertEqual(html_to_markdown(""), "")

if __name__ == "__main__":
    unittest.main() 