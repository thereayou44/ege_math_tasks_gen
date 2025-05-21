import re
import logging
from bs4 import BeautifulSoup

class ModelResponseProcessor:
    """
    Класс для обработки ответов модели.
    Извлекает блоки задачи, решения, подсказок и преобразует в HTML.
    """
    
    def __init__(self, raw_response):
        """Инициализирует процессор с сырым ответом модели"""
        self.raw_response = raw_response
        self.parsed_data = self._parse_response()
    
    def _parse_response(self):
        """Парсит ответ модели и извлекает блоки"""
        data = {
            "task": self._extract_block("ЗАДАЧА"),
            "solution": self._extract_block("РЕШЕНИЕ"),
            "hints": self._parse_hints(self._extract_block("ПОДСКАЗКИ")),
            "visualization_params": self._extract_block("ПАРАМЕТРЫ ДЛЯ ВИЗУАЛИЗАЦИИ")
        }
        
        # Проверяем, если блоки не найдены, то пробуем альтернативный формат
        if not data["task"]:
            data["task"] = self._extract_block("УСЛОВИЕ")
        if not data["visualization_params"]:
            data["visualization_params"] = self._extract_block("ПАРАМЕТРЫ ВИЗУАЛИЗАЦИИ")
        
        # Извлекаем ответ из решения
        data["answer"] = self._extract_answer(data["solution"])
        
        # Удаляем строку с ответом из решения
        if data["solution"] and data["answer"]:
            data["solution"] = self._remove_answer_from_solution(data["solution"])
        
        return data
    
    def _extract_block(self, block_name):
        """Извлекает блок текста между маркерами ---НАЗВАНИЕ_БЛОКА---"""
        pattern = f"---{block_name}---\\s*(.*?)(?=\\s*---[A-ZА-Я _]+---|\s*$)"
        match = re.search(pattern, self.raw_response, re.DOTALL)
        if match:
            return match.group(1).strip()
            
        # Альтернативный вариант поиска блоков
        alt_pattern = f"===(?:{block_name})===\\s*(.*?)(?=\\s*===|$)"
        alt_match = re.search(alt_pattern, self.raw_response, re.DOTALL | re.IGNORECASE)
        return alt_match.group(1).strip() if alt_match else ""
    
    def _extract_answer(self, solution):
        """Извлекает ответ из решения"""
        if not solution:
            return ""
        
        # Ищем ответ в блоке решения с различными вариантами форматирования
        answer_pattern = r'(?:Ответ|ОТВЕТ|ответ)\s*:(.+?)(?=$|\.|\.\s*$|\n\s*\n)'
        match = re.search(answer_pattern, solution, re.IGNORECASE | re.DOTALL)
        
        # Если нашли ответ в решении, возвращаем его
        if match:
            return match.group(1).strip()
            
        # Если не нашли в решении, ищем отдельный блок с ответом
        answer_block = self._extract_block("ОТВЕТ")
        if answer_block:
            # Удаляем возможный префикс "Ответ:"
            answer_block = re.sub(r'^(?:Ответ|ОТВЕТ|ответ)\s*:\s*', '', answer_block)
            return answer_block.strip()
            
        return ""
    
    def _parse_hints(self, hints_string):
        """Разделяет строку с подсказками на отдельные подсказки"""
        if not hints_string:
            return ["Подсказка недоступна"] * 3
        
        # Ищем подсказки в формате "1. [текст]"
        hint_pattern = r'(?:^|\n)(\d+\.)\s*(.*?)(?=(?:\n\d+\.)|$)'
        hint_matches = re.findall(hint_pattern, hints_string, re.DOTALL)
        
        if hint_matches:
            hints = [text.strip() for _, text in hint_matches]
        else:
            # Ищем подсказки в формате "Подсказка 1: [текст]"
            alt_hint_pattern = r'(?:подсказка|Подсказка)\s*(\d+)[:\.]\s*(.*?)(?=(?:\n(?:подсказка|Подсказка)\s*\d+)|$)'
            alt_hint_matches = re.findall(alt_hint_pattern, hints_string, re.DOTALL | re.IGNORECASE)
            
            if alt_hint_matches:
                hints = [text.strip() for _, text in alt_hint_matches]
            else:
                # Если не нашли подсказки в нужном формате, пробуем разделить по абзацам
                paragraphs = re.split(r'\n\s*\n', hints_string)
                hints = [p.strip() for p in paragraphs if p.strip()]
        
        # Дополняем или обрезаем до 3 подсказок
        while len(hints) < 3:
            hints.append("Подсказка недоступна")
        
        return hints[:3]
    
    def get_raw_structure(self):
        """
        Возвращает необработанную структуру ответа модели для отладки
        
        Returns:
            dict: Словарь с необработанными блоками
        """
        return self.parsed_data
    
    def process(self, format="html"):
        """
        Обрабатывает ответ модели и возвращает структурированный результат
        
        Args:
            format: Формат вывода - "html" или "original"
            
        Returns:
            dict: Словарь с обработанными элементами ответа
        """
        if format == "original":
            # Если требуется оригинальный формат, возвращаем исходные тексты
            return self.parsed_data
        else:
            # Преобразуем в HTML
            return {
                "task": self._convert_to_html(self.parsed_data["task"]),
                "solution": self._convert_to_html(self.parsed_data["solution"]),
                "hints": [self._convert_to_html(hint) for hint in self.parsed_data["hints"]],
                "answer": self._format_answer(self.parsed_data["answer"]),
                "visualization_params": self.parsed_data["visualization_params"]
            }
    
    def _convert_to_html(self, text):
        """
        Преобразует текст из Markdown в HTML, безопасно сохраняя LaTeX-формулы
        """
        if not text:
            return ""
        
        # Проверяем, содержит ли текст уже HTML-теги
        if re.search(r'<[a-z][a-z0-9]*(\s[^>]*)?>', text):
            return text
        
        # Защищаем LaTeX-формулы
        protected_text, formula_map = self._protect_latex(text)
        
        # Преобразуем Markdown в HTML
        html = self._markdown_to_html(protected_text)
        
        # Восстанавливаем LaTeX-формулы
        return self._restore_latex(html, formula_map)
    
    def _protect_latex(self, text):
        """
        Защищает LaTeX-формулы, заменяя их на маркеры
        
        Returns:
            tuple: (защищенный текст, словарь маркер -> формула)
        """
        formula_map = {}
        
        # Защищаем блочные LaTeX-формулы
        def protect_block_formula(match):
            key = f"__LATEX_BLOCK_{len(formula_map)}__"
            formula_map[key] = match.group(0)
            return key
        
        # Защищаем inline LaTeX-формулы
        def protect_inline_formula(match):
            key = f"__LATEX_INLINE_{len(formula_map)}__"
            formula_map[key] = match.group(0)
            return key
        
        # Ищем и заменяем формулы на маркеры
        protected_text = re.sub(r'\$\$(.*?)\$\$', protect_block_formula, text, flags=re.DOTALL)
        protected_text = re.sub(r'\$(.*?)\$', protect_inline_formula, protected_text, flags=re.DOTALL)
        
        return protected_text, formula_map
    
    def _restore_latex(self, html, formula_map):
        """
        Восстанавливает LaTeX-формулы из маркеров
        """
        for key, formula in formula_map.items():
            if key.startswith("__LATEX_BLOCK_"):
                html = html.replace(key, f'<div class="math-block">{formula}</div>')
            else:  # inline formula
                html = html.replace(key, f'<span class="math-inline">{formula}</span>')
        
        return html
    
    def _markdown_to_html(self, text):
        """
        Простое преобразование Markdown в HTML без внешних зависимостей
        """
        # Экранируем HTML-специальные символы
        text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        
        # Защищаем блоки кода
        def process_code_block(match):
            code = match.group(1).strip()
            # Экранируем возможные HTML-сущности внутри кода
            code = code.replace('&lt;', '<').replace('&gt;', '>')
            return f'<pre><code>{code}</code></pre>'
        
        # Заменяем блоки кода
        text = re.sub(r'```(.*?)```', process_code_block, text, flags=re.DOTALL)
        
        # Преобразуем заголовки
        text = re.sub(r'^# (.+)$', r'<h1>\1</h1>', text, flags=re.MULTILINE)
        text = re.sub(r'^## (.+)$', r'<h2>\1</h2>', text, flags=re.MULTILINE)
        text = re.sub(r'^### (.+)$', r'<h3>\1</h3>', text, flags=re.MULTILINE)
        
        # Преобразуем жирный и курсивный текст
        text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
        text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)
        
        # Преобразуем списки с нумерацией
        def process_numbered_list(match):
            items = match.group(1).split('\n')
            html = '<ol class="custom-numbered-list">\n'
            for item in items:
                # Извлекаем число из строки (например, из "1. Текст" получаем "1")
                num_match = re.match(r'^\s*(\d+)\.\s*(.*?)$', item)
                if num_match:
                    num = num_match.group(1)
                    item_content = num_match.group(2).strip()
                    if item_content:
                        html += f'  <li value="{num}">{item_content}</li>\n'
                else:
                    # Если формат не соответствует ожидаемому, обрабатываем как обычно
                    item_content = re.sub(r'^\s*\d+\.\s*', '', item).strip()
                    if item_content:
                        html += f'  <li>{item_content}</li>\n'
            html += '</ol>'
            return html
        
        text = re.sub(r'((?:^\s*\d+\.\s*.+\n?)+)', process_numbered_list, text, flags=re.MULTILINE)
        
        # Преобразуем ненумерованные списки
        def process_unordered_list(match):
            items = match.group(1).split('\n')
            html = '<ul>\n'
            for item in items:
                item_content = re.sub(r'^\s*[\-\*]\s*', '', item).strip()
                if item_content:
                    html += f'  <li>{item_content}</li>\n'
            html += '</ul>'
            return html
        
        text = re.sub(r'((?:^\s*[\-\*]\s*.+\n?)+)', process_unordered_list, text, flags=re.MULTILINE)
        
        # Преобразуем параграфы
        paragraphs = re.split(r'\n\s*\n', text)
        result = []
        
        for p in paragraphs:
            p = p.strip()
            if not p:
                continue
                
            # Пропускаем параграфы, которые уже содержат HTML-теги блочного уровня
            if re.match(r'^\s*<(?:h[1-6]|ul|ol|div|p|blockquote|pre)', p):
                result.append(p)
            else:
                # Заменяем одиночные переносы строк на <br>
                p = p.replace('\n', '<br>')
                result.append(f'<p>{p}</p>')
        
        return '\n'.join(result)
    
    def _format_answer(self, answer):
        """
        Форматирует ответ, добавляя LaTeX-окружение, если необходимо
        """
        if not answer or answer.strip() == "":
            return "Ответ не найден"
        
        # Удаляем префикс "Ответ:", если он есть
        answer = re.sub(r'^(?:Ответ|ОТВЕТ|ответ)\s*:\s*', '', answer)
        answer = answer.strip()
        
        # Если ответ уже в формате LaTeX, оставляем как есть
        if (answer.startswith('$') and answer.endswith('$')) or \
           (answer.startswith('$$') and answer.endswith('$$')):
            return answer
        
        # Если ответ содержит $$, заменяем на $
        if '$$' in answer:
            answer = answer.replace('$$', '')
            return f"${answer}$"
        
        # Проверяем, содержит ли ответ LaTeX-выражения
        if '\\' in answer or '^' in answer or '_' in answer or any(c in answer for c in '{[()]}'):
            # Оборачиваем в LaTeX
            return f"${answer}$"
        
        # Если ответ - число, оборачиваем его в LaTeX
        if re.match(r'^[-+]?\d*[.,]?\d+$', answer):
            return f"${answer}$"
        
        # По умолчанию также оборачиваем в LaTeX для единообразия
        return f"${answer}$"
    
    def _remove_answer_from_solution(self, solution):
        """Удаляет строку с ответом из решения"""
        # Ищем различные форматы строки с ответом
        answer_patterns = [
            r'(?:Ответ|ОТВЕТ|ответ)\s*:.*?(?=$|\n\s*\n)',  # Ответ: ... до конца строки или до пустой строки
            r'\n\s*(?:Ответ|ОТВЕТ|ответ)\s*:.*?(?=$|\n\s*\n)'  # Ответ на новой строке
        ]
        
        result = solution
        for pattern in answer_patterns:
            result = re.sub(pattern, '', result, flags=re.IGNORECASE | re.DOTALL)
        
        # Убираем лишние пустые строки в конце
        result = re.sub(r'\n\s*$', '', result)
        
        return result


def convert_model_response_to_html(raw_response, is_markdown=False):
    """
    Конвертирует сырой ответ модели в структурированный HTML или Markdown
    
    Args:
        raw_response: Сырой текст ответа модели
        is_markdown: Если True, возвращает ответы в формате Markdown вместо HTML
        
    Returns:
        dict: Словарь с обработанными компонентами (задача, решение, подсказки)
    """
    processor = ModelResponseProcessor(raw_response)
    return processor.process("original" if is_markdown else "html")