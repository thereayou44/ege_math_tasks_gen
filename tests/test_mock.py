import unittest
import sys
import os
import json
import re
from unittest.mock import patch, MagicMock

# Добавляем родительскую директорию в пути для импорта
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from task_generator import generate_complete_task, yandex_gpt_generate, extract_answer_with_latex, parse_hints

class TestTaskGeneratorWithMocks(unittest.TestCase):
    """Тесты для генератора задач с использованием моков."""
    
    @patch('task_generator.yandex_gpt_generate')
    def test_generate_complete_task_with_mock(self, mock_yandex_gpt):
        """Тестирует генерацию задачи с помощью мока API."""
        # Мокаем ответ от YandexGPT API
        mock_response = """
        ---ЗАДАЧА---
        Решите уравнение: $x^2 - 4x + 3 = 0$
        
        ---РЕШЕНИЕ---
        Рассмотрим уравнение $x^2 - 4x + 3 = 0$. 
        
        Используем формулу дискриминанта:
        $D = b^2 - 4ac = (-4)^2 - 4 \cdot 1 \cdot 3 = 16 - 12 = 4$
        
        Корни уравнения:
        $x_1 = \\frac{-b + \sqrt{D}}{2a} = \\frac{4 + 2}{2} = 3$
        $x_2 = \\frac{-b - \sqrt{D}}{2a} = \\frac{4 - 2}{2} = 1$
        
        Ответ: 1, 3.
        
        ---ПОДСКАЗКИ---
        1. Обратите внимание на то, что это квадратное уравнение, которое решается через дискриминант.
        2. Дискриминант вычисляется по формуле $D = b^2 - 4ac$.
        3. Корни квадратного уравнения находятся по формуле $x = \\frac{-b \pm \sqrt{D}}{2a}$.
        
        ---ПАРАМЕТРЫ ДЛЯ ВИЗУАЛИЗАЦИИ---
        Тип: график
        Функция: x**2 - 4*x + 3
        Диапазон X: -2,5
        Диапазон Y: -3,5
        """
        
        mock_yandex_gpt.return_value = mock_response
        
        # Мокаем функцию select_file
        with patch('task_generator.select_file') as mock_select_file:
            mock_data = {
                "task": "Решите уравнение: x^2 - 4x + 3 = 0",
                "solution": "Ответ: 1, 3."
            }
            mock_select_file.return_value = mock_data
            
            # Мокаем process_visualization_params
            with patch('task_generator.process_visualization_params') as mock_visualization:
                mock_visualization.return_value = ('static/images/generated/test.png', 'base64_data')
                
                # Генерируем задачу
                result = generate_complete_task("Уравнения")
                
                # Проверяем, что есть все ожидаемые элементы
                self.assertIn("task", result)
                self.assertIn("solution", result)
                self.assertIn("hints", result)
                self.assertIn("answer", result)
                self.assertIn("image_path", result)
                self.assertIn("image_base64", result)
                
                # Проверяем содержимое
                self.assertIn("Решите уравнение", result["task"])
                self.assertIn("корни", result["solution"].lower())
                self.assertEqual(len(result["hints"]), 3)
                self.assertEqual(result["answer"], "1, 3")
    
    def test_extract_answer_with_latex(self):
        """Проверяет извлечение ответа из решения."""
        # Простой ответ
        solution = "После всех преобразований получаем. Ответ: 5."
        answer = extract_answer_with_latex(solution)
        self.assertEqual(answer, "5")
        
        # Ответ с формулой LaTeX - обратите внимание, что функция возвращает двойные $$
        solution = "Итак, ответ: $\sqrt{2}$."
        answer = extract_answer_with_latex(solution)
        self.assertEqual(answer, "$$\sqrt{2}$$")
        
        # Альтернативная формулировка
        solution = "Подводя итог, получаем: 3 + 2 = 5."
        answer = extract_answer_with_latex(solution)
        self.assertEqual(answer, "См. решение")  # Нет явной метки "Ответ:"
    
    def test_parse_hints_with_mock(self):
        """Проверяет разбор подсказок из текста с моком."""
        # Подсказки в стандартном формате
        hints_text = "1. Подсказка первая\n2. Подсказка вторая\n3. Подсказка третья"
        hints = parse_hints(hints_text)
        self.assertEqual(len(hints), 3)
        self.assertEqual(hints[0], "Подсказка первая")
        self.assertEqual(hints[1], "Подсказка вторая")
        self.assertEqual(hints[2], "Подсказка третья")
        
        # Подсказки в другом формате
        hints_text = "Подсказка 1: Используйте формулу.\nПодсказка 2: Применить теорему.\nПодсказка 3: Вычислить ответ."
        hints = parse_hints(hints_text)
        self.assertEqual(len(hints), 3)
        
        # Меньше 3 подсказок
        hints_text = "Только одна подсказка - используйте формулу"
        hints = parse_hints(hints_text)
        self.assertEqual(len(hints), 3)
        self.assertEqual(hints[0], "Только одна подсказка - используйте формулу")
        self.assertEqual(hints[1], "Подсказка недоступна")
        self.assertEqual(hints[2], "Подсказка недоступна")

    @patch('requests.post')
    def test_yandex_gpt_generate_with_mock(self, mock_post):
        """Тестирует работу функции yandex_gpt_generate с моком requests."""
        # Создаем мок для ответа requests
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "result": {
                "alternatives": [
                    {
                        "message": {
                            "text": "Этот текст сгенерирован моком YandexGPT API"
                        }
                    }
                ]
            }
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response
        
        # Вызываем функцию с мокированным requests
        result = yandex_gpt_generate("Тестовый запрос")
        
        # Проверяем результат
        self.assertEqual(result, "Этот текст сгенерирован моком YandexGPT API")
        
        # Проверяем, что запрос был отправлен
        mock_post.assert_called_once()

if __name__ == "__main__":
    unittest.main() 