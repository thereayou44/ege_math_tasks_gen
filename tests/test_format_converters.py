import unittest
import sys
import os
import re
import json
from unittest.mock import patch, MagicMock

# Добавляем родительскую директорию в пути для импорта
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from task_generator import (
    generate_markdown_task,
    generate_json_task,
    generate_json_markdown_task
)

class TestFormatConverters(unittest.TestCase):
    """Тесты для функций конвертации форматов."""
    
    @patch('task_generator.yandex_gpt_generate')
    @patch('task_generator.select_file')
    def test_generate_markdown_task(self, mock_select_file, mock_yandex_gpt):
        """Проверяет генерацию задачи в формате Markdown."""
        # Мокаем данные для задачи
        mock_select_file.return_value = {
            "task": "Найдите корни уравнения x^2 - 5x + 6 = 0",
            "solution": "Решение: Факторизуем уравнение (x-2)(x-3)=0. Ответ: 2, 3."
        }
        
        # Мокаем ответ от API
        mock_response = """
        ---ЗАДАЧА---
        Найдите корни уравнения: $x^2 - 5x + 6 = 0$
        
        ---РЕШЕНИЕ---
        Решаем квадратное уравнение $x^2 - 5x + 6 = 0$.
        
        Для этого воспользуемся формулой разложения на множители.
        
        $x^2 - 5x + 6 = (x-2)(x-3) = 0$
        
        Отсюда получаем два корня:
        $x = 2$ или $x = 3$
        
        Ответ: 2, 3
        
        ---ПОДСКАЗКИ---
        1. Попробуйте разложить квадратный трехчлен на множители.
        2. Воспользуйтесь методом подбора или дискриминантом.
        3. Получив линейные множители, приравняйте каждый к нулю.
        
        ---ПАРАМЕТРЫ ДЛЯ ВИЗУАЛИЗАЦИИ---
        Тип: график
        Функция: x**2 - 5*x + 6
        Диапазон X: -1, 6
        Диапазон Y: -2, 10
        """
        
        mock_yandex_gpt.return_value = mock_response
        
        # Генерируем задачу в формате Markdown
        category = "Простейшие уравнения"
        result = generate_markdown_task(category)
        
        # Проверяем наличие ожидаемых полей
        self.assertIn("problem", result)
        self.assertIn("solution", result)
        self.assertIn("hint1", result)
        self.assertIn("hint2", result)
        self.assertIn("hint3", result)
        
        # Проверяем, что задача содержит корректную информацию, используя регулярные выражения
        import re
        # Проверяем наличие уравнения в задаче
        self.assertTrue(re.search(r'корн|уравнен|найдите', result["problem"], re.IGNORECASE), 
                       "Задача должна содержать ключевые слова, связанные с уравнением")
        
        # Проверяем наличие информации о решении
        self.assertTrue(re.search(r'решени|множител|корн', result["solution"], re.IGNORECASE),
                       "Решение должно содержать ключевую информацию")
        
        # Проверяем, что подсказки не пустые
        self.assertTrue(len(result["hint1"]) > 5, "Подсказка 1 должна быть непустой")
        self.assertTrue(len(result["hint2"]) > 5, "Подсказка 2 должна быть непустой")
        self.assertTrue(len(result["hint3"]) > 5, "Подсказка 3 должна быть непустой")
    
    @patch('task_generator.yandex_gpt_generate')
    @patch('task_generator.select_file')
    def test_generate_json_task(self, mock_select_file, mock_yandex_gpt):
        """Проверяет генерацию задачи в формате JSON."""
        # Мокаем данные и ответ API так же, как и выше
        mock_select_file.return_value = {
            "task": "Найдите корни уравнения x^2 - 5x + 6 = 0",
            "solution": "Решение: Факторизуем уравнение (x-2)(x-3)=0. Ответ: 2, 3."
        }
        
        mock_response = """
        ---ЗАДАЧА---
        Найдите корни уравнения: $x^2 - 5x + 6 = 0$
        
        ---РЕШЕНИЕ---
        Решаем квадратное уравнение $x^2 - 5x + 6 = 0$.
        
        Для этого воспользуемся формулой разложения на множители.
        
        $x^2 - 5x + 6 = (x-2)(x-3) = 0$
        
        Отсюда получаем два корня:
        $x = 2$ или $x = 3$
        
        Ответ: 2, 3
        
        ---ПОДСКАЗКИ---
        1. Попробуйте разложить квадратный трехчлен на множители.
        2. Воспользуйтесь методом подбора или дискриминантом.
        3. Получив линейные множители, приравняйте каждый к нулю.
        
        ---ПАРАМЕТРЫ ДЛЯ ВИЗУАЛИЗАЦИИ---
        Тип: график
        Функция: x**2 - 5*x + 6
        Диапазон X: -1, 6
        Диапазон Y: -2, 10
        """
        
        mock_yandex_gpt.return_value = mock_response
        
        # Мокаем вызов process_visualization_params для имитации изображения
        with patch('task_generator.process_visualization_params') as mock_viz:
            mock_viz.return_value = ('static/images/generated/test.png', 'base64_data')
            
            # Генерируем задачу в формате JSON
            category = "Простейшие уравнения"
            result = generate_json_task(category)
            
            # Проверяем наличие ожидаемых полей
            self.assertIn("task", result)
            self.assertIn("solution", result)
            self.assertIn("hints", result)
            self.assertIn("answer", result)
            
            # Более гибкие проверки с помощью регулярных выражений
            import re
            
            # Проверяем формат
            self.assertEqual(result["format"], "html")
            
            # Проверяем, что задача содержит текст и правильные данные
            self.assertIn("text", result["task"], "Задача должна содержать текстовое поле")
            self.assertTrue(re.search(r'корн|уравнен|найдите', result["task"]["text"], re.IGNORECASE), 
                           "Задача должна содержать ключевые слова, связанные с уравнением")
            
            # Проверяем, что ответ соответствует ожидаемому
            self.assertIn("2", result["answer"], "Ответ должен содержать 2")
            self.assertIn("3", result["answer"], "Ответ должен содержать 3")
            
            # Проверяем, что есть три подсказки
            self.assertEqual(len(result["hints"]), 3, "Должно быть три подсказки")
            
            # Проверяем, что подсказки не пустые
            for hint in result["hints"]:
                self.assertTrue(len(hint) > 5, "Подсказка должна быть непустой")
    
    @patch('task_generator.yandex_gpt_generate')
    @patch('task_generator.select_file')
    def test_generate_json_markdown_task(self, mock_select_file, mock_yandex_gpt):
        """Проверяет генерацию задачи в формате JSON с Markdown."""
        # Мокаем данные и ответ API так же, как и выше
        mock_select_file.return_value = {
            "task": "Найдите корни уравнения x^2 - 5x + 6 = 0",
            "solution": "Решение: Факторизуем уравнение (x-2)(x-3)=0. Ответ: 2, 3."
        }
        
        mock_response = """
        ---ЗАДАЧА---
        Найдите корни уравнения: $x^2 - 5x + 6 = 0$
        
        ---РЕШЕНИЕ---
        Решаем квадратное уравнение $x^2 - 5x + 6 = 0$.
        
        Для этого воспользуемся формулой разложения на множители.
        
        $x^2 - 5x + 6 = (x-2)(x-3) = 0$
        
        Отсюда получаем два корня:
        $x = 2$ или $x = 3$
        
        Ответ: 2, 3
        
        ---ПОДСКАЗКИ---
        1. Попробуйте разложить квадратный трехчлен на множители.
        2. Воспользуйтесь методом подбора или дискриминантом.
        3. Получив линейные множители, приравняйте каждый к нулю.
        
        ---ПАРАМЕТРЫ ДЛЯ ВИЗУАЛИЗАЦИИ---
        Тип: график
        Функция: x**2 - 5*x + 6
        Диапазон X: -1, 6
        Диапазон Y: -2, 10
        """
        
        mock_yandex_gpt.return_value = mock_response
        
        # Мокаем вызов process_visualization_params для имитации изображения
        with patch('task_generator.process_visualization_params') as mock_viz:
            mock_viz.return_value = ('static/images/generated/test.png', 'base64_data')
            
            # Генерируем задачу в формате JSON с Markdown
            category = "Простейшие уравнения"
            result = generate_json_markdown_task(category)
            
            # Проверяем наличие ожидаемых полей
            self.assertIn("task", result)
            self.assertIn("solution", result)
            self.assertIn("hints", result)
            self.assertIn("answer", result)
            
            # Более гибкие проверки с помощью регулярных выражений
            import re
            
            # Проверяем формат
            self.assertEqual(result["format"], "markdown")
            
            # Проверяем, что задача содержит текст и правильные данные
            self.assertIn("text", result["task"], "Задача должна содержать текстовое поле")
            
            # Проверяем текст задачи - он может включать разные варианты форматирования
            task_text = result["task"]["text"]
            self.assertTrue(
                re.search(r'корн|уравнен|найдите', task_text, re.IGNORECASE) or
                re.search(r'x\^2', task_text) or 
                re.search(r'x\s*=', task_text),
                "Задача должна содержать ключевые слова, связанные с уравнением"
            )
            
            # Проверяем, что ответ соответствует ожидаемому
            self.assertIn("2", result["answer"], "Ответ должен содержать 2")
            self.assertIn("3", result["answer"], "Ответ должен содержать 3")
            
            # Проверяем, что есть три подсказки
            self.assertEqual(len(result["hints"]), 3, "Должно быть три подсказки")
    
    @patch('task_generator.yandex_gpt_generate')
    @patch('task_generator.select_file')
    def test_json_serialization(self, mock_select_file, mock_yandex_gpt):
        """Проверяет, что сгенерированный JSON может быть сериализован."""
        # Мокаем данные и ответ API так же, как и в предыдущих тестах
        mock_select_file.return_value = {
            "task": "Найдите корни уравнения x^2 - 5x + 6 = 0",
            "solution": "Решение: Факторизуем уравнение (x-2)(x-3)=0. Ответ: 2, 3."
        }
        
        mock_response = """
        ---ЗАДАЧА---
        Найдите корни уравнения: $x^2 - 5x + 6 = 0$
        
        ---РЕШЕНИЕ---
        Решаем квадратное уравнение $x^2 - 5x + 6 = 0$.
        
        Для этого воспользуемся формулой разложения на множители.
        
        $x^2 - 5x + 6 = (x-2)(x-3) = 0$
        
        Отсюда получаем два корня:
        $x = 2$ или $x = 3$
        
        Ответ: 2, 3
        
        ---ПОДСКАЗКИ---
        1. Попробуйте разложить квадратный трехчлен на множители.
        2. Воспользуйтесь методом подбора или дискриминантом.
        3. Получив линейные множители, приравняйте каждый к нулю.
        
        ---ПАРАМЕТРЫ ДЛЯ ВИЗУАЛИЗАЦИИ---
        Тип: график
        Функция: x**2 - 5*x + 6
        Диапазон X: -1, 6
        Диапазон Y: -2, 10
        """
        
        mock_yandex_gpt.return_value = mock_response
        
        # Мокаем вызов process_visualization_params для имитации изображения
        with patch('task_generator.process_visualization_params') as mock_viz:
            mock_viz.return_value = ('static/images/generated/test.png', 'base64_data')
            
            category = "Простейшие уравнения"
            result = generate_json_task(category)
            
            try:
                # Пробуем сериализовать в JSON
                json_str = json.dumps(result, ensure_ascii=False)
                
                # Десериализовать обратно в словарь
                parsed_result = json.loads(json_str)
                
                # Проверяем, что десериализованный результат содержит те же поля
                self.assertIn("task", parsed_result)
                self.assertIn("solution", parsed_result)
                self.assertIn("hints", parsed_result)
                self.assertIn("answer", parsed_result)
                
            except Exception as e:
                self.fail(f"Ошибка при сериализации/десериализации JSON: {e}")

if __name__ == "__main__":
    unittest.main() 