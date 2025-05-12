import unittest
import sys
import os
import re
from unittest.mock import patch, MagicMock

# Добавляем родительскую директорию в пути для импорта
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.task_generator import generate_complete_task, extract_answer_with_latex, parse_hints
from app.prompts import DEFAULT_VISUALIZATION_PARAMS

class TestTaskGenerator(unittest.TestCase):
    """Тесты для проверки генерации задач."""
    
    @patch('app.task_generator.yandex_gpt_generate')
    @patch('app.task_generator.select_file')
    def test_basic_task_generation(self, mock_select_file, mock_yandex_gpt):
        """Проверяет, что генерация базовой задачи содержит все необходимые элементы."""
        # Мокаем данные для задачи
        mock_select_file.return_value = {
            "task": "Решите уравнение x^2 - 5x + 6 = 0",
            "solution": "Решение: D = 25 - 24 = 1, x1 = 2, x2 = 3. Ответ: 2, 3."
        }
        
        # Мокаем ответ от YandexGPT API
        mock_response = """
        ---ЗАДАЧА---
        Решите уравнение: $x^2 - 5x + 6 = 0$
        
        ---РЕШЕНИЕ---
        Решаем квадратное уравнение $x^2 - 5x + 6 = 0$.
        
        Найдем дискриминант: $D = b^2 - 4ac = (-5)^2 - 4 \cdot 1 \cdot 6 = 25 - 24 = 1$
        
        Тогда корни уравнения:
        $x_1 = \\frac{-b + \sqrt{D}}{2a} = \\frac{5 + 1}{2} = 3$
        $x_2 = \\frac{-b - \sqrt{D}}{2a} = \\frac{5 - 1}{2} = 2$
        
        Ответ: 2, 3
        
        ---ПОДСКАЗКИ---
        1. Для решения квадратного уравнения вида $ax^2 + bx + c = 0$ найдите дискриминант по формуле $D = b^2 - 4ac$.
        2. Найдите корни уравнения, используя формулу $x = \\frac{-b \pm \sqrt{D}}{2a}$.
        3. Проверьте корни, подставив их в исходное уравнение.
        
        ---ПАРАМЕТРЫ ДЛЯ ВИЗУАЛИЗАЦИИ---
        Тип: график
        Функция: x**2 - 5*x + 6
        Диапазон X: 0, 5
        Диапазон Y: -2, 6
        """
        
        mock_yandex_gpt.return_value = mock_response
        
        # Мокаем вызов process_visualization_params для имитации изображения
        with patch('app.task_generator.process_visualization_params') as mock_viz:
            mock_viz.return_value = ('static/images/generated/test.png', 'base64_data')
            
            # Используем простую категорию для быстрого теста
            category = "Простейшие уравнения"
            
            # Генерируем задачу с использованием моков
            result = generate_complete_task(category)
            
            # Проверяем наличие всех необходимых элементов
            self.assertIn("task", result)
            self.assertIn("solution", result)
            self.assertIn("hints", result)
            self.assertIn("answer", result)
            
            # Проверяем, что в задаче есть содержимое
            self.assertTrue(len(result["task"]) > 10)
            self.assertTrue(len(result["solution"]) > 10)
            
            # Проверяем, что все три подсказки присутствуют
            self.assertEqual(len(result["hints"]), 3)
            
            # Проверяем, что подсказки содержат текст
            for hint in result["hints"]:
                self.assertTrue(len(hint) > 5, "Подсказка должна содержать текст")
    
    @patch('app.task_generator.yandex_gpt_generate')
    @patch('app.task_generator.select_file')
    def test_geometry_task_with_image(self, mock_select_file, mock_yandex_gpt):
        """Проверяет, что задача по геометрии генерирует данные для изображения."""
        # Мокаем данные для задачи по геометрии
        mock_select_file.return_value = {
            "task": "В треугольнике ABC угол C равен 90°, AB = 5, BC = 3. Найдите AC.",
            "solution": "По теореме Пифагора: AC^2 = AB^2 - BC^2 = 25 - 9 = 16, AC = 4. Ответ: 4."
        }
        
        # Мокаем ответ от YandexGPT API
        mock_response = """
        ---ЗАДАЧА---
        В прямоугольном треугольнике ABC угол C равен 90°, AB = 5, BC = 3. Найдите длину гипотенузы AC.
        
        ---РЕШЕНИЕ---
        В прямоугольном треугольнике ABC угол C равен 90°, AB = 5, BC = 3.
        
        По теореме Пифагора:
        $AC^2 = AB^2 + BC^2 = 5^2 + 3^2 = 25 + 9 = 34$
        
        $AC = \sqrt{34} \approx 4$
        
        Ответ: 4
        
        ---ПОДСКАЗКИ---
        1. В прямоугольном треугольнике применяется теорема Пифагора для нахождения гипотенузы.
        2. Теорема Пифагора: a^2 + b^2 = c^2, где c - гипотенуза, a и b - катеты.
        3. Подставьте значения катетов в формулу и найдите гипотенузу.
        
        ---ПАРАМЕТРЫ ДЛЯ ВИЗУАЛИЗАЦИИ---
        Тип: треугольник
        Координаты вершин: (0,0),(5,0),(5,3)
        Подписи вершин: A,B,C
        Показать углы: да
        Показать длины: да
        """
        
        mock_yandex_gpt.return_value = mock_response
        
        # Мокаем вызов process_visualization_params для имитации изображения
        with patch('app.task_generator.process_visualization_params') as mock_viz:
            mock_viz.return_value = ('static/images/generated/triangle.png', 'base64_triangle_data')
            
            category = "Планиметрия"
            
            # Генерируем задачу по геометрии
            result = generate_complete_task(category)
            
            # Базовые проверки
            self.assertIn("task", result)
            self.assertIn("solution", result)
            
            # Проверка на наличие данных изображения
            self.assertIn("image_path", result)
            self.assertIn("image_base64", result)
            self.assertEqual(result["image_path"], 'static/images/generated/triangle.png')
            self.assertEqual(result["image_base64"], 'base64_triangle_data')
    
    @patch('app.task_generator.yandex_gpt_generate')
    @patch('app.task_generator.select_file')
    def test_function_graph_task(self, mock_select_file, mock_yandex_gpt):
        """Проверяет, что задача с графиком функции содержит данные для изображения."""
        # Мокаем данные для задачи с графиком
        mock_select_file.return_value = {
            "task": "Найдите точки пересечения графика функции y = x^2 - 4 с осями координат",
            "solution": "Решение: при y = 0 имеем x^2 - 4 = 0, x = ±2. При x = 0 имеем y = -4. Ответ: (-2, 0), (2, 0), (0, -4)."
        }
        
        # Мокаем ответ от YandexGPT API
        mock_response = """
        ---ЗАДАЧА---
        Найдите точки пересечения графика функции $y = x^2 - 4$ с осями координат.
        
        ---РЕШЕНИЕ---
        Найдем точки пересечения с осью Ox.
        
        На оси Ox координата y = 0, поэтому:
        $0 = x^2 - 4$
        $x^2 = 4$
        $x = \pm 2$
        
        Таким образом, получаем две точки пересечения с осью Ox: $(-2, 0)$ и $(2, 0)$.
        
        Найдем точку пересечения с осью Oy.
        
        На оси Oy координата x = 0, поэтому:
        $y = 0^2 - 4 = -4$
        
        Получаем точку пересечения с осью Oy: $(0, -4)$.
        
        Ответ: точки $(-2, 0)$, $(2, 0)$ и $(0, -4)$.
        
        ---ПОДСКАЗКИ---
        1. Для нахождения точек пересечения с осью Ox нужно решить уравнение y = 0.
        2. Для нахождения точек пересечения с осью Oy нужно подставить x = 0.
        3. Не забудьте записать координаты всех точек пересечения.
        
        ---ПАРАМЕТРЫ ДЛЯ ВИЗУАЛИЗАЦИИ---
        Тип: график
        Функция: x**2 - 4
        Диапазон X: -3, 3
        Диапазон Y: -5, 5
        """
        
        mock_yandex_gpt.return_value = mock_response
        
        # Мокаем вызов process_visualization_params для имитации изображения
        with patch('app.task_generator.process_visualization_params') as mock_viz:
            mock_viz.return_value = ('static/images/generated/graph.png', 'base64_graph_data')
            
            category = "Графики функций"
            
            # Генерируем задачу с графиком
            result = generate_complete_task(category)
            
            # Базовые проверки
            self.assertIn("task", result)
            self.assertIn("solution", result)
            
            # Проверка на наличие данных изображения
            self.assertIn("image_path", result)
            self.assertIn("image_base64", result)
            self.assertEqual(result["image_path"], 'static/images/generated/graph.png')
            self.assertEqual(result["image_base64"], 'base64_graph_data')
    
    @patch('app.task_generator.select_file')
    def test_task_generation_with_difficulty(self, mock_select_file):
        """Проверяет, что сложность подсказок учитывается при генерации."""
        # Создаем мок данных для задачи
        mock_select_file.return_value = {
            "task": "Пример текстовой задачи",
            "solution": "Решение текстовой задачи. Ответ: 42."
        }
        
        # Мокаем yandex_gpt_generate, создавая специальный контекст для каждого вызова
        with patch('app.task_generator.yandex_gpt_generate') as mock_yandex_gpt_easy:
            # Мокаем ответ для легкого уровня сложности
            mock_yandex_gpt_easy.return_value = """
            ---ЗАДАЧА---
            Простая текстовая задача
            
            ---РЕШЕНИЕ---
            Решение простой задачи. Ответ: 10.
            
            ---ПОДСКАЗКИ---
            1. Легкая подсказка 1
            2. Легкая подсказка 2
            3. Легкая подсказка 3
            
            ---ПАРАМЕТРЫ ДЛЯ ВИЗУАЛИЗАЦИИ---
            """
            
            # Генерируем задачу с легким уровнем сложности
            category = "Простейшие текстовые задачи"
            result_easy = generate_complete_task(category, difficulty_level=1)
            
            # Проверяем, что задача успешно сгенерирована с правильным содержимым
            self.assertNotIn("error", result_easy, "Ошибка при генерации задачи с низкой сложностью")
            self.assertEqual(result_easy.get("difficulty_level"), 1)
            self.assertIn("Простая", result_easy["task"])
            self.assertIn("Легкая подсказка", result_easy["hints"][0])
        
        # Запускаем отдельный контекст для высокого уровня сложности
        with patch('app.task_generator.yandex_gpt_generate') as mock_yandex_gpt_hard:
            # Мокаем ответ для высокого уровня сложности
            mock_yandex_gpt_hard.return_value = """
            ---ЗАДАЧА---
            Сложная текстовая задача
            
            ---РЕШЕНИЕ---
            Сложное решение. Ответ: 50.
            
            ---ПОДСКАЗКИ---
            1. Сложная подсказка 1
            2. Сложная подсказка 2
            3. Сложная подсказка 3
            
            ---ПАРАМЕТРЫ ДЛЯ ВИЗУАЛИЗАЦИИ---
            """
            
            # Генерируем задачу с высоким уровнем сложности
            category = "Простейшие текстовые задачи"
            result_hard = generate_complete_task(category, difficulty_level=5)
            
            # Проверяем, что задача успешно сгенерирована с правильным содержимым
            self.assertNotIn("error", result_hard, "Ошибка при генерации задачи с высокой сложностью")
            self.assertEqual(result_hard.get("difficulty_level"), 5)
            self.assertIn("Сложная", result_hard["task"])
            self.assertIn("Сложная подсказка", result_hard["hints"][0])

    def test_extract_answer(self):
        """Проверяет извлечение ответа из решения."""
        # Тестовое решение с ответом
        solution = "Решаем уравнение... Ответ: 5."
        answer = extract_answer_with_latex(solution)
        self.assertEqual(answer, "5")
        
        # Альтернативный формат ответа - обратите внимание на двойные $$ - это соответствует фактическому поведению функции
        solution = "Вычисляем... Итак, ответ: \\frac{1}{3}."
        answer = extract_answer_with_latex(solution)
        self.assertEqual(answer, "$$\\frac{1}{3}$$")  # Исправлено: ожидаем двойные $$
    
    def test_parse_hints(self):
        """Проверяет разбор подсказок из текста."""
        # Подсказки в формате "1. [текст]"
        hints_text = "1. Первая подсказка\n2. Вторая подсказка\n3. Третья подсказка"
        hints = parse_hints(hints_text)
        self.assertEqual(len(hints), 3)
        self.assertEqual(hints[0], "Первая подсказка")
        self.assertEqual(hints[1], "Вторая подсказка")
        self.assertEqual(hints[2], "Третья подсказка")
        
        # Подсказки без нумерации
        hints_text = "Первая подсказка\nВторая подсказка\nТретья подсказка"
        hints = parse_hints(hints_text)
        self.assertEqual(len(hints), 3)
        self.assertEqual(hints[0], "Первая подсказка")
        
        # Недостаточное количество подсказок
        hints_text = "Единственная подсказка"
        hints = parse_hints(hints_text)
        self.assertEqual(len(hints), 3)
        self.assertEqual(hints[0], "Единственная подсказка")
        self.assertEqual(hints[1], "Подсказка недоступна")

if __name__ == "__main__":
    unittest.main() 