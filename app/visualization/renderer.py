import os
import uuid
import matplotlib.pyplot as plt
import logging
from app.geometry.base import GeometricFigure
from app.geometry.trapezoid import Trapezoid
from app.geometry.triangle import Triangle
from app.geometry.rectangle import Rectangle
from app.geometry.parallelogram import Parallelogram
from app.geometry.circle import Circle

class GeometryRenderer:
    """
    Класс для рендеринга геометрических фигур.
    """
    
    @staticmethod
    def get_figure_class(figure_type):
        """
        Возвращает класс геометрической фигуры по ее типу.
        
        Args:
            figure_type (str): Тип фигуры ('trapezoid', 'triangle', 'rectangle', 'circle', 'parallelogram')
            
        Returns:
            class: Класс фигуры, наследующий от GeometricFigure
        """
        figure_classes = {
            'trapezoid': Trapezoid,
            'triangle': Triangle,
            'rectangle': Rectangle,
            'parallelogram': Parallelogram,
            'circle': Circle
        }
        
        return figure_classes.get(figure_type, GeometricFigure)
    
    @staticmethod
    def create_figure(figure_type, params=None):
        """
        Создает объект геометрической фигуры по ее типу.
        
        Args:
            figure_type (str): Тип фигуры ('trapezoid', 'triangle', 'rectangle', 'circle', 'parallelogram')
            params (dict): Параметры фигуры
            
        Returns:
            GeometricFigure: Объект геометрической фигуры
        """
        figure_class = GeometryRenderer.get_figure_class(figure_type)
        return figure_class(params)
    
    @staticmethod
    def render_figure(figure, filename=None):
        """
        Отрисовывает геометрическую фигуру и сохраняет ее в файл.
        
        Args:
            figure (GeometricFigure): Объект геометрической фигуры
            filename (str): Имя файла для сохранения. Если None, создается случайное имя.
            
        Returns:
            str: Путь к сохраненному файлу или None, если не удалось сохранить
        """
        try:
            if not isinstance(figure, GeometricFigure):
                raise ValueError("figure должен быть экземпляром GeometricFigure")
            
            # Создаем фигуру и рисуем ее
            ax = figure.draw()
            
            # Определяем имя файла
            if filename is None:
                filename = f"{figure.figure_type}_{uuid.uuid4().hex[:8]}.png"
            
            # Проверяем, содержит ли путь директорию
            directory = os.path.dirname(filename)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
            
            # Сохраняем изображение
            plt.savefig(filename, bbox_inches='tight', dpi=300)
            plt.close()
            
            return filename
        except Exception as e:
            logging.error(f"Ошибка при отрисовке фигуры: {e}")
            plt.close()  # Закрываем фигуру в случае ошибки
            return None
            
    @staticmethod
    def render_from_params(figure_type, params, filename=None):
        """
        Создает и отрисовывает геометрическую фигуру на основе параметров.
        
        Args:
            figure_type (str): Тип фигуры ('trapezoid', 'triangle', 'rectangle', 'circle', 'parallelogram')
            params (dict): Параметры фигуры
            filename (str): Имя файла для сохранения. Если None, создается случайное имя.
            
        Returns:
            str: Путь к сохраненному файлу или None, если не удалось сохранить
        """
        figure = GeometryRenderer.create_figure(figure_type, params)
        return GeometryRenderer.render_figure(figure, filename)
    
    @staticmethod
    def render_from_text(task_text, figure_type=None):
        """
        Извлекает параметры из текста задачи, создает и отрисовывает геометрическую фигуру.
        
        Args:
            task_text (str): Текст задачи
            figure_type (str): Тип фигуры. Если None, определяется из текста.
            
        Returns:
            str: Путь к сохраненному файлу или None, если не удалось сохранить
        """
        # Определяем тип фигуры, если не указан
        if figure_type is None:
            figure_type = GeometryRenderer.determine_figure_type(task_text)
            if not figure_type:
                logging.warning("Не удалось определить тип фигуры из текста задачи")
                return None
        
        # Создаем фигуру в зависимости от типа
        if figure_type == "trapezoid":
            figure = Trapezoid.from_text(task_text)
        elif figure_type == "triangle":
            figure = Triangle.from_text(task_text)
        elif figure_type == "rectangle":
            figure = Rectangle.from_text(task_text)
        elif figure_type == "parallelogram":
            figure = Parallelogram.from_text(task_text)
        elif figure_type == "circle":
            figure = Circle.from_text(task_text)
        else:
            logging.warning(f"Неизвестный тип фигуры: {figure_type}")
            return None
        
        # Отрисовываем фигуру
        return GeometryRenderer.render_figure(figure)
    
    @staticmethod
    def determine_figure_type(task_text):
        """
        Определяет тип геометрической фигуры из текста задачи.
        
        Args:
            task_text (str): Текст задачи
            
        Returns:
            str: Тип фигуры ('trapezoid', 'triangle', 'rectangle', 'circle', 'parallelogram') или None
        """
        import re
        
        # Словарь с ключевыми словами для каждого типа фигуры
        keywords = {
            'trapezoid': [r'трапеци', r'основани[а-я]+ трапеции', r'боковая сторона трапеции'],
            'triangle': [r'треугольник', r'прямоуголь?н[а-я]+ треуголь?ник', r'равносторон', r'равнобедрен'],
            'rectangle': [r'прямоугольник', r'квадрат', r'стор[а-я]+ прямоугольник', r'длин[а-я]+ прямоугольник'],
            'parallelogram': [r'параллелограмм', r'ромб', r'диагонал[а-я]+ параллелограмм'],
            'circle': [r'окружност', r'круг', r'радиус', r'диаметр', r'хорд[а-я]']
        }
        
        # Проверяем каждый тип фигуры
        for figure_type, patterns in keywords.items():
            for pattern in patterns:
                if re.search(pattern, task_text, re.IGNORECASE):
                    return figure_type
        
        return None 