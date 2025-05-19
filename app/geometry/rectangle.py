import numpy as np
import matplotlib.pyplot as plt
import re
import logging
from app.geometry.base import GeometricFigure

class Rectangle(GeometricFigure):
    """
    Класс для представления и визуализации прямоугольника, включая квадрат.
    """
    
    def __init__(self, params=None):
        """
        Инициализирует прямоугольник с заданными параметрами.
        
        Args:
            params (dict): Словарь параметров фигуры
        """
        super().__init__(params)
        self.figure_type = "rectangle"
        
        # Установка параметров по умолчанию, если не указаны
        if 'width' not in self.params:
            self.params['width'] = 4
        if 'height' not in self.params:
            self.params['height'] = 3
        if 'is_square' not in self.params:
            self.params['is_square'] = False
        if 'show_diagonals' not in self.params:
            self.params['show_diagonals'] = False
            
        # Если это квадрат, устанавливаем одинаковые параметры
        if self.params.get('is_square', False):
            side = self.params.get('width', 4)  # Используем width как размер стороны
            self.params['height'] = side
            
    def compute_points(self):
        """
        Вычисляет координаты точек прямоугольника.
        
        Returns:
            list: Список точек [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
        """
        # Проверяем, заданы ли конкретные координаты
        if 'coords' in self.params and len(self.params['coords']) >= 4:
            return self.params['coords'][:4]
        
        # Получаем основные параметры
        x = self.params.get('x', 0)
        y = self.params.get('y', 0)
        width = self.params.get('width', 4)
        height = self.params.get('height', 3)
        
        # Проверяем, является ли фигура квадратом
        if self.params.get('is_square', False):
            height = width  # Для квадрата высота равна ширине
            
        # Вычисляем координаты четырех вершин (против часовой стрелки)
        points = [
            (x, y),  # Нижний левый угол
            (x + width, y),  # Нижний правый угол
            (x + width, y + height),  # Верхний правый угол
            (x, y + height)  # Верхний левый угол
        ]
        
        return points
    
    def draw(self, ax=None):
        """
        Отрисовывает прямоугольник с дополнительными элементами.
        
        Args:
            ax (matplotlib.axes.Axes): Оси для отрисовки
            
        Returns:
            matplotlib.axes.Axes: Оси с нарисованной фигурой
        """
        # Используем базовый метод для рисования основной фигуры
        ax = super().draw(ax)
        
        # Добавляем диагонали, если нужно
        if self.params.get('show_diagonals', False):
            self._add_diagonals(ax)
            
        return ax
    
    def _add_diagonals(self, ax):
        """
        Добавляет диагонали прямоугольника.
        
        Args:
            ax (matplotlib.axes.Axes): Оси для отрисовки
        """
        points = self.points
        diagonal_lengths = self.params.get('diagonals_length', None)
        
        # Если у нас 4 точки, добавляем обе диагонали
        if len(points) == 4:
            # Диагональ от точки 0 к точке 2
            x1, y1 = points[0]
            x2, y2 = points[2]
            ax.plot([x1, x2], [y1, y2], 'g--', lw=1.5)
            
            # Вычисляем длину диагонали
            diag1_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            # Отображаем длину первой диагонали, если нужно
            if diagonal_lengths and len(diagonal_lengths) > 0 and diagonal_lengths[0] is not None:
                diag1_text = f"d₁ = {diagonal_lengths[0]}"
            else:
                diag1_text = f"d₁ = {diag1_length:.2f}"
                
            # Позиция для текста - середина диагонали
            mid_x1 = (x1 + x2) / 2
            mid_y1 = (y1 + y2) / 2
            
            ax.text(mid_x1, mid_y1, diag1_text, 
                    ha='center', va='center', color='green', fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', boxstyle='round,pad=0.3'))
            
            # Диагональ от точки 1 к точке 3
            x3, y3 = points[1]
            x4, y4 = points[3]
            ax.plot([x3, x4], [y3, y4], 'b--', lw=1.5)
            
            # Вычисляем длину второй диагонали
            diag2_length = np.sqrt((x4 - x3)**2 + (y4 - y3)**2)
            
            # Отображаем длину второй диагонали, если нужно
            if diagonal_lengths and len(diagonal_lengths) > 1 and diagonal_lengths[1] is not None:
                diag2_text = f"d₂ = {diagonal_lengths[1]}"
            else:
                diag2_text = f"d₂ = {diag2_length:.2f}"
                
            # Позиция для текста - середина диагонали
            mid_x2 = (x3 + x4) / 2
            mid_y2 = (y3 + y4) / 2
            
            ax.text(mid_x2, mid_y2, diag2_text, 
                    ha='center', va='center', color='blue', fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', boxstyle='round,pad=0.3'))
    
    def get_side_length_text(self, side_index, side_name, computed_length):
        """
        Возвращает текст для подписи длины стороны.
        
        Args:
            side_index (int): Индекс стороны
            side_name (str): Название стороны (например, "AB")
            computed_length (float): Вычисленная длина стороны
            
        Returns:
            str: Текст для подписи или None, если подпись не нужна
        """
        # Получаем параметры
        side_lengths = self.params.get('side_lengths', None)
        is_square = self.params.get('is_square', False)
        width = self.params.get('width', 4)
        height = self.params.get('height', 3)
        
        # Проверяем, нужно ли использовать конкретное значение из side_lengths
        if side_lengths and len(side_lengths) > side_index and side_lengths[side_index] is not None:
            if side_lengths[side_index] == "-":
                return None
            return f"{side_lengths[side_index]}"
        
        # Для квадрата все стороны равны
        if is_square:
            return f"{width}"
        
        # Для обычного прямоугольника
        if side_index == 0 or side_index == 2:  # Горизонтальные стороны (верхняя и нижняя)
            return f"{width}"
        elif side_index == 1 or side_index == 3:  # Вертикальные стороны (правая и левая)
            return f"{height}"
        
        # В других случаях отображаем вычисленную длину
        return f"{computed_length:.2f}".rstrip('0').rstrip('.')
    
    @staticmethod
    def from_text(params_text):
        """
        Создает объект прямоугольника из текста параметров.
        
        Args:
            params_text (str): Текст параметров
            
        Returns:
            Rectangle: Объект прямоугольника с параметрами из текста
        """
        from app.prompts.prompts import DEFAULT_VISUALIZATION_PARAMS, REGEX_PATTERNS
        
        # Создаем копию параметров по умолчанию
        params = DEFAULT_VISUALIZATION_PARAMS["rectangle"].copy()
        
        # Извлекаем параметры с помощью регулярных выражений
        
        # Проверяем, является ли прямоугольник квадратом
        is_square_match = re.search(r'Квадрат\s*:\s*\[(.*?)\]', params_text, re.IGNORECASE)
        if is_square_match:
            is_square_value = is_square_match.group(1).strip().lower()
            params['is_square'] = is_square_value in ['true', 'да', 'yes', '+']
        
        # Размеры прямоугольника
        dimensions_match = re.search(REGEX_PATTERNS['rectangle']['dimensions'], params_text, re.IGNORECASE)
        if dimensions_match:
            dimensions_str = dimensions_match.group(1).strip()
            dimensions = [dim.strip() for dim in dimensions_str.split(',')]
            
            if len(dimensions) >= 1 and dimensions[0] != "-":
                try:
                    width = float(dimensions[0])
                    params['width'] = width
                except ValueError:
                    logging.warning(f"Не удалось преобразовать ширину: {dimensions[0]}")
            
            if len(dimensions) >= 2 and dimensions[1] != "-":
                try:
                    height = float(dimensions[1])
                    params['height'] = height
                except ValueError:
                    logging.warning(f"Не удалось преобразовать высоту: {dimensions[1]}")
        
        # Если квадрат, устанавливаем одинаковую ширину и высоту
        if params.get('is_square', False):
            # Если указана только одна сторона в размерах, используем её
            if 'width' in params and 'height' not in params:
                params['height'] = params['width']
            elif 'height' in params and 'width' not in params:
                params['width'] = params['height']
            # Если указаны обе стороны, но они разные, используем ширину
            elif 'width' in params and 'height' in params and params['width'] != params['height']:
                params['height'] = params['width']
        
        # Метки вершин
        vertex_labels_match = re.search(REGEX_PATTERNS['rectangle']['vertex_labels'], params_text, re.IGNORECASE)
        if vertex_labels_match:
            vertex_labels_str = vertex_labels_match.group(1).strip()
            vertex_labels = [label.strip() for label in vertex_labels_str.split(',')]
            params['vertex_labels'] = vertex_labels
        
        # Длины сторон для отображения
        side_lengths_match = re.search(REGEX_PATTERNS['rectangle']['side_lengths'], params_text, re.IGNORECASE)
        if side_lengths_match:
            sides_str = side_lengths_match.group(1).strip()
            sides = []
            
            for side in sides_str.split(','):
                side = side.strip()
                if side == "-":
                    sides.append(None)
                else:
                    try:
                        sides.append(float(side))
                    except ValueError:
                        sides.append(None)
                        logging.warning(f"Не удалось преобразовать длину стороны: {side}")
            
            params['side_lengths'] = sides
            params['show_lengths'] = True
        
        # Отображение диагоналей
        show_diagonals_match = re.search(REGEX_PATTERNS['rectangle']['show_diagonals'], params_text, re.IGNORECASE)
        if show_diagonals_match:
            show_diagonals_value = show_diagonals_match.group(1).strip().lower()
            params['show_diagonals'] = show_diagonals_value in ['true', 'да', 'yes', '+']
        
        # Длины диагоналей
        diagonals_length_match = re.search(REGEX_PATTERNS['rectangle']['diagonals_length'], params_text, re.IGNORECASE)
        if diagonals_length_match:
            diagonals_str = diagonals_length_match.group(1).strip()
            diagonals = []
            
            for diagonal in diagonals_str.split(','):
                diagonal = diagonal.strip()
                if diagonal == "-":
                    diagonals.append(None)
                else:
                    try:
                        diagonals.append(float(diagonal))
                    except ValueError:
                        diagonals.append(None)
                        logging.warning(f"Не удалось преобразовать длину диагонали: {diagonal}")
            
            params['diagonals_length'] = diagonals
            # Если заданы диагонали, автоматически показываем их
            params['show_diagonals'] = True
        
        # Показывать углы
        show_angles_match = re.search(REGEX_PATTERNS['rectangle']['show_angles'], params_text, re.IGNORECASE)
        if show_angles_match:
            show_angles_value = show_angles_match.group(1).strip().lower()
            params['show_angles'] = show_angles_value in ['true', 'да', 'yes', '+']
        
        # Создаем объект прямоугольника
        rectangle = Rectangle(params)
        return rectangle 