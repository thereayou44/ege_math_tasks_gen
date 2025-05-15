import numpy as np
import re
from app.geometry.base import GeometricFigure

class Rectangle(GeometricFigure):
    """
    Класс для представления и визуализации прямоугольника.
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
        if 'x' not in self.params:
            self.params['x'] = 0
        if 'y' not in self.params:
            self.params['y'] = 0
        if 'show_heights' not in self.params:
            self.params['show_heights'] = False
            
    def compute_points(self):
        """
        Вычисляет координаты точек прямоугольника.
        
        Returns:
            list: Список точек [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
        """
        x = self.params.get('x', 0)
        y = self.params.get('y', 0)
        width = self.params.get('width', 4)
        height = self.params.get('height', 3)
        
        # Вычисляем координаты вершин прямоугольника
        return [(x, y), (x + width, y), (x + width, y + height), (x, y + height)]
    
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
        # Сначала проверяем, есть ли прямое указание длины в side_lengths
        side_lengths = self.params.get('side_lengths', None)
        if side_lengths and len(side_lengths) > side_index and side_lengths[side_index] is not None:
            return f"{side_lengths[side_index]}"
        
        # Если нет, используем стандартную логику для прямоугольника
        if side_index == 0 or side_index == 2:  # Горизонтальные стороны (AB и CD)
            if 'width' in self.params:
                return f"{self.params['width']}"
        elif side_index == 1 or side_index == 3:  # Вертикальные стороны (BC и DA)
            if 'height' in self.params:
                return f"{self.params['height']}"
        
        # В других случаях используем стандартную логику из базового класса
        return super().get_side_length_text(side_index, side_name, computed_length)
    
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
        
        # Добавляем высоты, если они должны быть отображены
        if self.params.get('show_heights', False):
            self._add_heights(ax)
            
        return ax
    
    def _add_heights(self, ax):
        """
        Добавляет отображение высот прямоугольника.
        
        Args:
            ax (matplotlib.axes.Axes): Оси для отрисовки
        """
        points = self.compute_points()
        x, y = self.params.get('x', 0), self.params.get('y', 0)
        width, height = self.params.get('width', 4), self.params.get('height', 3)
        
        # Рисуем высоту от нижней стороны
        height_offset = width * 0.1  # смещение для отображения высоты
        ax.plot([x + height_offset, x + height_offset], [y, y + height], 'r--', lw=1)
        
        # Добавляем подпись высоты
        height_text = f"h={height}"
        ax.text(x + height_offset * 0.5, y + height/2, height_text, ha='right', va='center', 
                color='red', fontsize=10, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        
        # При необходимости можно добавить маркеры прямых углов
        if self.params.get('show_right_angles', False):
            angle_size = min(width, height) * 0.1
            for i, point in enumerate(points):
                px, py = point
                if i == 0:  # Левый нижний угол
                    ax.plot([px, px + angle_size], [py, py], 'k-', lw=0.8)
                    ax.plot([px, px], [py, py + angle_size], 'k-', lw=0.8)
                elif i == 1:  # Правый нижний угол
                    ax.plot([px, px - angle_size], [py, py], 'k-', lw=0.8)
                    ax.plot([px, px], [py, py + angle_size], 'k-', lw=0.8)
                elif i == 2:  # Правый верхний угол
                    ax.plot([px, px - angle_size], [py, py], 'k-', lw=0.8)
                    ax.plot([px, px], [py, py - angle_size], 'k-', lw=0.8)
                elif i == 3:  # Левый верхний угол
                    ax.plot([px, px + angle_size], [py, py], 'k-', lw=0.8)
                    ax.plot([px, px], [py, py - angle_size], 'k-', lw=0.8)
    
    def add_vertex_labels(self, ax):
        """
        Добавляет подписи вершин фигуры.
        
        Args:
            ax (matplotlib.axes.Axes): Оси для отрисовки
        """
        if self.params.get('show_labels', True):
            show_specific_labels = self.params.get('show_specific_labels', None)
            labels = self.params.get('vertex_labels')
            
            # Важное исправление: Если метки не заданы, добавим их
            if not labels or len(labels) < len(self.points):
                labels = [chr(65+i) for i in range(len(self.points))]
                self.params['vertex_labels'] = labels  # Сохраняем для использования в других методах
            
            for i, ((x0,y0), lab) in enumerate(zip(self.points, labels)):
                # Проверяем, нужно ли отображать эту конкретную метку
                if show_specific_labels is None or lab in show_specific_labels:
                    ax.text(x0, y0, lab, ha='center', va='center', fontsize=14,
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
    def add_side_lengths(self, ax):
        """
        Добавляет подписи длин сторон фигуры.
        
        Args:
            ax (matplotlib.axes.Axes): Оси для отрисовки
        """
        side_lengths = self.params.get('side_lengths', None)
        show_lengths = self.params.get('show_lengths', False)
        show_specific_sides = self.params.get('show_specific_sides', None)
        
        if side_lengths or show_lengths:
            vertex_labels = self.params.get('vertex_labels', [chr(65+j) for j in range(len(self.points))])

            if not vertex_labels or len(vertex_labels) < len(self.points):
                vertex_labels = [chr(65+j) for j in range(len(self.points))]
                self.params['vertex_labels'] = vertex_labels
                    
            for i in range(len(self.points)):
                p1 = self.points[i]
                p2 = self.points[(i+1) % len(self.points)]
                
                # Получаем обозначения вершин для этой стороны
                v1 = vertex_labels[i]
                v2 = vertex_labels[(i+1) % len(self.points)]
                side_name = f"{v1}{v2}"
                side_name_rev = f"{v2}{v1}"  # Обратный порядок для проверки
                
                # Проверяем, нужно ли отображать эту конкретную сторону
                should_show = show_specific_sides is None or side_name in show_specific_sides or side_name_rev in show_specific_sides
                
                if should_show:
                    # Вычисляем длину стороны
                    L = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
                    
                    # Середина стороны для размещения подписи
                    mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
                    
                    # Вектор нормали к стороне для размещения текста
                    nx, ny = -(p2[1] - p1[1]) / L, (p2[0] - p1[0]) / L
                    offset = 0.2  # Уменьшенный отступ для лучшей видимости
                    
                    # Обработка специфичная для подклассов
                    text_value = self.get_side_length_text(i, side_name, L)
                    
                    # Отображаем значение
                    if text_value is not None:
                        ax.text(mx + nx*offset, my + ny*offset, text_value, 
                                ha='center', fontsize=12,
                                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
    def add_angles(self, ax):
        """
        Добавляет отображение углов фигуры.
        
        Args:
            ax (matplotlib.axes.Axes): Оси для отрисовки
        """
        if self.params.get('show_angles', False):
            angle_values = self.params.get('angle_values', None)
            show_angle_arcs = self.params.get('show_angle_arcs', False)
            show_specific_angles = self.params.get('show_specific_angles', None)
            
            vertex_labels = self.params.get('vertex_labels', [chr(65+j) for j in range(len(self.points))])
            
            for i in range(len(self.points)):
                # Получаем обозначение вершины для проверки
                vertex_label = vertex_labels[i]
                
                # Проверяем, нужно ли отображать этот угол
                if show_specific_angles is not None and vertex_label not in show_specific_angles:
                    continue
                
                # Получаем три последовательные точки для вычисления угла
                A = np.array(self.points[(i-1) % len(self.points)])
                B = np.array(self.points[i])
                C = np.array(self.points[(i+1) % len(self.points)])
                
                # Вычисляем векторы сторон
                v1 = A - B
                v2 = C - B
                
                # Для прямоугольника все углы должны быть 90 градусов
                angle_text = "90°"
                
                # Радиус дуги
                radius = min(np.linalg.norm(v1), np.linalg.norm(v2)) / 4
                
                # Выбираем координаты для текста угла - ближе к вершине
                angle_mid = (v1/np.linalg.norm(v1) + v2/np.linalg.norm(v2))
                angle_mid = angle_mid / np.linalg.norm(angle_mid) * radius * 0.8
                
                # Отображаем значение угла
                ax.text(B[0] + angle_mid[0], B[1] + angle_mid[1], angle_text, 
                        ha='center', fontsize=10,
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
    @staticmethod
    def from_text(task_text):
        """
        Создает объект прямоугольника из текста задачи.
        
        Args:
            task_text (str): Текст задачи
            
        Returns:
            Rectangle: Объект прямоугольника с параметрами из текста
        """
        import re
        from app.prompts import DEFAULT_VISUALIZATION_PARAMS
        
        params = DEFAULT_VISUALIZATION_PARAMS["rectangle"].copy()
        
        # Ищем размеры прямоугольника
        dimensions_pattern = r'(прямоугольник|ширин[а-я]+|длин[а-я]+)[а-я\s]*[=:]?\s*(\d+(?:[,.]\d+)?)[а-я\s,]*[=:]?\s*(?:и|[а-я]+\s+)?(\d+(?:[,.]\d+)?)'
        dimensions_match = re.search(dimensions_pattern, task_text, re.IGNORECASE)
        
        if dimensions_match:
            try:
                width = float(dimensions_match.group(2).replace(',', '.'))
                height = float(dimensions_match.group(3).replace(',', '.'))
                params['width'] = width
                params['height'] = height
            except Exception:
                pass
        
        # Проверяем, является ли прямоугольник квадратом
        if re.search(r'квадрат', task_text, re.IGNORECASE):
            # Ищем сторону квадрата
            square_side_pattern = r'квадрат[а-я\s]*[=:]?\s*(\d+(?:[,.]\d+)?)'
            square_side_match = re.search(square_side_pattern, task_text, re.IGNORECASE)
            
            if square_side_match:
                try:
                    side = float(square_side_match.group(1).replace(',', '.'))
                    params['width'] = side
                    params['height'] = side
                except Exception:
                    pass
        
        # Ищем высоту прямоугольника
        height_pattern = r'высот[а-я]*\s+прямоугольник[а-я]*\s*[=:]?\s*(\d+(?:[,.]\d+)?)'
        height_match = re.search(height_pattern, task_text, re.IGNORECASE)
        
        if height_match:
            try:
                height_value = float(height_match.group(1).replace(',', '.'))
                params['height'] = height_value
                params['show_heights'] = True
            except Exception:
                pass
        
        # Показываем длины сторон, если это указано в тексте или если это задача на вычисление площади/периметра
        if re.search(r'найд[а-я]+\s+сторон|найд[а-я]+\s+длин|длин[а-я]+\s+сторон|площадь|периметр', task_text, re.IGNORECASE):
            params['show_lengths'] = True
        
        # Показываем углы, если это указано в тексте
        if re.search(r'найд[а-я]+\s+угл|угл[а-я]+\s+прямоугольник|величин[а-я]+\s+угл', task_text, re.IGNORECASE):
            params['show_angles'] = True
            params['show_right_angles'] = True
        
        # Показываем высоты, если это указано в тексте
        if re.search(r'высот[а-я]+|найд[а-я]+\s+высот[а-я]+|вычисл[а-я]+\s+высот', task_text, re.IGNORECASE):
            params['show_heights'] = True
        
        # Определяем, какие конкретно стороны или углы нужно показать
        if "сторон" in task_text.lower():
            # Проверяем, упоминаются ли конкретные стороны
            side_names = re.findall(r'сторон[а-я]*\s+([A-Z]{2})', task_text)
            if side_names:
                params['show_specific_sides'] = side_names
        
        if "угол" in task_text.lower() or "углы" in task_text.lower():
            # Проверяем, упоминаются ли конкретные углы
            angle_names = re.findall(r'угл[а-я]*\s+([A-Z])', task_text)
            if angle_names:
                params['show_specific_angles'] = angle_names
        
        return Rectangle(params) 