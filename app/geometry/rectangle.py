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
        
        # Проверяем, равны ли ширина и высота для квадрата
        if abs(width - height) < 1e-6:
            # Для квадрата устанавливаем равные стороны
            self.params['is_square'] = True
        else:
            self.params['is_square'] = False
        
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
            
        # Добавляем диагонали, если они должны быть отображены
        if self.params.get('show_diagonals', False):
            self._add_diagonals(ax)
            
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
    
    def _add_diagonals(self, ax):
        """
        Добавляет отображение диагоналей прямоугольника.
        
        Args:
            ax (matplotlib.axes.Axes): Оси для отрисовки
        """
        points = self.compute_points()
        
        # Рисуем диагональ из нижнего левого угла (0) в верхний правый (2)
        ax.plot([points[0][0], points[2][0]], [points[0][1], points[2][1]], 'b--', lw=1)
        
        # Рисуем диагональ из нижнего правого угла (1) в верхний левый (3)
        ax.plot([points[1][0], points[3][0]], [points[1][1], points[3][1]], 'b--', lw=1)
        
        # Вычисляем длины диагоналей
        diag1_length = np.sqrt((points[2][0] - points[0][0])**2 + (points[2][1] - points[0][1])**2)
        diag2_length = np.sqrt((points[3][0] - points[1][0])**2 + (points[3][1] - points[1][1])**2)
        
        # Получаем значения диагоналей (если заданы явно)
        diagonals_length = self.params.get('diagonals_length', None)
        if diagonals_length and len(diagonals_length) >= 2:
            diag1_text = str(diagonals_length[0])
            diag2_text = str(diagonals_length[1])
        else:
            # Используем вычисленные значения
            diag1_text = f"{diag1_length:.2f}".rstrip('0').rstrip('.')
            diag2_text = f"{diag2_length:.2f}".rstrip('0').rstrip('.')
        
        # Отображаем значения диагоналей в центре
        mid_x = (points[0][0] + points[2][0]) / 2
        mid_y = (points[0][1] + points[2][1]) / 2
        
        # Размещаем метки диагоналей немного в стороне, чтобы они не накладывались
        offset = min(self.params.get('width', 4), self.params.get('height', 3)) * 0.1
        
        ax.text(mid_x - offset, mid_y - offset, diag1_text, ha='center', va='center', 
                color='blue', fontsize=10, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        
        ax.text(mid_x + offset, mid_y + offset, diag2_text, ha='center', va='center', 
                color='blue', fontsize=10, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
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
        from app.prompts import DEFAULT_VISUALIZATION_PARAMS, REGEX_PATTERNS
        
        params = DEFAULT_VISUALIZATION_PARAMS["rectangle"].copy()
        rect_patterns = REGEX_PATTERNS["rectangle"]
        
        # Функция для извлечения параметра по соответствующему регулярному выражению
        def extract_param(param_name, default=None, convert_type=None):
            pattern = rect_patterns.get(param_name)
            if not pattern:
                return default
            
            match = re.search(pattern, task_text, re.IGNORECASE)
            if not match:
                return default
            
            value = match.group(1).strip().replace(',', '.')
            if convert_type:
                try:
                    return convert_type(value)
                except:
                    return default
            return value
        
        # Функция для преобразования строки со списком чисел в список float
        def parse_numeric_list(input_str):
            if not input_str:
                return None
            try:
                # Ищем все числа в строке, включая десятичные и отрицательные
                numbers = re.findall(r'[-+]?\d*\.?\d+', input_str)
                return [float(num) for num in numbers] if numbers else None
            except:
                return None
        
        # Извлекаем размеры прямоугольника
        dimensions = extract_param("dimensions")
        if dimensions:
            dimensions_list = parse_numeric_list(dimensions)
            if dimensions_list and len(dimensions_list) >= 2:
                params['width'] = dimensions_list[0]
                params['height'] = dimensions_list[1]
        
        # Извлекаем метки вершин
        vertex_labels = extract_param("vertex_labels")
        if vertex_labels:
            # Извлекаем буквы из строки
            labels = re.findall(r'[A-Za-z]', vertex_labels)
            if len(labels) >= 4:
                params['vertex_labels'] = labels[:4]
        
        # Извлекаем параметры для отображения
        show_dimensions = extract_param("show_dimensions", "False", lambda x: x.lower() == "true")
        if show_dimensions:
            params['show_lengths'] = True
        
        # Извлекаем значения длин сторон
        side_lengths = extract_param("side_lengths")
        if side_lengths:
            lengths_list = parse_numeric_list(side_lengths)
            if lengths_list:
                params['side_lengths'] = lengths_list
        
        # Проверяем необходимость отображения углов
        show_angles = extract_param("show_angles", "False", lambda x: x.lower() == "true")
        if show_angles:
            params['show_angles'] = True
        
        # Извлекаем параметры для диагоналей
        show_diagonals = extract_param("show_diagonals", "False", lambda x: x.lower() == "true")
        if show_diagonals:
            params['show_diagonals'] = True
        
        # Извлекаем значения длин диагоналей
        diagonals_length = extract_param("diagonals_length")
        if diagonals_length:
            diag_lengths = parse_numeric_list(diagonals_length)
            if diag_lengths:
                params['diagonals_length'] = diag_lengths
        
        # Дополнительная проверка на ключевые слова в тексте
        if re.search(r'квадрат|равносторонний прямоугольник', task_text, re.IGNORECASE):
            # Делаем стороны равными
            side_length = max(params.get('width', 4), params.get('height', 3))
            params['width'] = side_length
            params['height'] = side_length
        
        # Проверяем на наличие упоминаний о диагоналях
        if re.search(r'диагонал[а-я]+', task_text, re.IGNORECASE):
            params['show_diagonals'] = True
        
        return Rectangle(params) 