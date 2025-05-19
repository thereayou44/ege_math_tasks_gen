import numpy as np
import re
from app.geometry.base import GeometricFigure
import matplotlib.pyplot as plt
import logging

class Triangle(GeometricFigure):
    """
    Класс для представления и визуализации треугольника.
    """
    
    def __init__(self, params=None):
        """
        Инициализирует треугольник с заданными параметрами.
        
        Args:
            params (dict): Словарь параметров фигуры
        """
        super().__init__(params)
        self.figure_type = "triangle"
        
        # Установка параметров по умолчанию, если не указаны
        if 'is_right' not in self.params:
            self.params['is_right'] = False
        if 'is_equilateral' not in self.params:
            self.params['is_equilateral'] = False
        if 'is_isosceles' not in self.params:
            self.params['is_isosceles'] = False
        if 'show_heights' not in self.params:
            self.params['show_heights'] = False
        if 'show_medians' not in self.params:
            self.params['show_medians'] = False
        if 'show_angle_bisectors' not in self.params:
            self.params['show_angle_bisectors'] = False
            
    def compute_points(self):
        """
        Вычисляет координаты точек треугольника.
        
        Returns:
            list: Список точек [(x1,y1), (x2,y2), (x3,y3)]
        """
        # Проверяем, были ли указаны конкретные координаты
        if 'coords' in self.params and len(self.params['coords']) >= 3:
            return self.params['coords'][:3]
        
        # Получаем параметры
        is_right = self.params.get('is_right', False)
        is_equilateral = self.params.get('is_equilateral', False)
        is_isosceles = self.params.get('is_isosceles', False)
        sides = self.params.get('sides', None)
        angles = self.params.get('angles', None)
        
        # Если треугольник прямоугольный
        if is_right:
            # Создаем прямоугольный треугольник с прямым углом в точке (0,0)
            return [(0, 0), (4, 0), (0, 3)]
            
        # Если треугольник равносторонний
        elif is_equilateral:
            # Создаем равносторонний треугольник
            side = 4  # Длина стороны
            height = side * np.sqrt(3) / 2
            return [(0, 0), (side, 0), (side/2, height)]
            
        # Если треугольник равнобедренный
        elif is_isosceles:
            # Создаем равнобедренный треугольник
            base = 4  # Длина основания
            height = 3  # Высота
            return [(0, 0), (base, 0), (base/2, height)]
            
        # По умолчанию - произвольный треугольник
        else:
            return [(0, 0), (4, 0), (1, 3)]
    
    def draw(self, ax=None):
        """
        Отрисовывает треугольник с дополнительными элементами.
        
        Args:
            ax (matplotlib.axes.Axes): Оси для отрисовки
            
        Returns:
            matplotlib.axes.Axes: Оси с нарисованной фигурой
        """
        # Используем базовый метод для рисования основной фигуры
        ax = super().draw(ax)
        
        # Добавляем высоты, если они должны быть отображены
        if self.params.get('show_heights', False):
            height_vertices = self.params.get('height_vertices', [])
            self._add_heights(ax, height_vertices)
            
        # Добавляем медианы, если нужно
        if self.params.get('show_medians', False):
            median_vertices = self.params.get('median_vertices', [])
            self._add_medians(ax, median_vertices)
            
        # Добавляем биссектрисы углов, если нужно
        if self.params.get('show_angle_bisectors', False):
            bisector_vertices = self.params.get('bisector_vertices', [])
            self._add_angle_bisectors(ax, bisector_vertices)
            
        # Отметка прямого угла, если треугольник прямоугольный
        if self.params.get('is_right', False):
            self._mark_right_angle(ax)
            
        return ax
    
    def _add_heights(self, ax, vertices=None):
        """
        Добавляет высоты треугольника.
        
        Args:
            ax (matplotlib.axes.Axes): Оси для отрисовки
            vertices (list): Список вершин, из которых проводятся высоты
        """
        points = self.points
        vertex_labels = self.params.get('vertex_labels', ['A', 'B', 'C'])
        
        # Если не указаны конкретные вершины, проводим все высоты
        if not vertices:
            vertices = vertex_labels
            
        for i, label in enumerate(vertex_labels):
            if label in vertices or not vertices:
                # Индексы двух других вершин
                j = (i + 1) % 3
                k = (i + 2) % 3
                
                # Координаты вершин
                A = np.array(points[i])
                B = np.array(points[j])
                C = np.array(points[k])
                
                # Вектор BC
                BC = C - B
                
                # Вычисляем проекцию
                BC_norm = BC / np.linalg.norm(BC)
                AB = A - B
                projection = B + np.dot(AB, BC_norm) * BC_norm
                
                # Рисуем высоту как прерывистую линию
                ax.plot([A[0], projection[0]], [A[1], projection[1]], 'r--', lw=1.5)
                
                # Отмечаем основание высоты
                ax.plot(projection[0], projection[1], 'ro', markersize=4)
                
                # Добавляем подпись высоты
                mid_x = (A[0] + projection[0]) / 2
                mid_y = (A[1] + projection[1]) / 2
                
                # Вычисляем длину высоты
                height_length = np.linalg.norm(A - projection)
                
                # Добавляем подпись
                height_text = f"h{label} = {height_length:.2f}"
                ax.text(mid_x, mid_y, height_text, 
                        ha='center', va='center', color='red', fontsize=10,
                        bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', boxstyle='round,pad=0.3'))
    
    def _add_medians(self, ax, vertices=None):
        """
        Добавляет медианы треугольника.
        
        Args:
            ax (matplotlib.axes.Axes): Оси для отрисовки
            vertices (list): Список вершин, из которых проводятся медианы
        """
        points = self.points
        vertex_labels = self.params.get('vertex_labels', ['A', 'B', 'C'])
        
        # Если не указаны конкретные вершины, проводим все медианы
        if not vertices:
            vertices = vertex_labels
            
        for i, label in enumerate(vertex_labels):
            if label in vertices or not vertices:
                # Индексы двух других вершин
                j = (i + 1) % 3
                k = (i + 2) % 3
                
                # Координаты вершин
                A = np.array(points[i])
                B = np.array(points[j])
                C = np.array(points[k])
                
                # Середина противоположной стороны
                midpoint = (B + C) / 2
                
                # Рисуем медиану как прерывистую линию
                ax.plot([A[0], midpoint[0]], [A[1], midpoint[1]], 'g--', lw=1.5)
                
                # Отмечаем середину стороны
                ax.plot(midpoint[0], midpoint[1], 'go', markersize=4)
                
                # Добавляем подпись медианы
                mid_x = (A[0] + midpoint[0]) / 2
                mid_y = (A[1] + midpoint[1]) / 2
                
                # Вычисляем длину медианы
                median_length = np.linalg.norm(A - midpoint)
                
                # Добавляем подпись
                median_text = f"m{label} = {median_length:.2f}"
                ax.text(mid_x, mid_y, median_text, 
                        ha='center', va='center', color='green', fontsize=10,
                        bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', boxstyle='round,pad=0.3'))
    
    def _add_angle_bisectors(self, ax, vertices=None):
        """
        Добавляет биссектрисы углов треугольника.
        
        Args:
            ax (matplotlib.axes.Axes): Оси для отрисовки
            vertices (list): Список вершин, из которых проводятся биссектрисы
        """
        points = self.points
        vertex_labels = self.params.get('vertex_labels', ['A', 'B', 'C'])
        
        # Если не указаны конкретные вершины, проводим все биссектрисы
        if not vertices:
            vertices = vertex_labels
            
        for i, label in enumerate(vertex_labels):
            if label in vertices or not vertices:
                # Индексы двух других вершин
                j = (i + 1) % 3
                k = (i + 2) % 3
                
                # Координаты вершин
                A = np.array(points[i])
                B = np.array(points[j])
                C = np.array(points[k])
                
                # Векторы от вершины к двум другим вершинам
                AB = B - A
                AC = C - A
                
                # Нормализуем векторы
                AB_norm = AB / np.linalg.norm(AB)
                AC_norm = AC / np.linalg.norm(AC)
                
                # Биссектриса угла
                bisector = AB_norm + AC_norm
                bisector = bisector / np.linalg.norm(bisector)
                
                # Находим точку пересечения с противоположной стороной
                # Решаем систему уравнений для нахождения параметра t
                # Уравнение прямой: P = B + t * (C - B)
                BC = C - B
                
                # Параметрические уравнения для точки на стороне BC и луча из A
                # A + s * bisector = B + t * BC, где 0 <= t <= 1
                
                # Преобразуем в матричное уравнение
                # [bisector, -BC] * [s, t]^T = B - A
                M = np.column_stack([bisector, -BC])
                b = B - A
                
                try:
                    # Решаем систему
                    s, t = np.linalg.solve(M, b)
                    
                    # Проверяем, что точка лежит на отрезке BC
                    if 0 <= t <= 1 and s > 0:
                        # Точка пересечения
                        intersection = B + t * BC
                        
                        # Рисуем биссектрису как прерывистую линию
                        ax.plot([A[0], intersection[0]], [A[1], intersection[1]], 'b--', lw=1.5)
                        
                        # Отмечаем точку пересечения
                        ax.plot(intersection[0], intersection[1], 'bo', markersize=4)
                        
                        # Добавляем подпись биссектрисы
                        mid_x = (A[0] + intersection[0]) / 2
                        mid_y = (A[1] + intersection[1]) / 2
                        
                        # Вычисляем длину биссектрисы
                        bisector_length = np.linalg.norm(A - intersection)
                        
                        # Добавляем подпись
                        bisector_text = f"l{label} = {bisector_length:.2f}"
                        ax.text(mid_x, mid_y, bisector_text, 
                                ha='center', va='center', color='blue', fontsize=10,
                                bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', boxstyle='round,pad=0.3'))
                except np.linalg.LinAlgError:
                    logging.warning(f"Не удалось вычислить биссектрису из вершины {label}")
    
    def _mark_right_angle(self, ax):
        """
        Отмечает прямой угол в прямоугольном треугольнике.
        
        Args:
            ax (matplotlib.axes.Axes): Оси для отрисовки
        """
        points = self.points
        vertex_labels = self.params.get('vertex_labels', ['A', 'B', 'C'])
        
        # Определяем прямой угол
        right_angle_idx = None
        for i in range(3):
            j = (i + 1) % 3
            k = (i + 2) % 3
            
            v1 = np.array(points[j]) - np.array(points[i])
            v2 = np.array(points[k]) - np.array(points[i])
            
            # Проверяем на перпендикулярность
            if abs(np.dot(v1, v2)) < 1e-10:
                right_angle_idx = i
                break
        
        if right_angle_idx is not None:
            # Координаты вершины с прямым углом
            p = np.array(points[right_angle_idx])
            
            # Координаты соседних вершин
            p1 = np.array(points[(right_angle_idx + 1) % 3])
            p2 = np.array(points[(right_angle_idx + 2) % 3])
            
            # Векторы к соседним вершинам
            v1 = p1 - p
            v2 = p2 - p
            
            # Нормализуем векторы
            v1_norm = v1 / np.linalg.norm(v1)
            v2_norm = v2 / np.linalg.norm(v2)
            
            # Масштаб для отметки прямого угла
            scale = min(np.linalg.norm(v1), np.linalg.norm(v2)) * 0.2
            
            # Точки для отметки прямого угла
            mark_p1 = p + v1_norm * scale
            mark_p2 = p + v2_norm * scale
            mark_p3 = p + v1_norm * scale + v2_norm * scale
            
            # Рисуем отметку прямого угла
            ax.plot([p[0], mark_p1[0], mark_p3[0], mark_p2[0], p[0]], 
                    [p[1], mark_p1[1], mark_p3[1], mark_p2[1], p[1]], 
                    'r-', lw=1.5)
    
    @staticmethod
    def from_text(params_text):
        """
        Создает объект треугольника из текста параметров.
        
        Args:
            params_text (str): Текст параметров
            
        Returns:
            Triangle: Объект треугольника с параметрами из текста
        """
        from app.prompts.prompts import DEFAULT_VISUALIZATION_PARAMS, REGEX_PATTERNS
        
        # Создаем копию параметров по умолчанию
        params = DEFAULT_VISUALIZATION_PARAMS["triangle"].copy()
        
        # Извлекаем параметры с помощью регулярных выражений
        
        # Метки вершин
        vertex_labels_match = re.search(REGEX_PATTERNS['triangle']['vertex_labels'], params_text, re.IGNORECASE)
        if vertex_labels_match:
            vertex_labels_str = vertex_labels_match.group(1).strip()
            vertex_labels = [label.strip() for label in vertex_labels_str.split(',')]
            params['vertex_labels'] = vertex_labels
            
        # Длины сторон
        sides_match = re.search(REGEX_PATTERNS['triangle']['sides'], params_text, re.IGNORECASE)
        if sides_match:
            sides_str = sides_match.group(1).strip()
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
                        logging.warning(f"Не удалось преобразовать значение стороны: {side}")
            
            params['side_lengths'] = sides
            params['show_lengths'] = True
            
        # Углы треугольника
        angles_match = re.search(REGEX_PATTERNS['triangle']['angles'], params_text, re.IGNORECASE)
        if angles_match:
            angles_str = angles_match.group(1).strip()
            angles = []
            
            for angle in angles_str.split(','):
                angle = angle.strip()
                if angle == "-":
                    angles.append(None)
                else:
                    try:
                        angles.append(float(angle))
                    except ValueError:
                        angles.append(None)
                        logging.warning(f"Не удалось преобразовать значение угла: {angle}")
            
            params['angle_values'] = angles
            params['show_angles'] = True
            
        # Проверяем на прямоугольный треугольник
        is_right_match = re.search(REGEX_PATTERNS['triangle']['is_right'], params_text, re.IGNORECASE)
        if is_right_match:
            is_right_value = is_right_match.group(1).strip().lower()
            params['is_right'] = is_right_value in ['true', 'да', 'yes', '+']
        
        # Показывать высоты
        show_heights_match = re.search(REGEX_PATTERNS['triangle']['show_heights'], params_text, re.IGNORECASE)
        if show_heights_match:
            heights_str = show_heights_match.group(1).strip()
            if heights_str.lower() in ['true', 'да', 'yes', '+']:
                params['show_heights'] = True
                params['height_vertices'] = []
            else:
                # Если указаны конкретные вершины
                params['show_heights'] = True
                params['height_vertices'] = [v.strip() for v in heights_str.split(',')]
                
        # Показывать медианы
        show_medians_match = re.search(REGEX_PATTERNS['triangle']['show_medians'], params_text, re.IGNORECASE)
        if show_medians_match:
            medians_str = show_medians_match.group(1).strip()
            if medians_str.lower() in ['true', 'да', 'yes', '+']:
                params['show_medians'] = True
                params['median_vertices'] = []
            else:
                # Если указаны конкретные вершины
                params['show_medians'] = True
                params['median_vertices'] = [v.strip() for v in medians_str.split(',')]
                
        # Показывать средние линии
        show_midlines_match = re.search(REGEX_PATTERNS['triangle']['show_midlines'], params_text, re.IGNORECASE)
        if show_midlines_match:
            params['show_midlines'] = show_midlines_match.group(1).strip().lower() in ['true', 'да', 'yes', '+']
        
        # Создаем объект треугольника
        triangle = Triangle(params)
        return triangle 