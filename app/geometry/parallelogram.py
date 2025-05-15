import numpy as np
import re
from app.geometry.base import GeometricFigure

class Parallelogram(GeometricFigure):
    """
    Класс для представления и визуализации параллелограмма.
    """
    
    def __init__(self, params=None):
        """
        Инициализирует параллелограмм с заданными параметрами.
        
        Args:
            params (dict): Словарь параметров фигуры
        """
        super().__init__(params)
        self.figure_type = "parallelogram"
        
        # Установка параметров по умолчанию, если не указаны
        if 'width' not in self.params:
            self.params['width'] = 4
        if 'height' not in self.params:
            self.params['height'] = 3
        if 'skew' not in self.params:
            self.params['skew'] = 60  # Угол наклона в градусах
        if 'x' not in self.params:
            self.params['x'] = 0
        if 'y' not in self.params:
            self.params['y'] = 0
        if 'show_heights' not in self.params:
            self.params['show_heights'] = False
        if 'angle_values' not in self.params:
            # Вычисляем значения углов на основе угла наклона
            skew_deg = self.params.get('skew', 60)
            acute_angle = skew_deg
            obtuse_angle = 180 - skew_deg
            self.params['angle_values'] = [acute_angle, obtuse_angle, acute_angle, obtuse_angle]
            
    def compute_points(self):
        """
        Вычисляет координаты точек параллелограмма.
        
        Returns:
            list: Список точек [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
        """
        x = self.params.get('x', 0)
        y = self.params.get('y', 0)
        width = self.params.get('width', 4)
        height = self.params.get('height', 3)
        skew_deg = self.params.get('skew', 60)
        
        # Вычисляем dx для наклона
        skew_rad = np.radians(skew_deg)
        dx = height / np.tan(skew_rad) if skew_deg != 90 else 0
        
        # Вычисляем координаты вершин параллелограмма
        raw_points = [(0, 0), (width, 0), (width + dx, height), (dx, height)]
        
        # Применяем смещение, если указано
        return [(px + x, py + y) for px, py in raw_points]
    
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
        
        # Если нет, используем стандартную логику для параллелограмма
        if side_index == 0 or side_index == 2:  # Горизонтальные стороны (AB и CD)
            if 'width' in self.params:
                return f"{self.params['width']}"
        elif side_index == 1 or side_index == 3:  # Наклонные стороны (BC и DA)
            # Вычисляем длину наклонной стороны по формуле
            height = self.params.get('height', 3)
            skew_deg = self.params.get('skew', 60)
            skew_rad = np.radians(skew_deg)
            dx = height / np.tan(skew_rad) if skew_deg != 90 else 0
            side_length = np.sqrt(height**2 + dx**2)
            
            # Округляем до 2 знаков после запятой
            return f"{side_length:.2f}".rstrip('0').rstrip('.')
            
        # В других случаях используем стандартную логику из базового класса
        return super().get_side_length_text(side_index, side_name, computed_length)
    
    def draw(self, ax=None):
        """
        Отрисовывает параллелограмм с дополнительными элементами.
        
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
        Добавляет отображение высот параллелограмма.
        
        Args:
            ax (matplotlib.axes.Axes): Оси для отрисовки
        """
        points = self.compute_points()
        height = self.params.get('height', 3)
        
        # Рисуем высоту от первой (нижней) стороны к противоположной
        x1, y1 = points[0]  # Нижняя левая точка
        x2, y2 = points[3]  # Верхняя левая точка
        
        # Рисуем высоту как прерывистую линию
        ax.plot([x1, x2], [y1, y2], 'r--', lw=1)
        
        # Добавляем подпись высоты
        height_text = f"h={height}"
        ax.text((x1 + x2) / 2 - 0.2, (y1 + y2) / 2, height_text, 
                ha='right', va='center', color='red', fontsize=10,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
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
            angle_values = self.params.get('angle_values')
            show_angle_arcs = self.params.get('show_angle_arcs', False)
            show_specific_angles = self.params.get('show_specific_angles', None)
            
            vertex_labels = self.params.get('vertex_labels', [chr(65+j) for j in range(len(self.points))])
            
            for i in range(len(self.points)):
                # Получаем обозначение вершины для проверки
                vertex_label = vertex_labels[i]
                
                # Проверяем, нужно ли отображать этот угол
                if show_specific_angles is not None and vertex_label not in show_specific_angles:
                    continue
                
                # В параллелограмме противоположные углы равны
                # 0 и 2 - это острые углы, 1 и 3 - тупые углы
                angle_idx = i % 2  # 0 или 1
                
                # Получаем значение угла из параметров
                if angle_values and len(angle_values) > angle_idx:
                    angle_text = f"{angle_values[angle_idx]}°"
                else:
                    # Вычисляем приблизительное значение
                    skew_deg = self.params.get('skew', 60)
                    angle = skew_deg if angle_idx == 0 else 180 - skew_deg
                    angle_text = f"{angle:.1f}°"
                
                # Получаем три последовательные точки для отображения
                A = np.array(self.points[(i-1) % len(self.points)])
                B = np.array(self.points[i])
                C = np.array(self.points[(i+1) % len(self.points)])
                
                # Вычисляем векторы сторон
                v1 = A - B
                v2 = C - B
                
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
        Создает объект параллелограмма из текста задачи.
        
        Args:
            task_text (str): Текст задачи
            
        Returns:
            Parallelogram: Объект параллелограмма с параметрами из текста
        """
        import re
        from app.prompts import DEFAULT_VISUALIZATION_PARAMS
        
        params = DEFAULT_VISUALIZATION_PARAMS["parallelogram"].copy()
        
        # Ищем размеры параллелограмма
        dimensions_pattern = r'(параллелограмм|сторон[а-я]+|основани[а-я]+)[а-я\s]*[=:]?\s*(\d+(?:[,.]\d+)?)[а-я\s,]*[=:]?\s*(?:и|[а-я]+\s+)?(\d+(?:[,.]\d+)?)'
        dimensions_match = re.search(dimensions_pattern, task_text, re.IGNORECASE)
        
        if dimensions_match:
            try:
                # Предполагаем, что первое число - это ширина, второе - другая сторона
                width = float(dimensions_match.group(2).replace(',', '.'))
                other_side = float(dimensions_match.group(3).replace(',', '.'))
                params['width'] = width
                
                # Храним вторую сторону для отображения
                if 'side_lengths' not in params:
                    params['side_lengths'] = [None, other_side, None, other_side]
                else:
                    params['side_lengths'][1] = other_side
                    params['side_lengths'][3] = other_side
            except Exception:
                pass
        
        # Ищем высоту параллелограмма
        height_pattern = r'высот[а-я]+\s*[=:]?\s*(\d+(?:[,.]\d+)?)'
        height_match = re.search(height_pattern, task_text, re.IGNORECASE)
        
        if height_match:
            try:
                height = float(height_match.group(1).replace(',', '.'))
                params['height'] = height
                params['show_heights'] = True
            except Exception:
                pass
        
        # Ищем углы параллелограмма
        angle_pattern = r'угол\s*[=:]?\s*(\d+(?:[,.]\d+)?)[°\s]'
        angle_match = re.search(angle_pattern, task_text, re.IGNORECASE)
        
        if angle_match:
            try:
                angle = float(angle_match.group(1).replace(',', '.'))
                params['skew'] = angle
                # Обновляем значения углов
                params['angle_values'] = [angle, 180 - angle, angle, 180 - angle]
            except Exception:
                pass
                
        # Ищем углы параллелограмма в формате "углы равны 60° и 120°"
        angles_pattern = r'угл[а-я]*\s+(?:параллелограмм[а-я]*\s+)?равн[а-я]*\s+(\d+(?:[,.]\d+)?)[°\s]+,?\s*и\s*(\d+(?:[,.]\d+)?)[°\s]+'
        angles_match = re.search(angles_pattern, task_text, re.IGNORECASE)
        
        if angles_match:
            try:
                angle1 = float(angles_match.group(1).replace(',', '.'))
                angle2 = float(angles_match.group(2).replace(',', '.'))
                
                # Проверяем, что сумма углов равна 180 градусов
                if abs(angle1 + angle2 - 180) < 1e-6:
                    params['angle_values'] = [angle1, angle2, angle1, angle2]
                    
                    # Определяем угол наклона (skew) на основе острого угла
                    acute_angle = min(angle1, angle2)
                    params['skew'] = acute_angle
            except Exception:
                pass
        
        # Проверяем, является ли параллелограмм ромбом
        if re.search(r'ромб', task_text, re.IGNORECASE):
            # Ищем сторону ромба
            rhombus_side_pattern = r'ромб[а-я\s]*[=:]?\s*(\d+(?:[,.]\d+)?)'
            rhombus_side_match = re.search(rhombus_side_pattern, task_text, re.IGNORECASE)
            
            if rhombus_side_match:
                try:
                    side = float(rhombus_side_match.group(1).replace(',', '.'))
                    params['width'] = side
                    # Устанавливаем все стороны равными
                    params['side_lengths'] = [side, side, side, side]
                except Exception:
                    pass
        
        # Показываем длины сторон, если это указано в тексте или задача на вычисление площади/периметра
        if re.search(r'найд[а-я]+\s+сторон|найд[а-я]+\s+длин|длин[а-я]+\s+сторон|площадь|периметр', task_text, re.IGNORECASE):
            params['show_lengths'] = True
        
        # Показываем углы, если это указано в тексте
        if re.search(r'найд[а-я]+\s+угл|угл[а-я]+\s+параллелограмм|величин[а-я]+\s+угл', task_text, re.IGNORECASE):
            params['show_angles'] = True
            
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
        
        return Parallelogram(params) 