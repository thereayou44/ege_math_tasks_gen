import numpy as np
import re
from app.geometry.base import GeometricFigure
import matplotlib.pyplot as plt

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
        if 'points' not in self.params:
            # Стандартный треугольник (близок к равностороннему)
            if 'is_right' not in self.params or not self.params['is_right']:
                self.params['points'] = [(0, 0), (1, 0), (0.5, 0.86)]
            else:
                # Прямоугольный треугольник
                self.params['points'] = [(0, 0), (0, 3), (4, 0)]
            
    def compute_points(self):
        """
        Вычисляет координаты точек треугольника.
        
        Returns:
            list: Список точек [(x1,y1), (x2,y2), (x3,y3)]
        """
        # Если точки уже заданы, используем их
        if 'points' in self.params:
            return self.params['points']
        
        # Иначе создаем стандартный треугольник
        if 'is_right' not in self.params or not self.params['is_right']:
            return [(0, 0), (1, 0), (0.5, 0.86)]
        else:
            # Прямоугольный треугольник
            return [(0, 0), (0, 3), (4, 0)]
    
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
                
                # Вычисляем угол в градусах
                if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:  # Проверка на нулевой вектор
                    cos_angle = v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    cos_angle = max(-1, min(1, cos_angle))  # Ограничиваем значение
                    ang = np.degrees(np.arccos(cos_angle))
                    
                    # Проверяем внутренний или внешний угол
                    cross_product = np.cross(np.append(v1, 0), np.append(v2, 0))[2]
                    if cross_product < 0:
                        ang = 360 - ang
                    
                    # Отображаем угол
                    if angle_values and i < len(angle_values) and angle_values[i] is not None:
                        angle_text = f"{angle_values[i]}°"
                    else:
                        angle_text = f"{ang:.1f}°"
                    
                    # Радиус дуги
                    radius = min(np.linalg.norm(v1), np.linalg.norm(v2)) / 4
                    
                    # Выбираем координаты для текста угла - ближе к вершине
                    angle_mid = (v1/np.linalg.norm(v1) + v2/np.linalg.norm(v2))
                    angle_mid = angle_mid / np.linalg.norm(angle_mid) * radius * 0.8
                    
                    # Рисуем дугу, если требуется
                    if show_angle_arcs:
                        # Вычисляем начальный и конечный углы для дуги
                        start_angle = np.degrees(np.arctan2(v1[1], v1[0]))
                        end_angle = np.degrees(np.arctan2(v2[1], v2[0]))
                        
                        # Корректируем углы для правильного направления дуги
                        if cross_product < 0:
                            start_angle, end_angle = end_angle, start_angle
                        
                        # Создаем дугу
                        arc = plt.matplotlib.patches.Arc(
                            (B[0], B[1]), radius*2, radius*2,
                            theta1=start_angle, theta2=end_angle,
                            angle=0, color='red', linewidth=1
                        )
                        ax.add_patch(arc)
                    
                    # Отображаем значение угла
                    ax.text(B[0] + angle_mid[0], B[1] + angle_mid[1], angle_text, 
                            ha='center', fontsize=10,
                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
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
        
        # Для треугольников часто нужно отображать вычисленную длину
        # Округляем до 2 знаков после запятой и убираем лишние нули
        return f"{computed_length:.2f}".rstrip('0').rstrip('.')
    
    @staticmethod
    def from_text(task_text):
        """
        Создает объект треугольника из текста задачи.
        
        Args:
            task_text (str): Текст задачи
            
        Returns:
            Triangle: Объект треугольника с параметрами из текста
        """
        import re
        from app.prompts import DEFAULT_VISUALIZATION_PARAMS, REGEX_PATTERNS
        
        params = DEFAULT_VISUALIZATION_PARAMS["triangle"].copy()
        triangle_patterns = REGEX_PATTERNS["triangle"]
        
        # Функция для извлечения параметра по соответствующему регулярному выражению
        def extract_param(param_name, default=None, convert_type=None):
            pattern = triangle_patterns.get(param_name)
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
        
        # Извлекаем метки вершин
        vertex_labels = extract_param("vertex_labels")
        if vertex_labels:
            # Извлекаем буквы из строки
            labels = re.findall(r'[A-Za-z]', vertex_labels)
            if len(labels) >= 3:
                params['vertex_labels'] = labels[:3]
        
        # Определяем тип треугольника
        is_right = extract_param("is_right", "False", lambda x: x.lower() == "true")
        if is_right or re.search(r'прямоугольн[а-я]+\s+треугольник', task_text, re.IGNORECASE):
            params['is_right'] = True
        
        # Извлекаем длины сторон
        sides = extract_param("sides")
        if sides:
            side_lengths = parse_numeric_list(sides)
            if side_lengths and len(side_lengths) >= 3:
                params['side_lengths'] = side_lengths[:3]
        
        # Извлекаем значения углов
        angles = extract_param("angles")
        if angles:
            angle_values = parse_numeric_list(angles)
            if angle_values and len(angle_values) >= 3:
                params['angle_values'] = angle_values[:3]
        
        # Обрабатываем показ высот
        show_heights = extract_param("show_heights")
        if show_heights:
            if show_heights.lower() == "true":
                params['show_heights'] = True
            else:
                # Извлекаем вершины, из которых нужно показать высоты
                height_vertices = re.findall(r'[A-Za-z]', show_heights)
                if height_vertices:
                    params['show_heights'] = True
                    params['specific_heights'] = height_vertices
        
        # Обрабатываем показ медиан
        show_medians = extract_param("show_medians")
        if show_medians:
            if show_medians.lower() == "true":
                params['show_medians'] = True
            else:
                # Извлекаем вершины, из которых нужно показать медианы
                median_vertices = re.findall(r'[A-Za-z]', show_medians)
                if median_vertices:
                    params['show_medians'] = True
                    params['specific_medians'] = median_vertices
        
        # Обрабатываем показ средних линий
        show_midlines = extract_param("show_midlines")
        if show_midlines:
            if show_midlines.lower() == "true":
                params['show_midlines'] = True
            else:
                # Извлекаем стороны, для которых нужно показать средние линии
                midline_sides = re.findall(r'[A-Za-z]{2}', show_midlines)
                if midline_sides:
                    params['show_midlines'] = True
                    params['specific_midlines'] = midline_sides
        
        # Извлекаем координаты вершин, если они указаны
        coords = extract_param("coords")
        if coords:
            try:
                # Извлекаем числа из строки и группируем по парам (x,y)
                numbers = re.findall(r'[-+]?\d*\.?\d+', coords)
                if len(numbers) >= 6:  # 3 точки по 2 координаты
                    points = []
                    for i in range(0, 6, 2):
                        points.append((float(numbers[i]), float(numbers[i+1])))
                    params['points'] = points
            except Exception:
                pass
        
        # Дополнительная проверка на ключевые слова в тексте
        # Показываем длины сторон, если это указано в тексте
        if re.search(r'найд[а-я]+\s+сторон|найд[а-я]+\s+длин|найд[а-я]+\s+периметр|сторон[а-я]+\s+треугольник|длин[а-я]+\s+сторон', task_text, re.IGNORECASE):
            params['show_lengths'] = True
        
        # Показываем углы, если это указано в тексте
        if re.search(r'найд[а-я]+\s+угл|угл[а-я]+\s+треугольник|величин[а-я]+\s+угл', task_text, re.IGNORECASE):
            params['show_angles'] = True
        
        # Показываем высоты, если это указано в тексте
        if re.search(r'высот[а-я]+\s+треугольник|найд[а-я]+\s+высот', task_text, re.IGNORECASE) and not params.get('show_heights'):
            params['show_heights'] = True
        
        # Показываем медианы, если это указано в тексте
        if re.search(r'медиан[а-я]+\s+треугольник|найд[а-я]+\s+медиан', task_text, re.IGNORECASE) and not params.get('show_medians'):
            params['show_medians'] = True
        
        # Показываем средние линии, если это указано в тексте
        if re.search(r'средн[а-я]+\s+лини[а-я]+|найд[а-я]+\s+средн[а-я]+\s+лини[а-я]+', task_text, re.IGNORECASE) and not params.get('show_midlines'):
            params['show_midlines'] = True
        
        return Triangle(params) 