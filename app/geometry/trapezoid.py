import numpy as np
from app.geometry.base import GeometricFigure

class Trapezoid(GeometricFigure):
    """
    Класс для представления и визуализации трапеции.
    """
    
    def __init__(self, params=None):
        """
        Инициализирует трапецию с заданными параметрами.
        
        Args:
            params (dict): Словарь параметров фигуры
        """
        super().__init__(params)
        self.figure_type = "trapezoid"
        
        # Установка параметров по умолчанию, если не указаны
        if 'bottom_width' not in self.params:
            self.params['bottom_width'] = 6
        if 'top_width' not in self.params:
            self.params['top_width'] = 3
        if 'height' not in self.params:
            self.params['height'] = 3
        if 'is_isosceles' not in self.params:
            self.params['is_isosceles'] = False
        if 'show_heights' not in self.params:
            self.params['show_heights'] = False
        if 'angle_values' not in self.params:
            # Значения углов по умолчанию вычисляются при отрисовке 
            self.params['angle_values'] = None
            
    def compute_points(self):
        """
        Вычисляет координаты точек трапеции.
        
        Returns:
            list: Список точек [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
        """
        bottom = self.params.get('bottom_width')
        top = self.params.get('top_width')
        height = self.params.get('height')
        is_isosceles = self.params.get('is_isosceles', False)
        x = self.params.get('x', 0)
        y = self.params.get('y', 0)
        
        if is_isosceles:
            # Для равнобедренной трапеции боковые стороны равны
            dx = (bottom - top) / 2
            raw_points = [(0, 0), (bottom, 0), (bottom - dx, height), (dx, height)]
        else:
            # Для произвольной трапеции (по умолчанию смещение верхнего основания слева)
            dx = (bottom - top)/2
            raw_points = [(0, 0), (bottom, 0), (bottom - dx, height), (dx, height)]
        
        # Применяем смещение, если указано
        return [(px+x, py+y) for px, py in raw_points]
    
    def compute_angles(self):
        """
        Вычисляет углы трапеции в градусах.
        
        Returns:
            list: Список углов в градусах [угол A, угол B, угол C, угол D]
        """
        # Если углы уже заданы, используем их
        if self.params.get('angle_values') is not None:
            return self.params['angle_values']
        
        points = self.compute_points()
        angles = []
        
        for i in range(4):
            # Получаем координаты текущей точки и соседних (предыдущей и следующей)
            prev_idx = (i - 1) % 4
            next_idx = (i + 1) % 4
            
            prev_point = points[prev_idx]
            current_point = points[i]
            next_point = points[next_idx]
            
            # Вычисляем векторы к соседним точкам
            vec1 = (prev_point[0] - current_point[0], prev_point[1] - current_point[1])
            vec2 = (next_point[0] - current_point[0], next_point[1] - current_point[1])
            
            # Вычисляем угол между векторами
            dot_product = vec1[0] * vec2[0] + vec1[1] * vec2[1]
            len1 = np.sqrt(vec1[0]**2 + vec1[1]**2)
            len2 = np.sqrt(vec2[0]**2 + vec2[1]**2)
            
            # Защита от деления на ноль
            if len1 * len2 < 1e-10:
                angle_rad = 0
            else:
                cos_angle = dot_product / (len1 * len2)
                # Защита от численных ошибок
                cos_angle = max(-1, min(1, cos_angle))
                angle_rad = np.arccos(cos_angle)
            
            # Преобразуем радианы в градусы
            angle_deg = np.degrees(angle_rad)
            angles.append(angle_deg)
        
        return angles
    
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
        
        # Если нет значения в side_lengths, используем стандартную логику для оснований и боковых сторон
        if side_index == 0:  # Нижнее основание (AB)
            if 'bottom_width' in self.params:
                return f"{self.params['bottom_width']}"
        elif side_index == 2:  # Верхнее основание (DC)
            if 'top_width' in self.params:
                return f"{self.params['top_width']}"
        elif side_index == 1 or side_index == 3:  # Боковые стороны (BC и AD)
            # Вычисляем длину боковой стороны, если не задана явно
            if self.params.get('is_isosceles', False):
                # Для равнобедренной трапеции боковые стороны равны
                bottom = self.params.get('bottom_width', 6)
                top = self.params.get('top_width', 3)
                height = self.params.get('height', 3)
                
                # Вычисляем длину боковой стороны по формуле
                dx = (bottom - top) / 2
                side_length = np.sqrt(dx**2 + height**2)
                
                # Округляем до 2 знаков после запятой
                return f"{side_length:.2f}".rstrip('0').rstrip('.')
            else:
                # Для обычной трапеции вычисляем каждую боковую сторону отдельно
                # Используем вычисленную длину, которую получили от вызывающего кода
                return f"{computed_length:.2f}".rstrip('0').rstrip('.')
                
        # В других случаях используем стандартную логику из базового класса
        return super().get_side_length_text(side_index, side_name, computed_length)
    
    def draw(self, ax=None):
        """
        Отрисовывает трапецию с дополнительными элементами.
        
        Args:
            ax (matplotlib.axes.Axes): Оси для отрисовки
            
        Returns:
            matplotlib.axes.Axes: Оси с нарисованной фигурой
        """
        # Используем базовый метод для рисования основной фигуры
        ax = super().draw(ax)
        
        # Добавляем высоты, если они должны быть отображены
        if self.params.get('show_height', False):
            self._add_heights(ax)
            
        # Добавляем среднюю линию, если она должна быть отображена
        if self.params.get('show_midline', False):
            self._add_midline(ax)
            
        return ax
    
    def _add_heights(self, ax):
        """
        Добавляет отображение высоты трапеции.
        
        Args:
            ax (matplotlib.axes.Axes): Оси для отрисовки
        """
        points = self.compute_points()
        height = self.params.get('height', 3)
        height_value = self.params.get('height_value', height)
        
        # Рисуем высоту от нижнего основания к верхнему
        # Выбираем точку в середине нижнего основания
        x1, y1 = points[0]
        x2, y2 = points[1]
        middle_x = (x1 + x2) / 2
        middle_y = y1
        
        # Точка на верхнем основании с тем же x (прямо вверх)
        top_y = y1 + height
        
        # Рисуем высоту как прерывистую линию
        ax.plot([middle_x, middle_x], [middle_y, top_y], 'r--', lw=1)
        
        # Добавляем подпись высоты
        height_text = f"h={height_value}"
        ax.text(middle_x - 0.2, (middle_y + top_y) / 2, height_text, 
                ha='right', va='center', color='red', fontsize=10,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
    def _add_midline(self, ax):
        """
        Добавляет отображение средней линии трапеции.
        
        Args:
            ax (matplotlib.axes.Axes): Оси для отрисовки
        """
        points = self.compute_points()
        
        # Вычисляем координаты средних точек боковых сторон
        left_side_mid = ((points[0][0] + points[3][0]) / 2, (points[0][1] + points[3][1]) / 2)
        right_side_mid = ((points[1][0] + points[2][0]) / 2, (points[1][1] + points[2][1]) / 2)
        
        # Рисуем среднюю линию
        ax.plot([left_side_mid[0], right_side_mid[0]], [left_side_mid[1], right_side_mid[1]], 'g-', lw=1.5)
        
        # Вычисляем длину средней линии (среднее арифметическое оснований)
        bottom = self.params.get('bottom_width', 6)
        top = self.params.get('top_width', 3)
        midline_length = (bottom + top) / 2
        
        # Используем заданное значение, если оно есть
        midline_value = self.params.get('midline_value', midline_length)
        
        # Добавляем подпись средней линии
        midline_text = f"m={midline_value}"
        mid_x = (left_side_mid[0] + right_side_mid[0]) / 2
        mid_y = left_side_mid[1]
        ax.text(mid_x, mid_y + 0.2, midline_text, 
                ha='center', va='bottom', color='green', fontsize=10,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
    @staticmethod
    def from_text(task_text):
        """
        Создает объект трапеции из текста задачи.
        
        Args:
            task_text (str): Текст задачи
            
        Returns:
            Trapezoid: Объект трапеции с параметрами из текста
        """
        import re
        from app.prompts import DEFAULT_VISUALIZATION_PARAMS, REGEX_PATTERNS
        
        params = DEFAULT_VISUALIZATION_PARAMS["trapezoid"].copy()
        trap_patterns = REGEX_PATTERNS["trapezoid"]
        
        # Функция для извлечения параметра по соответствующему регулярному выражению
        def extract_param(param_name, default=None, convert_type=None):
            pattern = trap_patterns.get(param_name)
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
        
        # Извлекаем основания трапеции
        bases = extract_param("bases")
        if bases:
            bases_list = parse_numeric_list(bases)
            if bases_list and len(bases_list) >= 2:
                params['bottom_width'] = max(bases_list[0], bases_list[1])
                params['top_width'] = min(bases_list[0], bases_list[1])
        
        # Извлекаем высоту
        height = extract_param("height", None, float)
        if height is not None:
            params['height'] = height
        
        # Извлекаем параметры для отображения высоты
        show_height = extract_param("show_height", "False", lambda x: x.lower() == "true")
        if show_height:
            params['show_height'] = True
        
        height_value = extract_param("height_value", None, float)
        if height_value is not None:
            params['height_value'] = height_value
            params['show_height'] = True
        
        # Извлекаем параметры для средней линии
        show_midline = extract_param("show_midline", "False", lambda x: x.lower() == "true")
        if show_midline:
            params['show_midline'] = True
        
        midline_value = extract_param("midline_value", None, float)
        if midline_value is not None:
            params['midline_value'] = midline_value
            params['show_midline'] = True
        
        # Извлекаем метки вершин
        vertex_labels = extract_param("vertex_labels")
        if vertex_labels:
            # Извлекаем буквы из строки
            labels = re.findall(r'[A-Za-z]', vertex_labels)
            if len(labels) >= 4:
                params['vertex_labels'] = labels[:4]
        
        # Извлекаем боковые стороны
        sides = extract_param("sides")
        if sides:
            sides_list = parse_numeric_list(sides)
            if sides_list and len(sides_list) >= 2:
                params['sides'] = sides_list
                if sides_list[0] == sides_list[1]:
                    params['is_isosceles'] = True
        
        # Извлекаем углы
        angles = extract_param("angles")
        if angles:
            angles_list = parse_numeric_list(angles)
            if angles_list and len(angles_list) >= 2:
                params['angle_values'] = angles_list
        
        # Распознаем равнобедренную трапецию
        if re.search(r'равнобедренн[а-я]*\s*трапеци', task_text, re.IGNORECASE):
            params['is_isosceles'] = True
        
        # Дополнительные проверки по ключевым словам
        if re.search(r'высот[а-я]*\s*трапеци|найд[а-я]+\s*высот[а-я]', task_text, re.IGNORECASE):
            params['show_height'] = True
        
        if re.search(r'средн[а-я]+\s*лини[а-я]', task_text, re.IGNORECASE):
            params['show_midline'] = True
        
        return Trapezoid(params)

    # Явно наследуем метод add_vertex_labels от базового класса
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

    # Явно реализуем метод add_side_lengths для трапеции
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
            
            # Если угловые значения не заданы, вычисляем их
            if angle_values is None:
                angle_values = self.compute_angles()
            
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
                
                # Получаем значение угла
                if i < len(angle_values):
                    angle_text = f"{angle_values[i]:.1f}°"
                else:
                    # Вычисляем угол на лету, если не указано в параметрах
                    if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                        cos_angle = v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                        cos_angle = max(-1, min(1, cos_angle))  # Ограничиваем значение
                        angle = np.degrees(np.arccos(cos_angle))
                        angle_text = f"{angle:.1f}°"
                    else:
                        continue  # Пропускаем, если векторы нулевые
                
                # Радиус дуги
                radius = min(np.linalg.norm(v1), np.linalg.norm(v2)) / 4
                
                # Выбираем координаты для текста угла - ближе к вершине
                angle_mid = (v1/np.linalg.norm(v1) + v2/np.linalg.norm(v2))
                angle_mid = angle_mid / np.linalg.norm(angle_mid) * radius * 0.8
                
                # Рисуем дугу, если требуется
                if show_angle_arcs:
                    import matplotlib.pyplot as plt
                    # Вычисляем начальный и конечный углы для дуги
                    start_angle = np.degrees(np.arctan2(v1[1], v1[0]))
                    end_angle = np.degrees(np.arctan2(v2[1], v2[0]))
                    
                    # Корректируем углы для правильного направления дуги
                    cross_product = np.cross(np.append(v1, 0), np.append(v2, 0))[2]
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