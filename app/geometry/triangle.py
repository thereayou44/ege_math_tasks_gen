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
        Всегда возвращает стандартный треугольник ФИКСИРОВАННОГО размера.
        Если треугольник прямоугольный, то возвращает прямоугольный треугольник.
        Параметры используются только для подписей, но не для определения формы и размера.
        
        Returns:
            list: Список точек [(x1,y1), (x2,y2), (x3,y3)]
        """
        # Проверяем, является ли треугольник прямоугольным
        is_right = self.params.get('is_right', False)
        
        # Смещение из параметров (сохраняем, чтобы при необходимости можно было сместить треугольник)
        x = self.params.get('x', 0)
        y = self.params.get('y', 0)
        
        # Фиксированные размеры для рисования
        if is_right:
            # Прямоугольный треугольник с прямым углом в точке (0,0)
            raw_points = [(0, 0), (4, 0), (0, 3)]
        else:
            # Обычный треугольник
            raw_points = [(0, 0), (5, 0), (2.5, 4)]
        
        # Применяем смещение
        return [(px+x, py+y) for px, py in raw_points]
    
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
        height_values = self.params.get('height_values', None)
        
        # Если не указаны конкретные вершины, проводим все высоты
        if not vertices:
            vertices = self.params.get('height_vertices', vertex_labels)
            
        for i, label in enumerate(vertex_labels):
            if label in vertices:
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
                
                # Значение высоты для подписи
                if height_values and i < len(height_values):
                    height_value = height_values[i]
                    # Если значение отмечено как "-", не отображаем его
                    if height_value == "-":
                        height_text = f"h{label}"
                    else:
                        height_text = f"h{label} = {height_value}"
                else:
                    height_text = f"h{label}"
                
                # Добавляем подпись с увеличенным размером шрифта
                ax.text(mid_x, mid_y, height_text, 
                        ha='center', va='center', color='red', fontsize=14,
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
        median_values = self.params.get('median_values', None)
        
        # Если не указаны конкретные вершины, проводим все медианы
        if not vertices:
            vertices = self.params.get('median_vertices', vertex_labels)
            
        for i, label in enumerate(vertex_labels):
            if label in vertices:
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
                
                # Значение медианы для подписи
                if median_values and i < len(median_values):
                    median_value = median_values[i]
                    # Если значение отмечено как "-", не отображаем его
                    if median_value == "-":
                        median_text = f"m{label}"
                    else:
                        median_text = f"m{label} = {median_value}"
                else:
                    median_text = f"m{label}"
                
                # Добавляем подпись с увеличенным размером шрифта
                ax.text(mid_x, mid_y, median_text, 
                        ha='center', va='center', color='green', fontsize=14,
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
        bisector_values = self.params.get('bisector_values', None)
        
        # Если не указаны конкретные вершины, проводим все биссектрисы
        if not vertices:
            vertices = self.params.get('bisector_vertices', vertex_labels)
            
        for i, label in enumerate(vertex_labels):
            if label in vertices:
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
                        
                        # Значение биссектрисы для подписи
                        if bisector_values and i < len(bisector_values):
                            bisector_value = bisector_values[i]
                            # Если значение отмечено как "-", не отображаем его
                            if bisector_value == "-":
                                bisector_text = f"l{label}"
                            else:
                                bisector_text = f"l{label} = {bisector_value}"
                        else:
                            bisector_text = f"l{label}"
                        
                        # Добавляем подпись с увеличенным размером шрифта
                        ax.text(mid_x, mid_y, bisector_text, 
                                ha='center', va='center', color='blue', fontsize=14,
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
        import logging
        
        # Создаем копию параметров по умолчанию
        params = DEFAULT_VISUALIZATION_PARAMS["triangle"].copy()
        
        # Добавляем логирование
        logging.info(f"Парсинг параметров треугольника из текста: {params_text}")
        
        # Извлекаем параметры с помощью регулярных выражений
        
        # Метки вершин
        vertex_labels_match = re.search(REGEX_PATTERNS['triangle']['vertex_labels'], params_text, re.IGNORECASE)
        if vertex_labels_match:
            vertex_labels_str = vertex_labels_match.group(1).strip()
            vertex_labels = [label.strip() for label in vertex_labels_str.split(',')]
            params['vertex_labels'] = vertex_labels
            logging.info(f"Установлены метки вершин: {params['vertex_labels']}")
            
        # Длины сторон
        sides_match = re.search(REGEX_PATTERNS['triangle']['sides'], params_text, re.IGNORECASE)
        if sides_match:
            sides_str = sides_match.group(1).strip()
            logging.info(f"Найдены стороны: {sides_str}")
            sides = []
            
            for side in sides_str.split(','):
                side = side.strip()
                if side == "-":
                    sides.append("-")
                    logging.info("Добавлена сторона: - (не отображать)")
                else:
                    try:
                        sides.append(float(side))
                        logging.info(f"Добавлена сторона: {side}")
                    except ValueError:
                        sides.append("-")
                        logging.warning(f"Не удалось преобразовать значение стороны: {side}")
            
            # Убедимся, что у нас ровно 3 значения
            while len(sides) < 3:
                sides.append("-")
            
            params['sides'] = sides
            params['show_lengths'] = True
            logging.info(f"Установлены длины сторон: {params['sides']}")
            
        # Углы треугольника
        angles_match = re.search(REGEX_PATTERNS['triangle']['angles'], params_text, re.IGNORECASE)
        if angles_match:
            angles_str = angles_match.group(1).strip()
            logging.info(f"Найдены углы: {angles_str}")
            angles = []
            
            for angle in angles_str.split(','):
                angle = angle.strip()
                if angle == "-":
                    angles.append("-")
                    logging.info("Добавлен угол: - (не отображать)")
                else:
                    try:
                        angles.append(float(angle))
                        # Если один из углов равен 90, отмечаем треугольник как прямоугольный
                        if float(angle) == 90:
                            params['is_right'] = True
                            logging.info("Треугольник отмечен как прямоугольный (угол 90°)")
                        logging.info(f"Добавлен угол: {angle}")
                    except ValueError:
                        angles.append("-")
                        logging.warning(f"Не удалось преобразовать значение угла: {angle}")
            
            # Убедимся, что у нас ровно 3 значения
            while len(angles) < 3:
                angles.append("-")
            
            params['angles'] = angles
            params['show_angles'] = True
            logging.info(f"Установлены значения углов: {params['angles']}")
            
        # Проверяем на прямоугольный треугольник
        is_right_match = re.search(REGEX_PATTERNS['triangle']['is_right'], params_text, re.IGNORECASE)
        if is_right_match:
            is_right_value = is_right_match.group(1).strip().lower()
            params['is_right'] = is_right_value in ['true', 'да', 'yes', '+']
            logging.info(f"Треугольник прямоугольный: {params['is_right']}")
        
        # Показывать высоты
        show_heights_match = re.search(REGEX_PATTERNS['triangle']['show_heights'], params_text, re.IGNORECASE)
        if show_heights_match:
            heights_str = show_heights_match.group(1).strip()
            if heights_str.lower() in ['true', 'да', 'yes', '+', '1']:
                params['show_heights'] = True
                params['height_vertices'] = []
                logging.info(f"Показывать все высоты")
            else:
                # Если указаны конкретные вершины
                params['show_heights'] = True
                params['height_vertices'] = [v.strip() for v in heights_str.split(',')]
                logging.info(f"Показывать высоты из вершин: {params['height_vertices']}")
                
        # Показывать медианы
        show_medians_match = re.search(REGEX_PATTERNS['triangle']['show_medians'], params_text, re.IGNORECASE)
        if show_medians_match:
            medians_str = show_medians_match.group(1).strip()
            if medians_str.lower() in ['true', 'да', 'yes', '+', '1']:
                params['show_medians'] = True
                params['median_vertices'] = []
                logging.info(f"Показывать все медианы")
            else:
                # Если указаны конкретные вершины
                params['show_medians'] = True
                params['median_vertices'] = [v.strip() for v in medians_str.split(',')]
                logging.info(f"Показывать медианы из вершин: {params['median_vertices']}")
                
        # Показывать средние линии
        show_midlines_match = re.search(REGEX_PATTERNS['triangle']['show_midlines'], params_text, re.IGNORECASE)
        if show_midlines_match:
            midlines_str = show_midlines_match.group(1).strip()
            params['show_midlines'] = midlines_str.lower() in ['true', 'да', 'yes', '+', '1']
            logging.info(f"Показывать средние линии: {params['show_midlines']}")
        
        # Показывать биссектрисы углов
        show_angle_bisectors_match = re.search(r'Показать биссектрисы\s*:?\s*\[?([^\]\n]*)\]?', params_text, re.IGNORECASE)
        if show_angle_bisectors_match:
            bisectors_str = show_angle_bisectors_match.group(1).strip()
            if bisectors_str.lower() in ['true', 'да', 'yes', '+', '1']:
                params['show_angle_bisectors'] = True
                params['bisector_vertices'] = []
                logging.info(f"Показывать все биссектрисы")
            else:
                # Если указаны конкретные вершины
                params['show_angle_bisectors'] = True
                params['bisector_vertices'] = [v.strip() for v in bisectors_str.split(',')]
                logging.info(f"Показывать биссектрисы из вершин: {params['bisector_vertices']}")
        
        # Значения биссектрис
        bisector_values_match = re.search(r'Значения биссектрис\s*:?\s*\[([^\]]*)\]', params_text, re.IGNORECASE)
        if bisector_values_match:
            bisector_values_str = bisector_values_match.group(1).strip()
            logging.info(f"Найдены значения биссектрис: {bisector_values_str}")
            bisector_values = []
            
            for value in bisector_values_str.split(','):
                value = value.strip()
                if value == "-":
                    bisector_values.append("-")
                    logging.info("Добавлено значение биссектрисы: - (не отображать)")
                else:
                    try:
                        bisector_values.append(float(value))
                        logging.info(f"Добавлено значение биссектрисы: {value}")
                    except ValueError:
                        bisector_values.append("-")
                        logging.warning(f"Не удалось преобразовать значение биссектрисы: {value}")
            
            # Убедимся, что у нас ровно 3 значения
            while len(bisector_values) < 3:
                bisector_values.append("-")
                
            params['bisector_values'] = bisector_values
            logging.info(f"Установлены значения биссектрис: {params['bisector_values']}")
        
        # Отладочная информация о финальных параметрах
        logging.info(f"Финальные параметры треугольника: {params}")
        
        # Создаем объект треугольника
        triangle = Triangle(params)
        return triangle 

    def add_side_lengths(self, ax):
        """
        Добавляет подписи длин сторон треугольника.
        
        Args:
            ax (matplotlib.axes.Axes): Оси для отрисовки
        """
        sides = self.params.get('sides', None)
        show_lengths = self.params.get('show_lengths', False)
        
        if not sides or not show_lengths:
            return
            
        vertex_labels = self.params.get('vertex_labels', ['A', 'B', 'C'])
        
        # Получаем средний размер стороны для расчета отступа
        side_sizes = []
        for i in range(3):
            p1 = self.points[i]
            p2 = self.points[(i+1) % 3]
            L = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            side_sizes.append(L)
        
        avg_side_size = sum(side_sizes) / len(side_sizes)
        offset = avg_side_size * 0.15  # Чуть больший отступ для треугольника
        
        for i in range(3):
            p1 = self.points[i]
            p2 = self.points[(i+1) % 3]
            
            # Проверяем, нужно ли отображать эту сторону
            if i < len(sides) and sides[i] == "-":
                continue  # Пропускаем стороны, помеченные как "-"
            
            # Вычисляем длину стороны и позицию для подписи
            L = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
            
            # Вектор нормали к стороне
            dx, dy = p2[0] - p1[0], p2[1] - p1[1]
            length = np.sqrt(dx*dx + dy*dy)
            
            # Нормаль всегда направлена "наружу" от треугольника
            # Для этого проверяем, направлена ли нормаль к центру треугольника или от него
            center_x = sum(p[0] for p in self.points) / len(self.points)
            center_y = sum(p[1] for p in self.points) / len(self.points)
            
            nx, ny = -dy/length, dx/length  # Стандартная нормаль
            
            # Проверяем, направлена ли нормаль к центру треугольника
            vector_to_center_x = center_x - mx
            vector_to_center_y = center_y - my
            dot_product = nx * vector_to_center_x + ny * vector_to_center_y
            
            # Если нормаль направлена к центру, инвертируем её
            if dot_product > 0:
                nx, ny = -nx, -ny
            
            # Получаем текст для подписи
            text_value = self.get_side_length_text(i)
            
            if text_value:
                # Смещаем текст от стороны
                text_x = mx + nx * offset
                text_y = my + ny * offset
                
                # Отображаем подпись с белым фоном
                ax.text(text_x, text_y, text_value, 
                        ha='center', va='center', fontsize=14,  # Увеличиваем размер шрифта
                        bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', 
                                boxstyle="round,pad=0.3"))
    
    def get_side_length_text(self, side_index):
        """
        Возвращает текст для подписи длины стороны треугольника.
        
        Args:
            side_index (int): Индекс стороны (0, 1 или 2)
            
        Returns:
            str: Текст для подписи или None, если подпись не нужна
        """
        sides = self.params.get('sides', None)
        
        if not sides or side_index >= len(sides):
            return None
            
        # Если сторона отмечена как "-", не отображаем подпись
        if sides[side_index] == "-":
            return None
            
        # Получаем вершины для обозначения стороны
        vertex_labels = self.params.get('vertex_labels', ['A', 'B', 'C'])
        v1 = vertex_labels[side_index]
        v2 = vertex_labels[(side_index + 1) % 3]
        
        # Возвращаем значение стороны
        return f"{sides[side_index]}"
        
    def add_angles(self, ax):
        """
        Добавляет отображение углов треугольника.
        
        Args:
            ax (matplotlib.axes.Axes): Оси для отрисовки
        """
        angles = self.params.get('angles', None)
        show_angles = self.params.get('show_angles', False)
        
        if not angles or not show_angles:
            return
            
        vertex_labels = self.params.get('vertex_labels', ['A', 'B', 'C'])
        
        # Вычисляем средний размер треугольника для масштабирования дуг
        all_x = [p[0] for p in self.points]
        all_y = [p[1] for p in self.points]
        width = max(all_x) - min(all_x)
        height = max(all_y) - min(all_y)
        avg_size = (width + height) / 2
        radius = avg_size * 0.12  # Радиус дуги угла
        
        for i in range(3):
            # Пропускаем углы, помеченные как "-"
            if i < len(angles) and angles[i] == "-":
                continue
                
            # Получаем три точки для вычисления угла
            A = np.array(self.points[(i-1) % 3])
            B = np.array(self.points[i])  # Вершина с углом
            C = np.array(self.points[(i+1) % 3])
            
            # Вычисляем векторы сторон
            v1 = A - B
            v2 = C - B
            
            # Нормализуем векторы
            v1_norm = np.linalg.norm(v1)
            v2_norm = np.linalg.norm(v2)
            
            if v1_norm > 1e-10 and v2_norm > 1e-10:
                v1 = v1 / v1_norm
                v2 = v2 / v2_norm
                
                # Вычисляем угол между векторами
                cos_angle = np.clip(np.dot(v1, v2), -1.0, 1.0)
                angle_rad = np.arccos(cos_angle)
                angle_deg = np.degrees(angle_rad)
                
                # Получаем значение угла для подписи
                if i < len(angles) and angles[i] != "-":
                    angle_text = f"{angles[i]}°"
                else:
                    angle_text = f"{angle_deg:.0f}°"
                
                # Вычисляем начальный угол для дуги
                start_angle = np.degrees(np.arctan2(v1[1], v1[0])) % 360
                end_angle = np.degrees(np.arctan2(v2[1], v2[0])) % 360
                
                # Убедимся, что рисуем внутренний угол, а не внешний
                # Проверяем, нужно ли изменить направление обхода
                if abs((end_angle - start_angle) % 360) > 180:
                    # Если разница больше 180 градусов, меняем начальный и конечный углы местами
                    start_angle, end_angle = end_angle, start_angle
                
                # Создаем и добавляем дугу
                arc = plt.matplotlib.patches.Arc(
                    B, radius * 2, radius * 2,  # центр и размеры
                    theta1=start_angle, theta2=end_angle,  # углы дуги
                    color='blue', linewidth=1.5, fill=False, zorder=1
                )
                ax.add_patch(arc)
                
                # Средний вектор для размещения текста
                v_mid = (v1 + v2) / 2
                if np.linalg.norm(v_mid) > 1e-10:
                    v_mid = v_mid / np.linalg.norm(v_mid)
                    
                # Положение текста угла
                text_pos = B + v_mid * radius * 1.5
                
                # Отображаем значение угла
                ax.text(text_pos[0], text_pos[1], angle_text, 
                        ha='center', va='center', fontsize=14,  # Увеличиваем размер шрифта
                        bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', 
                                boxstyle="round,pad=0.2")) 