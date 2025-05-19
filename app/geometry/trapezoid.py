import numpy as np
from app.geometry.base import GeometricFigure
import re
import logging

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
        
        # Установка параметров по умолчанию, только если они не заданы
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
        if 'show_midline' not in self.params:
            self.params['show_midline'] = False
        if 'angle_values' not in self.params:
            # Значения углов по умолчанию вычисляются при отрисовке 
            self.params['angle_values'] = None
            
        # Добавляем логирование
        logging.info(f"Инициализирована трапеция с параметрами: {self.params}")
        logging.info(f"top_width: {self.params.get('top_width')}")
        logging.info(f"bottom_width: {self.params.get('bottom_width')}")
        
    def compute_points(self):
        """
        Всегда возвращает стандартную равнобедренную трапецию ФИКСИРОВАННОГО размера.
        Параметры используются только для подписей, но не для определения формы и размера.
        """
        # Фиксированные размеры для рисования
        fixed_bottom = 6
        fixed_top = 3
        fixed_height = 3
        
        # Смещение из параметров (сохраняем, чтобы при необходимости можно было сместить трапецию)
        x = self.params.get('x', 0)
        y = self.params.get('y', 0)
        
        # Строим стандартную равнобедренную трапецию фиксированного размера
        dx = (fixed_bottom - fixed_top) / 2
        raw_points = [(0, 0), (fixed_bottom, 0), (fixed_bottom - dx, fixed_height), (dx, fixed_height)]
        
        # Применяем смещение
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
        if self.params.get('show_heights', False):
            self._add_heights(ax)
            
        # Добавляем среднюю линию, если нужно
        if self.params.get('show_midline', False):
            self._add_midline(ax)
            
        return ax
    
    def _add_heights(self, ax):
        """
        Высота всегда проводится из вершины D (тупой угол) на основание AB.
        """
        logging.info(f"Отрисовка высоты. Параметры: show_heights={self.params.get('show_heights')}, height={self.params.get('height')}, height_value={self.params.get('height_value')}")
        
        # Используем точки, рассчитанные с фиксированными размерами
        points = self.compute_points()
        
        # Значение высоты для подписи берём из параметров
        height_value = self.params.get('height_value')
        
        # D = points[3], основание AB = points[0]-points[1]
        xD, yD = points[3]
        xA, yA = points[0]
        xB, yB = points[1]
        # Опускаем перпендикуляр из D на AB
        dx = xB - xA
        dy = yB - yA
        if dx == 0 and dy == 0:
            return
        t = ((xD - xA) * dx + (yD - yA) * dy) / (dx*dx + dy*dy)
        xH = xA + t * dx
        yH = yA + t * dy
        ax.plot([xD, xH], [yD, yH], 'r--', lw=1.5)
        # Добавляем перпендикулярные черточки
        tick_width = (xB - xA) * 0.03
        ax.plot([xH - tick_width, xH + tick_width], [yH, yH], 'r-', lw=1.5)
        ax.plot([xD - tick_width, xD + tick_width], [yD, yD], 'r-', lw=1.5)
        # Подпись высоты - если значение не указано или None, то пишем просто "h"
        if height_value is None or height_value == "-":
            height_text = "h"
        else:
            height_text = f"h = {height_value}"
        ax.text((xD + xH)/2 - tick_width*3, (yD + yH)/2, height_text, ha='right', va='center', color='red', fontsize=14,
                bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', boxstyle='round,pad=0.3'))
    
    def _add_midline(self, ax):
        """
        Рисует среднюю линию только если явно указано.
        """
        # Используем точки, рассчитанные с фиксированными размерами
        points = self.compute_points()
        
        # Фиксированные размеры для рисования
        fixed_bottom = 6
        fixed_top = 3
        fixed_height = 3
        
        # Значение средней линии для подписи берём из параметров
        bottom_width = self.params.get('bottom_width', fixed_bottom)
        top_width = self.params.get('top_width', fixed_top)
        
        # Проверяем, чтобы оба значения были числами, иначе используем значения по умолчанию
        if isinstance(bottom_width, str) or isinstance(top_width, str):
            midline_length = (fixed_bottom + fixed_top) / 2
        else:
            midline_length = (bottom_width + top_width) / 2
        
        midline_value = self.params.get('midline_value', midline_length)
        
        # Средняя линия параллельна основаниям, проходит посередине по высоте
        mid_y = points[0][1] + fixed_height / 2
        left_x = points[0][0] + (points[3][0] - points[0][0]) / 2
        right_x = points[1][0] - (points[1][0] - points[2][0]) / 2
        ax.plot([left_x, right_x], [mid_y, mid_y], 'g-', lw=1.5)
        # Подпись средней линии - если значение не указано или None, то пишем просто "m"
        if midline_value is None or midline_value == "-":
            midline_text = "m"
        else:
            midline_text = f"m = {midline_value}"
        ax.text((left_x + right_x) / 2, mid_y + 0.2, midline_text, ha='center', va='bottom', color='green', fontsize=14,
                bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', boxstyle='round,pad=0.3'))
    
    @staticmethod
    def from_text(task_text):
        """
        Создает объект трапеции из текста задачи.
        Теперь поддерживает парсинг оснований и высоты без квадратных скобок.
        """
        from app.prompts import DEFAULT_VISUALIZATION_PARAMS
        import logging
        
        params = DEFAULT_VISUALIZATION_PARAMS["trapezoid"].copy()
        
        # Отладочная информация
        logging.info("Парсинг параметров трапеции из текста:")
        logging.info(f"Исходный текст: {task_text}")

        # Основания (ищем с или без скобок)
        bases_match = re.search(r'Основания\s*:?\s*(\[?)([\d\., \-]+)(\]?)', task_text, re.IGNORECASE)
        if bases_match:
            bases_str = bases_match.group(2).strip()
            logging.info(f"Найдены основания: {bases_str}")
            bases = [b.strip() for b in bases_str.split(',')]
            if len(bases) >= 1:
                if bases[0] != "-":
                    try:
                        params['bottom_width'] = float(bases[0])
                        logging.info(f"Установлено нижнее основание: {params['bottom_width']}")
                    except ValueError:
                        logging.warning(f"Не удалось преобразовать нижнее основание: {bases[0]}")
                else:
                    # Если явно указан прочерк, сохраняем его
                    params['bottom_width'] = "-"
                    logging.info("Нижнее основание не будет отображаться (-)") 
            
            if len(bases) >= 2:
                if bases[1] != "-":
                    try:
                        params['top_width'] = float(bases[1])
                        logging.info(f"Установлено верхнее основание: {params['top_width']}")
                    except ValueError:
                        logging.warning(f"Не удалось преобразовать верхнее основание: {bases[1]}")
                else:
                    # Если явно указан прочерк, сохраняем его
                    params['top_width'] = "-"
                    logging.info("Верхнее основание не будет отображаться (-)")

        # Высота (число)
        height_match = re.search(r'Высота\s*:?\s*(\[?)([\d\., \-]+)(\]?)', task_text, re.IGNORECASE)
        if height_match:
            height_str = height_match.group(2).strip()
            logging.info(f"Найдена высота: {height_str}")
            if height_str != "-":
                try:
                    params['height'] = float(height_str)
                    params['height_value'] = float(height_str)  # Устанавливаем также значение для отображения
                    logging.info(f"Установлена высота: {params['height']}")
                except ValueError:
                    logging.warning(f"Не удалось преобразовать высоту: {height_str}")
            else:
                params['height'] = "-"
                logging.info("Высота не будет отображаться (-)")

        # Показать высоту
        show_height_match = re.search(r'Показать высоту\s*:?\s*(\[?)([a-zA-Zа-яА-Я0-9]+)(\]?)', task_text, re.IGNORECASE)
        if show_height_match:
            show_height_str = show_height_match.group(2).strip().lower()
            params['show_heights'] = show_height_str in ['true', 'да', 'yes', '+', '1']
            logging.info(f"Показать высоту: {params['show_heights']}")

        # Значение высоты
        height_value_match = re.search(r'Значение высоты\s*:?\s*(\[?)([\d\., \-]+)(\]?)', task_text, re.IGNORECASE)
        if height_value_match:
            height_value_str = height_value_match.group(2).strip()
            logging.info(f"Найдено значение высоты: {height_value_str}")
            if height_value_str != "-":
                try:
                    params['height_value'] = float(height_value_str)
                    logging.info(f"Установлено значение высоты: {params['height_value']}")
                except ValueError:
                    logging.warning(f"Не удалось преобразовать значение высоты: {height_value_str}")
            else:
                params['height_value'] = "-"
                logging.info("Значение высоты не будет отображаться (-)")

        # Метки вершин
        vertex_labels_match = re.search(r'Метки вершин\s*:?\s*(\[?)([A-Za-zА-Яа-я, ]+)(\]?)', task_text, re.IGNORECASE)
        if vertex_labels_match:
            vertex_labels_str = vertex_labels_match.group(2).strip()
            vertex_labels = [label.strip() for label in vertex_labels_str.split(',')]
            params['vertex_labels'] = vertex_labels
            logging.info(f"Установлены метки вершин: {params['vertex_labels']}")

        # Боковые стороны
        sides_match = re.search(r'Боковые стороны\s*:?\s*(\[?)([\d\., \-]+)(\]?)', task_text, re.IGNORECASE)
        if sides_match:
            sides_str = sides_match.group(2).strip()
            logging.info(f"Найдены боковые стороны: {sides_str}")
            sides = [s.strip() for s in sides_str.split(',')]
            if len(sides) >= 2:
                side_lengths = [None, None, None, None]
                if 'bottom_width' in params and params['bottom_width'] != "-":
                    side_lengths[0] = params['bottom_width']
                if 'top_width' in params and params['top_width'] != "-":
                    side_lengths[2] = params['top_width']
                
                if sides[0] != "-":
                    try:
                        side_lengths[1] = float(sides[0])
                        logging.info(f"Установлена правая боковая сторона: {side_lengths[1]}")
                    except ValueError:
                        logging.warning(f"Не удалось преобразовать правую боковую сторону: {sides[0]}")
                else:
                    side_lengths[1] = "-"
                    logging.info("Правая боковая сторона не будет отображаться (-)")
                
                if sides[1] != "-":
                    try:
                        side_lengths[3] = float(sides[1])
                        logging.info(f"Установлена левая боковая сторона: {side_lengths[3]}")
                    except ValueError:
                        logging.warning(f"Не удалось преобразовать левую боковую сторону: {sides[1]}")
                else:
                    side_lengths[3] = "-"
                    logging.info("Левая боковая сторона не будет отображаться (-)")
                
                params['side_lengths'] = side_lengths
                logging.info(f"Установлены длины сторон: {params['side_lengths']}")

        # Углы
        angles_match = re.search(r'Углы\s*:?\s*(\[?)([\d\., \-]+)(\]?)', task_text, re.IGNORECASE)
        if angles_match:
            angles_str = angles_match.group(2).strip()
            logging.info(f"Найдены углы: {angles_str}")
            angles = [angle.strip() for angle in angles_str.split(',')]
            angle_values = []
            for angle in angles:
                if angle == "-":
                    angle_values.append("-")
                    logging.info("Добавлен угол: - (не отображать)")
                else:
                    try:
                        angle_values.append(float(angle))
                        logging.info(f"Добавлен угол: {angle}")
                    except ValueError:
                        angle_values.append("-")
                        logging.warning(f"Не удалось преобразовать значение угла: {angle}")
            while len(angle_values) < 4:
                angle_values.append("-")
            params['angle_values'] = angle_values
            params['show_angles'] = True
            logging.info(f"Установлены значения углов: {params['angle_values']}")

        # Показать среднюю линию
        show_midline_match = re.search(r'Показать среднюю линию\s*:?\s*(\[?)([a-zA-Zа-яА-Я0-9]+)(\]?)', task_text, re.IGNORECASE)
        if show_midline_match:
            show_midline_str = show_midline_match.group(2).strip().lower()
            params['show_midline'] = show_midline_str in ['true', 'да', 'yes', '+', '1']
            logging.info(f"Показать среднюю линию: {params['show_midline']}")

        # Значение средней линии
        midline_value_match = re.search(r'Значение средней линии\s*:?\s*(\[?)([\d\., \-]+)(\]?)', task_text, re.IGNORECASE)
        if midline_value_match:
            midline_value_str = midline_value_match.group(2).strip()
            logging.info(f"Найдено значение средней линии: {midline_value_str}")
            if midline_value_str != "-":
                try:
                    params['midline_value'] = float(midline_value_str)
                    logging.info(f"Установлено значение средней линии: {params['midline_value']}")
                except ValueError:
                    logging.warning(f"Не удалось преобразовать значение средней линии: {midline_value_str}")
            else:
                params['midline_value'] = "-"
                logging.info("Значение средней линии не будет отображаться (-)")

        # Отладочная информация о финальных параметрах
        logging.info(f"Финальные параметры трапеции: {params}")
        
        # Создаем и возвращаем объект трапеции
        trapezoid = Trapezoid(params)
        return trapezoid

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
                    ax.text(x0, y0, lab, ha='center', va='center', fontsize=16,
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')) 

    # Явно реализуем метод add_side_lengths для трапеции
    def add_side_lengths(self, ax):
        """
        Подписывает только те стороны, которые явно указаны (не "-").
        Основания: bottom_width/top_width, боковые: side_lengths.
        """
        import logging
        
        side_lengths = self.params.get('side_lengths', None)
        vertex_labels = self.params.get('vertex_labels', [chr(65+j) for j in range(len(self.points))])
        
        # Отладочная информация
        logging.info("Подписываем стороны трапеции:")
        logging.info(f"bottom_width: {self.params.get('bottom_width')}")
        logging.info(f"top_width: {self.params.get('top_width')}")
        logging.info(f"side_lengths: {side_lengths}")
        
        if not vertex_labels or len(vertex_labels) < len(self.points):
            vertex_labels = [chr(65+j) for j in range(len(self.points))]
            self.params['vertex_labels'] = vertex_labels
        for i in range(len(self.points)):
            p1 = self.points[i]
            p2 = self.points[(i+1) % len(self.points)]
            should_show = False
            text_value = None
            # Нижнее основание
            if i == 0:
                bottom_width = self.params.get('bottom_width')
                if bottom_width and bottom_width != "-":
                    should_show = True
                    text_value = str(bottom_width)
                    logging.info(f"Подписываем нижнее основание: {text_value}")
            # Верхнее основание
            elif i == 2:
                top_width = self.params.get('top_width')
                if top_width and top_width != "-":
                    should_show = True
                    text_value = str(top_width)
                    logging.info(f"Подписываем верхнее основание: {text_value}")
            # Правая боковая
            elif i == 1:
                if side_lengths and len(side_lengths) > 1 and side_lengths[1] != "-" and side_lengths[1] is not None:
                    should_show = True
                    text_value = str(side_lengths[1])
                    logging.info(f"Подписываем правую боковую: {text_value}")
            # Левая боковая
            elif i == 3:
                if side_lengths and len(side_lengths) > 3 and side_lengths[3] != "-" and side_lengths[3] is not None:
                    should_show = True
                    text_value = str(side_lengths[3])
                    logging.info(f"Подписываем левую боковую: {text_value}")
            if should_show and text_value is not None:
                L = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
                mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
                nx, ny = -(p2[1] - p1[1]) / L, (p2[0] - p1[0]) / L
                offset = 0.2
                ax.text(mx + nx*offset, my + ny*offset, text_value, ha='center', fontsize=14,
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    def add_angles(self, ax):
        """
        Подписывает только те углы, которые явно указаны (не "-").
        """
        angle_values = self.params.get('angle_values')
        vertex_labels = self.params.get('vertex_labels', [chr(65+j) for j in range(len(self.points))])
        if not vertex_labels or len(vertex_labels) < len(self.points):
            vertex_labels = [chr(65+j) for j in range(len(self.points))]
            self.params['vertex_labels'] = vertex_labels
        for i in range(len(self.points)):
            if angle_values and i < len(angle_values):
                angle_val = angle_values[i]
                if angle_val == "-" or angle_val is None:
                    continue
            else:
                continue
            A = np.array(self.points[(i-1) % len(self.points)])
            B = np.array(self.points[i])
            C = np.array(self.points[(i+1) % len(self.points)])
            v1 = A - B
            v2 = C - B
            angle_text = f"{angle_val:.1f}°"
            radius = min(np.linalg.norm(v1), np.linalg.norm(v2)) / 4
            angle_mid = (v1/np.linalg.norm(v1) + v2/np.linalg.norm(v2))
            angle_mid = angle_mid / np.linalg.norm(angle_mid) * radius * 0.8
            ax.text(B[0] + angle_mid[0], B[1] + angle_mid[1], angle_text, ha='center', fontsize=14,
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')) 