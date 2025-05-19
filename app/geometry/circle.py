import numpy as np
import matplotlib.pyplot as plt
import re
import logging
from app.geometry.base import GeometricFigure

class Circle(GeometricFigure):
    """
    Класс для представления и визуализации окружности.
    """
    
    def __init__(self, params=None):
        """
        Инициализирует окружность с заданными параметрами.
        
        Args:
            params (dict): Словарь параметров фигуры
        """
        super().__init__(params)
        self.figure_type = "circle"
        
        # Установка параметров по умолчанию, если не указаны
        if 'radius' not in self.params:
            self.params['radius'] = 3
        if 'center' not in self.params:
            self.params['center'] = (0, 0)
        if 'center_label' not in self.params:
            self.params['center_label'] = 'O'
        if 'show_center' not in self.params:
            self.params['show_center'] = True
        if 'show_radius' not in self.params:
            self.params['show_radius'] = False
        if 'show_diameter' not in self.params:
            self.params['show_diameter'] = False
        if 'show_chord' not in self.params:
            self.params['show_chord'] = False
            
    def compute_points(self):
        """
        Вычисляет координаты точек окружности.
        
        Returns:
            list: Список точек по контуру окружности
        """
        # Для окружности мы возвращаем только центр и несколько ключевых точек на контуре
        center = self.params.get('center', (0, 0))
        radius = self.params.get('radius', 3)
        
        # Создаем ключевые точки на контуре (север, восток, юг, запад)
        points = [
            center,  # Центр
            (center[0], center[1] + radius),  # Север
            (center[0] + radius, center[1]),  # Восток
            (center[0], center[1] - radius),  # Юг
            (center[0] - radius, center[1])   # Запад
        ]
        
        return points
    
    def draw(self, ax=None):
        """
        Отрисовывает окружность с дополнительными элементами.
        
        Args:
            ax (matplotlib.axes.Axes): Оси для отрисовки
            
        Returns:
            matplotlib.axes.Axes: Оси с нарисованной фигурой
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 10))
            ax.set_aspect('equal')
        
        # Отключаем оси для лучшего вида
        ax.axis('off')
        
        # Рассчитываем точки
        if not self.points:
            self.points = self.compute_points()
        
        # Получаем параметры для отрисовки
        center = self.params.get('center', (0, 0))
        radius = self.params.get('radius', 3)
        
        # Рисуем окружность
        circle = plt.Circle(center, radius, fill=False, color='blue', linewidth=2.5)
        ax.add_patch(circle)
        
        # Обозначаем центр
        if self.params.get('show_center', True):
            # Рисуем точку в центре
            ax.plot(center[0], center[1], 'bo', markersize=6)
            
            # Добавляем метку центра
            center_label = self.params.get('center_label', 'O')
            ax.text(center[0], center[1] + 0.2, center_label, 
                    ha='center', va='bottom', fontsize=14, fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.9, edgecolor='none'))
        
        # Добавляем радиус, если нужно
        if self.params.get('show_radius', False):
            self._add_radius(ax)
            
        # Добавляем диаметр, если нужно
        if self.params.get('show_diameter', False):
            self._add_diameter(ax)
            
        # Добавляем хорду, если нужно
        if self.params.get('show_chord', False):
            self._add_chord(ax)
            
        # Добавляем центральные углы, если нужно
        if self.params.get('show_central_angles', False):
            self._add_central_angles(ax)
            
        # Добавляем вписанные углы, если нужно
        if self.params.get('show_inscribed_angles', False):
            self._add_inscribed_angles(ax)
            
        # Добавляем касательную, если нужно
        if self.params.get('show_tangent', False):
            self._add_tangent(ax)
            
        # Определяем границы отображения с хорошим отступом
        ax.set_xlim(center[0] - radius * 1.5, center[0] + radius * 1.5)
        ax.set_ylim(center[1] - radius * 1.5, center[1] + radius * 1.5)
        
        # Добавляем сетку для лучшей ориентации
        ax.grid(True, linestyle='--', alpha=0.3)
        
        return ax
    
    def _add_radius(self, ax):
        """
        Добавляет отображение радиуса окружности.
        
        Args:
            ax (matplotlib.axes.Axes): Оси для отрисовки
        """
        center = self.params.get('center', (0, 0))
        radius = self.params.get('radius', 3)
        radius_value = self.params.get('radius_value', radius)
        
        # Рисуем радиус от центра до правой точки
        ax.plot([center[0], center[0] + radius], [center[1], center[1]], 'r-', lw=1.5)
        
        # Добавляем подпись радиуса
        radius_text = f"R = {radius_value}"
        ax.text(center[0] + radius/2, center[1] + 0.2, radius_text, 
                ha='center', va='bottom', color='red', fontsize=12,
                bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', boxstyle='round,pad=0.3'))
    
    def _add_diameter(self, ax):
        """
        Добавляет отображение диаметра окружности.
        
        Args:
            ax (matplotlib.axes.Axes): Оси для отрисовки
        """
        center = self.params.get('center', (0, 0))
        radius = self.params.get('radius', 3)
        diameter_value = self.params.get('diameter_value', 2 * radius)
        
        # Рисуем диаметр, проходящий через центр
        ax.plot([center[0] - radius, center[0] + radius], [center[1], center[1]], 'g-', lw=1.5)
        
        # Добавляем подпись диаметра
        diameter_text = f"D = {diameter_value}"
        ax.text(center[0], center[1] - 0.3, diameter_text, 
                ha='center', va='top', color='green', fontsize=12,
                bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', boxstyle='round,pad=0.3'))
    
    def _add_chord(self, ax):
        """
        Добавляет отображение хорды окружности.
        
        Args:
            ax (matplotlib.axes.Axes): Оси для отрисовки
        """
        center = self.params.get('center', (0, 0))
        radius = self.params.get('radius', 3)
        chord_value = self.params.get('chord_value', None)
        
        # Угол для точек хорды (по умолчанию 60 градусов)
        angle_rad = np.radians(60)
        
        # Вычисляем точки хорды
        x1 = center[0] + radius * np.cos(angle_rad)
        y1 = center[1] + radius * np.sin(angle_rad)
        x2 = center[0] + radius * np.cos(angle_rad + np.pi)
        y2 = center[1] + radius * np.sin(angle_rad + np.pi)
        
        # Рисуем хорду
        ax.plot([x1, x2], [y1, y2], 'b-', lw=1.5)
        
        # Вычисляем длину хорды, если не указана явно
        if chord_value is None:
            chord_length = 2 * radius * np.sin(angle_rad)
            chord_value = f"{chord_length:.2f}".rstrip('0').rstrip('.')
        
        # Добавляем подпись хорды
        chord_text = f"Chord = {chord_value}"
        # Размещаем текст посередине хорды
        ax.text((x1 + x2)/2, (y1 + y2)/2 + 0.2, chord_text, 
                ha='center', va='bottom', color='blue', fontsize=12,
                bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', boxstyle='round,pad=0.3'))
    
    def _add_central_angles(self, ax):
        """
        Добавляет отображение центральных углов окружности.
        
        Args:
            ax (matplotlib.axes.Axes): Оси для отрисовки
        """
        center = self.params.get('center', (0, 0))
        radius = self.params.get('radius', 3)
        angle_value = self.params.get('central_angle_value', 45)
        
        # Переводим значение угла в радианы
        angle_rad = np.radians(angle_value)
        
        # Вычисляем точки на окружности, образующие угол
        x1 = center[0] + radius * np.cos(0)
        y1 = center[1] + radius * np.sin(0)
        x2 = center[0] + radius * np.cos(angle_rad)
        y2 = center[1] + radius * np.sin(angle_rad)
        
        # Рисуем линии от центра до точек
        ax.plot([center[0], x1], [center[1], y1], 'r-', lw=1.5)
        ax.plot([center[0], x2], [center[1], y2], 'r-', lw=1.5)
        
        # Рисуем дугу для обозначения угла
        arc_radius = radius * 0.3  # Размер дуги
        arc = plt.matplotlib.patches.Arc(center, arc_radius * 2, arc_radius * 2,
                                       theta1=0, theta2=np.degrees(angle_rad),
                                       color='red', lw=1.5)
        ax.add_patch(arc)
        
        # Добавляем подпись угла
        angle_text = f"{angle_value}°"
        # Размещаем текст посередине дуги
        angle_mid_rad = angle_rad / 2
        text_r = arc_radius * 1.2  # Радиус для размещения текста
        ax.text(center[0] + text_r * np.cos(angle_mid_rad), 
                center[1] + text_r * np.sin(angle_mid_rad), 
                angle_text, ha='center', va='center', color='red', fontsize=12,
                bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', boxstyle='round,pad=0.3'))
    
    def _add_inscribed_angles(self, ax):
        """
        Добавляет отображение вписанных углов окружности.
        
        Args:
            ax (matplotlib.axes.Axes): Оси для отрисовки
        """
        center = self.params.get('center', (0, 0))
        radius = self.params.get('radius', 3)
        angle_value = self.params.get('inscribed_angle_value', 30)
        
        # Переводим значение угла в радианы
        angle_rad_central = np.radians(angle_value * 2)  # Центральный угол в 2 раза больше вписанного
        
        # Вычисляем точки на окружности - концы дуги
        x1 = center[0] + radius * np.cos(0)
        y1 = center[1] + radius * np.sin(0)
        x2 = center[0] + radius * np.cos(angle_rad_central)
        y2 = center[1] + radius * np.sin(angle_rad_central)
        
        # Вычисляем точку для вершины вписанного угла (противоположная сторона окружности)
        opposite_angle = np.pi + angle_rad_central / 2
        x3 = center[0] + radius * np.cos(opposite_angle)
        y3 = center[1] + radius * np.sin(opposite_angle)
        
        # Рисуем линии, образующие вписанный угол
        ax.plot([x3, x1], [y3, y1], 'b-', lw=1.5)
        ax.plot([x3, x2], [y3, y2], 'b-', lw=1.5)
        
        # Рисуем дугу для обозначения угла
        arc_radius = radius * 0.2  # Размер дуги
        
        # Вычисляем углы для дуги в градусах
        v1 = [x1 - x3, y1 - y3]
        v2 = [x2 - x3, y2 - y3]
        
        # Вычисляем углы векторов в градусах
        angle1 = np.degrees(np.arctan2(v1[1], v1[0])) % 360
        angle2 = np.degrees(np.arctan2(v2[1], v2[0])) % 360
        
        # Рисуем дугу для обозначения угла
        arc = plt.matplotlib.patches.Arc((x3, y3), arc_radius * 2, arc_radius * 2,
                                       theta1=angle1, theta2=angle2,
                                       color='blue', lw=1.5)
        ax.add_patch(arc)
        
        # Добавляем подпись угла
        angle_text = f"{angle_value}°"
        # Размещаем текст посередине дуги
        angle_mid = (angle1 + angle2) / 2
        if angle2 < angle1:  # Корректировка, если угол переходит через 360
            angle_mid = (angle1 + angle2 + 360) / 2
            if angle_mid >= 360:
                angle_mid -= 360
        
        text_r = arc_radius * 1.5  # Радиус для размещения текста
        angle_mid_rad = np.radians(angle_mid)
        ax.text(x3 + text_r * np.cos(angle_mid_rad), 
                y3 + text_r * np.sin(angle_mid_rad), 
                angle_text, ha='center', va='center', color='blue', fontsize=12,
                bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', boxstyle='round,pad=0.3'))
    
    def _add_tangent(self, ax):
        """
        Добавляет отображение касательной к окружности.
        
        Args:
            ax (matplotlib.axes.Axes): Оси для отрисовки
        """
        center = self.params.get('center', (0, 0))
        radius = self.params.get('radius', 3)
        tangent_point = self.params.get('tangent_point', None)
        
        # Если точка касания не указана, используем точку справа
        if tangent_point is None:
            tangent_point = (center[0] + radius, center[1])
        
        # Вычисляем вектор от центра к точке касания
        cx, cy = center
        tx, ty = tangent_point
        
        # Вектор касательной перпендикулярен радиусу
        # Поворачиваем радиус-вектор на 90 градусов против часовой стрелки
        perpendicular_x = -(ty - cy)
        perpendicular_y = tx - cx
        
        # Нормализуем вектор
        length = np.sqrt(perpendicular_x**2 + perpendicular_y**2)
        perpendicular_x /= length
        perpendicular_y /= length
        
        # Длина касательной линии
        tangent_length = radius * 1.5
        
        # Вычисляем концы касательной
        x1 = tx - perpendicular_x * tangent_length
        y1 = ty - perpendicular_y * tangent_length
        x2 = tx + perpendicular_x * tangent_length
        y2 = ty + perpendicular_y * tangent_length
        
        # Рисуем касательную
        ax.plot([x1, x2], [y1, y2], 'g-', lw=1.5)
        
        # Рисуем радиус к точке касания
        ax.plot([cx, tx], [cy, ty], 'r--', lw=1.5)
        
        # Отмечаем точку касания
        ax.plot(tx, ty, 'ro', markersize=6)
        
        # Добавляем подпись касательной
        ax.text(tx + perpendicular_x * 0.5, ty + perpendicular_y * 0.5, 
                "Tangent", ha='center', va='center', color='green', fontsize=12,
                bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', boxstyle='round,pad=0.3'))
    
    @staticmethod
    def from_text(params_text):
        """
        Создает объект окружности из текста параметров.
        
        Args:
            params_text (str): Текст параметров
            
        Returns:
            Circle: Объект окружности с параметрами из текста
        """
        from app.prompts.prompts import DEFAULT_VISUALIZATION_PARAMS, REGEX_PATTERNS
        
        # Создаем копию параметров по умолчанию
        params = DEFAULT_VISUALIZATION_PARAMS["circle"].copy()
        
        # Извлекаем параметры с помощью регулярных выражений
        
        # Радиус окружности
        radius_match = re.search(REGEX_PATTERNS['circle']['radius'], params_text, re.IGNORECASE)
        if radius_match:
            radius_str = radius_match.group(1).strip()
            try:
                radius = float(radius_str)
                params['radius'] = radius
            except ValueError:
                logging.warning(f"Не удалось преобразовать значение радиуса: {radius_str}")
        
        # Центр окружности
        center_match = re.search(REGEX_PATTERNS['circle']['center'], params_text, re.IGNORECASE)
        if center_match:
            center_str = center_match.group(1).strip()
            try:
                # Извлекаем координаты центра
                coord_match = re.search(r'\((.*?),(.*?)\)', center_str)
                if coord_match:
                    x = float(coord_match.group(1).strip())
                    y = float(coord_match.group(2).strip())
                    params['center'] = (x, y)
            except ValueError:
                logging.warning(f"Не удалось преобразовать координаты центра: {center_str}")
        
        # Метка центра
        center_label_match = re.search(REGEX_PATTERNS['circle']['center_label'], params_text, re.IGNORECASE)
        if center_label_match:
            center_label = center_label_match.group(1).strip()
            params['center_label'] = center_label
        
        # Показывать центр
        show_center_match = re.search(REGEX_PATTERNS['circle']['show_center'], params_text, re.IGNORECASE)
        if show_center_match:
            show_center_value = show_center_match.group(1).strip().lower()
            params['show_center'] = show_center_value in ['true', 'да', 'yes', '+']
        
        # Показывать радиус
        show_radius_match = re.search(REGEX_PATTERNS['circle']['show_radius'], params_text, re.IGNORECASE)
        if show_radius_match:
            show_radius_value = show_radius_match.group(1).strip().lower()
            params['show_radius'] = show_radius_value in ['true', 'да', 'yes', '+']
        
        # Значение радиуса для отображения
        radius_value_match = re.search(REGEX_PATTERNS['circle']['radius_value'], params_text, re.IGNORECASE)
        if radius_value_match:
            radius_value_str = radius_value_match.group(1).strip()
            try:
                radius_value = float(radius_value_str)
                params['radius_value'] = radius_value
            except ValueError:
                logging.warning(f"Не удалось преобразовать значение радиуса для отображения: {radius_value_str}")
        
        # Показывать диаметр
        show_diameter_match = re.search(REGEX_PATTERNS['circle']['show_diameter'], params_text, re.IGNORECASE)
        if show_diameter_match:
            show_diameter_value = show_diameter_match.group(1).strip().lower()
            params['show_diameter'] = show_diameter_value in ['true', 'да', 'yes', '+']
        
        # Значение диаметра для отображения
        diameter_value_match = re.search(REGEX_PATTERNS['circle']['diameter_value'], params_text, re.IGNORECASE)
        if diameter_value_match:
            diameter_value_str = diameter_value_match.group(1).strip()
            try:
                diameter_value = float(diameter_value_str)
                params['diameter_value'] = diameter_value
            except ValueError:
                logging.warning(f"Не удалось преобразовать значение диаметра: {diameter_value_str}")
        
        # Показывать хорду
        show_chord_match = re.search(REGEX_PATTERNS['circle']['show_chord'], params_text, re.IGNORECASE)
        if show_chord_match:
            show_chord_value = show_chord_match.group(1).strip().lower()
            params['show_chord'] = show_chord_value in ['true', 'да', 'yes', '+']
        
        # Значение хорды для отображения
        chord_value_match = re.search(REGEX_PATTERNS['circle']['chord_value'], params_text, re.IGNORECASE)
        if chord_value_match:
            chord_value_str = chord_value_match.group(1).strip()
            try:
                chord_value = float(chord_value_str)
                params['chord_value'] = chord_value
            except ValueError:
                logging.warning(f"Не удалось преобразовать значение хорды: {chord_value_str}")
        
        # Показывать центральные углы
        show_central_angles_match = re.search(REGEX_PATTERNS['circle']['show_central_angles'], params_text, re.IGNORECASE)
        if show_central_angles_match:
            show_central_angles_value = show_central_angles_match.group(1).strip().lower()
            params['show_central_angles'] = show_central_angles_value in ['true', 'да', 'yes', '+']
        
        # Значение центрального угла для отображения
        central_angle_value_match = re.search(REGEX_PATTERNS['circle']['central_angle_value'], params_text, re.IGNORECASE)
        if central_angle_value_match:
            central_angle_value_str = central_angle_value_match.group(1).strip()
            try:
                central_angle_value = float(central_angle_value_str)
                params['central_angle_value'] = central_angle_value
            except ValueError:
                logging.warning(f"Не удалось преобразовать значение центрального угла: {central_angle_value_str}")
        
        # Показывать вписанные углы
        show_inscribed_angles_match = re.search(REGEX_PATTERNS['circle']['show_inscribed_angles'], params_text, re.IGNORECASE)
        if show_inscribed_angles_match:
            show_inscribed_angles_value = show_inscribed_angles_match.group(1).strip().lower()
            params['show_inscribed_angles'] = show_inscribed_angles_value in ['true', 'да', 'yes', '+']
        
        # Значение вписанного угла для отображения
        inscribed_angle_value_match = re.search(REGEX_PATTERNS['circle']['inscribed_angle_value'], params_text, re.IGNORECASE)
        if inscribed_angle_value_match:
            inscribed_angle_value_str = inscribed_angle_value_match.group(1).strip()
            try:
                inscribed_angle_value = float(inscribed_angle_value_str)
                params['inscribed_angle_value'] = inscribed_angle_value
            except ValueError:
                logging.warning(f"Не удалось преобразовать значение вписанного угла: {inscribed_angle_value_str}")
        
        # Показывать касательную
        show_tangent_match = re.search(REGEX_PATTERNS['circle']['show_tangent'], params_text, re.IGNORECASE)
        if show_tangent_match:
            show_tangent_value = show_tangent_match.group(1).strip().lower()
            params['show_tangent'] = show_tangent_value in ['true', 'да', 'yes', '+']
        
        # Точка касания
        tangent_point_match = re.search(REGEX_PATTERNS['circle']['tangent_point'], params_text, re.IGNORECASE)
        if tangent_point_match:
            tangent_point_str = tangent_point_match.group(1).strip()
            try:
                # Извлекаем координаты точки касания
                point_match = re.search(r'\((.*?),(.*?)\)', tangent_point_str)
                if point_match:
                    x = float(point_match.group(1).strip())
                    y = float(point_match.group(2).strip())
                    params['tangent_point'] = (x, y)
            except ValueError:
                logging.warning(f"Не удалось преобразовать координаты точки касания: {tangent_point_str}")
        
        # Создаем объект окружности
        circle = Circle(params)
        return circle 