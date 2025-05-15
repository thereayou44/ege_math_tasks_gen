import numpy as np
import matplotlib.pyplot as plt
import re
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
        if 'center' not in self.params:
            self.params['center'] = (0, 0)
        if 'radius' not in self.params:
            self.params['radius'] = 3
        if 'center_label' not in self.params:
            self.params['center_label'] = 'O'
        if 'show_center' not in self.params:
            self.params['show_center'] = True
        if 'show_central_angles' not in self.params:
            self.params['show_central_angles'] = False
        if 'show_inscribed_angles' not in self.params:
            self.params['show_inscribed_angles'] = False
        if 'show_tangent' not in self.params:
            self.params['show_tangent'] = False
        if 'tangent_point' not in self.params:
            self.params['tangent_point'] = None
            
    def compute_points(self):
        """
        Вычисляет координаты точек окружности для отображения.
        В случае окружности, возвращает точки для визуализации ограничивающего прямоугольника.
        
        Returns:
            list: Список угловых точек ограничивающего прямоугольника
        """
        cx, cy = self.params.get('center', (0, 0))
        r = self.params.get('radius', 3)
        
        # Возвращаем углы ограничивающего прямоугольника
        return [(cx - r, cy - r), (cx + r, cy - r), (cx + r, cy + r), (cx - r, cy + r)]
    
    def draw(self, ax=None):
        """
        Отрисовывает окружность на заданных осях.
        Переопределяет метод базового класса для специфичной отрисовки окружности.
        
        Args:
            ax (matplotlib.axes.Axes): Оси для отрисовки, если None - создаются новые
            
        Returns:
            matplotlib.axes.Axes: Оси с нарисованной окружностью
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 8))
            ax.set_aspect('equal')
            ax.axis('off')
        
        # Получаем параметры окружности
        cx, cy = self.params.get('center', (0, 0))
        r = self.params.get('radius', 3)
        
        # Рисуем окружность
        circle = plt.Circle((cx, cy), r, fill=False, edgecolor='blue', linewidth=2)
        ax.add_patch(circle)
        
        # Устанавливаем границы области отображения
        ax.set_xlim(cx - r * 1.2, cx + r * 1.2)
        ax.set_ylim(cy - r * 1.2, cy + r * 1.2)
        
        # Отображение центра
        if self.params.get('show_center', True):
            ax.text(cx, cy, self.params.get('center_label', 'O'),
                   ha='center', va='center', fontsize=14,
                   bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        
        # Отображение радиуса
        self._add_radius(ax)
        
        # Отображение диаметра
        self._add_diameter(ax)
        
        # Отображение хорды
        self._add_chord(ax)
        
        # Отображение центральных углов
        if self.params.get('show_central_angles', False):
            self._add_central_angles(ax)
            
        # Отображение вписанных углов
        if self.params.get('show_inscribed_angles', False):
            self._add_inscribed_angles(ax)
            
        # Отображение касательной
        if self.params.get('show_tangent', False):
            self._add_tangent(ax)
        
        return ax
    
    def _add_radius(self, ax):
        """
        Добавляет отображение радиуса окружности.
        
        Args:
            ax (matplotlib.axes.Axes): Оси для отрисовки
        """
        radius_value = self.params.get('radius_value', None)
        
        if self.params.get('show_radius', False) or radius_value is not None:
            cx, cy = self.params.get('center', (0, 0))
            r = self.params.get('radius', 3)
            
            # Рисуем линию радиуса
            ax.plot([cx, cx + r], [cy, cy], 'r-', lw=1.5)
            
            # Отображаем значение радиуса
            displayed_radius = radius_value if radius_value is not None else r
            ax.text(cx + r/2, cy + 0.3, f"r={displayed_radius}", ha='center', fontsize=12,
                   bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
    def _add_diameter(self, ax):
        """
        Добавляет отображение диаметра окружности.
        
        Args:
            ax (matplotlib.axes.Axes): Оси для отрисовки
        """
        diameter_value = self.params.get('diameter_value', None)
        
        if self.params.get('show_diameter', False) or diameter_value is not None:
            cx, cy = self.params.get('center', (0, 0))
            r = self.params.get('radius', 3)
            
            # Рисуем линию диаметра
            ax.plot([cx - r, cx + r], [cy, cy], 'g-', lw=1.5)
            
            # Отображаем значение диаметра
            displayed_diameter = diameter_value if diameter_value is not None else 2*r
            ax.text(cx, cy - 0.3, f"d={displayed_diameter}", ha='center', fontsize=12,
                   bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
    def _add_chord(self, ax):
        """
        Добавляет отображение хорды окружности.
        
        Args:
            ax (matplotlib.axes.Axes): Оси для отрисовки
        """
        chord_value = self.params.get('chord_value', None)
        
        if self.params.get('show_chord', False) or chord_value is not None:
            cx, cy = self.params.get('center', (0, 0))
            r = self.params.get('radius', 3)
            
            if chord_value is not None:
                # Проверяем, что хорда не больше диаметра
                chord_value = min(chord_value, 2*r)
                
                # Вычисляем положение хорды
                half_chord = chord_value / 2
                
                # Расстояние от центра до хорды (по теореме Пифагора)
                if half_chord < r:  # Защита от ошибок вычисления
                    h = np.sqrt(r**2 - half_chord**2)
                else:
                    h = 0
                
                # Рисуем хорду горизонтально ниже центра
                chord_y = cy - h
                chord_start_x = cx - half_chord
                chord_end_x = cx + half_chord
                
                # Рисуем хорду
                ax.plot([chord_start_x, chord_end_x], [chord_y, chord_y], 'b-', lw=1.5)
                
                # Подпись располагаем четко под хордой
                ax.text((chord_start_x + chord_end_x)/2, chord_y - 0.4, 
                        f"{chord_value}", ha='center', fontsize=12,
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
    def _add_central_angles(self, ax):
        """
        Добавляет отображение центральных углов окружности.
        
        Args:
            ax (matplotlib.axes.Axes): Оси для отрисовки
        """
        central_angle_value = self.params.get('central_angle_value', None)
        
        if central_angle_value is not None:
            cx, cy = self.params.get('center', (0, 0))
            r = self.params.get('radius', 3)
            
            # Преобразуем угол в радианы
            angle_rad = np.radians(central_angle_value)
            
            # Рисуем дугу центрального угла
            theta1 = 0
            theta2 = central_angle_value
            
            # Рисуем два радиуса
            x1 = cx + r * np.cos(0)
            y1 = cy + r * np.sin(0)
            x2 = cx + r * np.cos(angle_rad)
            y2 = cy + r * np.sin(angle_rad)
            
            ax.plot([cx, x1], [cy, y1], 'r-', lw=1)
            ax.plot([cx, x2], [cy, y2], 'r-', lw=1)
            
            # Рисуем дугу
            arc = plt.matplotlib.patches.Arc(
                (cx, cy), 2*r*0.3, 2*r*0.3, 
                theta1=theta1, theta2=theta2, 
                color='red', lw=1.5
            )
            ax.add_patch(arc)
            
            # Подписываем угол
            mid_angle_rad = angle_rad / 2
            text_x = cx + r * 0.2 * np.cos(mid_angle_rad)
            text_y = cy + r * 0.2 * np.sin(mid_angle_rad)
            ax.text(text_x, text_y, f"{central_angle_value}°", 
                    ha='center', va='center', color='red', fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
    def _add_inscribed_angles(self, ax):
        """
        Добавляет отображение вписанных углов окружности.
        
        Args:
            ax (matplotlib.axes.Axes): Оси для отрисовки
        """
        inscribed_angle_value = self.params.get('inscribed_angle_value', None)
        
        if inscribed_angle_value is not None:
            cx, cy = self.params.get('center', (0, 0))
            r = self.params.get('radius', 3)
            
            # Преобразуем угол в радианы
            angle_rad = np.radians(inscribed_angle_value * 2)  # Центральный угол в 2 раза больше вписанного
            
            # Рисуем точки на окружности
            x1 = cx + r * np.cos(0)
            y1 = cy + r * np.sin(0)
            x2 = cx + r * np.cos(angle_rad)
            y2 = cy + r * np.sin(angle_rad)
            
            # Выбираем точку на окружности для вершины вписанного угла
            # Например, берем точку на окружности, противоположную середине дуги
            midpoint_angle = angle_rad / 2 + np.pi
            x3 = cx + r * np.cos(midpoint_angle)
            y3 = cy + r * np.sin(midpoint_angle)
            
            # Рисуем треугольник вписанного угла
            ax.plot([x1, x3], [y1, y3], 'g-', lw=1)
            ax.plot([x2, x3], [y2, y3], 'g-', lw=1)
            
            # Рисуем дугу вписанного угла
            # Вычисляем углы относительно вершины вписанного угла
            vec1 = (x1 - x3, y1 - y3)
            vec2 = (x2 - x3, y2 - y3)
            
            # Вычисляем угол между векторами для определения направления дуги
            angle1 = np.arctan2(vec1[1], vec1[0])
            angle2 = np.arctan2(vec2[1], vec2[0])
            
            # Нормализуем углы
            if angle1 < 0:
                angle1 += 2 * np.pi
            if angle2 < 0:
                angle2 += 2 * np.pi
                
            # Обеспечиваем правильный порядок углов для дуги
            if angle1 > angle2:
                angle1, angle2 = angle2, angle1
                
            # Преобразуем в градусы для matplotlib
            theta1 = np.degrees(angle1)
            theta2 = np.degrees(angle2)
            
            # Рисуем дугу
            arc_radius = r * 0.2
            arc = plt.matplotlib.patches.Arc(
                (x3, y3), 2*arc_radius, 2*arc_radius, 
                theta1=theta1, theta2=theta2, 
                color='green', lw=1.5
            )
            ax.add_patch(arc)
            
            # Подписываем угол
            mid_angle = (angle1 + angle2) / 2
            text_x = x3 + arc_radius * 0.7 * np.cos(mid_angle)
            text_y = y3 + arc_radius * 0.7 * np.sin(mid_angle)
            ax.text(text_x, text_y, f"{inscribed_angle_value}°", 
                    ha='center', va='center', color='green', fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
            
            # Добавляем точки на окружности
            ax.plot([x1, x2], [y1, y2], 'go', markersize=5)
            ax.plot(x3, y3, 'go', markersize=5)
    
    def _add_tangent(self, ax):
        """
        Добавляет отображение касательной к окружности.
        
        Args:
            ax (matplotlib.axes.Axes): Оси для отрисовки
        """
        tangent_point = self.params.get('tangent_point', None)
        
        if tangent_point is not None:
            cx, cy = self.params.get('center', (0, 0))
            r = self.params.get('radius', 3)
            
            tx, ty = tangent_point
            
            # Проверяем, что точка находится на окружности
            distance = np.sqrt((tx - cx)**2 + (ty - cy)**2)
            if abs(distance - r) > 0.01 * r:  # Допуск на округление
                # Проецируем точку на окружность
                angle = np.arctan2(ty - cy, tx - cx)
                tx = cx + r * np.cos(angle)
                ty = cy + r * np.sin(angle)
            
            # Вектор от центра к точке касания
            vx, vy = tx - cx, ty - cy
            
            # Нормализуем вектор
            length = np.sqrt(vx**2 + vy**2)
            vx, vy = vx / length, vy / length
            
            # Вектор касательной перпендикулярен радиус-вектору
            tx1, ty1 = tx - vy * r, ty + vx * r
            tx2, ty2 = tx + vy * r, ty - vx * r
            
            # Рисуем касательную
            ax.plot([tx1, tx2], [ty1, ty2], 'g-', lw=1.5)
            
            # Отображаем меткой точку касания
            ax.plot(tx, ty, 'go', markersize=5)
            ax.text(tx + 0.1, ty + 0.1, 'T', ha='left', va='bottom', fontsize=12)
    
    def add_side_lengths(self, ax):
        """
        Пустая реализация метода add_side_lengths для совместимости с базовым классом.
        У окружности нет "сторон" в традиционном понимании, поэтому метод ничего не делает.
        
        Args:
            ax (matplotlib.axes.Axes): Оси для отрисовки
        """
        pass
    
    def add_vertex_labels(self, ax):
        """
        Пустая реализация метода add_vertex_labels для совместимости с базовым классом.
        У окружности нет "вершин" в традиционном понимании, метка центра добавляется отдельно.
        
        Args:
            ax (matplotlib.axes.Axes): Оси для отрисовки
        """
        pass
    
    def add_angles(self, ax):
        """
        Пустая реализация метода add_angles для совместимости с базовым классом.
        У окружности нет "углов" в традиционном понимании, центральные 
        и вписанные углы добавляются отдельными методами.
        
        Args:
            ax (matplotlib.axes.Axes): Оси для отрисовки
        """
        pass
    
    @staticmethod
    def from_text(task_text):
        """
        Создает объект окружности из текста задачи.
        
        Args:
            task_text (str): Текст задачи
            
        Returns:
            Circle: Объект окружности с параметрами из текста
        """
        import re
        from app.prompts import DEFAULT_VISUALIZATION_PARAMS
        
        params = DEFAULT_VISUALIZATION_PARAMS["circle"].copy()
        
        # Ищем радиус
        radius_pattern = r'радиус\s*[=:]?\s*(\d+(?:[,.]\d+)?)'
        radius_match = re.search(radius_pattern, task_text, re.IGNORECASE)
        
        if radius_match:
            try:
                radius = float(radius_match.group(1).replace(',', '.'))
                params['radius'] = radius
                params['radius_value'] = radius
                params['show_radius'] = True
            except Exception:
                pass
        
        # Ищем диаметр
        diameter_pattern = r'диаметр\s*[=:]?\s*(\d+(?:[,.]\d+)?)'
        diameter_match = re.search(diameter_pattern, task_text, re.IGNORECASE)
        
        if diameter_match:
            try:
                diameter = float(diameter_match.group(1).replace(',', '.'))
                params['radius'] = diameter / 2
                params['diameter_value'] = diameter
                params['show_diameter'] = True
            except Exception:
                pass
        
        # Ищем хорду
        chord_pattern = r'хорда\s*[=:]?\s*(\d+(?:[,.]\d+)?)'
        chord_match = re.search(chord_pattern, task_text, re.IGNORECASE)
        
        if chord_match:
            try:
                chord = float(chord_match.group(1).replace(',', '.'))
                params['chord_value'] = chord
                params['show_chord'] = True
            except Exception:
                pass
                
        # Ищем центральный угол
        central_angle_pattern = r'центральный\s+угол\s*[=:]?\s*(\d+(?:[,.]\d+)?)[°\s]'
        central_angle_match = re.search(central_angle_pattern, task_text, re.IGNORECASE)
        
        if central_angle_match:
            try:
                angle = float(central_angle_match.group(1).replace(',', '.'))
                params['central_angle_value'] = angle
                params['show_central_angles'] = True
            except Exception:
                pass
                
        # Ищем вписанный угол
        inscribed_angle_pattern = r'вписанный\s+угол\s*[=:]?\s*(\d+(?:[,.]\d+)?)[°\s]'
        inscribed_angle_match = re.search(inscribed_angle_pattern, task_text, re.IGNORECASE)
        
        if inscribed_angle_match:
            try:
                angle = float(inscribed_angle_match.group(1).replace(',', '.'))
                params['inscribed_angle_value'] = angle
                params['show_inscribed_angles'] = True
            except Exception:
                pass
        
        # Ищем касательную
        if re.search(r'касательн[а-я]+', task_text, re.IGNORECASE):
            params['show_tangent'] = True
            # Устанавливаем точку касания справа от окружности (0 градусов)
            params['tangent_point'] = 0
        
        # Показываем касательную, если это указано в тексте
        if re.search(r'касательн[а-я]+|точк[а-я]+\s+касани[а-я]+', task_text, re.IGNORECASE):
            params['show_tangent'] = True
            
        # Показываем углы, если это указано в тексте
        if re.search(r'угл[а-я]+|вписанн[а-я]+\s+угл|центральн[а-я]+\s+угл', task_text, re.IGNORECASE):
            # По умолчанию показываем центральные углы при упоминании углов и окружности
            params['show_central_angles'] = True
            # Если конкретно упоминается вписанный угол
            if re.search(r'вписанн[а-я]+\s+угл', task_text, re.IGNORECASE):
                params['show_inscribed_angles'] = True
                params['inscribed_angle_value'] = params.get('inscribed_angle_value', 30)
        
        # Определяем, какие конкретно углы нужно показать
        if "угол" in task_text.lower() or "углы" in task_text.lower():
            # Проверяем, упоминаются ли конкретные углы
            angle_names = re.findall(r'угл[а-я]*\s+([A-Z])', task_text)
            if angle_names:
                params['show_specific_angles'] = angle_names
        
        return Circle(params) 