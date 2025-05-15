import numpy as np
import matplotlib.pyplot as plt
import uuid
import os
import logging

class GeometricFigure:
    """
    Базовый класс для всех геометрических фигур.
    """
    
    def __init__(self, params=None):
        """
        Инициализирует фигуру с базовыми параметрами.
        
        Args:
            params (dict): Словарь параметров фигуры
        """
        self.params = params or {}
        self.points = []
        self.figure_type = "base"
    
    def compute_points(self):
        """
        Вычисляет координаты точек фигуры.
        Должен быть переопределен в дочерних классах.
        
        Returns:
            list: Список точек [(x1,y1), (x2,y2), ...]
        """
        raise NotImplementedError("Метод должен быть переопределен в дочернем классе")
    
    def draw(self, ax=None):
        """
        Отрисовывает фигуру на заданных осях.
        
        Args:
            ax (matplotlib.axes.Axes): Оси для отрисовки, если None - создаются новые
            
        Returns:
            matplotlib.axes.Axes: Оси с нарисованной фигурой
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 8))
            ax.set_aspect('equal')
            ax.axis('off')
        
        if not self.points:
            self.points = self.compute_points()
        
        # Проверка валидности точек
        if len(self.points) < 2:
            logging.warning(f"Недостаточно точек для отрисовки фигуры: {len(self.points)}")
            return ax
        
        # Отрисовка фигуры - рисуем замкнутый полигон
        if len(self.points) >= 3:  # Полигон для 3+ точек
            # Закрываем полигон
            points_closed = self.points.copy()
            if points_closed[0] != points_closed[-1]:
                points_closed.append(points_closed[0])
                
            # Извлекаем x и y координаты для отрисовки
            xs, ys = zip(*points_closed)
            
            # Рисуем полигон
            ax.plot(xs, ys, 'blue', linewidth=2)
            
            # Также добавляем сам полигон для визуализации площади, если нужно
            if self.params.get('fill', False):
                patch = plt.Polygon(self.points, fill=True, 
                                   facecolor=self.params.get('fill_color', 'lightblue'), 
                                   alpha=self.params.get('fill_alpha', 0.3), 
                                   edgecolor='blue', linewidth=2)
                ax.add_patch(patch)
        elif len(self.points) == 2:  # Линия для 2 точек
            x1, y1 = self.points[0]
            x2, y2 = self.points[1]
            ax.plot([x1, x2], [y1, y2], 'blue', linewidth=2)
        
        # Определяем границы отображения
        min_x = min(p[0] for p in self.points) - 1
        max_x = max(p[0] for p in self.points) + 1
        min_y = min(p[1] for p in self.points) - 1
        max_y = max(p[1] for p in self.points) + 1
        
        # Устанавливаем границы с небольшим отступом
        width = max_x - min_x
        height = max_y - min_y
        padding = max(width, height) * 0.1
        
        ax.set_xlim(min_x - padding, max_x + padding)
        ax.set_ylim(min_y - padding, max_y + padding)
            
        # Добавляем подписи вершин
        self.add_vertex_labels(ax)
        
        # Добавляем длины сторон
        self.add_side_lengths(ax)
        
        # Добавляем углы
        self.add_angles(ax)
        
        return ax
    
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
    
    def get_side_length_text(self, side_index, side_name, computed_length):
        """
        Возвращает текст для подписи длины стороны.
        Может быть переопределен в дочерних классах.
        
        Args:
            side_index (int): Индекс стороны
            side_name (str): Название стороны (например, "AB")
            computed_length (float): Вычисленная длина стороны
            
        Returns:
            str: Текст для подписи или None, если подпись не нужна
        """
        side_lengths = self.params.get('side_lengths', None)
        
        # Если указаны конкретные длины сторон, используем их
        if side_lengths and side_index < len(side_lengths) and side_lengths[side_index] is not None:
            return f"{side_lengths[side_index]}"
        
        # Иначе отображаем вычисленную длину
        return f"{computed_length:.2f}".rstrip('0').rstrip('.')
    
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
    
    def save(self, filename=None):
        """
        Сохраняет изображение фигуры в файл.
        
        Args:
            filename (str): Имя файла для сохранения. Если None, создается случайное имя.
            
        Returns:
            str: Путь к сохраненному файлу
        """
        if filename is None:
            # Создаем случайное имя файла
            filename = f"{self.figure_type}_{uuid.uuid4().hex[:8]}.png"
        
        # Проверяем, содержит ли путь директорию
        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        
        # Если путь не указан, сохраняем в директорию static/images/generated
        if not directory:
            os.makedirs("static/images/generated", exist_ok=True)
            filename = os.path.join("static/images/generated", filename)
        
        # Создаем фигуру и отрисовываем
        _, ax = plt.subplots(figsize=(8, 8))
        ax.set_aspect('equal')
        ax.axis('off')
        self.draw(ax)
        
        # Сохраняем изображение
        plt.savefig(filename, dpi=100, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        
        return filename 