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
            _, ax = plt.subplots(figsize=(10, 10))
            ax.set_aspect('equal')
        
        # Отключаем оси для лучшего вида
        ax.axis('off')
        
        # Рассчитываем точки, если они еще не рассчитаны
        if not self.points:
            self.points = self.compute_points()
        
        # Проверка валидности точек
        if len(self.points) < 2:
            logging.warning(f"Недостаточно точек для отрисовки фигуры: {len(self.points)}")
            return ax
        
        # Отрисовка фигуры
        if len(self.points) >= 3:  # Полигон для 3+ точек
            # Закрываем полигон
            points_closed = self.points.copy()
            if points_closed[0] != points_closed[-1]:
                points_closed.append(points_closed[0])
                
            # Извлекаем x и y координаты для отрисовки
            xs, ys = zip(*points_closed)
            
            # Рисуем полигон более жирной линией для лучшей видимости
            ax.plot(xs, ys, color='blue', linewidth=2.5)
            
            # Добавляем заливку, если нужно
            if self.params.get('fill', False):
                patch = plt.Polygon(self.points, fill=True, 
                                   facecolor=self.params.get('fill_color', 'lightblue'), 
                                   alpha=self.params.get('fill_alpha', 0.2), 
                                   edgecolor='blue', linewidth=2.5)
                ax.add_patch(patch)
        elif len(self.points) == 2:  # Линия для 2 точек
            x1, y1 = self.points[0]
            x2, y2 = self.points[1]
            ax.plot([x1, x2], [y1, y2], color='blue', linewidth=2.5)
        
        # Определяем границы отображения с хорошим отступом
        min_x = min(p[0] for p in self.points) - 1
        max_x = max(p[0] for p in self.points) + 1
        min_y = min(p[1] for p in self.points) - 1
        max_y = max(p[1] for p in self.points) + 1
        
        # Устанавливаем более широкие границы с более значительным отступом
        width = max_x - min_x
        height = max_y - min_y
        padding = max(width, height) * 0.25  # Увеличиваем отступ для лучшего обзора
        
        ax.set_xlim(min_x - padding, max_x + padding)
        ax.set_ylim(min_y - padding, max_y + padding)
        
        # Добавляем сетку для лучшей ориентации
        ax.grid(True, linestyle='--', alpha=0.3)
            
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
            
            # Если метки не заданы, добавим их
            if not labels or len(labels) < len(self.points):
                labels = [chr(65+i) for i in range(len(self.points))]
                self.params['vertex_labels'] = labels  # Сохраняем для использования в других методах
            
            for i, ((x0,y0), lab) in enumerate(zip(self.points, labels)):
                # Проверяем, нужно ли отображать эту конкретную метку
                if show_specific_labels is None or lab in show_specific_labels:
                    # Добавляем контрастный белый круг под меткой для лучшей видимости
                    white_circle = plt.Circle((x0, y0), 0.25, color='white', alpha=0.9, zorder=2)
                    ax.add_patch(white_circle)
                    
                    # Добавляем метку с повышенным zorder, чтобы она была поверх круга
                    ax.text(x0, y0, lab, ha='center', va='center', fontsize=14, 
                            fontweight='bold', color='blue', zorder=3)
    
    def add_side_lengths(self, ax):
        """
        Добавляет подписи длин сторон фигуры.
        
        Args:
            ax (matplotlib.axes.Axes): Оси для отрисовки
        """
        side_lengths = self.params.get('side_lengths', None)
        show_lengths = self.params.get('show_lengths', True)  # По умолчанию показываем
        show_specific_sides = self.params.get('show_specific_sides', None)
        
        if side_lengths or show_lengths:
            vertex_labels = self.params.get('vertex_labels', [chr(65+j) for j in range(len(self.points))])

            if not vertex_labels or len(vertex_labels) < len(self.points):
                vertex_labels = [chr(65+j) for j in range(len(self.points))]
                self.params['vertex_labels'] = vertex_labels
                
            # Получаем средний размер стороны для расчета отступа
            side_sizes = []
            for i in range(len(self.points)):
                p1 = self.points[i]
                p2 = self.points[(i+1) % len(self.points)]
                L = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
                side_sizes.append(L)
            
            avg_side_size = sum(side_sizes) / len(side_sizes) if side_sizes else 1
            offset = avg_side_size * 0.1  # Масштабируем отступ относительно размера фигуры
                    
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
                    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
                    length = np.sqrt(dx*dx + dy*dy)
                    nx, ny = -dy/length, dx/length  # Нормализованная нормаль
                    
                    # Обработка специфичная для подклассов
                    text_value = self.get_side_length_text(i, side_name, L)
                    
                    # Отображаем значение с фоном для лучшей видимости
                    if text_value is not None:
                        # Смещаем текст от стороны фигуры
                        text_x = mx + nx * offset
                        text_y = my + ny * offset
                        
                        # Белый прямоугольник под текстом
                        ax.text(text_x, text_y, text_value, 
                                ha='center', va='center', fontsize=12, color='black',
                                bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', 
                                          boxstyle="round,pad=0.3"))
    
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
            # Проверяем, не является ли значение строкой "-", что означает "не отображать"
            if side_lengths[side_index] == "-":
                return None
            return f"{side_lengths[side_index]}"
        
        # Иначе отображаем вычисленную длину, округленную до 2 знаков
        return f"{computed_length:.2f}".rstrip('0').rstrip('.')
    
    def add_angles(self, ax):
        """
        Добавляет отображение углов фигуры.
        
        Args:
            ax (matplotlib.axes.Axes): Оси для отрисовки
        """
        if self.params.get('show_angles', False):
            angle_values = self.params.get('angle_values', None)
            show_angle_arcs = self.params.get('show_angle_arcs', True)  # По умолчанию показываем дуги
            show_specific_angles = self.params.get('show_specific_angles', None)
            
            vertex_labels = self.params.get('vertex_labels', [chr(65+j) for j in range(len(self.points))])
            
            # Вычисляем средний размер фигуры для масштабирования дуг
            if self.points:
                all_x = [p[0] for p in self.points]
                all_y = [p[1] for p in self.points]
                width = max(all_x) - min(all_x)
                height = max(all_y) - min(all_y)
                avg_size = (width + height) / 2
                radius = avg_size * 0.1  # 10% от среднего размера фигуры
            else:
                radius = 0.5  # Значение по умолчанию
            
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
                
                # Проверяем, что векторы не нулевые
                if np.linalg.norm(v1) > 1e-10 and np.linalg.norm(v2) > 1e-10:
                    # Нормализуем векторы
                    v1 = v1 / np.linalg.norm(v1)
                    v2 = v2 / np.linalg.norm(v2)
                    
                    # Вычисляем угол между векторами
                    cos_angle = np.clip(np.dot(v1, v2), -1.0, 1.0)
                    angle_rad = np.arccos(cos_angle)
                    angle_deg = np.degrees(angle_rad)
                    
                    # Определяем, является ли угол внутренним или внешним
                    # Используем векторное произведение для определения знака поворота
                    cross_prod = np.cross(np.append(v1, 0), np.append(v2, 0))[2]
                    if cross_prod < 0:
                        angle_deg = 360 - angle_deg
                    
                    # Получаем значение угла для отображения
                    if angle_values and i < len(angle_values) and angle_values[i] is not None:
                        # Проверяем, не является ли значение строкой "-", что означает "не отображать"
                        if angle_values[i] == "-":
                            continue
                        angle_text = f"{angle_values[i]}°"
                    else:
                        angle_text = f"{angle_deg:.1f}°"
                    
                    # Отображаем дугу угла, если нужно
                    if show_angle_arcs:
                        # Вычисляем начальный угол для дуги (угол первого вектора)
                        start_angle = np.degrees(np.arctan2(v1[1], v1[0])) % 360
                        
                        # Вычисляем угол дуги (разница между углами векторов)
                        arc_angle = angle_deg if cross_prod >= 0 else -angle_deg
                        
                        # Создаем и добавляем дугу
                        arc = plt.matplotlib.patches.Arc(
                            B, radius * 2, radius * 2,  # центр и размеры
                            theta1=start_angle, theta2=start_angle + arc_angle,  # углы дуги
                            color='red', linewidth=1.5, fill=False, zorder=1
                        )
                        ax.add_patch(arc)
                    
                    # Определяем положение для текста угла
                    # Средний вектор между v1 и v2
                    v_mid = (v1 + v2) / 2
                    v_mid = v_mid / np.linalg.norm(v_mid) if np.linalg.norm(v_mid) > 1e-10 else np.array([1, 0])
                    
                    # Положение текста угла
                    text_pos = B + v_mid * radius * 1.5
                    
                    # Отображаем значение угла с фоном для лучшей видимости
                    ax.text(text_pos[0], text_pos[1], angle_text, 
                            ha='center', va='center', fontsize=10, color='red',
                            bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', 
                                      boxstyle="round,pad=0.2"))
    
    def save(self, filename=None):
        """
        Сохраняет изображение фигуры в файл.
        
        Args:
            filename (str): Имя файла для сохранения. Если None, создается случайное имя.
            
        Returns:
            str: Путь к сохраненному файлу или None, если не удалось сохранить
        """
        try:
            # Создаем директорию для сохранения, если её нет
            output_dir = os.path.join("static", "images", "generated")
            os.makedirs(output_dir, exist_ok=True)
            
            # Генерируем имя файла, если не указано
            if filename is None:
                figure_type = self.figure_type or self.__class__.__name__.lower()
                filename = f"{figure_type}_{uuid.uuid4().hex[:8]}.png"
                
            # Полный путь к файлу
            output_path = os.path.join(output_dir, filename)
            
            # Создаем оси и рисуем фигуру
            fig, ax = plt.subplots(figsize=(12, 12))
            ax.set_aspect('equal')
            self.draw(ax)
            
            # Сохраняем изображение с высоким разрешением
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)  # Закрываем фигуру для освобождения памяти
            
            return output_path
        except Exception as e:
            logging.error(f"Ошибка при сохранении изображения: {e}")
            return None 