import os
import uuid
import logging
import traceback

from app.visualization.parsers.graph_parsers import parse_graph_params, remove_latex_markup
from app.visualization.parsers.coordinate_parsers import process_coordinate_params
from app.visualization.graphs.function_graphs import generate_multi_function_graph
from app.visualization.coordinates.coordinate_system import generate_coordinate_system

def process_visualization_params(params_text):
    """
    Обрабатывает параметры для визуализации из текста.
    
    Args:
        params_text: Текст с параметрами визуализации
        
    Returns:
        tuple: (изображение, тип визуализации) или (None, None) в случае ошибки
    """
    try:
        if not params_text:
            return None, None
        
        import re
        lines = params_text.strip().split('\n')
        viz_type = None
        
        # Ищем тип визуализации
        for line in lines:
            if line.lower().startswith('тип:'):
                viz_type = line.split(':', 1)[1].strip().lower()
                break
            if line.lower().startswith('тип фигуры:'):  
                viz_type = line.split(':', 1)[1].strip().lower()
                break
            if re.search(r'фигура\s*:', line.lower()):
                viz_type = re.split(r'фигура\s*:', line.lower(), 1)[1].strip()
                break
        
        if not viz_type:
            logging.warning("Не удалось определить тип визуализации")
            return None, None
        
        # Функция для извлечения параметра по шаблону
        def extract_param(pattern, text, default=None):
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                try:
                    value = match.group(1).strip()
                    # Очищаем от LaTeX разметки
                    value = remove_latex_markup(value)
                    return value
                except Exception as e:
                    logging.error(f"Ошибка при извлечении параметра: {e}")
            return default
        
        if "график" in viz_type:
            try:
                # Используем общую функцию parse_graph_params для парсинга параметров
                funcs_to_plot, x_range, y_range, special_points = parse_graph_params(params_text)
                
                # Создаем директорию для сохранения изображения, если она не существует
                output_dir = os.path.join("static", "images", "generated")
                os.makedirs(output_dir, exist_ok=True)
                
                # Генерируем имя файла с уникальным идентификатором
                filename = f'multifunction_{uuid.uuid4().hex[:8]}.png'
                output_path = os.path.join(output_dir, filename)
                
                # Логируем для отладки
                x_min, x_max = x_range
                y_min, y_max = (None, None) if y_range is None else y_range
                logging.info(f"Обработка функций с параметрами: x: [{x_min}, {x_max}], " +
                             f"y: [{y_min if y_min is not None else 'auto'}, {y_max if y_max is not None else 'auto'}]")
                
                # Отрисовываем график
                filepath = generate_multi_function_graph(funcs_to_plot, x_range=x_range, y_range=y_range, special_points=special_points, filename=filename)
                return filepath, "graph"
            except Exception as e:
                logging.error(f"Ошибка при создании графика функций: {e}")
                logging.error(traceback.format_exc())
                return None, None
                
        elif "треугольник" in viz_type or "прямоугольник" in viz_type or "параллелограмм" in viz_type or "трапеция" in viz_type or "окружность" in viz_type:
            # Для этих типов визуализаций используем классы геометрии и GeometryRenderer
            from app.visualization.renderer import GeometryRenderer
            figure_type = None
            
            if "треугольник" in viz_type:
                figure_type = "triangle"
            elif "прямоугольник" in viz_type:
                figure_type = "rectangle"
            elif "параллелограмм" in viz_type:
                figure_type = "parallelogram"
            elif "трапеция" in viz_type:
                figure_type = "trapezoid"
            elif "окружность" in viz_type:
                figure_type = "circle"
                
            # Отрисовываем геометрическую фигуру
            image_path = GeometryRenderer.render_from_text(params_text, figure_type)
            return image_path, figure_type
            
        elif "координатная плоскость" in viz_type:
            return process_coordinate_visualization(params_text), "coordinate"
        else:
            logging.warning(f"Неизвестный тип визуализации: {viz_type}")
            return None, None
            
    except Exception as e:
        logging.error(f"Ошибка при создании графика функции: {e}")
        logging.error(traceback.format_exc())
        return None, None

def process_coordinate_visualization(params_text):
    """Обрабатывает параметры для координатной плоскости и создает визуализацию"""
    # Извлекаем параметры для координатной плоскости
    points, functions, vectors = process_coordinate_params(params_text)
    
    # Создаем директорию для сохранения изображения, если она не существует
    output_dir = os.path.join("static", "images", "generated")
    os.makedirs(output_dir, exist_ok=True)
    
    # Генерируем имя файла с уникальным идентификатором
    filename = f'coordinate_{uuid.uuid4().hex[:8]}.png'
    output_path = os.path.join(output_dir, filename)
    
    # Генерируем координатную плоскость
    return generate_coordinate_system(points, functions, vectors, filename=output_path) 