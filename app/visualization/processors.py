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
            logging.warning("Пустой текст параметров для визуализации")
            return None, None
        
        import re
        lines = params_text.strip().split('\n')
        viz_type = None
        
        logging.info(f"Обработка параметров визуализации. Количество строк: {len(lines)}")
        logging.info(f"Начало параметров: {lines[0:min(5, len(lines))]}")
        
        # Ищем тип визуализации
        for line in lines:
            if line.lower().startswith('тип:'):
                viz_type = line.split(':', 1)[1].strip().lower()
                logging.info(f"Найден тип визуализации (строка 'тип:'): {viz_type}")
                break
            if line.lower().startswith('тип фигуры:'):  
                viz_type = line.split(':', 1)[1].strip().lower()
                logging.info(f"Найден тип визуализации (строка 'тип фигуры:'): {viz_type}")
                break
            if re.search(r'фигура\s*:', line.lower()):
                viz_type = re.split(r'фигура\s*:', line.lower(), 1)[1].strip()
                logging.info(f"Найден тип визуализации (строка 'фигура:'): {viz_type}")
                break
            # Проверяем наличие строки "Визуализация не требуется"
            if re.search(r'визуализация\s+не\s+требуется', line.lower()):
                logging.info("Найдена строка 'Визуализация не требуется'")
                return None, "not_required"
        
        if not viz_type:
            # Если в первых строках не найден тип, ищем дальше в тексте
            visualization_not_required = re.search(r'визуализация\s+не\s+требуется', params_text.lower())
            if visualization_not_required:
                logging.info("Найдена строка 'Визуализация не требуется' в тексте параметров")
                return None, "not_required"
                
            logging.warning("Не удалось определить тип визуализации")
            logging.warning(f"Содержимое параметров: {params_text[:200]}...")
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
                logging.info("Обработка параметров для графика функции")
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
                logging.info(f"Функции для построения: {funcs_to_plot}")
                
                # Отрисовываем график
                filepath = generate_multi_function_graph(funcs_to_plot, x_range=x_range, y_range=y_range, special_points=special_points, filename=filename)
                logging.info(f"Создан файл графика: {filepath}")
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
                logging.info("Обработка параметров для треугольника")
            elif "прямоугольник" in viz_type:
                figure_type = "rectangle"
                logging.info("Обработка параметров для прямоугольника")
            elif "параллелограмм" in viz_type:
                figure_type = "parallelogram"
                logging.info("Обработка параметров для параллелограмма")
            elif "трапеция" in viz_type:
                figure_type = "trapezoid"
                logging.info("Обработка параметров для трапеции")
            elif "окружность" in viz_type:
                figure_type = "circle"
                logging.info("Обработка параметров для окружности")
                
            # Отрисовываем геометрическую фигуру
            image_path = GeometryRenderer.render_from_text(params_text, figure_type)
            if image_path:
                logging.info(f"Создан файл геометрической фигуры: {image_path}")
            else:
                logging.error(f"Не удалось создать изображение для {figure_type}")
                logging.error(f"Параметры: {params_text[:200]}...")
            return image_path, figure_type
            
        elif "координатная плоскость" in viz_type:
            logging.info("Обработка параметров для координатной плоскости")
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