import os
import re
import logging
import traceback
from app.task_generator import generate_complete_task, extract_visualization_params, create_image_from_params

def _process_images(result, category, subcategory):
    """
    Обрабатывает изображения для задачи
    
    Args:
        result: Результат генерации задачи
        category: Категория задачи
        subcategory: Подкатегория задачи
        
    Returns:
        list: Список изображений для задачи
    """
    task_images = []
    
    # Если есть изображение в результате, добавляем его
    if "image_path" in result:
        image_path = result["image_path"]
        image_filename = os.path.basename(image_path)
        image_url = f"/static/images/generated/{image_filename}"
        
        # Добавляем изображение к задаче
        task_images.append({
            "url": image_url,
            "alt": "Изображение к задаче"
        })
    elif "image_url" in result:
        # Если есть прямая ссылка на изображение
        image_url = result["image_url"]
        task_images.append({
            "url": image_url,
            "alt": "Изображение к задаче"
        })
    else:
        # Проверяем, нужно ли генерировать изображение
        task = result.get("task", "")
        viz_params = extract_visualization_params(task, category, subcategory)
        if viz_params and viz_params.get("type") != "none":
            try:
                image_path = create_image_from_params(viz_params)
                if image_path:
                    image_filename = os.path.basename(image_path)
                    image_url = f"/static/images/generated/{image_filename}"
                    task_images.append({
                        "url": image_url,
                        "alt": "Изображение к задаче"
                    })
            except Exception as viz_error:
                logging.error(f"Ошибка при создании изображения: {viz_error}")
    
    return task_images

def _extract_answer(solution):
    """
    Извлекает ответ из решения
    
    Args:
        solution: Текст решения
        
    Returns:
        str: Извлеченный ответ
    """
    # Если ответ не был успешно извлечен, пробуем найти его снова
    answer_match = re.search(r"(?:Ответ|ОТВЕТ|ответ)\s*:(.+?)(?=$|\.|\.\s*$|\n\s*\n|\<\/p\>)", solution, re.IGNORECASE | re.DOTALL)
    if answer_match:
        return answer_match.group(1).strip()
    return ""

def generate_markdown_task(category, subcategory="", difficulty_level=3, is_basic_level=False):
    """
    Генерирует задачу в формате Markdown для API запроса
    
    Args:
        category: Категория задачи
        subcategory: Подкатегория задачи (опционально)
        difficulty_level: Уровень сложности подсказок (1-5)
        is_basic_level: Выбор между базовым и профильным уровнем ЕГЭ
        
    Returns:
        dict: Словарь с задачей и подсказками в Markdown
    """
    try:
        # Используем существующую функцию для генерации задачи с markdown форматом
        result = generate_complete_task(category, subcategory, difficulty_level, is_basic_level, is_markdown=True)
        
        # Проверяем на ошибки
        if "error" in result:
            return result
        
        # Берем текст задачи из результата
        task = result.get("task", "")
        solution = result.get("solution", "")
        hints = result.get("hints", [])
        
        # Формируем результат
        markdown_result = {
            "problem": task,
            "problem_picture": "",
            "solution": solution,
            "hint1": hints[0] if len(hints) > 0 else "",
            "hint2": hints[1] if len(hints) > 1 else "",
            "hint3": hints[2] if len(hints) > 2 else "",
            "difficulty_level": result.get("difficulty_level", difficulty_level),
            "is_basic_level": is_basic_level
        }
        
        # Если есть изображение, добавляем его URL
        if "image_url" in result:
            markdown_result["problem_picture"] = result["image_url"]
        elif "image_path" in result:
            image_path = result["image_path"]
            image_filename = os.path.basename(image_path)
            image_url = f"/static/images/generated/{image_filename}"
            markdown_result["problem_picture"] = image_url
        else:
            # Проверяем, нужно ли генерировать изображение
            viz_params = extract_visualization_params(task, category, subcategory)
            if viz_params and viz_params.get("type") != "none":
                try:
                    image_path = create_image_from_params(viz_params)
                    if image_path:
                        image_filename = os.path.basename(image_path)
                        image_url = f"/static/images/generated/{image_filename}"
                        markdown_result["problem_picture"] = image_url
                except Exception as viz_error:
                    logging.error(f"Ошибка при создании изображения: {viz_error}")
        
        return markdown_result
    except Exception as e:
        logging.error(f"Ошибка при генерации задачи в формате Markdown: {e}")
        logging.error(traceback.format_exc())
        return {"error": f"Ошибка при генерации задачи в формате Markdown: {str(e)}"}

def generate_json_task(category, subcategory="", difficulty_level=3, is_basic_level=False):
    """
    Генерирует задачу в формате JSON для API запроса
    
    Args:
        category: Категория задачи
        subcategory: Подкатегория задачи (опционально)
        difficulty_level: Уровень сложности подсказок (1-5)
        is_basic_level: Выбор между базовым и профильным уровнем ЕГЭ
        
    Returns:
        dict: Словарь с задачей в формате JSON
    """
    try:
        # Используем существующую функцию для генерации задачи
        result = generate_complete_task(category, subcategory, difficulty_level, is_basic_level)
        
        # Проверяем на ошибки
        if "error" in result:
            return result
        
        # Извлекаем ответ из решения для отдельного поля
        solution = result.get("solution", "")
        answer = result.get("answer", "")
        
        # Если ответ не был успешно извлечен, пробуем найти его снова
        if not answer or answer == "См. решение":
            answer = _extract_answer(solution)
        
        # Обрабатываем изображения
        task_images = _process_images(result, category, subcategory)
        
        # Формируем JSON-результат
        json_result = {
            "task": {
                "text": result.get("task", ""),
                "images": task_images
            },
            "solution": {
                "text": solution,
                "images": [] # Не добавляем изображения к решению
            },
            "answer": answer,
            "hints": result.get("hints", []),
            "difficulty_level": result.get("difficulty_level", difficulty_level),
            "category": category,
            "subcategory": subcategory,
            "is_basic_level": is_basic_level,
            "format": "html"
        }
        
        return json_result
    except Exception as e:
        logging.error(f"Ошибка при генерации задачи в формате JSON: {e}")
        logging.error(traceback.format_exc())
        return {"error": f"Ошибка при генерации задачи в формате JSON: {str(e)}"}

def generate_json_markdown_task(category, subcategory="", difficulty_level=3, is_basic_level=False):
    """
    Генерирует задачу в формате JSON с Markdown для API запроса
    
    Args:
        category: Категория задачи
        subcategory: Подкатегория задачи (опционально)
        difficulty_level: Уровень сложности подсказок (1-5)
        is_basic_level: Выбор между базовым и профильным уровнем ЕГЭ
        
    Returns:
        dict: Словарь с задачей в формате JSON с Markdown
    """
    try:
        # Используем существующую функцию для генерации задачи с параметром is_markdown=True
        result = generate_complete_task(category, subcategory, difficulty_level, is_basic_level, is_markdown=True)
        
        # Проверяем на ошибки
        if "error" in result:
            return result
        
        # Извлекаем ответ из решения для отдельного поля
        task_text = result.get("task", "")
        solution_text = result.get("solution", "")
        answer = result.get("answer", "")
        
        # Если ответ не был успешно извлечен, пробуем найти его снова
        if not answer or answer == "См. решение":
            answer = _extract_answer(solution_text)
        
        # Обрабатываем изображения
        task_images = _process_images(result, category, subcategory)
        
        # Формируем JSON-результат
        json_result = {
            "task": {
                "text": task_text,
                "images": task_images
            },
            "solution": {
                "text": solution_text,
                "images": [] # Не добавляем изображения к решению
            },
            "answer": answer,
            "hints": result.get("hints", []),
            "difficulty_level": result.get("difficulty_level", difficulty_level),
            "category": category,
            "subcategory": subcategory,
            "is_basic_level": is_basic_level,
            "format": "markdown"  # Указываем формат данных
        }
        
        return json_result
    except Exception as e:
        logging.error(f"Ошибка при генерации задачи в формате JSON с Markdown: {e}")
        logging.error(traceback.format_exc())
        return {"error": f"Ошибка при генерации задачи в формате JSON с Markdown: {str(e)}"} 