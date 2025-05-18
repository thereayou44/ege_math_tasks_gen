import base64
import logging

def get_image_base64(image_path):
    """
    Преобразует изображение в строку base64 для встраивания в HTML.
    
    Args:
        image_path: Путь к изображению
        
    Returns:
        str: Строка в формате base64
    """
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except Exception as e:
        logging.error(f"Ошибка при конвертации изображения в base64: {e}")
        return None 