import os
import logging
import sys

# Настраиваем логирование
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s',
                   handlers=[logging.StreamHandler(sys.stdout)])

# Добавляем корневую директорию проекта в путь для импорта
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Импортируем модули визуализации
from app.visualization.processors import process_visualization_params

def test_debug_response():
    """Тестирует визуализацию функции из debug_response.txt"""
    try:
        # Загружаем параметры из файла
        file_path = 'debug_files/debug_response.txt'
        with open(file_path, 'r', encoding='utf-8') as f:
            debug_params = f.read()
        
        # Извлекаем секцию параметров
        params_start = debug_params.find('---ПАРАМЕТРЫ ДЛЯ ВИЗУАЛИЗАЦИИ---')
        if params_start == -1:
            logging.error("Секция параметров для визуализации не найдена!")
            return
        
        params_text = debug_params[params_start:]
        logging.info(f"Параметры визуализации:\n{params_text}")
        
        # Обрабатываем параметры и создаем визуализацию
        image_path, image_type = process_visualization_params(params_text)
        
        # Выводим результат
        if image_path and os.path.exists(image_path):
            logging.info(f"Визуализация успешно создана: {image_path}")
            logging.info(f"Тип изображения: {image_type}")
            
            # Показываем изображение (опционально)
            try:
                import matplotlib.pyplot as plt
                import matplotlib.image as mpimg
                
                # Загружаем и отображаем изображение
                img = mpimg.imread(image_path)
                plt.figure(figsize=(10, 8))
                plt.imshow(img)
                plt.axis('off')
                plt.title("Результат визуализации")
                plt.show()
            except Exception as e:
                logging.warning(f"Не удалось отобразить изображение: {e}")
        else:
            logging.error(f"Не удалось создать визуализацию. Путь: {image_path}, Тип: {image_type}")
    
    except Exception as e:
        logging.error(f"Ошибка при тестировании: {e}")
        import traceback
        logging.error(traceback.format_exc())

if __name__ == "__main__":
    test_debug_response() 