#!/usr/bin/env python3
import time
import random
import statistics
import argparse
import json
from app.task_generator import generate_complete_task
from app.json_api_helpers import generate_json_task

def load_categories():
    """Загружает категории заданий из файла"""
    import json
    try:
        with open('data/categories/categories_list.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        # Если файл не найден, возвращаем минимальный список
        return ["Неравенства", "Уравнения", "Текстовые задачи"]

def get_random_category_subcategory(categories):
    """Выбирает случайную категорию и подкатегорию из списка"""
    if isinstance(categories, list):
        # Выбираем случайную категорию из списка
        category_data = random.choice(categories)
        
        # Извлекаем имя категории
        if isinstance(category_data, dict) and 'category' in category_data:
            category_name = category_data['category']
            
            # Извлекаем список подкатегорий, если они есть
            if 'subcategories' in category_data and isinstance(category_data['subcategories'], list):
                subcategories = category_data['subcategories']
                if subcategories:
                    subcategory_data = random.choice(subcategories)
                    if isinstance(subcategory_data, dict) and 'name' in subcategory_data:
                        return category_name, subcategory_data['name']
            
            return category_name, ""
    
    # Если структура не соответствует ожидаемой или categories не список
    return "Неравенства", ""

def benchmark_task_generation(num_tasks=5, use_json_api=False):
    """
    Измеряет среднее время генерации задачи
    
    Args:
        num_tasks: Количество задач для тестирования
        use_json_api: Использовать ли JSON API или обычную генерацию
    
    Returns:
        dict: Результаты тестирования
    """
    categories = load_categories()
    generation_times = []
    successful = 0
    failed = 0
    results = []
    
    print(f"Начинаю измерение времени генерации {num_tasks} задач...")
    
    for i in range(num_tasks):
        category, subcategory = get_random_category_subcategory(categories)
        difficulty_level = random.randint(1, 5)
        is_basic_level = random.choice([True, False])
        
        print(f"\nЗадача {i+1}/{num_tasks}:")
        print(f"Категория: {category}")
        print(f"Подкатегория: {subcategory}")
        print(f"Уровень сложности: {difficulty_level}")
        print(f"Базовый уровень: {'Да' if is_basic_level else 'Нет'}")
        
        start_time = time.time()
        
        try:
            if use_json_api:
                result = generate_json_task(category, subcategory, difficulty_level, is_basic_level)
            else:
                result = generate_complete_task(category, subcategory, difficulty_level, is_basic_level)
                
            end_time = time.time()
            generation_time = end_time - start_time
            
            if "error" in result:
                print(f"Ошибка: {result['error']}")
                failed += 1
            else:
                print(f"Успешно сгенерирована за {generation_time:.2f} секунд")
                generation_times.append(generation_time)
                successful += 1
                
            # Сохраняем детали каждой генерации
            results.append({
                "category": category,
                "subcategory": subcategory,
                "difficulty_level": difficulty_level,
                "is_basic_level": is_basic_level,
                "time": generation_time,
                "status": "success" if "error" not in result else "error"
            })
            
        except Exception as e:
            end_time = time.time()
            generation_time = end_time - start_time
            print(f"Исключение при генерации: {str(e)}")
            failed += 1
            
            results.append({
                "category": category,
                "subcategory": subcategory,
                "difficulty_level": difficulty_level,
                "is_basic_level": is_basic_level,
                "time": generation_time,
                "status": "exception",
                "error": str(e)
            })
    
    # Анализ результатов
    if generation_times:
        avg_time = sum(generation_times) / len(generation_times)
        median_time = statistics.median(generation_times)
        min_time = min(generation_times)
        max_time = max(generation_times)
        std_dev = statistics.stdev(generation_times) if len(generation_times) > 1 else 0
    else:
        avg_time = median_time = min_time = max_time = std_dev = 0
    
    # Формируем итоговую статистику
    stats = {
        "total_tasks": num_tasks,
        "successful": successful,
        "failed": failed,
        "success_rate": (successful / num_tasks) * 100 if num_tasks > 0 else 0,
        "average_time": avg_time,
        "median_time": median_time,
        "min_time": min_time,
        "max_time": max_time,
        "std_deviation": std_dev,
        "detailed_results": results
    }
    
    return stats

def print_stats(stats):
    """Печатает статистику тестирования"""
    print("\n" + "="*50)
    print("РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ ВРЕМЕНИ ГЕНЕРАЦИИ ЗАДАЧ")
    print("="*50)
    print(f"Всего задач: {stats['total_tasks']}")
    print(f"Успешно сгенерировано: {stats['successful']}")
    print(f"Ошибок: {stats['failed']}")
    print(f"Процент успеха: {stats['success_rate']:.1f}%")
    print("\nВРЕМЯ ГЕНЕРАЦИИ:")
    print(f"Среднее время: {stats['average_time']:.2f} секунд")
    print(f"Медианное время: {stats['median_time']:.2f} секунд")
    print(f"Минимальное время: {stats['min_time']:.2f} секунд")
    print(f"Максимальное время: {stats['max_time']:.2f} секунд")
    print(f"Стандартное отклонение: {stats['std_deviation']:.2f} секунд")
    print("="*50)
    
    # Дополнительный анализ по категориям если есть успешные генерации
    if stats['successful'] > 0:
        category_times = {}
        
        for result in stats['detailed_results']:
            if result['status'] == 'success':
                category = result['category']
                if category not in category_times:
                    category_times[category] = []
                category_times[category].append(result['time'])
        
        if category_times:
            print("\nВРЕМЯ ПО КАТЕГОРИЯМ:")
            for category, times in category_times.items():
                avg_time = sum(times) / len(times)
                print(f"{category}: {avg_time:.2f} сек (в среднем по {len(times)} задачам)")

def save_results_to_file(stats, filename="benchmark_results.txt"):
    """Сохраняет результаты в текстовый файл"""
    with open(filename, "w", encoding="utf-8") as f:
        f.write("РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ ВРЕМЕНИ ГЕНЕРАЦИИ ЗАДАЧ\n")
        f.write("="*50 + "\n")
        f.write(f"Всего задач: {stats['total_tasks']}\n")
        f.write(f"Успешно сгенерировано: {stats['successful']}\n")
        f.write(f"Ошибок: {stats['failed']}\n")
        f.write(f"Процент успеха: {stats['success_rate']:.1f}%\n\n")
        f.write("ВРЕМЯ ГЕНЕРАЦИИ:\n")
        f.write(f"Среднее время: {stats['average_time']:.2f} секунд\n")
        f.write(f"Медианное время: {stats['median_time']:.2f} секунд\n")
        f.write(f"Минимальное время: {stats['min_time']:.2f} секунд\n")
        f.write(f"Максимальное время: {stats['max_time']:.2f} секунд\n")
        f.write(f"Стандартное отклонение: {stats['std_deviation']:.2f} секунд\n")
        f.write("="*50 + "\n\n")
        
        # Детальные результаты каждой задачи
        f.write("ДЕТАЛЬНЫЕ РЕЗУЛЬТАТЫ:\n")
        for i, result in enumerate(stats['detailed_results']):
            f.write(f"Задача {i+1}:\n")
            f.write(f"  Категория: {result['category']}\n")
            f.write(f"  Подкатегория: {result['subcategory']}\n")
            f.write(f"  Сложность: {result['difficulty_level']}\n")
            f.write(f"  Базовый уровень: {'Да' if result['is_basic_level'] else 'Нет'}\n")
            f.write(f"  Время: {result['time']:.2f} секунд\n")
            f.write(f"  Статус: {result['status']}\n")
            if result['status'] != 'success' and 'error' in result:
                f.write(f"  Ошибка: {result['error']}\n")
            f.write("\n")
    
    print(f"Результаты сохранены в файл: {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Тестирование времени генерации задач")
    parser.add_argument("-n", "--num-tasks", type=int, default=5, 
                        help="Количество задач для тестирования (по умолчанию: 5)")
    parser.add_argument("--json", action="store_true",
                        help="Использовать JSON API для генерации задач")
    parser.add_argument("-o", "--output", type=str, default="benchmark_results.txt",
                        help="Файл для сохранения результатов (по умолчанию: benchmark_results.txt)")
    
    args = parser.parse_args()
    
    # Запускаем тестирование
    stats = benchmark_task_generation(args.num_tasks, args.json)
    
    # Выводим результаты
    print_stats(stats)
    
    # Сохраняем результаты в файл
    save_results_to_file(stats, args.output) 