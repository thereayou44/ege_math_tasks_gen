#!/usr/bin/env python3
"""
Скрипт для запуска всех тестов качества генерации задач ЕГЭ по математике.
Проверяет качество генерации уравнений.

"""

import sys
import os
import unittest
import logging
from dotenv import load_dotenv

# Добавляем родительский каталог в sys.path для импорта модулей
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Загружаем переменные окружения из .env файла
load_dotenv()

# Настраиваем логирование
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("quality_tests.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

def run_all_quality_tests():
    """
    Запускает все тесты качества и выводит общую статистику.
    """
    from tests.quality_tests.test_equations import EquationQualityTest
    
    # Создаем экземпляры тестов
    equation_test = EquationQualityTest()
    
    # Выводим заголовок
    print("\n" + "="*80)
    print("ЗАПУСК ТЕСТОВ КАЧЕСТВА ГЕНЕРАТОРА МАТЕМАТИЧЕСКИХ ЗАДАЧ ЕГЭ")
    print("Версия: 1.3.0 (только тесты уравнений)")
    print("="*80)
    
    # Запускаем тесты уравнений
    print("\n" + "="*70)
    print("ТЕСТИРОВАНИЕ КАЧЕСТВА ГЕНЕРАЦИИ УРАВНЕНИЙ")
    print("="*70)
    print("Улучшенный парсер математических выражений:")
    print("- Корректная обработка отрицательных чисел и отрицательных дробей")
    print("- Поддержка степеней в LaTeX формате a^{bx+c}")
    print("- Обработка тригонометрических выражений с pi")
    print("- Исправление несбалансированных скобок")
    print("- Поддержка логарифмов с коэффициентами (например, 2log_3(2))")
    print("- Преобразование вложенных математических конструкций")
    print("- Обработка сложных дробей с корнями (например, \\frac{1 + \\sqrt{133}}{2})")
    print("- Корректная обработка десятичных чисел с запятой (0,5 -> 0.5)")
    print("- Улучшенное извлечение корней из ответов")
    print("\nПроверка ответов:")
    print("- Гибкая проверка наличия хотя бы одного корректного ответа")
    print("- Сравнение с решениями SymPy с учетом погрешности")
    
    try:
        equation_test.test_equation_generation_quality()
    except Exception as e:
        logging.error(f"Ошибка при тестировании уравнений: {e}")
    
    print("\n" + "="*70)
    print("СТАТИСТИКА ТЕСТИРОВАНИЯ КАЧЕСТВА")
    print("="*70)
    try:
        # Собираем статистику из выполненных тестов
        equation_stats = equation_test.get_test_statistics() if hasattr(equation_test, 'get_test_statistics') else {}
        
        # Выводим общую статистику
        print(f"Уравнения: проверено задач - {equation_stats.get('total_tasks', 'N/A')}")
        print(f"Уравнения: успешно распознаны - {equation_stats.get('successful_parsings', 'N/A')}")
        print(f"Уравнения: корректно верифицированы - {equation_stats.get('correct_verifications', 'N/A')}")
        
        # Вычисляем и выводим процент успеха
        if equation_stats.get('successful_parsings', 0) > 0:
            equation_success_rate = equation_stats.get('correct_verifications', 0) / equation_stats.get('successful_parsings', 1) * 100
            print(f"\nПроцент успеха для уравнений: {equation_success_rate:.2f}%")
            
    except Exception as e:
        logging.error(f"Ошибка при выводе статистики: {e}")
    
    print("\n" + "="*70)
    print("ТЕСТИРОВАНИЕ КАЧЕСТВА ГЕНЕРАЦИИ ЗАВЕРШЕНО")
    print("="*70)

if __name__ == "__main__":
    # Проверяем наличие API ключа Яндекса
    if not os.getenv("YANDEX_API_KEY") or not os.getenv("YANDEX_FOLDER_ID"):
        print("ВНИМАНИЕ: Не указаны YANDEX_API_KEY или YANDEX_FOLDER_ID в файле .env!")
        print("Генерация задач не будет работать.")
        sys.exit(1)
    
    run_all_quality_tests() 