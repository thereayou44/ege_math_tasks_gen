#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Запускает все тесты из пакета tests.
"""

import unittest
import sys
import os
import argparse

# Добавляем родительскую директорию в пути для импорта
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Импортируем все тесты с правильными путями
from tests.test_task_generator import TestTaskGenerator
from tests.test_visualization import TestVisualization
from tests.test_parsers import TestParsers
from tests.test_format_converters import TestFormatConverters
from tests.test_mock import TestTaskGeneratorWithMocks

def run_unit_tests():
    """Запускает стандартные модульные тесты"""
    # Создаем набор тестов
    test_suite = unittest.TestSuite()
    
    # Добавляем классы тестов
    test_suite.addTest(unittest.makeSuite(TestTaskGenerator))
    test_suite.addTest(unittest.makeSuite(TestVisualization))
    test_suite.addTest(unittest.makeSuite(TestParsers))
    test_suite.addTest(unittest.makeSuite(TestFormatConverters))
    test_suite.addTest(unittest.makeSuite(TestTaskGeneratorWithMocks))
    
    # Запускаем все тесты
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()

def run_quality_tests():
    """Запускает тесты качества генерации"""
    try:
        # Импортируем функцию запуска тестов качества
        from tests.quality_tests.run_quality_tests import run_all_quality_tests
        
        # Запускаем тесты качества
        run_all_quality_tests()
        return True
    except Exception as e:
        print(f"Ошибка при запуске тестов качества: {e}")
        return False

if __name__ == "__main__":
    # Настраиваем аргументы командной строки
    parser = argparse.ArgumentParser(description='Запуск тестов для генератора задач ЕГЭ')
    parser.add_argument('--quality', action='store_true', 
                        help='Запустить тесты качества генерации уравнений')
    parser.add_argument('--all', action='store_true', 
                        help='Запустить все тесты, включая модульные и тесты качества')
    
    args = parser.parse_args()
    
    success = True
    
    # Решаем, какие тесты запускать
    if args.quality:
        # Запускаем только тесты качества
        success = run_quality_tests()
    elif args.all:
        # Запускаем и модульные тесты, и тесты качества
        unit_success = run_unit_tests()
        quality_success = run_quality_tests()
        success = unit_success and quality_success
    else:
        # По умолчанию запускаем только модульные тесты
        success = run_unit_tests()
    
    # Выходим с ненулевым кодом, если есть ошибки или сбои
    sys.exit(not success) 