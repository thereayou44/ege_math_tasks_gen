#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Главный скрипт для запуска всех тестов проекта генератора задач ЕГЭ.
Запускает функциональные тесты, тесты визуализации и тесты качества.
"""

import os
import sys
import unittest
import logging
from datetime import datetime

# Добавляем корневую директорию проекта в sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Настраиваем логирование
log_filename = os.path.join(project_root, 'test_results.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def run_functional_tests():
    """Запускает функциональные тесты модулей."""
    logger.info("Запуск функциональных тестов")
    
    test_modules = [
        'tests.test_task_generator',
        'tests.test_parsers',
        'tests.test_format_converters'
    ]
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    for module_name in test_modules:
        try:
            module = __import__(module_name, fromlist=[''])
            suite.addTests(loader.loadTestsFromModule(module))
        except ImportError as e:
            logger.error(f"Не удалось импортировать модуль {module_name}: {e}")
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result

def run_visualization_tests():
    """Запускает тесты визуализации."""
    logger.info("Запуск тестов визуализации")
    
    test_modules = [
        'tests.test_visualization',
        'tests.test_shapes',
        'tests.test_graph_visualization',
        'tests.test_all_figures_visual'
    ]
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    for module_name in test_modules:
        try:
            module = __import__(module_name, fromlist=[''])
            suite.addTests(loader.loadTestsFromModule(module))
        except ImportError as e:
            logger.error(f"Не удалось импортировать модуль {module_name}: {e}")
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result

def run_mock_tests():
    """Запускает тесты с моками."""
    logger.info("Запуск тестов с моками")
    
    try:
        from tests import test_mock
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromModule(test_mock)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        return result
    except ImportError as e:
        logger.error(f"Не удалось импортировать модуль test_mock: {e}")
        return None

def run_quality_tests():
    """Запускает тесты качества генерации задач."""
    logger.info("Запуск тестов качества генерации задач")
    
    try:
        # Здесь запускаем только тесты, которые не требуют API вызовов
        from tests.test_real_tasks import RealTasksTest
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromTestCase(RealTasksTest)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        return result
    except ImportError as e:
        logger.error(f"Не удалось импортировать модуль test_real_tasks: {e}")
        return None

def main():
    """Основная функция для запуска всех тестов."""
    start_time = datetime.now()
    logger.info(f"Начало тестирования: {start_time}")
    
    # Запускаем все типы тестов
    functional_result = run_functional_tests()
    visualization_result = run_visualization_tests()
    mock_result = run_mock_tests()
    quality_result = run_quality_tests()
    
    # Выводим сводную информацию
    end_time = datetime.now()
    duration = end_time - start_time
    
    logger.info(f"Окончание тестирования: {end_time}")
    logger.info(f"Продолжительность: {duration}")
    
    # Выводим общую статистику
    if functional_result and visualization_result:
        total_tests = (functional_result.testsRun + visualization_result.testsRun + 
                      (mock_result.testsRun if mock_result else 0) + 
                      (quality_result.testsRun if quality_result else 0))
                      
        total_errors = (len(functional_result.errors) + len(visualization_result.errors) + 
                       (len(mock_result.errors) if mock_result else 0) + 
                       (len(quality_result.errors) if quality_result else 0))
                       
        total_failures = (len(functional_result.failures) + len(visualization_result.failures) + 
                         (len(mock_result.failures) if mock_result else 0) + 
                         (len(quality_result.failures) if quality_result else 0))
        
        logger.info(f"Всего тестов: {total_tests}")
        logger.info(f"Ошибок: {total_errors}")
        logger.info(f"Неудач: {total_failures}")
        logger.info(f"Успешно: {total_tests - total_errors - total_failures}")
        
        success_rate = ((total_tests - total_errors - total_failures) / total_tests) * 100 if total_tests > 0 else 0
        logger.info(f"Процент успеха: {success_rate:.2f}%")
        
        # Определяем код возврата для CI/CD
        if total_errors == 0 and total_failures == 0:
            logger.info("Результат тестирования: УСПЕШНО")
            return 0
        else:
            logger.warning("Результат тестирования: НЕУДАЧА")
            return 1
    else:
        logger.error("Не удалось запустить все необходимые тесты")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 