#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Запускает все тесты из пакета tests.
"""

import unittest
import sys
import os

# Добавляем родительскую директорию в пути для импорта
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Импортируем все тесты с правильными путями
from test_task_generator import TestTaskGenerator
from test_visualization import TestVisualization
from test_parsers import TestParsers
from test_format_converters import TestFormatConverters
from test_mock import TestTaskGeneratorWithMocks

if __name__ == "__main__":
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
    
    # Выходим с ненулевым кодом, если есть ошибки или сбои
    sys.exit(not result.wasSuccessful()) 