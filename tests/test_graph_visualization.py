#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Тестовый скрипт для проверки функциональности генерации графиков с особыми точками.
"""

import os
import sys
import re
import matplotlib.pyplot as plt
import numpy as np

# Добавляем путь к модулю в sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.task_generator import generate_multi_function_graph

def test_special_points_graph():
    """Генерация тестового графика с функцией и особыми точками для задачи анализа графика функции"""
    
    # Создаем функцию для графика
    function = "x**3 - 6*x**2 + 9*x - 4"
    
    # Создаем список функций для generate_multi_function_graph
    functions = [(function, 'blue', 'f(x)')]
    
    # Создаем особые точки для интервалов
    special_points = [
        (-1, 0, 'a'),
        (1, 0, 'b'),
        (3, 0, 'c'),
        (5, 0, 'd'),
        (7, 0, 'e')
    ]
    
    # Генерируем график
    filepath = generate_multi_function_graph(functions, x_range=(-2, 8), y_range=(-5, 5), special_points=special_points)
    
    if filepath:
        print(f"График успешно создан: {filepath}")
    else:
        print("Ошибка при создании графика")

if __name__ == "__main__":
    test_special_points_graph() 