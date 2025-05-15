#!/usr/bin/env python3
"""
Тестовый скрипт для проверки всех геометрических фигур.
"""

import os
import sys
from app.geometry import Triangle, Rectangle, Parallelogram, Trapezoid, Circle
from app.visualization import GeometryRenderer

def test_triangle():
    """
    Тестирует создание треугольника.
    """
    print("Тест: Треугольник")
    
    # Создаем параметры треугольника
    params = {
        'points': [(0, 0), (4, 0), (2, 3)],
        'show_labels': True,
        'show_lengths': True,
        'show_angles': True,
        'vertex_labels': ['A', 'B', 'C']
    }
    
    # Создаем треугольник
    triangle = Triangle(params)
    
    # Отрисовываем и сохраняем
    os.makedirs('test_images', exist_ok=True)
    output_path = GeometryRenderer.render_figure(triangle, 'test_images/triangle.png')
    print(f"Изображение сохранено: {output_path}")
    
    # Тестируем создание прямоугольного треугольника
    params_right = {
        'is_right': True,
        'show_labels': True,
        'show_lengths': True,
        'show_angles': True
    }
    triangle_right = Triangle(params_right)
    output_path_right = GeometryRenderer.render_figure(triangle_right, 'test_images/triangle_right.png')
    print(f"Изображение прямоугольного треугольника сохранено: {output_path_right}")
    
    # Тестируем создание из текста
    test_text = "В треугольнике ABC стороны равны 3, 4 и 5."
    triangle_from_text = Triangle.from_text(test_text)
    output_path_text = GeometryRenderer.render_figure(triangle_from_text, 'test_images/triangle_from_text.png')
    print(f"Изображение треугольника из текста сохранено: {output_path_text}")
    
    return all(path is not None for path in [output_path, output_path_right, output_path_text])

def test_rectangle():
    """
    Тестирует создание прямоугольника.
    """
    print("\nТест: Прямоугольник")
    
    # Создаем параметры прямоугольника
    params = {
        'width': 5,
        'height': 3,
        'show_labels': True,
        'show_lengths': True,
        'vertex_labels': ['A', 'B', 'C', 'D']
    }
    
    # Создаем прямоугольник
    rectangle = Rectangle(params)
    
    # Отрисовываем и сохраняем
    output_path = GeometryRenderer.render_figure(rectangle, 'test_images/rectangle.png')
    print(f"Изображение сохранено: {output_path}")
    
    # Тестируем создание квадрата
    params_square = {
        'width': 4,
        'height': 4,
        'show_labels': True,
        'show_lengths': True,
        'show_angles': True
    }
    square = Rectangle(params_square)
    output_path_square = GeometryRenderer.render_figure(square, 'test_images/square.png')
    print(f"Изображение квадрата сохранено: {output_path_square}")
    
    # Тестируем создание из текста
    test_text = "Прямоугольник имеет стороны 6 и 4. Найдите его площадь."
    rectangle_from_text = Rectangle.from_text(test_text)
    output_path_text = GeometryRenderer.render_figure(rectangle_from_text, 'test_images/rectangle_from_text.png')
    print(f"Изображение прямоугольника из текста сохранено: {output_path_text}")
    
    return all(path is not None for path in [output_path, output_path_square, output_path_text])

def test_parallelogram():
    """
    Тестирует создание параллелограмма.
    """
    print("\nТест: Параллелограмм")
    
    # Создаем параметры параллелограмма
    params = {
        'width': 5,
        'height': 3,
        'skew': 60,
        'show_labels': True,
        'show_lengths': True,
        'vertex_labels': ['A', 'B', 'C', 'D']
    }
    
    # Создаем параллелограмм
    parallelogram = Parallelogram(params)
    
    # Отрисовываем и сохраняем
    output_path = GeometryRenderer.render_figure(parallelogram, 'test_images/parallelogram.png')
    print(f"Изображение сохранено: {output_path}")
    
    # Тестируем создание ромба
    params_rhombus = {
        'width': 4,
        'height': 3,
        'skew': 45,
        'side_lengths': [4, 4, 4, 4],
        'show_labels': True,
        'show_lengths': True,
        'show_angles': True
    }
    rhombus = Parallelogram(params_rhombus)
    output_path_rhombus = GeometryRenderer.render_figure(rhombus, 'test_images/rhombus.png')
    print(f"Изображение ромба сохранено: {output_path_rhombus}")
    
    # Тестируем создание из текста
    test_text = "В параллелограмме стороны равны 5 и 3, а угол между ними 60 градусов."
    parallelogram_from_text = Parallelogram.from_text(test_text)
    output_path_text = GeometryRenderer.render_figure(parallelogram_from_text, 'test_images/parallelogram_from_text.png')
    print(f"Изображение параллелограмма из текста сохранено: {output_path_text}")
    
    return all(path is not None for path in [output_path, output_path_rhombus, output_path_text])

def test_trapezoid():
    """
    Тестирует создание трапеции.
    """
    print("\nТест: Трапеция")
    
    # Создаем параметры трапеции
    params = {
        'bottom_width': 6,
        'top_width': 3,
        'height': 3,
        'show_labels': True,
        'show_lengths': True,
        'vertex_labels': ['A', 'B', 'C', 'D']
    }
    
    # Создаем трапецию
    trapezoid = Trapezoid(params)
    
    # Отрисовываем и сохраняем
    output_path = GeometryRenderer.render_figure(trapezoid, 'test_images/trapezoid.png')
    print(f"Изображение сохранено: {output_path}")
    
    # Тестируем создание равнобедренной трапеции
    params_isosceles = {
        'bottom_width': 8,
        'top_width': 4,
        'height': 3,
        'is_isosceles': True,
        'show_labels': True,
        'show_lengths': True,
        'show_angles': True
    }
    isosceles_trapezoid = Trapezoid(params_isosceles)
    output_path_isosceles = GeometryRenderer.render_figure(isosceles_trapezoid, 'test_images/trapezoid_isosceles.png')
    print(f"Изображение равнобедренной трапеции сохранено: {output_path_isosceles}")
    
    # Тестируем создание из текста
    test_text = "В трапеции основания равны 10 и 6, а высота равна 4."
    trapezoid_from_text = Trapezoid.from_text(test_text)
    output_path_text = GeometryRenderer.render_figure(trapezoid_from_text, 'test_images/trapezoid_from_text.png')
    print(f"Изображение трапеции из текста сохранено: {output_path_text}")
    
    return all(path is not None for path in [output_path, output_path_isosceles, output_path_text])

def test_circle():
    """
    Тестирует создание окружности.
    """
    print("\nТест: Окружность")
    
    # Создаем параметры окружности
    params = {
        'center': (0, 0),
        'radius': 3,
        'center_label': 'O',
        'show_center': True,
        'show_radius': True
    }
    
    # Создаем окружность
    circle = Circle(params)
    
    # Отрисовываем и сохраняем
    output_path = GeometryRenderer.render_figure(circle, 'test_images/circle.png')
    print(f"Изображение сохранено: {output_path}")
    
    # Тестируем отображение диаметра и хорды
    params_with_all = {
        'center': (0, 0),
        'radius': 3,
        'center_label': 'O',
        'show_center': True,
        'show_radius': True,
        'show_diameter': True,
        'chord_value': 4.5,
        'show_chord': True
    }
    circle_with_all = Circle(params_with_all)
    output_path_with_all = GeometryRenderer.render_figure(circle_with_all, 'test_images/circle_with_all.png')
    print(f"Изображение окружности с радиусом, диаметром и хордой сохранено: {output_path_with_all}")
    
    # Тестируем создание из текста
    test_text = "Окружность с центром O и радиусом 5. Хорда равна 8."
    circle_from_text = Circle.from_text(test_text)
    output_path_text = GeometryRenderer.render_figure(circle_from_text, 'test_images/circle_from_text.png')
    print(f"Изображение окружности из текста сохранено: {output_path_text}")
    
    return all(path is not None for path in [output_path, output_path_with_all, output_path_text])

def test_from_text_auto():
    """
    Тестирует автоматическое определение типа фигуры из текста и создание соответствующей фигуры.
    """
    print("\nТест: Автоматическое определение типа фигуры")
    
    # Тексты для разных типов фигур
    texts = [
        "В треугольнике ABC углы равны 60, 60 и 60 градусов.",
        "Прямоугольник ABCD имеет стороны 5 см и 3 см.",
        "В параллелограмме ABCD диагонали пересекаются в точке O.",
        "В трапеции ABCD основания равны 8 см и 4 см, а высота 3 см.",
        "Окружность с центром O и радиусом 4 см. Найдите длину хорды, которая находится на расстоянии 2 см от центра."
    ]
    
    results = []
    for i, text in enumerate(texts):
        figure_type = GeometryRenderer.determine_figure_type(text)
        print(f"Текст: {text}")
        print(f"Определенный тип: {figure_type}")
        
        if figure_type:
            output_path = GeometryRenderer.render_from_text(text, figure_type)
            print(f"Изображение сохранено: {output_path}")
            results.append(output_path is not None)
        else:
            print("Не удалось определить тип фигуры")
            results.append(False)
    
    return all(results)

def run_all_tests():
    """
    Запускает все тесты и выводит результаты.
    """
    print("Запуск тестов всех геометрических фигур\n" + "="*60)
    
    tests = [
        test_triangle,
        test_rectangle,
        test_parallelogram,
        test_trapezoid,
        test_circle,
        test_from_text_auto
    ]
    
    results = []
    for test_func in tests:
        result = test_func()
        results.append(result)
    
    # Вывод результатов
    print("\n" + "="*60 + "\nРезультаты тестов:")
    for i, (test_func, result) in enumerate(zip(tests, results)):
        status = "УСПЕШНО" if result else "ОШИБКА"
        print(f"{i+1}. {test_func.__name__}: {status}")
    
    # Общий результат
    if all(results):
        print("\nВсе тесты прошли успешно!")
        return 0
    else:
        print("\nНекоторые тесты завершились с ошибками.")
        return 1

if __name__ == '__main__':
    sys.exit(run_all_tests()) 