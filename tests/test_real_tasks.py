#!/usr/bin/env python3
"""
Тестовый скрипт для проверки отображения фигур на примере реальных заданий ЕГЭ.
"""

import os
import sys
import matplotlib.pyplot as plt

# Получаем корневую директорию проекта
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.geometry import Triangle, Rectangle, Parallelogram, Trapezoid, Circle
from app.visualization import GeometryRenderer

def test_parallelogram_task():
    """
    Тестирует создание и отображение параллелограмма для типичного задания ЕГЭ.
    """
    # Текст тестового задания
    task_text = """
    В параллелограмме ABCD стороны AB = 5 и AD = 3.46, а углы A и C равны 60 градусов. 
    Найдите высоту параллелограмма.
    """
    
    # Создаем параллелограмм из текста задания
    parallelogram = Parallelogram.from_text(task_text)
    
    # Создаем директорию для тестовых изображений, если она не существует
    os.makedirs('test_images', exist_ok=True)
    
    # Отрисовываем и сохраняем изображение
    filename = "test_images/parallelogram_task.png"
    output_path = GeometryRenderer.render_figure(parallelogram, filename)
    
    print(f"Изображение параллелограмма для задания сохранено: {output_path}")
    return output_path is not None

def test_triangle_task():
    """
    Тестирует создание и отображение треугольника для типичного задания ЕГЭ.
    """
    # Текст тестового задания
    task_text = """
    В треугольнике ABC угол C равен 90 градусов, AB = 5, AC = 4. 
    Найдите высоту, опущенную на сторону AC.
    """
    
    # Создаем треугольник из текста задания
    triangle = Triangle.from_text(task_text)
    
    # Отрисовываем и сохраняем изображение
    filename = "test_images/triangle_task.png"
    output_path = GeometryRenderer.render_figure(triangle, filename)
    
    print(f"Изображение треугольника для задания сохранено: {output_path}")
    return output_path is not None

def test_trapezoid_task():
    """
    Тестирует создание и отображение трапеции для типичного задания ЕГЭ.
    """
    # Текст тестового задания
    task_text = """
    В равнобедренной трапеции ABCD основания AD и BC равны 8 и 4 соответственно, 
    а боковые стороны AB и CD равны 5. Найдите площадь трапеции.
    """
    
    # Создаем трапецию из текста задания
    trapezoid = Trapezoid.from_text(task_text)
    
    # Отрисовываем и сохраняем изображение
    filename = "test_images/trapezoid_task.png"
    output_path = GeometryRenderer.render_figure(trapezoid, filename)
    
    print(f"Изображение трапеции для задания сохранено: {output_path}")
    return output_path is not None

def test_circle_task():
    """
    Тестирует создание и отображение окружности для типичного задания ЕГЭ.
    """
    # Текст тестового задания
    task_text = """
    Окружность с центром O имеет радиус 6. Хорда AB равна 8. 
    Найдите расстояние от центра окружности до хорды AB.
    """
    
    # Создаем окружность из текста задания
    circle = Circle.from_text(task_text)
    
    # Отрисовываем и сохраняем изображение
    filename = "test_images/circle_task.png"
    output_path = GeometryRenderer.render_figure(circle, filename)
    
    print(f"Изображение окружности для задания сохранено: {output_path}")
    return output_path is not None

def test_rectangle_task():
    """
    Тестирует создание и отображение прямоугольника для типичного задания ЕГЭ.
    """
    # Текст тестового задания
    task_text = """
    Диагонали прямоугольника ABCD пересекаются в точке O. Периметр прямоугольника равен 28, 
    а диагональ AC равна 10. Найдите площадь прямоугольника.
    """
    
    # Создаем прямоугольник из текста задания
    rectangle = Rectangle.from_text(task_text)
    
    # Отрисовываем и сохраняем изображение
    filename = "test_images/rectangle_task.png"
    output_path = GeometryRenderer.render_figure(rectangle, filename)
    
    print(f"Изображение прямоугольника для задания сохранено: {output_path}")
    return output_path is not None

def test_combined_real_tasks():
    """
    Создает комбинированное изображение всех фигур из реальных заданий.
    """
    # Тексты заданий
    tasks = {
        "triangle": """
        В треугольнике ABC угол C равен 90 градусов, AB = 5, AC = 4. 
        Найдите высоту, опущенную на сторону AC.
        """,
        "rectangle": """
        Диагонали прямоугольника ABCD пересекаются в точке O. Периметр прямоугольника равен 28, 
        а диагональ AC равна 10. Найдите площадь прямоугольника.
        """,
        "parallelogram": """
        В параллелограмме ABCD стороны AB = 5 и AD = 3.46, а углы A и C равны 60 градусов. 
        Найдите высоту параллелограмма.
        """,
        "trapezoid": """
        В равнобедренной трапеции ABCD основания AD и BC равны 8 и 4 соответственно, 
        а боковые стороны AB и CD равны 5. Найдите площадь трапеции.
        """,
        "circle": """
        Окружность с центром O имеет радиус 6. Хорда AB равна 8. 
        Найдите расстояние от центра окружности до хорды AB.
        """
    }
    
    # Создаем фигуры
    triangle = Triangle.from_text(tasks["triangle"])
    rectangle = Rectangle.from_text(tasks["rectangle"])
    parallelogram = Parallelogram.from_text(tasks["parallelogram"])
    trapezoid = Trapezoid.from_text(tasks["trapezoid"])
    circle = Circle.from_text(tasks["circle"])
    
    # Создаем изображение с подграфиками
    fig, axs = plt.subplots(3, 2, figsize=(18, 24))
    fig.suptitle('Визуализация фигур из реальных заданий ЕГЭ', fontsize=16)
    
    # Отрисовываем фигуры
    triangle.draw(axs[0, 0])
    axs[0, 0].set_title('Прямоугольный треугольник')
    
    rectangle.draw(axs[0, 1])
    axs[0, 1].set_title('Прямоугольник')
    
    parallelogram.draw(axs[1, 0])
    axs[1, 0].set_title('Параллелограмм')
    
    trapezoid.draw(axs[1, 1])
    axs[1, 1].set_title('Равнобедренная трапеция')
    
    circle.draw(axs[2, 0])
    axs[2, 0].set_title('Окружность с хордой')
    
    # Скрываем пустой график
    axs[2, 1].axis('off')
    
    # Выравниваем и настраиваем макет
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Сохраняем комбинированное изображение
    output_path = 'test_images/all_real_tasks_combined.png'
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"Комбинированное изображение фигур из реальных заданий сохранено: {output_path}")
    return True

def test_all_real_tasks():
    """
    Запускает все тестовые примеры учебных заданий.
    """
    print("Запуск тестовых примеров реальных заданий ЕГЭ\n" + "="*60)
    
    results = []
    results.append(test_triangle_task())
    results.append(test_rectangle_task())
    results.append(test_parallelogram_task())
    results.append(test_trapezoid_task())
    results.append(test_circle_task())
    results.append(test_combined_real_tasks())
    
    # Вывод результатов
    print("\n" + "="*60 + "\nРезультаты тестов:")
    tests = ["test_triangle_task", "test_rectangle_task", "test_parallelogram_task", 
             "test_trapezoid_task", "test_circle_task", "test_combined_real_tasks"]
    
    for i, (test, result) in enumerate(zip(tests, results)):
        status = "УСПЕШНО" if result else "ОШИБКА"
        print(f"{i+1}. {test}: {status}")
    
    # Общий результат
    if all(results):
        print("\nВсе тесты прошли успешно!")
        return 0
    else:
        print("\nНекоторые тесты завершились с ошибками.")
        return 1

if __name__ == '__main__':
    sys.exit(test_all_real_tasks()) 