#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Скрипт для генерации файлов категорий из директорий.
Создает JSON файлы base_categories_list.json и categories_list.json
на основе структуры каталогов.
"""

import os
import json

def generate_categories_list(base_dir, output_file, catalog_file=None):
    """
    Генерирует JSON файл с категориями на основе структуры директорий.
    
    Args:
        base_dir: Путь к базовой директории с категориями
        output_file: Путь к выходному JSON файлу
        catalog_file: Путь к каталогу с подкатегориями (опционально)
    """
    print(f"Создаем список категорий из {base_dir}")
    
    # Проверяем существование директории
    if not os.path.exists(base_dir):
        print(f"Ошибка: директория {base_dir} не существует")
        return False
    
    # Получаем список директорий (категорий)
    try:
        categories = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
    except Exception as e:
        print(f"Ошибка при чтении директории {base_dir}: {e}")
        return False
    
    # Загружаем дополнительный каталог с подкатегориями, если указан
    catalog_data = None
    if catalog_file and os.path.exists(catalog_file):
        try:
            with open(catalog_file, 'r', encoding='utf-8') as f:
                catalog_data = json.load(f)
            print(f"Загружены дополнительные данные из {catalog_file}")
        except Exception as e:
            print(f"Ошибка при чтении каталога {catalog_file}: {e}")
    
    # Формируем структуру JSON
    categories_data = []
    for i, category in enumerate(categories, 1):
        category_data = {
            "category_number": str(i),
            "category": category,
            "subcategories": []
        }
        
        # Путь к директории категории
        category_dir = os.path.join(base_dir, category)
        
        # Проверяем наличие подкатегорий в файле
        subcategories_file = os.path.join(category_dir, "subcategories.json")
        if os.path.exists(subcategories_file):
            try:
                with open(subcategories_file, 'r', encoding='utf-8') as f:
                    subcategories_data = json.load(f)
                
                # Добавляем подкатегории, если они есть
                if isinstance(subcategories_data, list):
                    for j, subcat in enumerate(subcategories_data, 1):
                        if isinstance(subcat, dict) and "name" in subcat:
                            subcat["number"] = f"{i}.{j}"
                            category_data["subcategories"].append(subcat)
            except Exception as e:
                print(f"Ошибка при чтении {subcategories_file}: {e}")
        
        # Если есть дополнительный каталог, ищем подкатегории там
        if catalog_data:
            for cat_entry in catalog_data:
                if cat_entry.get("name") == category and "categories" in cat_entry:
                    # Если в подкатегориях уже есть данные, пропускаем
                    if category_data["subcategories"]:
                        continue
                        
                    for j, subcat in enumerate(cat_entry["categories"], 1):
                        if "name" in subcat:
                            # Формируем структуру подкатегории
                            subcat_data = {
                                "name": subcat["name"],
                                "number": f"{i}.{j}"
                            }
                            category_data["subcategories"].append(subcat_data)
        
        categories_data.append(category_data)
    
    # Записываем результат в файл
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(categories_data, f, ensure_ascii=False, indent=2)
        print(f"Файл {output_file} успешно создан с {len(categories_data)} категориями")
        return True
    except Exception as e:
        print(f"Ошибка при записи в файл {output_file}: {e}")
        return False

if __name__ == "__main__":
    # Генерируем список категорий для базового уровня
    base_success = generate_categories_list(
        "Data/math_base_catalog_subcategories", 
        "Data/base_categories_list.json",
        "Data/math_base_catalog.json"
    )
    
    # Генерируем список категорий для профильного уровня
    advanced_success = generate_categories_list(
        "Data/math_catalog_subcategories", 
        "Data/categories_list.json"
    )
    
    if base_success and advanced_success:
        print("Все файлы категорий успешно созданы!")
    else:
        print("Были ошибки при создании файлов категорий.") 