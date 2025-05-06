#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Скрипт для создания файлов subcategories.json для всех категорий профильного уровня
на основе существующих поддиректорий.
"""

import os
import json

def create_subcategories_files():
    base_dir = "Data/math_catalog_subcategories"
    
    # Проверяем существование директории
    if not os.path.exists(base_dir):
        print(f"Ошибка: директория {base_dir} не существует")
        return False
    
    # Получаем все категории (директории верхнего уровня)
    categories = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    print(f"Найдено {len(categories)} категорий в {base_dir}")
    
    for category in categories:
        category_path = os.path.join(base_dir, category)
        
        # Получаем все подкатегории (поддиректории категории)
        subcategories = [d for d in os.listdir(category_path) if os.path.isdir(os.path.join(category_path, d))]
        
        if subcategories:
            print(f"Категория: {category}, найдено {len(subcategories)} подкатегорий")
            
            # Создаем JSON-структуру для подкатегорий
            subcategories_data = []
            for subcat in subcategories:
                subcategories_data.append({"name": subcat})
            
            # Путь к файлу subcategories.json
            subcategories_file = os.path.join(category_path, "subcategories.json")
            
            # Записываем файл
            try:
                with open(subcategories_file, 'w', encoding='utf-8') as f:
                    json.dump(subcategories_data, f, ensure_ascii=False, indent=2)
                print(f"  Создан файл {subcategories_file}")
            except Exception as e:
                print(f"  Ошибка при создании файла {subcategories_file}: {e}")
        else:
            print(f"Категория: {category}, подкатегории не найдены")
    
    return True

if __name__ == "__main__":
    print("Создание файлов subcategories.json для профильного уровня ЕГЭ")
    if create_subcategories_files():
        print("Файлы subcategories.json успешно созданы")
        
        # После создания файлов запускаем генерацию списка категорий
        print("Обновляем списки категорий...")
        os.system("python3 generate_categories_list.py")
    else:
        print("Были ошибки при создании файлов subcategories.json") 