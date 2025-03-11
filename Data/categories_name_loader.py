import os
import json

def get_category_subcategory_list():
    categories = []
    base_path = "/home/thereayou/dimploma/diploma/Data/math_catalog_subcategories"


    category_number = 1  # Начальный номер категории

    # Получаем список категорий (папок)
    for category in os.listdir(base_path):
        category_path = os.path.join(base_path, category)
        if os.path.isdir(category_path):  # Проверяем, что это директория
            subcategories = []
            subcategory_number = 1  # Начальный номер подкатегории для каждой категории

            # Список подкатегорий
            for subcategory in os.listdir(category_path):
                subcategories.append({
                    'name': subcategory,
                    'number': f"{category_number}.{subcategory_number}"  # Формат: 1.1, 1.2 и т.д.
                })
                subcategory_number += 1

            categories.append({
                'category': category,
                'category_number': category_number,  # Номер категории
                'subcategories': subcategories
            })

            category_number += 1  # Увеличиваем номер категории для следующей категории

    return categories

def save_categories_to_file():
    categories = get_category_subcategory_list()
    with open('categories_list.json', 'w', encoding='utf-8') as file:
        json.dump(categories, file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    save_categories_to_file()