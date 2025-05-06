import json
import os

class Categories:
    """Класс для работы с категориями задач ЕГЭ"""
    
    @staticmethod
    def load_categories(is_basic_level=False):
        """
        Загружает категории из соответствующего файла в зависимости от уровня ЕГЭ
        
        Args:
            is_basic_level: Если True, загружает категории для базового уровня, иначе для профильного
            
        Returns:
            list: Список категорий
        """
        try:
            # Выбираем файл в зависимости от уровня
            if is_basic_level:
                filename = 'data/categories/base_categories_list.json'
            else:
                filename = 'data/categories/categories_list.json'
                
            with open(filename, 'r', encoding='utf-8') as f:
                categories = json.load(f)
            return categories
        except FileNotFoundError:
            print(f"Файл категорий не найден: {filename}")
            return []
        except json.JSONDecodeError:
            print("Ошибка формата файла категорий!")
            return []
    
    @staticmethod
    def get_subcategories(category, is_basic_level=False):
        """
        Получает список подкатегорий для указанной категории
        
        Args:
            category: Название категории
            is_basic_level: Если True, работаем с базовым уровнем, иначе с профильным
            
        Returns:
            list: Список подкатегорий для указанной категории
        """
        categories = Categories.load_categories(is_basic_level)
        for cat in categories:
            if cat["category"] == category:
                return cat.get("subcategories", [])
        return []
    
    @staticmethod
    def validate_category(category, subcategory=None, is_basic_level=False):
        """
        Проверяет существование категории и подкатегории
        
        Args:
            category: Название категории
            subcategory: Название подкатегории (опционально)
            is_basic_level: Если True, проверяем в базовом уровне, иначе в профильном
            
        Returns:
            bool: True если категория и подкатегория существуют, иначе False
        """
        categories = Categories.load_categories(is_basic_level)
        
        # Проверяем существование категории
        category_exists = any(cat["category"] == category for cat in categories)
        
        # Если подкатегория не указана, достаточно проверки категории
        if not subcategory:
            return category_exists
        
        # Если указана подкатегория, проверяем ее наличие
        for cat in categories:
            if cat["category"] == category:
                return any(subcat["name"] == subcategory for subcat in cat.get("subcategories", []))
        
        return False 