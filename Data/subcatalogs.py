import json
import os
import asyncio
from sdamgia import SdamgiaAPI
from sdamgia.enums import GiaType, Subject

async def fetch_and_save_subcategories(subject: Subject, catalog: list, base_folder: str) -> None:
    sdamgia = SdamgiaAPI(gia_type=GiaType.EGE, subject=subject)

    for topic in catalog:
        topic_name = topic['name']
        for category in topic['categories']:
            category_name = category['name']
            category_id = category['id']

            print(f"Получаем подкатегории для категории: {category_name}")

            try:
                # Получаем список подкатегорий с помощью get_theme
                subcategories = await sdamgia.get_theme(category_id)

                # Создаем папку для категории
                category_folder = os.path.join(base_folder, topic_name, category_name)
                os.makedirs(category_folder, exist_ok=True)

                # Сохраняем информацию о подкатегориях в файл JSON
                subcategories_file = os.path.join(category_folder, 'subcategories.json')
                with open(subcategories_file, 'w', encoding='utf-8') as f:
                    json.dump(subcategories, f, ensure_ascii=False, indent=4)

                print(f"Подкатегории для категории '{category_name}' сохранены.")

            except Exception as e:
                print(f"Ошибка при получении подкатегорий для категории {category_name}: {e}")

    await sdamgia.close()

async def main() -> None:
    with open('math_base_catalog.json', 'r', encoding='utf-8') as f:
        catalog = json.load(f)

    base_folder = 'math_base_catalog_subcategories'

    # Скачиваем подкатегории и сохраняем их
    await fetch_and_save_subcategories(Subject.MATH, catalog, base_folder)

# Запуск основной функции
if __name__ == "__main__":
    asyncio.run(main())
