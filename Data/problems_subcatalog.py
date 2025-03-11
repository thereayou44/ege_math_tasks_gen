import os
import json
import asyncio
from sdamgia import SdamgiaAPI
from sdamgia.enums import Subject, GiaType
import dataclasses

banned_problems = [508108, 500213]
def problem_to_json(problem) -> str:
    return json.dumps(dataclasses.asdict(problem), indent=4, ensure_ascii=False)

async def fetch_problem_data(problem_id: int, sdamgia: SdamgiaAPI):
    problem = await sdamgia.get_problem(problem_id, subject=Subject.MATH)
    return problem_to_json(problem)

async def process_subcategory(subcategory_path: str, sdamgia: SdamgiaAPI):
    # Открываем subcategories.json и извлекаем id задач
    with open(subcategory_path, 'r', encoding='utf-8') as f:
        problem_ids = json.load(f)

    # Для каждого ID задачи получаем данные и сохраняем в соответствующий файл
    for problem_id in problem_ids:
        if problem_id in banned_problems:
            print(f"Пропускаем задачу {problem_id}")
            continue

        json_file_path = os.path.join(os.path.dirname(subcategory_path), f'{problem_id}.json')

        # Проверяем, существует ли уже файл для этой задачи
        if os.path.exists(json_file_path):
            print(f"Problem {problem_id} already processed, skipping.")
            continue  # Пропускаем задачу, если файл уже существует

        print(f"Processing problem {problem_id}")
        problem_data = await fetch_problem_data(problem_id, sdamgia)

        # Сохраняем данные задачи в JSON файл
        with open(json_file_path, 'w', encoding='utf-8') as json_file:
            json_file.write(problem_data)

async def process_category(category_path: str, sdamgia: SdamgiaAPI):
    for subcategory in os.listdir(category_path):
        subcategory_path = os.path.join(category_path, subcategory)
        if os.path.isdir(subcategory_path):
            # Ищем subcategories.json в каждой подкатегории
            subcategories_json_path = os.path.join(subcategory_path, 'subcategories.json')
            if os.path.exists(subcategories_json_path):
                await process_subcategory(subcategories_json_path, sdamgia)

async def main():
    async with SdamgiaAPI(gia_type=GiaType.EGE, subject=Subject.MATH) as sdamgia:
        base_dir = "math_base_catalog_subcategories"  # Указываем путь к вашей папке с категориями
        for category in os.listdir(base_dir):
            print(f"Processing category: {category}")
            category_path = os.path.join(base_dir, category)
            if os.path.isdir(category_path):
                await process_category(category_path, sdamgia)

if __name__ == "__main__":
    asyncio.run(main())
