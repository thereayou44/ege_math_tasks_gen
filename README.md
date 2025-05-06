# Генератор задач ЕГЭ по математике

Веб-приложение для генерации тренировочных задач ЕГЭ по математике с использованием искусственного интеллекта.

## Описание

Проект представляет собой веб-приложение, позволяющее генерировать тренировочные задачи по математике для подготовки к ЕГЭ. 

Основные возможности:
- Генерация задач разных типов и уровней сложности
- Получение полного решения задач
- Возможность получения пошаговых подсказок
- Поддержка базового и профильного уровней ЕГЭ
- Визуализация геометрических задач и функций

## Структура проекта

```
/
├── app/                       # Основной код приложения
│   ├── __init__.py           
│   ├── routes.py              # Flask маршруты
│   ├── task_generator.py      # Логика генерации задач
│   ├── prompts.py             # Промпты для AI
│   ├── utils/                 # Вспомогательные функции
│   │   ├── __init__.py
│   │   ├── converters.py      # Конвертеры форматов
│   │   ├── parsers.py         # Парсеры текста и LaTeX
│   │   └── visualization.py   # Визуализация фигур и графиков
│   └── models/                # Модели данных
│       ├── __init__.py
│       └── categories.py      # Структуры данных категорий
│
├── data/                      # Управление данными
│   ├── categories/            # Файлы с данными категорий
│   └── problems/              # Наборы задач
│
├── scripts/                   # Вспомогательные скрипты
│   ├── download_mathb_problems.py  # Скачивание задач
│   ├── generate_categories_list.py # Генерация списка категорий
│   └── create_subcategories.py     # Создание подкатегорий
│
├── static/                    # Статические ресурсы
│   ├── css/
│   ├── js/
│   └── images/
│       └── generated/         # Генерируемые изображения
│
├── templates/                 # Flask шаблоны
│
├── tests/                     # Тесты
│   ├── __init__.py
│   ├── run_all_tests.py
│   ├── test_task_generator.py
│   ├── test_visualization.py
│   ├── test_parsers.py
│   ├── test_format_converters.py
│   └── test_mock.py
│
├── model_tuning/              # Файлы для fine-tuning модели
│   ├── fine_tune.py
│   ├── start-tuning.py
│   └── tuning_dataset_final.jsonl
│
├── app.py                     # Точка входа в приложение
├── requirements.txt           # Зависимости проекта
└── README.md                  # Данный файл
```

## Установка и запуск

### Предварительные требования

- Python 3.8 или выше
- API-ключ Yandex GPT (для генерации задач)

### Настройка

1. Клонируйте репозиторий:
   ```
   git clone https://github.com/yourusername/ege-math-tasks-gen.git
   cd ege-math-tasks-gen
   ```

2. Создайте виртуальное окружение:
   ```
   python -m venv venv
   source venv/bin/activate  # На Windows: venv\Scripts\activate
   ```

3. Установите зависимости:
   ```
   pip install -r requirements.txt
   ```

4. Создайте файл .env в корневой директории и добавьте ваши API-ключи:
   ```
   YANDEX_API_KEY=ваш_ключ_api
   YANDEX_FOLDER_ID=ваш_folder_id
   ```

### Запуск приложения

```
python app.py
```

После запуска откройте в браузере адрес http://localhost:5000

## API

Приложение предоставляет API для генерации задач:

- `GET /api/task` - генерация задачи с параметрами
- `GET /api/json_task` - генерация задачи в формате JSON
- `POST /api/generate_markdown_task` - генерация задачи в формате Markdown

## Тестирование

Для запуска тестов выполните:

```
python -m tests.run_all_tests
```

## Лицензия

Этот проект распространяется под лицензией MIT. См. файл LICENSE для получения дополнительной информации.
