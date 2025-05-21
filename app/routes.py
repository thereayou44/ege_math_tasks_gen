from flask import render_template, request, jsonify
import json
import os
import traceback
import re
from bs4 import BeautifulSoup
import random
from dotenv import load_dotenv
from app.task_generator import generate_complete_task, DEBUG_FILES_DIR
from app.model_processor import ModelResponseProcessor, convert_model_response_to_html
from app.json_api_helpers import generate_json_task, generate_json_markdown_task, generate_markdown_task
from app.utils.latex_formatter import normalize_latex_in_text, escape_html_in_text, format_solution_punctuation
import html

try:
    from app.task_generator import generate_complete_task
    print("Используется YandexGPT API для генерации задач")
except ImportError as e:
    print(f"Ошибка при импорте модуля task_generator: {e}")

def decode_html_entities(text):
    """
    Декодирует HTML-сущности в тексте
    
    Args:
        text: Текст с возможными HTML-сущностями
        
    Returns:
        str: Текст с декодированными HTML-сущностями
    """
    return html.unescape(text)

def load_categories_from_file(is_basic_level=False):
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

# Функция для форматирования ответа в LaTeX
def format_latex_answer(answer):
    """
    Форматирует ответ, добавляя LaTeX-окружение, если необходимо
    """
    if not answer:
        return "Ответ не указан"
    
    # Нормализуем LaTeX в ответе
    answer = normalize_latex_in_text(answer)
    
    # Если ответ уже содержит LaTeX, возвращаем как есть
    if '$' in answer:
        return answer
    
    # Иначе оборачиваем в LaTeX
    return f"${answer}$"

# Функция для извлечения ответа из решения
def extract_answer_from_solution(solution):
    """
    Извлекает ответ из решения
    """
    if not solution:
        return ""
    
    answer_pattern = r'(?:Ответ|ОТВЕТ|ответ)\s*:(.+?)(?=$|\.|\.\s*$|\n\s*\n)'
    match = re.search(answer_pattern, solution, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else ""

def init_routes(app):
    @app.route('/')
    def index():
        # Загружаем категории для обоих уровней
        advanced_categories = load_categories_from_file(is_basic_level=False)
        basic_categories = load_categories_from_file(is_basic_level=True)
        
        return render_template('index.html', 
                            advanced_categories=advanced_categories,
                            basic_categories=basic_categories)

    @app.route('/generate_task', methods=['POST'])
    def generate_task():
        try:
            data = request.get_json()
            category = data.get("category")
            subcategory = data.get("subcategory", "")
            difficulty_level = int(data.get("difficulty_level", 3))
            is_basic_level = data.get("is_basic_level", False)
            
            # Проверяем входные данные
            if not category:
                return jsonify({"error": "Категория не указана"}), 400
            
            # Генерируем полный пакет: задачу, решение и подсказки
            result = generate_complete_task(category, subcategory, difficulty_level, is_basic_level)
            
            # Проверяем на ошибки
            if "error" in result:
                return jsonify({"error": result["error"]}), 400
            
            # Проверка наличия основных компонентов результата
            if not result.get("task") or not result.get("solution"):
                return jsonify({"error": "Не удалось сгенерировать полный материал"}), 400
            
            # Проверка наличия решения и предупреждение, если оно отсутствует
            solution_warning = False
            if not result.get("solution") or result.get("solution") == "Не удалось извлечь решение":
                print("Предупреждение: для задачи не удалось сгенерировать решение. Попробуйте другую задачу.")
                solution_warning = True
            elif len(result.get("solution", "").strip()) < 10:
                print("Предупреждение: решение слишком короткое или пустое")
                solution_warning = True
            
            # Возвращаем данные клиенту
            response_data = {
                "task": result["task"],
                "solution": result["solution"],
                "solution_warning": solution_warning,
                "answer": result["answer"],
                "hints": result["hints"],
                "difficulty_level": result["difficulty_level"]
            }
            
            # Добавляем информацию об изображении, если оно есть
            if "image_url" in result:
                response_data["image_url"] = result["image_url"]
            elif "image_path" in result:
                image_path = result["image_path"]
                image_url = f"/static/images/generated/{os.path.basename(image_path)}"
                response_data["image_url"] = image_url
            
            return jsonify(response_data)
        except Exception as e:
            error_details = traceback.format_exc()
            print(f"Ошибка при генерации задачи: {e}")
            print(error_details)
            return jsonify({"error": f"Произошла ошибка при генерации задачи: {str(e)}"}), 500

    @app.route('/get_hint', methods=['POST'])
    def get_hint():
        try:
            data = request.get_json()
            hint_index = int(data.get("hint_index", 0))
            hints = data.get("hints", [])
            
            if not hints or hint_index >= len(hints):
                return jsonify({"error": "Подсказка недоступна"}), 400
            
            # Преобразуем подсказку в HTML перед отправкой и декодируем HTML-сущности
            hint = hints[hint_index]
            
            # Декодируем HTML-сущности (например, &lt; в <)
            hint = decode_html_entities(hint)
            
            # Преобразуем в HTML, если нужно
            hint_html = convert_markdown_to_html(hint)
            
            return jsonify({"hint": hint_html})
        except Exception as e:
            return jsonify({"error": f"Ошибка при получении подсказки: {str(e)}"}), 500

    @app.route('/solution', methods=['GET'])
    def view_solution():
        try:
            # Получаем решение из последнего сгенерированного ответа модели
            debug_response_file = os.path.join(DEBUG_FILES_DIR, "debug_response.txt")
            
            if os.path.exists(debug_response_file):
                with open(debug_response_file, 'r', encoding='utf-8') as f:
                    debug_content = f.read()
                
                # Используем ModelResponseProcessor для извлечения решения
                processor = ModelResponseProcessor(debug_content)
                solution = processor.process(format="html")["solution"]
                answer = processor.process(format="html")["answer"]
            else:
                return "Файл с решением не найден. Сначала сгенерируйте задачу."
            
            return f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Текущее решение</title>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
                <link href="/static/css/styles.css" rel="stylesheet">
                <!-- MathJax для отображения формул -->
                <script type="text/x-mathjax-config">
                    MathJax.Hub.Config({{
                        tex2jax: {{
                            inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
                            displayMath: [['$$', '$$'], ['\\\\[', '\\\\]'], ['\\\\begin{{equation}}', '\\\\end{{equation}}']],
                            processEscapes: true,
                            processEnvironments: true,
                            preview: "none"
                        }},
                        TeX: {{
                            extensions: ["AMSmath.js", "AMSsymbols.js", "noErrors.js", "noUndefined.js"],
                            equationNumbers: {{ autoNumber: "AMS" }}
                        }},
                        CommonHTML: {{ linebreaks: {{ automatic: false }} }},
                        "HTML-CSS": {{ 
                            linebreaks: {{ automatic: false }},
                            availableFonts: ["TeX"],
                            scale: 100,
                            styles: {{
                                ".MathJax_Display": {{ "text-align": "center !important" }}
                            }}
                        }},
                        SVG: {{ linebreaks: {{ automatic: false }} }},
                        showProcessingMessages: false,
                        messageStyle: "none"
                    }});
                </script>
                <script type="text/javascript" async
                        src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML"></script>
                <style>
                    .solution-container {{
                        max-width: 800px;
                        margin: 0 auto;
                        padding: 20px;
                    }}
                    .answer {{
                        margin-top: 20px;
                        padding: 10px;
                        background-color: #d4edda;
                        border-radius: 5px;
                    }}
                    .step-number {{
                        font-size: 1.2em;
                        font-weight: bold;
                        color: #3273dc;
                        display: inline-block;
                        min-width: 1.5em;
                        background-color: #f0f8ff;
                        border-radius: 50%;
                        text-align: center;
                        margin-right: 5px;
                        padding: 2px 5px;
                    }}
                    .full-solution p {{
                        margin-bottom: 10px;
                    }}
                    /* Правильное отображение блочных формул */
                    .MathJax_Display {{
                        display: block !important;
                        margin: 1em 0 !important;
                        text-align: center !important;
                    }}
                    /* Отображение инлайновых формул */
                    .MathJax {{
                        display: inline-block !important;
                    }}
                    /* Стиль для знаков препинания после формул */
                    .math-punctuation {{
                        display: inline-block;
                        margin-left: 1px;
                    }}
                    /* Стили для пользовательской нумерации списков */
                    ol.custom-numbered-list {{
                        counter-reset: none;
                        list-style-type: none;
                    }}
                    ol.custom-numbered-list li {{
                        position: relative;
                        padding-left: 30px;
                        margin-bottom: 8px;
                    }}
                    ol.custom-numbered-list li::before {{
                        content: attr(value) ".";
                        position: absolute;
                        left: 0;
                        font-weight: bold;
                        color: #3273dc;
                    }}
                </style>
            </head>
            <body>
                <div class="solution-container">
                    <h1 class="text-center">Решение задачи</h1>
                    <div class="full-solution">
                        {solution}
                    </div>
                    <div class="answer">
                        <h3>Ответ:</h3>
                        {answer}
                    </div>
                </div>
                <script>
                    // При загрузке страницы обрабатываем формулы
                    window.onload = function() {{
                        // Ждем загрузки MathJax
                        setTimeout(function() {{
                            // После рендеринга формул добавляем стили
                            var mathElements = document.querySelectorAll('.MathJax');
                            for (var i = 0; i < mathElements.length; i++) {{
                                mathElements[i].style.display = 'inline-block';
                            }}
                        }}, 500);
                    }};
                </script>
            </body>
            </html>
            """
        except Exception as e:
            error_detail = traceback.format_exc()
            return f"Ошибка при отображении решения: {str(e)}<br><pre>{error_detail}</pre>"

    @app.route('/debug')
    def debug_page():
        """
        Отладочная страница для проверки формата данных категорий
        """
        # Загружаем оба типа категорий
        advanced_categories = load_categories_from_file(is_basic_level=False)
        basic_categories = load_categories_from_file(is_basic_level=True)
        
        return render_template('debug.html', 
                              advanced_categories=advanced_categories,
                              basic_categories=basic_categories)

    @app.route('/debug_categories')
    def debug_categories():
        """
        Отладочная страница для просмотра категорий и их структуры
        """
        # Загружаем оба типа категорий
        advanced_categories = load_categories_from_file(is_basic_level=False)
        basic_categories = load_categories_from_file(is_basic_level=True)
        
        return render_template('debug_categories.html', 
                              advanced_categories=advanced_categories,
                              basic_categories=basic_categories)

    @app.route('/debug_categories_console')
    def debug_categories_console():
        """
        Расширенная отладочная страница для категорий с выводом в консоль
        """
        # Загружаем оба типа категорий
        advanced_categories = load_categories_from_file(is_basic_level=False)
        basic_categories = load_categories_from_file(is_basic_level=True)
        
        print("Отладка категорий в консоли:")
        print(f"- Профильный уровень: {len(advanced_categories)} категорий")
        print(f"- Базовый уровень: {len(basic_categories)} категорий")
        
        return render_template('debug_categories_console.html', 
                              advanced_categories=advanced_categories,
                              basic_categories=basic_categories)

    @app.route('/api/generate_markdown_task', methods=['POST'])
    def api_generate_markdown_task():
        try:
            data = request.get_json()
            category = data.get("category")
            subcategory = data.get("subcategory", "")
            difficulty_level = int(data.get("difficulty_level", 3))
            is_basic_level = data.get("is_basic_level", False)
            
            if not category:
                return jsonify({"error": "Категория не указана"}), 400
            
            # Генерируем задачу с опциональной автоматической генерацией изображения
            result = generate_markdown_task(category, subcategory, difficulty_level, is_basic_level)
            
            # Проверяем на ошибки
            if "error" in result:
                return jsonify({"error": result["error"]}), 400
                
            return jsonify(result)
        except Exception as e:
            return jsonify({"error": f"Ошибка при генерации задачи в формате Markdown: {str(e)}"}), 500

    @app.route('/api/task', methods=['GET'])
    def api_get_task():
        try:
            from app.task_generator import generate_complete_task
            
            category = request.args.get("category")
            subcategory = request.args.get("subcategory", "")
            difficulty_level = int(request.args.get("difficulty_level", 3))
            is_basic_level = request.args.get("is_basic_level", "false").lower() == "true"
            
            if not category:
                return jsonify({"error": "Категория не указана"}), 400
            
            # Генерируем полный пакет: задачу, решение и подсказки
            result = generate_complete_task(category, subcategory, difficulty_level, is_basic_level)
            
            # Проверяем на ошибки
            if "error" in result:
                return jsonify({"error": result["error"]}), 400
                
            return jsonify(result)
        except Exception as e:
            return jsonify({"error": f"Ошибка при генерации задачи: {str(e)}"}), 500

    @app.route('/api/json_task', methods=['GET'])
    def api_get_json_task():
        """
        API-маршрут для получения задачи в формате JSON
        с полями task, solution, answer, hints и метаданными
        
        Поддерживает параметр format: 
        - html (по умолчанию) - текст в формате HTML
        - markdown - текст в формате Markdown
        """
        try:
            category = request.args.get("category")
            subcategory = request.args.get("subcategory", "")
            difficulty_level = int(request.args.get("difficulty_level", 3))
            is_basic_level = request.args.get("is_basic_level", "false").lower() == "true"
            output_format = request.args.get("format", "html").lower()
            
            if not category:
                return jsonify({"error": "Категория не указана"}), 400
            
            # Выбираем функцию в зависимости от запрошенного формата
            if output_format == "markdown":
                # Используем формат Markdown
                result = generate_json_markdown_task(category, subcategory, difficulty_level, is_basic_level)
            else:
                # По умолчанию используем формат HTML
                result = generate_json_task(category, subcategory, difficulty_level, is_basic_level)
            
            # Проверяем на ошибки
            if "error" in result:
                return jsonify({"error": result["error"]}), 400
                
            return jsonify(result)
        except Exception as e:
            # Логируем ошибку для отладки
            error_details = traceback.format_exc()
            print(f"Ошибка при получении задачи в JSON через API: {e}")
            print(error_details)
            return jsonify({"error": f"Ошибка при генерации задачи в формате JSON: {str(e)}"}), 500

    # Проверка необходимых файлов и ключей при инициализации маршрутов
    print("=" * 50)
    print("Инициализация генератора задач ЕГЭ по математике")

    # Проверяем наличие необходимых файлов и ключей
    if not os.getenv("YANDEX_API_KEY") or not os.getenv("YANDEX_FOLDER_ID"):
        print("ВНИМАНИЕ: Не указаны YANDEX_API_KEY или YANDEX_FOLDER_ID в файле .env!")
        print("Генерация задач может не работать.")

    print("=" * 50)
    
    return app 

def convert_markdown_to_html(markdown_text):
    """
    Простая обертка для конвертации markdown в HTML
    """
    processor = ModelResponseProcessor(markdown_text)
    return processor._convert_to_html(markdown_text) 