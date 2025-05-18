from flask import render_template, request, jsonify
import json
import os
import traceback
import re
from bs4 import BeautifulSoup
import random
from dotenv import load_dotenv
from app.task_generator import generate_complete_task, convert_markdown_to_html

try:
    from app.task_generator import generate_complete_task
    print("Используется YandexGPT API для генерации задач")
except ImportError as e:
    print(f"Ошибка при импорте модуля task_generator: {e}")

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

def format_latex_in_answer(answer):
    """
    Проверяет и корректирует формат LaTeX в ответе
    
    Args:
        answer: Текст ответа
        
    Returns:
        str: Ответ с правильно отформатированным LaTeX
    """
    if not answer or answer.strip() == "":
        return ""
    
    # Убираем \( и \) из ответа и заменяем на доллары, если они есть
    if '\\(' in answer and '\\)' in answer:
        answer = answer.replace('\\(', '$').replace('\\)', '$')
    
    # Если ответ уже содержит знаки доллара, считаем что LaTeX уже оформлен
    if '$' in answer:
        # Проверяем парность долларов
        open_count = answer.count('$')
        if open_count % 2 != 0:
            answer += '$'  # Добавляем закрывающий доллар если не хватает
        return answer
    
    # Расширенный список LaTeX-команд для проверки
    latex_patterns = [
        r'\\frac', r'\\sqrt', r'\\sum', r'\\prod', r'\\int', 
        r'\\lim', r'\\sin', r'\\cos', r'\\tan', r'\\log', r'\\ln',
        r'\\alpha', r'\\beta', r'\\gamma', r'\\delta', r'\\pi',
        r'\\le', r'\\ge', r'\\neq', r'\\approx', r'\\cdot',
        r'\\left', r'\\right', r'\\mathbb', r'\\mathcal', r'\\partial',
        r'\\begin\{.*?\}', r'\\end\{.*?\}', r'\\overline', r'\\underline',
        r'\\times', r'\\div', r'\\equiv', r'\\cup', r'\\cap', r'\\in', r'\\infty',
        r'\\tg', r'\\ctg', r'\\arctg'
    ]
    
    # Ищем любой паттерн LaTeX в ответе
    has_latex = any(re.search(pattern, answer) for pattern in latex_patterns)
    
    # Если ответ содержит LaTeX-команды, но не обернут в доллары
    if has_latex and '$' not in answer:
        return f"${answer}$"
    
    # Экранируем угловые скобки, если они не являются частью HTML-тега
    if '<' in answer and not re.search(r'<[a-z/]', answer):
        answer = answer.replace('<', '&lt;').replace('>', '&gt;')
    
    return answer

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
                
            # Проверка наличия решения
            if not result.get("solution") or result.get("solution") == "Не удалось извлечь решение":
                print("Предупреждение: для задачи не удалось сгенерировать решение. Попробуйте другую задачу.")
                result["solution_warning"] = True
            else:
                result["solution_warning"] = False
                # Дополнительная проверка решения - должен быть не пустым и содержать текст
                if len(result.get("solution", "").strip()) < 10:
                    print("Предупреждение: решение слишком короткое или пустое")
                    result["solution_warning"] = True
            
            # Проверяем формат ответа и корректируем LaTeX
            if "answer" in result:
                result["answer"] = format_latex_in_answer(result["answer"])
            
            # Проверяем, что все подсказки существуют и имеют корректный формат
            hints = result.get("hints", [])
            if len(hints) < 3:
                # Дополняем список подсказок до 3, если их меньше
                while len(hints) < 3:
                    hints.append("Подсказка недоступна")
                result["hints"] = hints
            
            # Дополнительная проверка и конвертация Markdown в HTML, если ещё не конвертировано
            task = result.get("task", "")
            solution = result.get("solution", "")
            
            # Извлекаем ответ из решения, чтобы переместить его в отдельную рамку
            answer_match = re.search(r'(?:Ответ|ОТВЕТ|ответ)\s*:(.+?)(?=$|\.|\.\s*$|\n\s*\n|\<\/p\>)', solution, re.IGNORECASE | re.DOTALL)
            if answer_match:
                # Если нашли "Ответ:" в решении, извлекаем его
                answer_text = answer_match.group(1).strip()
                # Удаляем "Ответ:" из текста решения
                solution = re.sub(r'(?:Ответ|ОТВЕТ|ответ)\s*:.+?(?=$|\.|\.\s*$|\n\s*\n|\<\/p\>)', '', solution, flags=re.IGNORECASE | re.DOTALL)
                # Обновляем ответ в результате
                result["answer"] = format_latex_in_answer(answer_text)

            # Проверяем, содержит ли текст Markdown-форматирование
            if re.search(r'\*\*.*?\*\*|\*.*?\*', task) or "\n" in task:
                task = convert_markdown_to_html(task)
            
            if re.search(r'\*\*.*?\*\*|\*.*?\*', solution) or "\n" in solution:
                solution = convert_markdown_to_html(solution)
            
            # Проверяем подсказки
            for i in range(len(hints)):
                if re.search(r'\*\*.*?\*\*|\*.*?\*', hints[i]) or "\n" in hints[i]:
                    hints[i] = convert_markdown_to_html(hints[i])
                    
            # Убеждаемся, что решение не потерялось
            if not solution or solution.strip() == "":
                solution = "<p>Решение не удалось сгенерировать. Пожалуйста, попробуйте перегенерировать задачу.</p>"
                result["solution_warning"] = True
            
            # Обновляем результат
            result["task"] = task
            result["solution"] = solution
            result["hints"] = hints
            
            # Логируем информацию для отладки
            print(f"Отправка данных клиенту:")
            print(f"- Длина задачи: {len(task)}")
            print(f"- Длина решения: {len(solution)}")
            print(f"- Количество подсказок: {len(hints)}")
            
            # Возвращаем данные клиенту
            response_data = {
                "task": result["task"],
                "solution": result["solution"],
                "solution_warning": result.get("solution_warning", False),
                "answer": result["answer"],
                "hints": result["hints"],
                "difficulty_level": result["difficulty_level"]
            }
            
            # Добавляем информацию об изображении, если оно есть
            if "image_path" in result:
                image_path = result["image_path"]
                image_url = f"/static/images/generated/{os.path.basename(image_path)}"
                response_data["image_url"] = image_url
            
            return jsonify(response_data)
        except Exception as e:
            # Логируем ошибку для отладки
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
            
            return jsonify({"hint": hints[hint_index]})
        except Exception as e:
            return jsonify({"error": f"Ошибка при получении подсказки: {str(e)}"}), 500

    @app.route('/solution', methods=['GET'])
    def view_solution():
        try:
            with open("last_generated_task.txt", 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Ищем решение в тексте
            solution_match = re.search(r'===РЕШЕНИЕ===\s*(.*?)(?=\n===)', content, re.DOTALL)
            
            # Ищем ответ в решении
            solution = solution_match.group(1).strip() if solution_match else "Решение не найдено"
            
            # Извлекаем ответ из решения
            answer_match = re.search(r'(?:Ответ|ОТВЕТ|ответ)\s*:(.+?)(?=$|\.|\.\s*$|\n\s*\n)', solution, re.IGNORECASE | re.DOTALL)
            answer = answer_match.group(1).strip() if answer_match else ""
            
            # Удаляем "Ответ:" из текста решения, если он найден
            if answer_match:
                solution = re.sub(r'(?:Ответ|ОТВЕТ|ответ)\s*:.+?(?=$|\.|\.\s*$|\n\s*\n)', '', solution, flags=re.IGNORECASE | re.DOTALL)
            
            # Форматируем ответ с LaTeX, если есть
            answer = format_latex_in_answer(answer) if answer else ""
            
            # Оставляем HTML-теги как есть для корректного отображения
            
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
                            displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
                            processEscapes: true
                        }},
                        TeX: {{
                            extensions: ["AMSmath.js", "AMSsymbols.js", "noErrors.js", "noUndefined.js"],
                            equationNumbers: {{ autoNumber: "AMS" }}
                        }},
                        CommonHTML: {{ linebreaks: {{ automatic: true }} }},
                        "HTML-CSS": {{ linebreaks: {{ automatic: true }} }},
                        SVG: {{ linebreaks: {{ automatic: true }} }}
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
                </style>
            </head>
            <body>
                <div class="solution-container">
                    <h1 class="mb-4">Решение задачи</h1>
                    <p><a href="/" class="btn btn-primary mb-3">← Вернуться на главную</a></p>
                    
                    <div class="card">
                        <div class="card-header bg-success text-white">
                            <h5 class="card-title mb-0">Полное решение</h5>
                        </div>
                        <div class="card-body">
                            <div id="solutionContent" class="full-solution">
                                {solution}
                            </div>
                            
                            {f'<div class="answer alert alert-success mt-3"><h5>Ответ:</h5><span>{answer}</span></div>' if answer else ''}
                        </div>
                    </div>
                </div>
                
                <script>
                    // Перерендериваем MathJax для корректного отображения формул
                    if (typeof MathJax !== 'undefined') {{
                        MathJax.Hub.Queue(["Typeset", MathJax.Hub]);
                    }}
                </script>
            </body>
            </html>
            """
        except FileNotFoundError:
            return "Файл с решением не найден. Сначала сгенерируйте задачу."
        except Exception as e:
            return f"Ошибка при чтении решения: {str(e)}"

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
            from app.json_api_helpers import generate_markdown_task
            
            data = request.get_json()
            category = data.get("category")
            subcategory = data.get("subcategory", "")
            difficulty_level = int(data.get("difficulty_level", 3))
            is_basic_level = data.get("is_basic_level", False)
            
            if not category:
                return jsonify({"error": "Категория не указана"}), 400
            
            # Генерируем задачу в формате Markdown
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
            from app.json_api_helpers import generate_json_task, generate_json_markdown_task
            
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