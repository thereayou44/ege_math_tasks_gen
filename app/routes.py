from flask import render_template, request, jsonify
import json
import os
import traceback
import re
from bs4 import BeautifulSoup
import random
from dotenv import load_dotenv
from app.task_generator import generate_complete_task, convert_markdown_to_html
from app.json_api_helpers import generate_json_task, generate_json_markdown_task, generate_markdown_task
from app.utils.latex_formatter import format_latex_answer, extract_answer_from_solution, normalize_latex_in_text, escape_html_in_text, format_solution_punctuation
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

# Заменяем функцию format_latex_in_answer на новый импорт format_latex_answer из utils.latex_formatter
format_latex_in_answer = format_latex_answer

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
            
            # Пытаемся получить более качественное решение из debug_response.txt
            solution_from_debug = ""
            debug_response_file = "debug_files/debug_response.txt"
            if os.path.exists(debug_response_file):
                try:
                    with open(debug_response_file, 'r', encoding='utf-8') as f:
                        debug_content = f.read()
                    
                    # Ищем решение в debug_response.txt
                    solution_match = re.search('---РЕШЕНИЕ---\\s*(.*?)(?=\\n---|\\Z)', debug_content, re.DOTALL)
                    if solution_match:
                        solution = solution_match.group(1).strip()
                        print("Решение получено из debug_response.txt")
                        if solution and len(solution) > 20:
                            print("Используем решение из debug_response.txt")
                            # Заменяем решение на более качественное
                            result["solution"] = solution
                            
                            # Сразу ищем ответ в конце решения
                            answer_match = re.search(r'(?:Ответ|ОТВЕТ|ответ):\s*(.+?)(?:$|\.|\.\s*$)', solution, re.IGNORECASE)
                            if answer_match:
                                result["answer"] = format_latex_in_answer(answer_match.group(1).strip())
                except Exception as e:
                    print(f"Ошибка при чтении debug_response.txt: {e}")
            
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
                # Декодируем HTML-сущности (например, &lt; в <)
                hints[i] = decode_html_entities(hints[i])
                # Всегда преобразуем подсказки в HTML, а не только если они содержат markdown
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
            debug_response_file = "debug_files/debug_response.txt"
            solution = ""
            
            # Сначала пытаемся получить решение из debug_response.txt, где форматирование лучше
            if os.path.exists(debug_response_file):
                with open(debug_response_file, 'r', encoding='utf-8') as f:
                    debug_content = f.read()
                    
                # Ищем решение в debug_response.txt
                solution_match = re.search('---РЕШЕНИЕ---\\s*(.*?)(?=\\n---|\\Z)', debug_content, re.DOTALL)
                if solution_match:
                    solution = solution_match.group(1).strip()
                    print("Решение получено из debug_response.txt")
            
            # Если решение не найдено в debug-файле, используем last_generated_task.txt
            if not solution:
                with open("last_generated_task.txt", 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Ищем решение в тексте
                solution_match = re.search('===РЕШЕНИЕ===\\s*(.*?)(?=\\n===)', content, re.DOTALL)
                solution = solution_match.group(1).strip() if solution_match else "Решение не найдено"
                print("Решение получено из last_generated_task.txt")
            
            # Нормализуем LaTeX в тексте решения
            solution = normalize_latex_in_text(solution)
            
            # Экранируем HTML-специальные символы, сохраняя LaTeX
            solution = escape_html_in_text(solution)
            
            # Извлекаем ответ из решения
            answer = extract_answer_from_solution(solution)
            
            # Удаляем "Ответ:" из текста решения, если он найден
            if answer != "См. решение" and answer != "Ответ не найден":
                # Создаем комбинированные шаблоны для удаления ответа из решения
                answer_patterns = [
                    '(?:Ответ|ОТВЕТ|ответ|Окончательный ответ|Итоговый ответ)\\s*:(.+?)(?=$|\\.|\\.\s*$|\\n\\s*\\n)',
                    '\\n\\s*(?:Ответ|ОТВЕТ|ответ|Окончательный ответ|Итоговый ответ)\\s*:\\s*(.+?)(?=$|\\n\\s*\\n|\\n[0-9])',
                    '\\n[0-9]+\\.\\s*(?:Ответ|ОТВЕТ|ответ|Окончательный ответ|Итоговый ответ)\\s*:\\s*(.+?)(?=$|\\n\\s*\\n|\\n[0-9])',
                    '\\n\\s*\\d+\\.\\s*(?:Ответ|ОТВЕТ|ответ|Окончательный ответ|Итоговый ответ)\\s*:\\s*\\n\\s*(.+?)(?=$|\\n\\s*\\n|\\n\\s*\\d+\\.)',
                    '(?:Ответ|ОТВЕТ|ответ|Окончательный ответ|Итоговый ответ)\\s*:\\s*\\n\\s*(.+?)(?=$|\\n\\s*\\n)'
                ]
                for pattern in answer_patterns:
                    solution = re.sub(pattern, '', solution, flags=re.IGNORECASE | re.DOTALL)
            
            # Преобразуем текст решения в HTML для обработки переносов строк и списков
            solution_lines = solution.strip().split('\n')
            solution_html = []
            
            # Сначала обработаем блочные формулы 
            in_block_formula = False
            temp_solution = []
            
            for line in solution_lines:
                stripped_line = line.strip()
                
                # Проверяем, является ли строка началом или концом блочной формулы
                if stripped_line == '$$' or stripped_line.startswith('$$') and stripped_line.endswith('$$'):
                    if in_block_formula:
                        # Конец блочной формулы
                        in_block_formula = False
                    else:
                        # Начало блочной формулы
                        in_block_formula = True
                
                # Если мы в блочной формуле или это отдельная формула, обрабатываем особым образом
                if in_block_formula or (stripped_line.startswith('$$') and stripped_line.endswith('$$')):
                    temp_solution.append(line)
                else:
                    temp_solution.append(line)
            
            solution_lines = temp_solution
            
            for i, line in enumerate(solution_lines):
                # Проверяем, начинается ли строка с нумерованного пункта (1., 2., и т.д.)
                step_match = re.match(r'^(\d+)\.\s+(.+)$', line)
                if step_match:
                    number = step_match.group(1)
                    content = step_match.group(2)
                    # Форматируем знаки препинания после формул
                    content = format_solution_punctuation(content)
                    solution_html.append(f'<p><span class="step-number">{number}.</span> {content}</p>')
                else:
                    # Если это пустая строка, добавляем разрыв параграфа
                    if not line.strip():
                        solution_html.append('<br>')
                    else:
                        # Форматируем знаки препинания после формул
                        line = format_solution_punctuation(line)
                        solution_html.append(f'<p>{line}</p>')
            
            solution_html = '\n'.join(solution_html)
            
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
                                {solution_html}
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