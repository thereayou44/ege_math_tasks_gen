from flask import Flask, render_template, request, jsonify
import json
from generate_task import process_random_json

app = Flask(__name__)

def load_categories_from_file():
    with open('Data/categories_list.json', 'r', encoding='utf-8') as f:
        categories = json.load(f)
    return categories

@app.route('/')
def index():
    categories = load_categories_from_file()
    return render_template('index.html', categories=categories)

@app.route('/generate_task', methods=['POST'])
def generate_task():
    data = request.get_json()
    category = data.get("category")
    subcategory = data.get("subcategory", "")

    # Если подкатегория не выбрана, передаём None
    task_html = process_random_json(category, subcategory=subcategory if subcategory else None)

    # Возвращаем сгенерированный HTML в виде строки
    return jsonify({"task": str(task_html)})

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
