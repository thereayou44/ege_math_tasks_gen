from dotenv import load_dotenv
from app import create_app

# Загружаем переменные окружения
load_dotenv()

# Создаем приложение Flask
app = create_app()

if __name__ == "__main__":
    app.run(debug=True)