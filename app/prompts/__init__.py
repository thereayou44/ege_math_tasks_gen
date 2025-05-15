"""
Модуль с константами и шаблонами для системы.
"""
# Этот файл намеренно оставлен пустым, чтобы избежать циклического импорта
# Все константы и функции находятся в prompts.py 

# Импортируем и экспортируем все константы и функции из prompts.py,
# чтобы поддерживать существующие импорты в проекте
from app.prompts.prompts import (
    HINT_PROMPTS,
    SYSTEM_PROMPT,
    HINT_SYSTEM_PROMPT,
    REGEX_PATTERNS,
    DEFAULT_VISUALIZATION_PARAMS,
    create_complete_task_prompt,
    check_visualization_requirement
) 