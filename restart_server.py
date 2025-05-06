#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Скрипт для перезапуска Flask-сервера
"""

import os
import sys
import time
import signal
import subprocess
import psutil

def find_flask_process():
    """Находит процесс Flask"""
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.info['name'] == 'python3' or proc.info['name'] == 'python':
                cmdline = proc.info['cmdline']
                if cmdline and 'app.py' in cmdline:
                    return proc.pid
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return None

def kill_flask_process():
    """Убивает процесс Flask, если он запущен"""
    pid = find_flask_process()
    if pid:
        print(f"Найден процесс Flask с PID {pid}, останавливаем...")
        try:
            os.kill(pid, signal.SIGTERM)
            time.sleep(1)  # Даем процессу время на завершение
            
            # Проверяем, завершился ли процесс
            if psutil.pid_exists(pid):
                print(f"Процесс {pid} все еще работает, применяем SIGKILL...")
                os.kill(pid, signal.SIGKILL)
                time.sleep(1)
        except Exception as e:
            print(f"Ошибка при остановке процесса: {e}")
    else:
        print("Процесс Flask не найден")

def start_flask_server():
    """Запускает Flask-сервер в фоновом режиме"""
    print("Запуск Flask-сервера...")
    
    # Создаем файл для логов
    log_file = open("flask.log", "w")
    
    # Запускаем сервер с перенаправлением вывода в файл логов
    process = subprocess.Popen(
        ["python3", "app.py"],
        stdout=log_file,
        stderr=log_file,
        start_new_session=True  # Запускаем в новой сессии, чтобы процесс продолжал работать после завершения скрипта
    )
    
    print(f"Flask-сервер запущен с PID {process.pid}, вывод перенаправлен в flask.log")
    print("ВНИМАНИЕ: Сервер запущен в фоновом режиме, используйте restart_server.py для перезапуска")

def main():
    print("=== Перезапуск Flask-сервера ===")
    
    # Останавливаем сервер, если он запущен
    kill_flask_process()
    
    # Запускаем сервер
    start_flask_server()
    
    print("Перезапуск выполнен успешно!")
    print("Откройте браузер и перейдите по адресу: http://localhost:5000/")

if __name__ == "__main__":
    main() 