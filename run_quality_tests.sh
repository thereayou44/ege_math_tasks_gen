#!/bin/bash

# Запуск тестов качества с проверкой 100 уравнений
cd $(dirname $0)
echo 'Запуск тестов качества генерации уравнений (100 тестов)'
python3 -m tests.quality_tests.run_quality_tests 