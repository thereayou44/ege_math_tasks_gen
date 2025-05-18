
// Функция для корректной обработки LaTeX-формул
function improveLatexRendering() {
    // Функция для повторной обработки формул через MathJax
    function retypeset(element) {
        if (typeof MathJax !== 'undefined') {
            // Сначала сбрасываем предыдущее форматирование
            if (element) {
                MathJax.Hub.Queue(["Remove", MathJax.Hub, element]);
            }
            
            // Затем применяем новое форматирование
            MathJax.Hub.Queue(["Typeset", MathJax.Hub, element]);
        }
    }
    
    // Функция для обработки формул с двойными долларами
    function fixDoubleDelimiters(container) {
        // Находим все элементы, которые содержат $$
        if (!container) return;
        
        // Найти все текстовые узлы в контейнере
        const walker = document.createTreeWalker(
            container,
            NodeFilter.SHOW_TEXT,
            null,
            false
        );
        
        const nodesToReplace = [];
        let textNode;
        
        // Собираем все текстовые узлы
        while (textNode = walker.nextNode()) {
            if (textNode.nodeValue.includes('$$')) {
                nodesToReplace.push(textNode);
            }
        }
        
        // Заменяем текстовые узлы формулами
        nodesToReplace.forEach(node => {
            const parent = node.parentNode;
            const content = node.nodeValue;
            
            // Разбиваем содержимое текстового узла на формулы и обычный текст
            const parts = content.split(/((?:\$\$)[^$]+(?:\$\$))/g);
            
            // Создаем фрагмент документа для новых узлов
            const fragment = document.createDocumentFragment();
            
            parts.forEach(part => {
                if (part.startsWith('$$') && part.endsWith('$$')) {
                    // Создаем элемент для формулы
                    const formula = document.createElement('div');
                    formula.className = 'math-formula';
                    formula.innerHTML = part;
                    fragment.appendChild(formula);
                } else if (part.trim()) {
                    // Создаем текстовый узел для обычного текста
                    fragment.appendChild(document.createTextNode(part));
                }
            });
            
            // Заменяем старый узел новыми
            parent.replaceChild(fragment, node);
        });
    }
    
    // Корректируем двойные разделители в контейнерах
    fixDoubleDelimiters(document.getElementById('taskOutput'));
    fixDoubleDelimiters(document.getElementById('hintOutput'));
    fixDoubleDelimiters(document.getElementById('answerOutput'));
    fixDoubleDelimiters(document.getElementById('solutionContent'));
    
    // Повторная обработка формул через MathJax
    retypeset('taskOutput');
    retypeset('hintOutput');
    retypeset('answerOutput');
    retypeset('solutionContent');
}

// Модификация существующих функций для вызова улучшенной обработки формул
// Модификация функции showHint
function enhanceShowHint(originalShowHint) {
    return function() {
        originalShowHint.apply(this, arguments);
        
        // Таймаут для гарантированной отработки после обновления DOM
        setTimeout(improveLatexRendering, 100);
    }
}

// Модификация функции showSolution
function enhanceShowSolution(originalShowSolution) {
    return function() {
        originalShowSolution.apply(this, arguments);
        
        // Таймаут для гарантированной отработки после обновления DOM
        setTimeout(improveLatexRendering, 100);
    }
}

// Модификация функции generateTask
function enhanceGenerateTask(originalGenerateTask) {
    return function() {
        const result = originalGenerateTask.apply(this, arguments);
        
        // Таймаут для гарантированной отработки после обновления DOM
        setTimeout(improveLatexRendering, 100);
        
        return result;
    }
}

// Назначение улучшенных версий функций
document.addEventListener('DOMContentLoaded', function() {
    // Сохраняем оригинальные функции
    const originalGenerateTask = window.generateTask;
    const originalShowHint = window.showHint;
    const originalShowSolution = window.showSolution;
    
    // Заменяем на улучшенные версии
    if (typeof originalGenerateTask === 'function') {
        window.generateTask = enhanceGenerateTask(originalGenerateTask);
    }
    
    if (typeof originalShowHint === 'function') {
        window.showHint = enhanceShowHint(originalShowHint);
    }
    
    if (typeof originalShowSolution === 'function') {
        window.showSolution = enhanceShowSolution(originalShowSolution);
    }
});