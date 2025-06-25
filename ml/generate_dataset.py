import json
import random
# from faker import Faker # Faker не используется, можно будет удалить, если не понадобится в будущем
# from datetime import datetime, timedelta # datetime не используется

# fake = Faker('ru_RU') # Faker не используется

# Базовые параметры для генерации
DEFAULT_TASK_DURATION_MAP = {
    "позвонить": 15,
    "написать": 20,
    "встреча": 60,
    "совещание": 90,
    "поработать над": 120,
    "сходить в магазин": 45,
    "приготовить ужин": 60,
    "почитать книгу": 60,
    "посмотреть фильм": 120,
    "сделать уборку": 90,
    "заняться спортом": 60,
    "погулять": 45,
    "проверить почту": 10,
    "запланировать": 20,
    "подготовить отчет": 180,
    "провести исследование": 240,
    "обсудить проект": 60,
    "купить билеты": 30,
    "записаться к врачу": 15,
    "починить": 60,
    "изучить": 90,
}

HIGH_PRIORITY_KEYWORDS = ["срочно", "важно", "немедленно", "в первую очередь", "обязательно"]
LOW_PRIORITY_KEYWORDS = ["если будет время", "потом", "не горит", "можно отложить"]

def get_default_duration(action_text):
    action_text_lower = action_text.lower()
    for keyword, duration in DEFAULT_TASK_DURATION_MAP.items():
        if keyword in action_text_lower:
            return duration
    return 60 # Длительность по умолчанию, если не найдено ключевое слово

def determine_priority(action_text, explicit_priority_phrase=None):
    action_text_lower = action_text.lower()
    if explicit_priority_phrase:
        phrase_lower = explicit_priority_phrase.lower()
        if any(k in phrase_lower for k in HIGH_PRIORITY_KEYWORDS):
            return "high"
        if any(k in phrase_lower for k in LOW_PRIORITY_KEYWORDS):
            return "low"

    if any(k in action_text_lower for k in HIGH_PRIORITY_KEYWORDS):
        return "high"
    if any(k in action_text_lower for k in LOW_PRIORITY_KEYWORDS):
        return "low"
    
    # Простая логика на основе содержания задачи для примера
    if "отчет" in action_text_lower or "презентац" in action_text_lower or "документ" in action_text_lower:
        return "high"
    if "посмотреть фильм" in action_text_lower or "почитать книгу" in action_text_lower or "погулять" in action_text_lower:
        return "low"
    return "medium"


def generate_task_item():
    # Действия
    actions = [
        "пойти в магазин за продуктами", "сделать домашку по математике", "поработать над новым проектом",
        "посмотреть интересный фильм", "почитать новую книгу", "погулять в парке",
        "позвонить маме и поздравить с праздником", "написать отчет по командировке", "подготовить презентацию для встречи",
        "разобраться с накопившимися документами", "починить сломанный стул", "сходить в банк и оплатить счета",
        "проверить рабочую почту", "запланировать задачи на неделю", "провести исследование рынка",
        "обсудить проект с коллегами", "купить билеты на поезд", "записаться к врачу на осмотр",
        "изучить документацию по API"
    ]

    # Временные указания и соответствующие минуты
    time_options = [
        ("часик", 60), ("пару часов", 120), ("часов", 60), ("минут 30", 30), ("минут 45", 45),
        ("полчаса", 30), ("часа полтора", 90), ("два часа", 120), ("три часа", 180),
        ("минут 20", 20), ("минут 15", 15), ("минут 50", 50),
        ("на полчасика", 30), ("на часок-другой", 90), # усредняем или берем первое
        ("", None) # Вариант без явного указания времени
    ]

    priority_phrases = ["", "это очень важно", "нужно сделать срочно", "это не горит", "можно сделать потом"]

    action_text = random.choice(actions)
    
    # 50% шанс, что время будет указано явно
    if random.random() < 0.7:
        duration_phrase, duration_minutes = random.choice(time_options)
        if duration_minutes is None: # Если выбран вариант "без явного указания времени"
            duration_minutes = get_default_duration(action_text)
            duration_phrase = "" # Убедимся, что фраза пустая
    else:
        duration_phrase = ""
        duration_minutes = get_default_duration(action_text)

    # 20% шанс на явное указание приоритета фразой
    explicit_priority_phrase = ""
    if random.random() < 0.3:
        explicit_priority_phrase = random.choice(priority_phrases)

    priority = determine_priority(action_text, explicit_priority_phrase)

    return action_text, duration_phrase, duration_minutes, priority, explicit_priority_phrase


def generate_freeform_sentence():
    starters = [
        "Сегодня мне нужно", "Мой план на день:", "Надо не забыть",
        "Вот что я должен сделать:", "Задачи на сегодня:", "Хочу успеть",
        "Мои дела:", "Планы:", "Необходимо выполнить:"
    ]
    
    # Шаблоны для сборки предложения с задачей
    # {action} - сама задача
    # {duration_phrase} - фраза о времени, например "полчаса"
    # {priority_phrase} - фраза о приоритете, например "это очень важно"
    sentence_templates = [
        "{action} {duration_phrase} {priority_phrase}",
        "{action} {priority_phrase} {duration_phrase}",
        "{priority_phrase}, {action} {duration_phrase}",
        "{action} {duration_phrase}",
        "{action} {priority_phrase}",
        "{action}",
        "{duration_phrase} на {action} {priority_phrase}",
        "{duration_phrase} хочу потратить на {action}",
        "Займусь {action} {duration_phrase}",
        "Нужно {action}",
        "Обязательно {action} {duration_phrase}",
        "Не забыть: {action}",
    ]
    
    full_text = random.choice(starters) + " "
    num_tasks = random.randint(1, 4) # Уменьшил макс число задач для читаемости
    entities = []
    current_pos = len(full_text)

    for i in range(num_tasks):
        action, duration_ph, duration_min, task_priority, priority_ph = generate_task_item()
        
        # Выбираем шаблон для текущей задачи
        template = random.choice(sentence_templates)
        
        # Собираем текст задачи, убирая лишние пробелы, если фразы пустые
        task_sentence_parts = []
        
        # Подготавливаем части для форматирования
        format_args = {
            "action": action,
            "duration_phrase": duration_ph.strip(),
            "priority_phrase": priority_ph.strip()
        }
        
        # Форматируем шаблон, но обрабатываем его по частям, чтобы правильно расставить start/end
        # Это упрощенный подход. Для более точного определения start/end может понадобиться более сложная логика
        # или использование инструментов токенизации после сборки строки.
        # Пока что, будем считать, что 'action' это основная сущность.
        
        # Собираем полный текст текущего элемента списка дел
        current_task_full_phrase = template.format(**format_args)
        current_task_full_phrase = ' '.join(current_task_full_phrase.split()) # Убираем двойные пробелы
        
        # Добавляем в общий текст
        full_text += current_task_full_phrase

        # Определяем start/end для action
        # Ищем action в только что добавленном current_task_full_phrase
        # Это важно, если action может повторяться в общем тексте
        action_start_in_segment = current_task_full_phrase.lower().find(action.lower())
        if action_start_in_segment != -1:
            action_start_global = current_pos + action_start_in_segment
            action_end_global = action_start_global + len(action)
            entities.append({
                "text": action, # Текст самой задачи
                "start": action_start_global,
                "end": action_end_global,
                "label": "TASK", # Для NER модели все еще нужен label
                "priority": task_priority,
                "duration_minutes": duration_min,
                "duration_phrase_original": duration_ph # Сохраняем оригинальную фразу для информации
            })

        current_pos += len(current_task_full_phrase)

        # Добавление разделителей
        if i < num_tasks - 1:
            sep = random.choice([", ", ", потом ", ", а еще ", " и "])
            full_text += sep
            current_pos += len(sep)
        else:
            full_text += "."
            current_pos += 1

    return {
        "text": full_text,
        "entities": entities
    }

# Генерация датасета
dataset = [generate_freeform_sentence() for _ in range(300)]

# Сохранение
with open('freeform_task_dataset.json', 'w', encoding='utf-8') as f:
    json.dump(dataset, f, ensure_ascii=False, indent=2)

print(f"Создан датасет ({len(dataset)} примеров) с задачами, приоритетами и длительностью.")

# Пример вывода одного элемента
if dataset:
    print("\nПример сгенерированного элемента:")
    print(json.dumps(dataset[0], ensure_ascii=False, indent=2))
