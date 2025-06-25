import json
import random
from faker import Faker
from datetime import datetime, timedelta

fake = Faker('ru_RU')

def generate_freeform_task():
    # Варианты начала предложений
    starters = [
        "Сегодня мне нужно",
        "Мой план на день:",
        "Надо не забыть",
        "Вот что я должен сделать:",
        "Задачи на сегодня:",
        "Хочу успеть",
        "Мои дела:",
        "Планы:",
        "Необходимо выполнить:"
    ]
    
    # Шаблоны задач с разным порядком времени
    task_templates = [
        "{action} {duration}",
        "{action} где-то {duration}",
        "{action} минут {duration_num}",
        "{action} ({duration})",
        "{action} - {duration}",
        "{duration} выделю на {action}",
        "на {action} уйдет {duration}",
        "{action} займет {duration}",
        "где-то {duration} потрачу на {action}"
    ]
    
    # Действия
    actions = [
        "пойти в магазин", "сделать домашку", "поделать проект", 
        "посмотреть фильм", "почитать книгу", "погулять", 
        "позвонить маме", "написать отчет", "подготовить презентацию",
        "разобраться с документами", "починить компьютер", "сходить в банк"
    ]
    
    # Временные указания
    time_phrases = [
        "часик", "часа", "часов", "минут", "минутки", "часа полтора",
        "полчаса", "часок", "где-то час", "примерно час", "минут 30",
        "часа два", "три часа", "минут 20", "часа 4", "полтора часа"
    ]
    
    # Генерация текста
    text = random.choice(starters) + " "
    num_tasks = random.randint(2, 5)
    tasks = []
    
    for _ in range(num_tasks):
        action = random.choice(actions)
        duration_phrase = random.choice(time_phrases)
        
        # Преобразование в минуты
        if "час" in duration_phrase:
            if "полтора" in duration_phrase:
                duration = 90
            elif "пол" in duration_phrase:
                duration = 30
            else:
                numbers = [int(s) for s in duration_phrase.split() if s.isdigit()]
                duration = (numbers[0] if numbers else 1) * 60
        else:
            numbers = [int(s) for s in duration_phrase.split() if s.isdigit()]
            duration = numbers[0] if numbers else 30
        
        # Выбор шаблона
        template = random.choice(task_templates)
        task_text = template.format(
            action=action,
            duration=duration_phrase,
            duration_num=duration
        )
        
        # Добавление разделителей
        if random.random() > 0.5:
            task_text += random.choice([", ", ", потом ", ", затем ", ", а еще ", " и "])
        else:
            task_text += random.choice([". ", "! ", "? "])
        
        text += task_text
        
        # Разметка сущностей
        action_start = text.find(action)
        if action_start != -1:
            tasks.append({
                "text": action,
                "start": action_start,
                "end": action_start + len(action),
                "label": "TASK",
                "duration": duration
            })
        
        time_start = text.find(duration_phrase)
        if time_start != -1:
            tasks.append({
                "text": duration_phrase,
                "start": time_start,
                "end": time_start + len(duration_phrase),
                "label": "DURATION",
                "duration": duration
            })
    
    # Удаление последнего разделителя
    text = text.rstrip(", .!?") + "."
    
    # Уникальные сущности (иногда одна задача может быть найдена несколько раз)
    unique_entities = []
    seen = set()
    for ent in tasks:
        key = (ent["start"], ent["end"], ent["label"])
        if key not in seen:
            seen.add(key)
            unique_entities.append({
                "text": ent["text"],
                "label": ent["label"],
                "duration": ent["duration"]
            })
    
    return {
        "text": text,
        "entities": unique_entities
    }

# Генерация датасета
dataset = [generate_freeform_task() for _ in range(300)]

# Сохранение
with open('freeform_task_dataset.json', 'w', encoding='utf-8') as f:
    json.dump(dataset, f, ensure_ascii=False, indent=2)

print(f"Создан датасет со свободными формулировками ({len(dataset)} примеров)")
