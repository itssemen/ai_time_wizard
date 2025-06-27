import json
import random
# from faker import Faker # Faker не используется, можно будет удалить, если не понадобится в будущем
# from datetime import datetime, timedelta # datetime не используется

# fake = Faker('ru_RU') # Faker не используется

# --- Расширенные параметры для генерации ---

# Категории задач и примерные действия с типичными длительностями (min, max) в минутах
TASK_CATEGORIES = {
    "короткие_коммуникации": {
        "actions": [
            "позвонить {кому}", "написать сообщение {кому}", "ответить на письмо {от кого}",
            "скинуть файл {кому}", "уточнить детали по {чему}", "согласовать {что} с {кем}",
            "проверить почту", "быстро ответить {кому}"
        ],
        "duration_range": (5, 25),
        "keywords_for_duration": {"позвонить": (5,15), "написать": (5,20), "проверить почту": (5,10)}
    },
    "быт_еда": {
        "actions": [
            "приготовить {что}", "заказать еду", "помыть посуду", "загрузить стирку",
            "развесить белье", "полить цветы", "вынести мусор", "сходить в магазин за {чем}",
            "купить продукты"
        ],
        "duration_range": (15, 75),
        "keywords_for_duration": {"приготовить": (30,90), "сходить в магазин": (20,60)}
    },
    "уборка_организация": {
        "actions": [
            "сделать уборку в {где}", "пропылесосить", "вытереть пыль", "разобрать {что}",
            "навести порядок на столе", "организовать файлы на компьютере", "рассортировать документы"
        ],
        "duration_range": (20, 120),
        "keywords_for_duration": {"сделать уборку": (45,180), "разобрать": (30,90)}
    },
    "работа_проекты": {
        "actions": [
            "поработать над проектом {каким}", "написать код для {чего}", "отладить программу",
            "подготовить отчет по {теме}", "составить план {чего}", "провести исследование {темы}",
            "проанализировать данные {какие}", "написать статью о {чем}", "решить рабочую задачу"
        ],
        "duration_range": (45, 240),
        "keywords_for_duration": {"отчет": (90,300), "исследование": (120,360), "код": (60,240)}
    },
    "встречи_созвоны": {
        "actions": [
            "провести встречу с {кем}", "созвониться с {кем} по {вопросу}", "участвовать в совещании",
            "обсудить {что} с командой", "планерка", "стендап"
        ],
        "duration_range": (30, 120),
        "keywords_for_duration": {"совещание": (45,120), "планерка": (20,60)}
    },
    "обучение_развитие": {
        "actions": [
            "изучить {что}", "пройти онлайн-курс по {теме}", "посмотреть вебинар о {чем}",
            "почитать профессиональную литературу", "сделать упражнения по {предмету}", "выучить новые слова на {языке}"
        ],
        "duration_range": (30, 180),
        "keywords_for_duration": {"изучить": (45,120), "курс": (60,180)}
    },
    "личные_дела_поручения": {
        "actions": [
            "записаться к {кому}", "оплатить счета", "сходить в {место}", "забрать {что} из {откуда}",
            "починить {что}", "купить билеты на {что}", "подать документы в {куда}", "поздравить {кого} с {чем}"
        ],
        "duration_range": (15, 90),
        "keywords_for_duration": {"записаться": (10,30), "оплатить счета": (15,45)}
    },
    "отдых_хобби": {
        "actions": [
            "почитать книгу", "посмотреть фильм {какой}", "послушать музыку", "поиграть в {игру}",
            "заняться хобби ({каким})", "погулять {где}", "помедитировать", "поспать {сколько} часов", "отдохнуть"
        ],
        "duration_range": (30, 180),
        "keywords_for_duration": {"посмотреть фильм": (90,150), "погулять": (30,90), "поспать": (360, 540)} # Сон - особая категория
    },
    "спорт_здоровье": {
        "actions": [
            "сделать зарядку", "пойти на тренировку в {место}", "пробежка на {дистанция}", "заняться йогой",
            "посетить врача", "принять лекарства"
        ],
        "duration_range": (20, 120),
        "keywords_for_duration": {"тренировка": (45,90), "пробежка": (30,75)}
    },
}

# Заполнители для действий
PLACEHOLDERS = {
    "кому": ["маме", "другу", "коллеге", "начальнику", "клиенту", "подруге", "брату"],
    "от кого": ["банка", "партнера", "руководства", "клиента"],
    "что": ["этот вопрос", "детали проекта", "план работ", "условия договора", "презентацию", "ужин", "отчет"],
    "чем": ["хлебом", "молоком", "овощами", "фруктами", "необходимым"],
    "где": ["комнате", "квартире", "на кухне", "в парке", "в центре города", "на рабочем месте"],
    "каким": ["новым", "интересным", "сложным", "важным", "текущим"],
    "чего": ["следующей недели", "маркетинговой кампании", "разработки фичи", "отпуска"],
    "теме": ["продаж", "нового продукта", "анализа конкурентов", "квартального отчета"],
    "какие": ["финансовые", "статистические", "пользовательские"],
    "вопросу": ["сотрудничества", "поставки", "текущих задач", "обновлений"],
    "предмету": ["английскому языку", "программированию", "истории"],
    "языке": ["английском", "немецком", "испанском"],
    "место": ["банк", "почту", "химчистку", "спортзал", "бассейн"],
    "откуда": ["ремонта", "прачечной", "пункта выдачи"],
    "какой": ["новый", "документальный", "любимый"],
    "игру": ["компьютерную игру", "настолку", "шахматы"],
    "дистанция": ["5 км", "короткую дистанцию", "обычную дистанцию"],
    "сколько": ["7", "8", "немного", "час"] # для сна
}

HIGH_PRIORITY_KEYWORDS = ["срочно", "важно", "немедленно", "в первую очередь", "обязательно", "критично", "дедлайн"]
LOW_PRIORITY_KEYWORDS = ["если будет время", "потом", "не горит", "можно отложить", "второстепенно", "когда-нибудь"]

def get_realistic_duration(action_text_template, action_text_concrete):
    action_text_lower = action_text_concrete.lower()

    # Сначала ищем по ключевым словам в конкретной фразе, если они есть в категории
    for category_name, cat_data in TASK_CATEGORIES.items():
        if action_text_template in cat_data["actions"]: # Проверяем, что шаблон из этой категории
            for keyword, (dur_min, dur_max) in cat_data.get("keywords_for_duration", {}).items():
                if keyword in action_text_lower:
                    return random.randint(dur_min // 5, dur_max // 5) * 5 # Возвращаем с шагом 5 минут
            # Если специфичных ключевых слов не найдено, используем общий диапазон категории
            cat_min, cat_max = cat_data["duration_range"]
            return random.randint(cat_min // 5, cat_max // 5) * 5

    # Общий дефолт, если категория не найдена (маловероятно при текущей логике)
    return random.choice([30, 45, 60, 90])


def determine_priority(action_text_template, action_text_concrete, explicit_priority_phrase=None):
    action_text_concrete_lower = action_text_concrete.lower()
    # high: 2, medium: 1, low: 0
    if explicit_priority_phrase:
        phrase_lower = explicit_priority_phrase.lower()
        if any(k in phrase_lower for k in HIGH_PRIORITY_KEYWORDS):
            return 2 # high
        if any(k in phrase_lower for k in LOW_PRIORITY_KEYWORDS):
            return 0 # low

    if any(k in action_text_concrete_lower for k in HIGH_PRIORITY_KEYWORDS):
        return 2 # high
    if any(k in action_text_concrete_lower for k in LOW_PRIORITY_KEYWORDS):
        return 0 # low
    
    # Эвристики на основе категории задачи, если нет явных ключевых слов
    # (можно добавить, если TASK_CATEGORIES будут содержать информацию о типичном приоритете)
    # Например, задачи из "работа_проекты" могут чаще быть high/medium
    # Задачи из "отдых_хобби" чаще low/medium

    # Пример простой логики на основе содержания задачи (можно оставить или улучшить)
    if any(kw in action_text_concrete_lower for kw in ["отчет", "презентац", "документ", "дедлайн", "срочная задача"]):
        return 2 # high
    if any(kw in action_text_concrete_lower for kw in ["посмотреть фильм", "почитать книгу", "погулять", "отдохнуть", "хобби"]):
        return 0 # low
    return 1 # medium


def generate_task_item():
    # Выбираем случайную категорию и из нее случайный шаблон действия
    chosen_category_name = random.choice(list(TASK_CATEGORIES.keys()))
    chosen_category = TASK_CATEGORIES[chosen_category_name]
    action_template = random.choice(chosen_category["actions"])

    # Заполняем плейсхолдеры в шаблоне действия
    action_text_concrete = action_template
    # Ищем все плейсхолдеры вида {key}
    placeholders_in_template = [ph[1:-1] for ph in action_template.split() if ph.startswith("{") and ph.endswith("}")]

    filled_placeholders = {}
    for ph_key in placeholders_in_template:
        if ph_key in PLACEHOLDERS:
            chosen_value = random.choice(PLACEHOLDERS[ph_key])
            action_text_concrete = action_text_concrete.replace("{" + ph_key + "}", chosen_value, 1)
            filled_placeholders[ph_key] = chosen_value

    # Временные указания и соответствующие минуты
    # Расширенный список, включающий более разнообразные и числовые варианты
    time_options = [
        ("часик", 60), ("пару часов", 120), ("полчаса", 30), ("часа полтора", 90),
        ("два часа", 120), ("три часа", 180), ("четыре часа", 240),
        ("минут 10", 10), ("минут 15", 15), ("минут 20", 20), ("минут 25", 25),
        ("минут 30", 30), ("минут 40", 40), ("минут 45", 45), ("минут 50", 50),
        ("1 час", 60), ("2 часа", 120), ("3 часа", 180), ("1.5 часа", 90), ("2.5 часа", 150),
        ("1ч", 60), ("2ч", 120), ("1ч30м", 90), ("45м", 45), ("90 мин", 90),
        ("на полчасика", 30), ("на часок-другой", random.choice([60, 90, 120])),
        ("немного времени на", random.choice([15, 20, 30])),
        ("посвятить этому где-то", random.choice([45, 60, 75, 90])),
        ("", None) # Вариант без явного указания времени (будет использован get_realistic_duration)
    ]
    
    # Вероятность явного указания времени (например, 70%)
    if random.random() < 0.7:
        duration_phrase, duration_minutes_from_phrase = random.choice(time_options)
        if duration_minutes_from_phrase is None: # Если выбран вариант "без явного указания времени"
            actual_duration_minutes = get_realistic_duration(action_template, action_text_concrete)
            duration_phrase = "" # Убедимся, что фраза пустая
        else:
            actual_duration_minutes = duration_minutes_from_phrase
            # Небольшая вариативность для фраз типа "часик", "пару часов"
            if duration_phrase in ["часик", "пару часов", "на часок-другой"]:
                 actual_duration_minutes = random.randint(max(15, actual_duration_minutes - 15) // 5, (actual_duration_minutes + 15) // 5) * 5
    else:
        duration_phrase = ""
        actual_duration_minutes = get_realistic_duration(action_template, action_text_concrete)

    # Фразы приоритета
    priority_phrases = ["", "это очень важно", "нужно сделать срочно", "это не горит", "можно сделать потом",
                        "обязательно выполни", "критически важно", "если останется время"]

    # Вероятность явного указания приоритета фразой (например, 30%)
    explicit_priority_phrase = ""
    if random.random() < 0.3:
        explicit_priority_phrase = random.choice(priority_phrases)

    priority = determine_priority(action_template, action_text_concrete, explicit_priority_phrase)

    # Убедимся, что текст задачи не пустой, если плейсхолдеры не заполнились (маловероятно)
    if not action_text_concrete.strip():
        action_text_concrete = "какое-то дело" # Запасной вариант

    return action_text_concrete, duration_phrase, actual_duration_minutes, priority, explicit_priority_phrase


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
    num_tasks = random.randint(1, 5) # Немного увеличим максимальное число задач
    entities = []
    current_pos = len(full_text)

    for i in range(num_tasks):
        action_text_concrete, duration_ph, duration_min, task_priority, priority_ph = generate_task_item()
        
        # Выбираем шаблон для текущей задачи
        template = random.choice(sentence_templates)
        
        # Подготавливаем части для форматирования
        format_args = {
            "action": action_text_concrete, # Используем конкретный текст задачи
            "duration_phrase": duration_ph.strip(),
            "priority_phrase": priority_ph.strip()
        }
        
        # Собираем полный текст текущего элемента списка дел
        # Убираем лишние пробелы, которые могли возникнуть из-за пустых фраз
        current_task_full_phrase = template.format(**format_args)
        current_task_full_phrase = ' '.join(filter(None, current_task_full_phrase.split(' ')))
        current_task_full_phrase = current_task_full_phrase.replace(" ,", ",").replace(" .", ".") # Чистим пунктуацию
        
        # Добавляем в общий текст
        full_text += current_task_full_phrase

        # Определяем start/end для action_text_concrete
        # Ищем action_text_concrete в только что добавленном current_task_full_phrase
        action_start_in_segment = -1
        # Пробуем найти точное совпадение сначала
        # Для более надежного поиска можно использовать fuzzy matching или учитывать возможные вариации из-за форматирования
        # но для генерации, где мы контролируем процесс, это должно работать нормально.

        # Чтобы поиск был точнее, мы ищем action_text_concrete в последнем добавленном сегменте (current_task_full_phrase)
        # и затем смещаем на current_pos.
        # Однако, если action_text_concrete сам является результатом форматирования (например, содержит duration_phrase),
        # то лучше искать его "как есть".

        # Ищем action_text_concrete в full_text, начиная с current_pos.
        # Это более надежно, если action_text_concrete не содержит других фраз.
        # Если action_text_concrete может содержать duration_phrase или priority_phrase (из-за шаблона),
        # то нужно извлекать "чистую" задачу. Для простоты, сейчас считаем, что action_text_concrete - это и есть задача.

        search_text_lower = full_text[current_pos:].lower()
        action_to_find_lower = action_text_concrete.lower()

        action_start_in_segment = search_text_lower.find(action_to_find_lower)

        if action_start_in_segment != -1:
            action_start_global = current_pos + action_start_in_segment
            action_end_global = action_start_global + len(action_text_concrete) # Длина оригинального, не lower-case

            # Проверка на пересечение с предыдущими сущностями (простой вариант)
            valid_entity = True
            for prev_entity in entities:
                if max(prev_entity['start'], action_start_global) < min(prev_entity['end'], action_end_global):
                    valid_entity = False
                    # print(f"Warning: Overlapping entity found for '{action_text_concrete}'. Skipping.") # Для отладки
                    break

            if valid_entity:
                entities.append({
                    "text": action_text_concrete, # Текст самой задачи
                    "start": action_start_global,
                    "end": action_end_global,
                    "label": "TASK",
                    "priority": task_priority,
                    "duration_minutes": duration_min,
                    "duration_phrase_original": duration_ph
                })

        current_pos += len(current_task_full_phrase)

        # Добавление разделителей
        if i < num_tasks - 1:
            sep = random.choice([", ", "; ", ", потом ", ", а еще ", " и ", ". Затем ", ". После этого "])
            full_text += sep
            current_pos += len(sep)
        else:
            if not full_text.endswith((".", "!", "?")):
                 full_text += random.choice([".", "!"])
                 current_pos += 1

    # Финальная очистка текста (убрать лишние пробелы по краям, двойные пробелы)
    full_text = ' '.join(full_text.split())

    return {
        "text": full_text,
        "entities": entities # Отдаем только непересекающиеся сущности
    }

# Генерация датасета
NUM_EXAMPLES = 1200 # Увеличено количество примеров
dataset = []
generated_count = 0
attempt_limit = NUM_EXAMPLES * 2 # Чтобы не уйти в бесконечный цикл, если много коллизий

while generated_count < NUM_EXAMPLES and attempt_limit > 0:
    example = generate_freeform_sentence()
    # Проверяем, что есть хотя бы одна сущность, чтобы избежать пустых примеров
    if example["entities"]:
        dataset.append(example)
        generated_count += 1
    attempt_limit -=1

if attempt_limit <= 0:
    print(f"Предупреждение: Достигнут лимит попыток генерации. Сгенерировано {generated_count} из {NUM_EXAMPLES} примеров.")

# Сохранение
output_file_path = 'freeform_task_dataset.json'
with open(output_file_path, 'w', encoding='utf-8') as f:
    json.dump(dataset, f, ensure_ascii=False, indent=2)

print(f"Создан датасет ({len(dataset)} примеров) и сохранен в '{output_file_path}'.")

# Пример вывода одного элемента
if dataset:
    print("\nПример сгенерированного элемента:")
    print(json.dumps(dataset[0], ensure_ascii=False, indent=2))
