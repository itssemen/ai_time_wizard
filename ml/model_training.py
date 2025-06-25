import spacy
from spacy.training import Example
from spacy.util import minibatch, compounding
import random
import json
# sklearn и joblib больше не нужны для этого скрипта в его текущей конфигурации
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.ensemble import RandomForestClassifier
# import joblib
# from datetime import datetime, timedelta # не используется
import warnings
warnings.filterwarnings('ignore') # Оставим, т.к. spacy может выдавать ворнинги

# 1. Загрузка данных
with open('freeform_task_dataset.json', 'r', encoding='utf-8') as f:
    dataset = json.load(f)

# Пример из датасета
if dataset:
    print("Пример текста из датасета:", dataset[0]["text"])
    print("Сущности из датасета:", dataset[0]["entities"])
else:
    print("Датасет пуст или не загружен.")
    exit()

# 2. Подготовка данных для NER
# Теперь каждая сущность в датасете - это TASK с атрибутами priority и duration_minutes.
# Для NER нам нужны только start, end и label ("TASK").
def convert_to_spacy_format(data):
    spacy_data = []
    for item in data:
        entities_for_spacy = []
        for ent in item["entities"]:
            # Убедимся, что start и end существуют и являются числами
            if isinstance(ent.get("start"), int) and isinstance(ent.get("end"), int) and ent.get("label"):
                entities_for_spacy.append((ent["start"], ent["end"], ent["label"]))
            else:
                print(f"Пропущена некорректная сущность в элементе: {item['text']}, сущность: {ent}")

        if not item["text"]:
            print(f"Пропущен элемент с пустым текстом.")
            continue

        spacy_data.append((item["text"], {"entities": entities_for_spacy}))
    return spacy_data

# Разделение данных
random.shuffle(dataset)
split_ratio = 0.8
# Убедимся, что датасет не слишком мал для разделения
if len(dataset) < 5: # Например, минимальный размер для разделения
    print("Датасет слишком мал для разделения на обучающую и тестовую выборки.")
    # Можно либо завершить выполнение, либо использовать весь датасет для обучения
    # Для примера, используем все для обучения если данных мало
    train_data_source = dataset
    test_data_source = dataset
else:
    split_idx = int(split_ratio * len(dataset))
    train_data_source = dataset[:split_idx]
    test_data_source = dataset[split_idx:]

train_data = convert_to_spacy_format(train_data_source)
test_data = convert_to_spacy_format(test_data_source)


if not train_data:
    print("Нет данных для обучения после конвертации. Проверьте формат 'freeform_task_dataset.json'.")
    exit()

# 3. Создание и обучение NER модели
# Загружаем предобученную модель ru_core_news_sm как основу
try:
    nlp = spacy.load("ru_core_news_sm")
    print("Модель 'ru_core_news_sm' загружена.")
except OSError:
    print("Не удалось загрузить модель 'ru_core_news_sm'. Убедитесь, что она установлена: python -m spacy download ru_core_news_sm")
    exit()

ner_pipe_name = "ner"
if ner_pipe_name in nlp.pipe_names:
    ner = nlp.get_pipe(ner_pipe_name)
    print(f"Используется существующий NER компонент из 'ru_core_news_sm'. Текущие метки: {ner.labels}")
else:
    ner = nlp.add_pipe(ner_pipe_name, last=True)
    print("Добавлен новый NER компонент.")

# Добавление нашего лейбла, если его еще нет
if "TASK" not in ner.labels:
    ner.add_label("TASK")
    print("Метка 'TASK' добавлена в NER.")
# ner.add_label("DURATION") # DURATION больше не извлекается как отдельная сущность


# Инициализация или дообучение
# Для SpaCy 3.x, nlp.initialize() используется для инициализации весов новых компонентов
# и для подготовки к обучению. Если мы дообучаем, существующие веса сохраняются.
# Передача get_examples помогает инициализировать компоненты, такие как 'tok2vec', если они есть.
# Предоставляем train_data как примеры для инициализации
def get_training_examples_for_initialize():
    examples = []
    for text, annotations in train_data: # train_data это список кортежей (text, annotation_dict)
        if not text: continue
        doc = nlp.make_doc(text) # Создаем Doc объект из текста
        examples.append(Example.from_dict(doc, annotations))
    if not examples:
        print("Внимание: get_training_examples_for_initialize не сгенерировал ни одного Example. Проверьте train_data.")
        # Возвращаем пустой список, чтобы избежать ошибки, но это указывает на проблему с данными.
        # Или можно вызвать исключение, если это критично.
        # Для nlp.initialize лучше, если он получит хотя бы один пример.
        # Если train_data пуст, это вызовет проблемы раньше.
        # Если train_data не пуст, но все тексты пустые, это тоже проблема.

        # Попробуем создать один фиктивный Example, если train_data пуст, чтобы избежать ошибки,
        # но это костыль и указывает на проблемы с данными.
        # Лучше убедиться, что train_data всегда содержит валидные данные.
        # Сейчас, если examples пуст, nlp.initialize может выдать ошибку дальше.
        # Пусть пока так, чтобы увидеть, если это произойдет.
        pass
    return examples

optimizer = nlp.initialize(get_examples=get_training_examples_for_initialize)


# Функция для оценки (исправленная)
def evaluate_ner_model(nlp_model, examples):
    scorer = spacy.scorer.Scorer()
    example_list = []
    for text, annotations in examples:
        if not text: # Пропускаем пустые тексты, если они как-то попали
            continue
        pred_doc = nlp_model(text)
        # Создаем Example объект для оценки
        # Убедимся, что annotations это словарь {'entities': [(start, end, label), ...]}
        if isinstance(annotations, dict) and "entities" in annotations:
            example = Example.from_dict(nlp_model.make_doc(text), annotations)
            example_list.append(example)
        else:
            print(f"Некорректный формат аннотаций для текста: {text}")

    if not example_list:
        return {"ents_p": 0, "ents_r": 0, "ents_f": 0, "ents_per_type": {}}
        
    scores = scorer.score(example_list)
    return scores


# Обучение с прогрессом
print("Начало обучения NER модели...")
n_iter = 30 # Количество эпох

# Создаем директорию для моделей, если она не существует
import os
output_dir = "models/ner_model"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

best_f_score = 0.0

for epoch in range(n_iter):
    random.shuffle(train_data)
    losses = {}
    
    # Пакетная обработка данных
    batches = minibatch(train_data, size=compounding(4.0, 32.0, 1.001))
    for batch in batches:
        examples = []
        for text, annotations in batch:
            if not text: continue # Пропуск пустых текстов
            # Создаем Example объект для обучения
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, annotations)
            examples.append(example)
        
        if examples: # Убедимся, что есть что обновлять
            nlp.update(examples, sgd=optimizer, losses=losses, drop=0.35)
    
    # Оценка после каждой эпохи (на небольшом подмножестве для скорости)
    # Важно: передавать корректно отформатированные данные в evaluate_ner_model
    if test_data:
        # Используем весь тестовый набор для более точной оценки
        scores = evaluate_ner_model(nlp, test_data)
        f_score = scores.get('ents_f', 0)
        print(
            f"Epoch {epoch + 1}/{n_iter}, Loss: {losses.get('ner', 0.0):.3f}, "
            f"P: {scores.get('ents_p', 0.0):.3f}, R: {scores.get('ents_r', 0.0):.3f}, F: {f_score:.3f}"
        )
        # Сохранение лучшей модели
        if f_score > best_f_score:
            best_f_score = f_score
            nlp.to_disk(output_dir)
            print(f"Сохранена новая лучшая модель с F-мерой: {f_score:.3f} в {output_dir}")
        else:
            print(f"F-мера {f_score:.3f} не лучше предыдущей {best_f_score:.3f}. Модель не сохранена в эту эпоху.")
    else:
         print(f"Epoch {epoch + 1}/{n_iter}, Loss: {losses.get('ner',0.0):.3f}. Тестовые данные отсутствуют для оценки.")

model_saved_during_training = best_f_score > 0

# Если тестовых данных не было, но обучение проводилось, сохраняем модель после последней эпохи
if not test_data and n_iter > 0 and not model_saved_during_training:
    nlp.to_disk(output_dir)
    print(f"Модель сохранена после {n_iter} эпох обучения в {output_dir} (тестовых данных не было, сохранение по F-мере не производилось).")
    model_saved_during_training = True
elif not test_data and n_iter > 0 and model_saved_during_training:
    # Это условие может быть избыточным, если best_f_score инициализирован < 0
    # и сохранение происходит только при улучшении.
    # Но если вдруг best_f_score остался на начальном 0.0 и test_data не было,
    # то первое сохранение произойдет выше. Если же test_data не было,
    # но каким-то образом best_f_score стал >0 (что невозможно без test_data в текущей логике),
    # то модель уже сохранена.
    pass


# Раздел обучения классификатора приоритетов и векторизатора УДАЛЕН,
# так как приоритеты и длительность теперь являются частью генерируемого датасета
# и извлекаются вместе с задачей.

# 5. Финальное сообщение о сохранении
if n_iter == 0:
    print("Обучение не проводилось (0 эпох). Модель не сохранена.")
elif not model_saved_during_training:
    # Это может произойти, если F-мера никогда не улучшалась и test_data были,
    # или если n_iter > 0, test_data не было, но сохранение все равно не произошло (маловероятно с текущей логикой).
    # В таком случае, можно принудительно сохранить последнюю версию.
    if not os.path.exists(output_dir): # На всякий случай, если директория не была создана
        os.makedirs(output_dir)
    nlp.to_disk(output_dir)
    print(f"Финальная NER модель принудительно сохранена в {output_dir} (F-мера не улучшалась или не оценивалась).")
else:
    if best_f_score > 0 : # Если оценка была
        print(f"Лучшая NER модель была сохранена в {output_dir} с F-мерой: {best_f_score:.3f}")
    else: # Если оценки не было (нет test_data), но модель сохранена
        print(f"NER модель была сохранена в {output_dir} (оценка F-меры не проводилась).")


print("Обучение NER модели завершено.")

# Пример использования обученной модели для извлечения задач и их атрибутов
# Этот блок можно раскомментировать для проверки загрузки и использования модели
# print("\n--- Пример использования обученной модели ---")
# if os.path.exists(output_dir):
#     print(f"Загрузка модели из {output_dir}...")
#     nlp_loaded = spacy.load(output_dir)
#     print("Модель успешно загружена.")

#     # Возьмем пример из тестового набора (если он есть) или из тренировочного
#     sample_text_data = test_data[0] if test_data else (train_data[0] if train_data else None)

#     if sample_text_data:
#         sample_text = sample_text_data[0]
#         print(f"\nИсходный текст для теста: \"{sample_text}\"")

#         doc = nlp_loaded(sample_text)
#         print("Извлеченные задачи (сущности NER):")
#         if not doc.ents:
#             print("  Модель не извлекла ни одной задачи из этого текста.")

#         for ent in doc.ents:
#             print(f"- Текст задачи: '{ent.text}', Метка: {ent.label_} (позиции: {ent.start_char}-{ent.end_char})")

#             # Поиск оригинальной сущности для демонстрации атрибутов
#             original_entity_info = None
#             # Ищем в исходном полном датасете (train_data_source + test_data_source) или просто dataset
#             # Это упрощенный поиск для примера. В реальном приложении структура данных может быть другой.
#             source_to_search = dataset # Используем весь исходный датасет
#             for item in source_to_search:
#                 if item["text"] == sample_text:
#                     for original_ent in item["entities"]:
#                         if original_ent["start"] == ent.start_char and \
#                            original_ent["end"] == ent.end_char and \
#                            original_ent["text"] == ent.text: # Сравниваем и текст сущности
#                             original_entity_info = original_ent
#                             break
#                     if original_entity_info:
#                         break

#             if original_entity_info:
#                 print(f"  Priority (из исходных данных): {original_entity_info.get('priority')}")
#                 print(f"  Duration (из исходных данных): {original_entity_info.get('duration_minutes')} минут")
#                 print(f"  Original duration phrase: {original_entity_info.get('duration_phrase_original')}")
#             else:
#                 # Это может произойти, если модель выделила сущность, которой не было в исходной разметке
#                 # или если текст был модифицирован/не найден в исходном датасете.
#                 print("  Дополнительная информация (priority/duration) не найдена в исходных данных для этой сущности.")
#     else:
#         print("Нет данных для демонстрации использования модели.")
# else:
#     print(f"Директория с моделью {output_dir} не найдена. Не удалось загрузить модель для примера.")

print("\nСкрипт model_training.py завершен.")
