import json
import os
import nltk
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
import joblib
import numpy as np

# --- Загрузка ресурсов NLTK ---
def download_nltk_resource(resource_name, resource_path_to_check):
    try:
        nltk.data.find(resource_path_to_check)
        print(f"Ресурс nltk '{resource_path_to_check}' найден.")
    except LookupError:
        print(f"Загрузка ресурса nltk '{resource_name}' (для '{resource_path_to_check}')...")
        nltk.download(resource_name, quiet=True)
        try:
            nltk.data.find(resource_path_to_check)
            print(f"Ресурс nltk '{resource_name}' успешно загружен и найден по пути '{resource_path_to_check}'.")
        except LookupError:
            print(f"ПРЕДУПРЕЖДЕНИЕ: Ресурс nltk '{resource_name}' был загружен, но все еще не найден по пути '{resource_path_to_check}'.")

download_nltk_resource('punkt', 'tokenizers/punkt')

# --- 1. Загрузка данных ---
script_dir = os.path.dirname(os.path.abspath(__file__))
data_file_path = os.path.join(script_dir, 'freeform_task_dataset.json')

dataset = []
try:
    with open(data_file_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    print(f"Загружено {len(dataset)} записей из '{data_file_path}'")
except FileNotFoundError:
    print(f"ОШИБКА: Файл данных '{data_file_path}' не найден.")
    exit()
except json.JSONDecodeError:
    print(f"ОШИБКА: Не удалось декодировать JSON из файла '{data_file_path}'.")
    exit()

if not dataset:
    print("Датасет пуст. Завершение работы.")
    exit()

# --- 2. Подготовка данных для NER (токенизация и IOB-тегирование) ---
def get_tokens_with_char_spans(text):
    tokens = []
    # Используем простой WhitespaceTokenizer, т.к. он сохраняет соответствие символов
    # Для более сложных случаев можно использовать nltk.word_tokenize и align_tokens
    tokenizer = nltk.tokenize.WhitespaceTokenizer()
    for start, end in tokenizer.span_tokenize(text):
        token_text = text[start:end]
        tokens.append({'text': token_text, 'start': start, 'end': end})
    return tokens

def create_iob_tags(text_tokens_with_spans, entities):
    tags = ['O'] * len(text_tokens_with_spans)
    for entity in entities:
        ent_start = entity['start']
        ent_end = entity['end']
        ent_label = entity['label']
        first_token_in_entity = True
        for i, token in enumerate(text_tokens_with_spans):
            token_start = token['start']
            token_end = token['end']
            # Проверка на пересечение токена и сущности
            if max(token_start, ent_start) < min(token_end, ent_end):
                if first_token_in_entity:
                    tags[i] = f'B-{ent_label}'
                    first_token_in_entity = False
                else:
                    tags[i] = f'I-{ent_label}'
    return tags

sents_tokens_text_for_ner = []
sents_iob_tags_for_ner = []
sents_tokens_info_for_ner = [] # Для отладки и анализа NER

for item in dataset:
    text = item['text']
    entities = item['entities'] # Здесь entities содержат 'label': 'TASK'

    tokens_with_spans = get_tokens_with_char_spans(text)
    if not tokens_with_spans:
        continue

    current_item_tokens_text = [t['text'] for t in tokens_with_spans]
    current_item_iob_tags = create_iob_tags(tokens_with_spans, entities)

    sents_tokens_text_for_ner.append(current_item_tokens_text)
    sents_iob_tags_for_ner.append(current_item_iob_tags)
    sents_tokens_info_for_ner.append(tokens_with_spans)

print(f"Обработано {len(sents_tokens_text_for_ner)} предложений для IOB-разметки (NER).")

# --- 3. Разделение данных для NER ---
X_train_sents_ner, X_test_sents_ner, \
y_train_sents_ner, y_test_sents_ner, \
X_train_sents_tokens_info_ner, X_test_sents_tokens_info_ner = train_test_split(
    sents_tokens_text_for_ner, sents_iob_tags_for_ner, sents_tokens_info_for_ner,
    test_size=0.2, random_state=42, stratify=None # Stratify может быть проблематичен для списков списков
)

print(f"NER Обучающая выборка: {len(X_train_sents_ner)} предложений.")
print(f"NER Тестовая выборка: {len(X_test_sents_ner)} предложений.")


# --- 4. Векторизация и Обучение NER модели ---
def word2features(sent, i):
    word = sent[i]
    features = {
        'bias': 1.0, 'word_lower': word.lower(), 'word_istitle': word.istitle(),
        'word_isupper': word.isupper(), 'word_isdigit': word.isdigit(),
        'word_suffix_2': word[-2:], 'word_prefix_2': word[:2],
    }
    if i > 0:
        prev_word = sent[i-1]
        features.update({
            'prev_word_lower': prev_word.lower(), 'prev_word_istitle': prev_word.istitle(),
            'prev_word_isupper': prev_word.isupper(), 'prev_word_isdigit': prev_word.isdigit(),
        })
    else:
        features['BOS'] = True # Начало предложения
    if i < len(sent)-1:
        next_word = sent[i+1]
        features.update({
            'next_word_lower': next_word.lower(), 'next_word_istitle': next_word.istitle(),
            'next_word_isupper': next_word.isupper(), 'next_word_isdigit': next_word.isdigit(),
        })
    else:
        features['EOS'] = True # Конец предложения
    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

print("\nNER: Извлечение признаков из обучающей выборки...")
X_train_features_ner = [sent2features(s) for s in X_train_sents_ner]
print("NER: Извлечение признаков из тестовой выборки...")
X_test_features_ner = [sent2features(s) for s in X_test_sents_ner]

# "Распрямляем" списки для обучения
X_train_flat_ner = [item for sublist in X_train_features_ner for item in sublist]
y_train_flat_ner = [item for sublist in y_train_sents_ner for item in sublist]
X_test_flat_ner = [item for sublist in X_test_features_ner for item in sublist]
y_test_flat_ner = [item for sublist in y_test_sents_ner for item in sublist]

ner_vectorizer = None
ner_model = None
y_pred_flat_ner = None

if not X_train_flat_ner:
    print("ОШИБКА (NER): Нет признаков для обучения после обработки.")
else:
    print("\nNER: Обучение DictVectorizer...")
    ner_vectorizer = DictVectorizer(sparse=True)
    X_train_vectorized_ner = ner_vectorizer.fit_transform(X_train_flat_ner)
    print("NER: Векторизация тестовых данных...")
    X_test_vectorized_ner = ner_vectorizer.transform(X_test_flat_ner)

    print("\nNER: Обучение модели LogisticRegression...")
    # Можно добавить GridSearchCV для подбора C
    ner_model = LogisticRegression(solver='liblinear', multi_class='auto', random_state=42, C=0.1, max_iter=200)
    try:
        ner_model.fit(X_train_vectorized_ner, y_train_flat_ner)
        print("NER: Модель успешно обучена.")
        if X_test_vectorized_ner.shape[0] > 0:
            print("\nNER: Предсказание на тестовой выборке...")
            y_pred_flat_ner = ner_model.predict(X_test_vectorized_ner)
        else:
            print("NER: Тестовая выборка пуста после векторизации, предсказание не выполняется.")
            y_pred_flat_ner = []

    except Exception as e:
        print(f"ОШИБКА (NER) при обучении или предсказании модели: {e}")

# --- 5. Подготовка данных для классификаторов длительности и приоритета ---
task_texts_all = []
duration_labels_all = []
priority_labels_all = []

def duration_to_category(minutes):
    if minutes <= 15: return "0-15 min"
    if minutes <= 30: return "16-30 min"
    if minutes <= 60: return "31-60 min"
    if minutes <= 120: return "61-120 min"
    return ">120 min"

for item in dataset: # Используем весь датасет для сбора текстов задач
    for entity in item['entities']:
        if entity['label'] == 'TASK': # Убедимся, что это задача
            task_texts_all.append(entity['text'])
            duration_labels_all.append(duration_to_category(entity['duration_minutes']))
            priority_labels_all.append(entity['priority'])

print(f"\nСобрано {len(task_texts_all)} задач для обучения классификаторов длительности/приоритета.")

if not task_texts_all:
    print("ОШИБКА: Не найдено ни одной задачи в датасете для обучения классификаторов длительности/приоритета.")
    # В этом случае дальнейшее обучение классификаторов бессмысленно
else:
    # --- 6. Обучение классификатора длительности ---
    print("\n--- Обучение классификатора длительности ---")
    X_train_dur, X_test_dur, y_train_dur, y_test_dur = train_test_split(
        task_texts_all, duration_labels_all, test_size=0.2, random_state=42,
        stratify=duration_labels_all if len(set(duration_labels_all)) > 1 else None
    )

    # Пайплайн для удобства: TF-IDF + Классификатор
    # Попробуем LinearSVC, он часто хорошо работает на текстовых данных
    duration_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1,2), min_df=2)), # min_df=2 чтобы отсеять редкие слова
        ('clf', LinearSVC(random_state=42, C=0.1, dual="auto")) # dual="auto" or False for newer sklearn
    ])

    # Параметры для GridSearchCV (опционально, но полезно)
    # duration_parameters = {
    #     'tfidf__max_df': (0.75, 1.0),
    #     'clf__C': (0.1, 1, 10)
    # }
    # duration_gs_clf = GridSearchCV(duration_pipeline, duration_parameters, cv=3, n_jobs=-1, verbose=1)
    # duration_gs_clf.fit(X_train_dur, y_train_dur)
    # duration_model = duration_gs_clf.best_estimator_
    # print(f"Лучшие параметры для длительности: {duration_gs_clf.best_params_}")

    try:
        if not X_train_dur:
            print("ОШИБКА (Длительность): Обучающая выборка пуста.")
            duration_model = None
        else:
            duration_model = duration_pipeline
            duration_model.fit(X_train_dur, y_train_dur)
            print("Классификатор длительности обучен.")
            if X_test_dur:
                y_pred_dur = duration_model.predict(X_test_dur)
                print("\nОтчет по классификации длительности (на тестовой выборке):")
                print(classification_report(y_test_dur, y_pred_dur, zero_division=0))
            else:
                print("Тестовая выборка для длительности пуста.")
    except Exception as e:
        print(f"ОШИБКА при обучении классификатора длительности: {e}")
        duration_model = None


    # --- 7. Обучение классификатора приоритета ---
    print("\n--- Обучение классификатора приоритета ---")
    X_train_pri, X_test_pri, y_train_pri, y_test_pri = train_test_split(
        task_texts_all, priority_labels_all, test_size=0.2, random_state=42,
        stratify=priority_labels_all if len(set(priority_labels_all)) > 1 else None
    )

    priority_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1,2), min_df=2)),
        ('clf', LinearSVC(random_state=42, C=0.1, dual="auto"))
    ])

    # priority_parameters = {
    #     'tfidf__max_df': (0.75, 1.0),
    #     'clf__C': (0.1, 1, 10)
    # }
    # priority_gs_clf = GridSearchCV(priority_pipeline, priority_parameters, cv=3, n_jobs=-1, verbose=1)
    # priority_gs_clf.fit(X_train_pri, y_train_pri)
    # priority_model = priority_gs_clf.best_estimator_
    # print(f"Лучшие параметры для приоритета: {priority_gs_clf.best_params_}")
    try:
        if not X_train_pri:
            print("ОШИБКА (Приоритет): Обучающая выборка пуста.")
            priority_model = None
        else:
            priority_model = priority_pipeline
            priority_model.fit(X_train_pri, y_train_pri)
            print("Классификатор приоритета обучен.")
            if X_test_pri:
                y_pred_pri = priority_model.predict(X_test_pri)
                print("\nОтчет по классификации приоритета (на тестовой выборке):")
                print(classification_report(y_test_pri, y_pred_pri, zero_division=0))
            else:
                print("Тестовая выборка для приоритета пуста.")
    except Exception as e:
        print(f"ОШИБКА при обучении классификатора приоритета: {e}")
        priority_model = None


# --- Сохранение всех моделей ---
output_model_dir = os.path.join(script_dir, "models_sklearn")
if not os.path.exists(output_model_dir):
    os.makedirs(output_model_dir)
    print(f"Создана директория: {output_model_dir}")

# NER модели
ner_vectorizer_path = os.path.join(output_model_dir, "ner_vectorizer.pkl")
ner_model_path = os.path.join(output_model_dir, "ner_model.pkl")
if ner_vectorizer: joblib.dump(ner_vectorizer, ner_vectorizer_path); print(f"NER Векторизатор сохранен в: {ner_vectorizer_path}")
if ner_model: joblib.dump(ner_model, ner_model_path); print(f"NER Модель сохранена в: {ner_model_path}")

# Модели классификации (длительность и приоритет)
# Векторизаторы TF-IDF являются частью пайплайнов, поэтому сохраняем только пайплайны
duration_model_path = os.path.join(output_model_dir, "duration_model.pkl")
priority_model_path = os.path.join(output_model_dir, "priority_model.pkl")

if 'duration_model' in locals() and duration_model:
    joblib.dump(duration_model, duration_model_path)
    print(f"Модель (пайплайн) для длительности сохранена в: {duration_model_path}")
if 'priority_model' in locals() and priority_model:
    joblib.dump(priority_model, priority_model_path)
    print(f"Модель (пайплайн) для приоритета сохранена в: {priority_model_path}")


# --- Функции для использования NER модели и оценки на уровне сущностей (для отладки NER) ---
# (Оставлено для возможной отладки NER, если понадобится)
def iob_tags_to_entities(tokens_info_list, predicted_tags_list):
    # ... (реализация этой функции не меняется, можно скопировать из предыдущей версии, если нужна)
    # Эта функция в основном для оценки NER, не для обучения классификаторов атрибутов
    entities = []
    current_entity_tokens = []
    current_entity_start_char = -1
    current_entity_label = None
    if len(tokens_info_list) != len(predicted_tags_list):
        return entities

    for i, token_info in enumerate(tokens_info_list):
        tag = predicted_tags_list[i]
        token_text = token_info['text']
        if tag.startswith('B-'):
            if current_entity_tokens:
                entities.append({
                    "text": " ".join(current_entity_tokens), "label": current_entity_label,
                    "start": current_entity_start_char, "end": tokens_info_list[i-1]['end']
                })
            current_entity_tokens = [token_text]
            current_entity_start_char = token_info['start']
            current_entity_label = tag[2:]
        elif tag.startswith('I-'):
            if current_entity_tokens and tag[2:] == current_entity_label:
                current_entity_tokens.append(token_text)
            else: # Ошибка разметки или начало новой сущности без B-
                if current_entity_tokens:
                     entities.append({
                        "text": " ".join(current_entity_tokens), "label": current_entity_label,
                        "start": current_entity_start_char, "end": tokens_info_list[i-1]['end']
                    })
                # Начать новую сущность, даже если это I- без B-
                current_entity_tokens = [token_text]
                current_entity_start_char = token_info['start']
                current_entity_label = tag[2:]
        elif tag == 'O':
            if current_entity_tokens:
                entities.append({
                    "text": " ".join(current_entity_tokens), "label": current_entity_label,
                    "start": current_entity_start_char, "end": tokens_info_list[i-1]['end']
                })
                current_entity_tokens = []
                current_entity_start_char = -1
                current_entity_label = None
    if current_entity_tokens: # Завершить последнюю сущность
        entities.append({
            "text": " ".join(current_entity_tokens), "label": current_entity_label,
            "start": current_entity_start_char, "end": tokens_info_list[-1]['end']
        })
    return entities

# --- Основной блок выполнения (проверка NER) ---
if __name__ == '__main__':
    print("\n--- Проверка NER модели ---")
    if ner_model and ner_vectorizer and y_test_flat_ner and y_pred_flat_ner is not None and len(y_pred_flat_ner) == len(y_test_flat_ner):
        print("\nОтчет по классификации IOB-тегов NER (на тестовой выборке):")
        labels_ner = sorted(list(set(y_test_flat_ner) | set(y_pred_flat_ner)))
        if not labels_ner:
            print("Нет меток NER для оценки.")
        else:
            print(classification_report(y_test_flat_ner, y_pred_flat_ner, labels=labels_ner, zero_division=0))

        print("\n--- Анализ предсказаний NER на нескольких тестовых примерах ---")
        num_samples_to_check = min(3, len(X_test_sents_ner))
        for i in range(num_samples_to_check):
            print(f"\nNER Пример {i+1}:")
            current_tokens_info_for_sample = X_test_sents_tokens_info_ner[i]
            current_token_features_for_sample = X_test_features_ner[i]

            if not current_token_features_for_sample:
                print("  Нет признаков для этого примера.")
                continue

            current_vectorized_features_for_sample = ner_vectorizer.transform(current_token_features_for_sample)
            current_predicted_tags_for_sample = ner_model.predict(current_vectorized_features_for_sample)

            print(f"  Токены: {[t['text'] for t in current_tokens_info_for_sample]}")
            print(f"  Эталонные IOB: {y_test_sents_ner[i]}")
            print(f"  Предсказанные IOB: {list(current_predicted_tags_for_sample)}")

            # predicted_entities = iob_tags_to_entities(current_tokens_info_for_sample, list(current_predicted_tags_for_sample))
            # print(f"  Предсказанные сущности (NER): {predicted_entities}")
    elif not X_train_flat_ner:
         print("ОШИБКА (NER в main): Не удалось извлечь признаки для обучающей выборки NER.")
    elif ner_model is None or y_pred_flat_ner is None:
         print("ОШИБКА (NER в main): Модель NER не была успешно обучена или предсказания NER не были сделаны.")
    else:
        print("Проверка NER не может быть выполнена из-за отсутствия данных или модели.")

    print("\nСкрипт model_training.py завершен.")
