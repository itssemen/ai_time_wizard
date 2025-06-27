import json
import os
import warnings # Add this line
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.validation")
import nltk
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
# from sklearn.ensemble import RandomForestRegressor # Заменим на LGBMRegressor
import lightgbm as lgb # Импортируем LightGBM
from sklearn.metrics import classification_report, mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
import joblib
import numpy as np
import pymorphy2 # для лемматизации

# --- Инициализация лемматизатора ---
morph = pymorphy2.MorphAnalyzer()

def lemmatize_text(text):
    words = text.split() # Простое разделение по пробелам, можно улучшить токенизацией nltk
    lemmatized_words = [morph.parse(word)[0].normal_form for word in words]
    return " ".join(lemmatized_words)

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
# Исправляем путь к файлу данных: он должен быть относительно script_dir, а не script_dir + 'ml/'
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
task_features_for_duration = [] # Будет содержать [лемматизированный_текст_задачи, has_explicit_duration_phrase, длина_текста_задачи]
duration_labels_all = []
task_texts_for_priority_lemmatized = [] # Лемматизированные тексты для приоритета
priority_labels_all = []


for item in dataset: # Используем весь датасет для сбора текстов задач
    for entity in item['entities']:
        if entity['label'] == 'TASK': # Убедимся, что это задача
            original_text = entity['text']
            lemmatized_text = lemmatize_text(original_text)
            text_length = len(original_text.split()) # Длина текста в словах

            task_texts_for_priority_lemmatized.append(lemmatized_text)
            priority_labels_all.append(entity['priority'])

            has_explicit_duration = 1 if entity.get('has_explicit_duration_phrase', False) else 0
            task_features_for_duration.append([lemmatized_text, has_explicit_duration, text_length])
            duration_labels_all.append(entity['duration_minutes'])


print(f"\nСобрано {len(task_features_for_duration)} задач для обучения модели длительности.")
print(f"Собрано {len(task_texts_for_priority_lemmatized)} задач для обучения модели приоритета.")

if not task_features_for_duration:
    print("ОШИБКА: Не найдено ни одной задачи в датасете для обучения модели длительности.")
else:
    # --- 6. Обучение модели регрессии длительности ---
    from sklearn.compose import ColumnTransformer
    # from sklearn.preprocessing import StandardScaler # Может понадобиться для text_length, если разброс большой
    from sklearn.preprocessing import MinMaxScaler


    print("\n--- Обучение модели регрессии длительности ---")
    X_train_dur_features, X_test_dur_features, y_train_dur, y_test_dur = train_test_split(
        task_features_for_duration, duration_labels_all, test_size=0.2, random_state=42
    )

    # Препроцессор для ColumnTransformer
    # Элемент 0: лемматизированный текст
    # Элемент 1: has_explicit_duration_phrase (бинарный)
    # Элемент 2: text_length (числовой)
    duration_preprocessor = ColumnTransformer(
        transformers=[
            ('tfidf', TfidfVectorizer(ngram_range=(1,2), min_df=3), 0), # к лемматизированному тексту
            ('explicit_time_feat', 'passthrough', [1]), # бинарный признак как есть
            ('text_length_feat', MinMaxScaler(), [2]) # числовой признак длины текста, масштабируем
        ],
        remainder='drop'
    )

    duration_pipeline = Pipeline([
        ('preprocessor', duration_preprocessor),
        ('reg', lgb.LGBMRegressor(random_state=42, verbose=-1)) # Используем LGBMRegressor
    ])

    # Расширенная сетка параметров для LGBMRegressor
    duration_parameters = {
        'preprocessor__tfidf__ngram_range': [(1, 1), (1, 2), (1,3)],
        'preprocessor__tfidf__min_df': [3, 5, 7],
        'reg__n_estimators': [100, 200, 300],
        'reg__learning_rate': [0.01, 0.05, 0.1],
        'reg__num_leaves': [20, 31, 40], # Типичные значения для LGBM
        'reg__max_depth': [-1, 5, 10], # -1 означает без ограничений
        'reg__colsample_bytree': [0.7, 0.8, 1.0],
        'reg__subsample': [0.7, 0.8, 1.0],
    }

    duration_model = None
    if not X_train_dur_features:
        print("ОШИБКА (Длительность): Обучающая выборка пуста.")
    else:
        print("Запуск GridSearchCV для модели регрессии длительности (LGBM)...")
        # Увеличим cv до 3, если позволит время, можно и 5. Начнем с 3.
        duration_gs_reg = GridSearchCV(duration_pipeline, duration_parameters, cv=3,
                                       n_jobs=-1, verbose=1, scoring='neg_mean_absolute_error')
        try:
            duration_gs_reg.fit(X_train_dur_features, y_train_dur)
            duration_model = duration_gs_reg.best_estimator_
            print(f"Лучшие параметры для регрессии длительности (LGBM): {duration_gs_reg.best_params_}")
            print(f"Лучший MAE (кросс-валидация, LGBM): {-duration_gs_reg.best_score_:.2f}")

            if X_test_dur_features:
                y_pred_dur = duration_model.predict(X_test_dur_features)
                print("\nОценка регрессии длительности (LGBM, на тестовой выборке):")
                print(f"  Mean Squared Error (MSE): {mean_squared_error(y_test_dur, y_pred_dur):.2f}")
                print(f"  Mean Absolute Error (MAE): {mean_absolute_error(y_test_dur, y_pred_dur):.2f}")
                print(f"  Root Mean Squared Error (RMSE): {np.sqrt(mean_squared_error(y_test_dur, y_pred_dur)):.2f}")
            else:
                print("Тестовая выборка для длительности пуста.")
        except Exception as e:
            print(f"ОШИБКА при обучении модели регрессии длительности (LGBM): {e}")
            duration_model = None


    # --- 7. Обучение классификатора приоритета (с числовыми метками) ---
    print("\n--- Обучение классификатора приоритета ---")
    # Используем task_texts_for_priority_lemmatized
    if not task_texts_for_priority_lemmatized:
        print("ОШИБКА (Приоритет): Нет текстов задач для обучения.")
        priority_model = None
    else:
        X_train_pri, X_test_pri, y_train_pri, y_test_pri = train_test_split(
            task_texts_for_priority_lemmatized, priority_labels_all, test_size=0.2, random_state=42,
            stratify=priority_labels_all if len(set(priority_labels_all)) > 1 else None
        )

        # Пайплайн для классификации приоритета с лемматизацией (уже применена к данным)
        # и LogisticRegression с class_weight='balanced'
        priority_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            # LogisticRegression с class_weight='balanced' и увеличенным max_iter
            ('clf', LogisticRegression(random_state=42, class_weight='balanced', solver='liblinear', max_iter=1000))
        ])

        # Расширенная сетка параметров для LogisticRegression
        priority_parameters = {
            'tfidf__ngram_range': [(1, 1), (1, 2), (1,3)],
            'tfidf__min_df': [3, 5, 7],
            'tfidf__max_df': [0.7, 0.85, 0.95],
            'clf__C': [0.01, 0.1, 1, 10, 100],
            'clf__penalty': ['l1', 'l2'] # liblinear поддерживает l1 и l2
        }

        priority_model = None
        if not X_train_pri:
            print("ОШИБКА (Приоритет): Обучающая выборка пуста.")
        else:
            print("Запуск GridSearchCV для классификатора приоритета (LogisticRegression)...")
            # Увеличим cv до 3 (или 5)
            priority_gs_clf = GridSearchCV(priority_pipeline, priority_parameters, cv=3,
                                         n_jobs=-1, verbose=1, scoring='f1_weighted')
            try:
                priority_gs_clf.fit(X_train_pri, y_train_pri)
                priority_model = priority_gs_clf.best_estimator_
                print(f"Лучшие параметры для приоритета (LogisticRegression): {priority_gs_clf.best_params_}")
                print(f"Лучший F1-weighted (кросс-валидация, LogReg): {priority_gs_clf.best_score_:.2f}")

                if X_test_pri:
                    y_pred_pri = priority_model.predict(X_test_pri)
                    print("\nОтчет по классификации приоритета (LogReg, на тестовой выборке с лучшими параметрами):")
                    labels_pri = sorted(list(set(y_test_pri) | set(y_pred_pri)))
                    print(classification_report(y_test_pri, y_pred_pri, labels=labels_pri, zero_division=0))
                else:
                    print("Тестовая выборка для приоритета пуста.")
            except Exception as e:
                print(f"ОШИБКА при GridSearchCV или оценке классификатора приоритета (LogReg): {e}")
                # Попытка с более простой моделью или параметрами, если основная не удалась
                # (Эта часть может быть упрощена или удалена, если основная модель стабильна)
                print("Попытка обучения классификатора приоритета (LogReg) с параметрами по умолчанию (упрощенными)...")
                try:
                    default_priority_pipeline = Pipeline([
                        ('tfidf', TfidfVectorizer(ngram_range=(1,2), min_df=5)),
                        ('clf', LogisticRegression(random_state=42, class_weight='balanced', solver='liblinear', C=1.0, max_iter=500))
                    ])
                    default_priority_pipeline.fit(X_train_pri, y_train_pri)
                    priority_model = default_priority_pipeline
                    print("Классификатор приоритета (LogReg) с параметрами по умолчанию обучен.")
                    if X_test_pri:
                        y_pred_pri_default = priority_model.predict(X_test_pri)
                        print("\nОтчет по классификации приоритета (LogReg, на тестовой выборке, модель по умолчанию):")
                        labels_pri_default = sorted(list(set(y_test_pri) | set(y_pred_pri_default)))
                        print(classification_report(y_test_pri, y_pred_pri_default, labels=labels_pri_default, zero_division=0))
                except Exception as e_default:
                    print(f"ОШИБКА при обучении классификатора приоритета (LogReg) с параметрами по умолчанию: {e_default}")
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
