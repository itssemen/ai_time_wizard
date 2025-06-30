import json
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning, message="X does not have valid feature names, but RandomForestRegressor was fitted with feature names")
import nltk
import time
from tqdm import tqdm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report, mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
import joblib
import numpy as np
import pymorphy3


morph = pymorphy3.MorphAnalyzer()

def lemmatize_text(text):
    words = text.split()
    lemmatized_words = [morph.parse(word)[0].normal_form for word in words]
    return " ".join(lemmatized_words)


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


def get_tokens_with_char_spans(text):
    tokens = []


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

            if max(token_start, ent_start) < min(token_end, ent_end):
                if first_token_in_entity:
                    tags[i] = f'B-{ent_label}'
                    first_token_in_entity = False
                else:
                    tags[i] = f'I-{ent_label}'
    return tags

sents_tokens_text_for_ner = []
sents_iob_tags_for_ner = []
sents_tokens_info_for_ner = []

for item in dataset:
    text = item['text']
    entities = item['entities']

    tokens_with_spans = get_tokens_with_char_spans(text)
    if not tokens_with_spans:
        continue

    current_item_tokens_text = [t['text'] for t in tokens_with_spans]
    current_item_iob_tags = create_iob_tags(tokens_with_spans, entities)

    sents_tokens_text_for_ner.append(current_item_tokens_text)
    sents_iob_tags_for_ner.append(current_item_iob_tags)
    sents_tokens_info_for_ner.append(tokens_with_spans)

print(f"Обработано {len(sents_tokens_text_for_ner)} предложений для IOB-разметки (NER).")


X_train_sents_ner, X_test_sents_ner, \
y_train_sents_ner, y_test_sents_ner, \
X_train_sents_tokens_info_ner, X_test_sents_tokens_info_ner = train_test_split(
    sents_tokens_text_for_ner, sents_iob_tags_for_ner, sents_tokens_info_for_ner,
    test_size=0.2, random_state=42, stratify=None
)

print(f"NER Обучающая выборка: {len(X_train_sents_ner)} предложений.")
print(f"NER Тестовая выборка: {len(X_test_sents_ner)} предложений.")


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
        features['BOS'] = True
    if i < len(sent)-1:
        next_word = sent[i+1]
        features.update({
            'next_word_lower': next_word.lower(), 'next_word_istitle': next_word.istitle(),
            'next_word_isupper': next_word.isupper(), 'next_word_isdigit': next_word.isdigit(),
        })
    else:
        features['EOS'] = True
    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

print("\nNER: Извлечение признаков из обучающей выборки...")
X_train_features_ner = [sent2features(s) for s in X_train_sents_ner]
print("NER: Извлечение признаков из тестовой выборки...")
X_test_features_ner = [sent2features(s) for s in X_test_sents_ner]


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


task_features_for_duration = []
duration_labels_all = []
duration_phrases_original_all = []
task_texts_for_priority_lemmatized = []
priority_labels_all = []


for item in dataset:
    for entity in item['entities']:
        if entity['label'] == 'TASK':
            original_text = entity['text']
            lemmatized_text = lemmatize_text(original_text)
            text_length = len(original_text.split())

            task_texts_for_priority_lemmatized.append(lemmatized_text)
            priority_labels_all.append(entity['priority'])

            has_explicit_duration = 1 if entity.get('has_explicit_duration_phrase', False) else 0
            explicit_duration_parsed = entity.get('explicit_duration_parsed_minutes', 0)
            task_features_for_duration.append([lemmatized_text, has_explicit_duration, text_length, explicit_duration_parsed])
            duration_labels_all.append(entity['duration_minutes'])
            duration_phrases_original_all.append(entity.get('duration_phrase_original', ''))


print(f"\nСобрано {len(task_features_for_duration)} задач для обучения модели длительности.")
print(f"Собрано {len(task_texts_for_priority_lemmatized)} задач для обучения модели приоритета.")

if not task_features_for_duration:
    print("ОШИБКА: Не найдено ни одной задачи в датасете для обучения модели длительности.")
else:

    from sklearn.compose import ColumnTransformer

    from sklearn.preprocessing import MinMaxScaler


    print("\n--- Обучение модели регрессии длительности ---")

    X_train_dur_features, X_test_dur_features, \
    y_train_dur, y_test_dur, \
    X_train_original_phrases, X_test_original_phrases = train_test_split(
        task_features_for_duration, duration_labels_all, duration_phrases_original_all,
        test_size=0.2, random_state=42
    )






    duration_preprocessor = ColumnTransformer(
        transformers=[
            ('tfidf', TfidfVectorizer(ngram_range=(1,2), min_df=3), 0),
            ('explicit_time_flag_feat', 'passthrough', [1]),
            ('text_length_feat', MinMaxScaler(), [2]),
            ('parsed_duration_feat', MinMaxScaler(), [3])
        ],
        remainder='drop'
    )

    duration_pipeline = Pipeline([
        ('preprocessor', duration_preprocessor),
        ('reg', RandomForestRegressor(random_state=42))
    ])



    duration_parameters = {
        'preprocessor__tfidf__ngram_range': [(1, 1), (1, 2)],
        'preprocessor__tfidf__min_df': [3, 5],

        'reg__n_estimators': [100, 200],
        'reg__max_depth': [None, 10, 20],
        'reg__min_samples_split': [2, 5],
        'reg__min_samples_leaf': [1, 2]
    }

    duration_model = None
    if not X_train_dur_features:
        print("ОШИБКА (Длительность): Обучающая выборка пуста.")
    else:
        print("Запуск GridSearchCV для модели регрессии длительности (RandomForestRegressor)...")

        duration_gs_reg = GridSearchCV(duration_pipeline, duration_parameters, cv=3,
                                       n_jobs=-1, verbose=1, scoring='neg_mean_absolute_error')
        try:
            start_time = time.time()
            with tqdm(total=len(X_train_dur_features), desc="Обучение модели длительности") as pbar:
                duration_gs_reg.fit(X_train_dur_features, y_train_dur)
                pbar.update(len(X_train_dur_features))
            end_time = time.time()
            print(f"Обучение модели длительности заняло: {end_time - start_time:.2f} секунд")
            duration_model = duration_gs_reg.best_estimator_
            print(f"Лучшие параметры для регрессии длительности (RandomForestRegressor): {duration_gs_reg.best_params_}")
            print(f"Лучший MAE (кросс-валидация, RandomForestRegressor): {-duration_gs_reg.best_score_:.2f}")

            if X_test_dur_features:
                y_pred_dur = duration_model.predict(X_test_dur_features)
                print("\nОценка регрессии длительности (RandomForestRegressor, на тестовой выборке):")
                print(f"  Mean Squared Error (MSE): {mean_squared_error(y_test_dur, y_pred_dur):.2f}")
                print(f"  Mean Absolute Error (MAE): {mean_absolute_error(y_test_dur, y_pred_dur):.2f}")
                print(f"  Root Mean Squared Error (RMSE): {np.sqrt(mean_squared_error(y_test_dur, y_pred_dur)):.2f}")


                if X_test_dur_features and y_test_dur and X_test_original_phrases:
                    print("\n--- Анализ предсказаний модели длительности (первые N тестовых примеров) ---")
                    num_duration_samples_to_check = min(10, len(X_test_dur_features))

                    for i in range(num_duration_samples_to_check):
                        original_feature_values = X_test_dur_features[i]
                        true_duration = y_test_dur[i]
                        original_phrase = X_test_original_phrases[i]

                        predicted_duration = y_pred_dur[i]

                        print(f"\n  Пример {i+1}:")
                        print(f"    Лемматизированный текст: '{original_feature_values[0]}'")
                        print(f"    Флаг явной длительности (has_explicit_duration_phrase): {original_feature_values[1]}")
                        print(f"    Длина текста (исходная, до обработки ColumnTransformer): {original_feature_values[2]}")
                        print(f"    Распарсенная явная длительность (минуты, 0 если нет): {original_feature_values[3]}")
                        print(f"    Оригинальная фраза длительности: '{original_phrase}'")
                        print(f"    Эталонная длительность: {true_duration} минут")
                        print(f"    Предсказанная длительность: {predicted_duration:.2f} минут")
                        print(f"    Абсолютная ошибка: {abs(true_duration - predicted_duration):.2f} минут")
                        print("-" * 30)
            else:
                print("Тестовая выборка для длительности пуста (или отсутствуют оригинальные фразы для анализа).")
        except Exception as e:
            print(f"ОШИБКА при обучении модели регрессии длительности (RandomForestRegressor): {e}")
            duration_model = None



    print("\n--- Обучение классификатора приоритета ---")

    if not task_texts_for_priority_lemmatized:
        print("ОШИБКА (Приоритет): Нет текстов задач для обучения.")
        priority_model = None
    else:
        X_train_pri, X_test_pri, y_train_pri, y_test_pri = train_test_split(
            task_texts_for_priority_lemmatized, priority_labels_all, test_size=0.2, random_state=42,
            stratify=priority_labels_all if len(set(priority_labels_all)) > 1 else None
        )



        priority_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),

            ('clf', LinearSVC(random_state=42, class_weight='balanced', max_iter=10000, dual=True))
        ])


        priority_parameters = {
            'tfidf__ngram_range': [(1, 1), (1, 2)],
            'tfidf__min_df': [3, 5],

            'clf__C': [0.1, 1, 10],
            'clf__penalty': ['l2'],
            'clf__loss': ['hinge', 'squared_hinge']
        }

        priority_model = None
        if not X_train_pri:
            print("ОШИБКА (Приоритет): Обучающая выборка пуста.")
        else:
            print("Запуск GridSearchCV для классификатора приоритета (LinearSVC)...")
            priority_gs_clf = GridSearchCV(priority_pipeline, priority_parameters, cv=3,
                                         n_jobs=-1, verbose=1, scoring='f1_weighted')
            try:
                start_time = time.time()
                with tqdm(total=len(X_train_pri), desc="Обучение модели приоритета (LinearSVC)") as pbar:
                    priority_gs_clf.fit(X_train_pri, y_train_pri)
                    pbar.update(len(X_train_pri))
                end_time = time.time()
                print(f"Обучение модели приоритета (LinearSVC) заняло: {end_time - start_time:.2f} секунд")
                priority_model = priority_gs_clf.best_estimator_
                print(f"Лучшие параметры для приоритета (LinearSVC): {priority_gs_clf.best_params_}")
                print(f"Лучший F1-weighted (кросс-валидация, LinearSVC): {priority_gs_clf.best_score_:.2f}")

                if X_test_pri:
                    y_pred_pri = priority_model.predict(X_test_pri)
                    print("\nОтчет по классификации приоритета (LinearSVC, на тестовой выборке с лучшими параметрами):")

                    labels_pri = sorted(list(set(y_train_pri)))


                    print(classification_report(y_test_pri, y_pred_pri, labels=labels_pri, zero_division=0))
                else:
                    print("Тестовая выборка для приоритета пуста.")
            except Exception as e:
                print(f"ОШИБКА при GridSearchCV или оценке классификатора приоритета (LinearSVC): {e}")

                print("Попытка обучения классификатора приоритета (LinearSVC) с параметрами по умолчанию (упрощенными)...")
                try:
                    default_priority_pipeline = Pipeline([
                        ('tfidf', TfidfVectorizer(ngram_range=(1,2), min_df=5)),
                        ('clf', LinearSVC(random_state=42, class_weight='balanced', C=1.0, max_iter=5000, dual=True))
                    ])
                    default_priority_pipeline.fit(X_train_pri, y_train_pri)
                    priority_model = default_priority_pipeline
                    print("Классификатор приоритета (LinearSVC) с параметрами по умолчанию обучен.")
                    if X_test_pri:
                        y_pred_pri_default = priority_model.predict(X_test_pri)
                        print("\nОтчет по классификации приоритета (LinearSVC, на тестовой выборке, модель по умолчанию):")
                        labels_pri_default = sorted(list(set(y_train_pri)))

                        print(classification_report(y_test_pri, y_pred_pri_default, labels=labels_pri_default, zero_division=0))
                except Exception as e_default:
                    print(f"ОШИБКА при обучении классификатора приоритета (LinearSVC) с параметрами по умолчанию: {e_default}")
                    priority_model = None



output_model_dir = os.path.join(script_dir, "models_sklearn")
if not os.path.exists(output_model_dir):
    os.makedirs(output_model_dir)
    print(f"Создана директория: {output_model_dir}")


ner_vectorizer_path = os.path.join(output_model_dir, "ner_vectorizer.pkl")
ner_model_path = os.path.join(output_model_dir, "ner_model.pkl")
if ner_vectorizer: joblib.dump(ner_vectorizer, ner_vectorizer_path); print(f"NER Векторизатор сохранен в: {ner_vectorizer_path}")
if ner_model: joblib.dump(ner_model, ner_model_path); print(f"NER Модель сохранена в: {ner_model_path}")


duration_model_path = os.path.join(output_model_dir, "duration_model.pkl")
priority_model_path = os.path.join(output_model_dir, "priority_model.pkl")

if 'duration_model' in locals() and duration_model:
    joblib.dump(duration_model, duration_model_path)
    print(f"Модель (пайплайн) для длительности сохранена в: {duration_model_path}")
if 'priority_model' in locals() and priority_model:
    joblib.dump(priority_model, priority_model_path)
    print(f"Модель (пайплайн) для приоритета сохранена в: {priority_model_path}")



def iob_tags_to_entities(tokens_info_list, predicted_tags_list):


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
            else:
                if current_entity_tokens:
                     entities.append({
                        "text": " ".join(current_entity_tokens), "label": current_entity_label,
                        "start": current_entity_start_char, "end": tokens_info_list[i-1]['end']
                    })

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
    if current_entity_tokens:
        entities.append({
            "text": " ".join(current_entity_tokens), "label": current_entity_label,
            "start": current_entity_start_char, "end": tokens_info_list[-1]['end']
        })
    return entities


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



    elif not X_train_flat_ner:
         print("ОШИБКА (NER в main): Не удалось извлечь признаки для обучающей выборки NER.")
    elif ner_model is None or y_pred_flat_ner is None:
         print("ОШИБКА (NER в main): Модель NER не была успешно обучена или предсказания NER не были сделаны.")
    else:
        print("Проверка NER не может быть выполнена из-за отсутствия данных или модели.")

    print("\nСкрипт model_training.py завершен.")
