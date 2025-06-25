import json
import os
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

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
script_dir = os.path.dirname(os.path.abspath(__file__)) # Используем abspath для надежности
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

# --- 2. Подготовка данных (токенизация и IOB-тегирование) ---
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

sents_tokens_text = []
sents_iob_tags = []
sents_tokens_info = []

for item in dataset:
    text = item['text']
    entities = item['entities']
    tokens_with_spans = get_tokens_with_char_spans(text)
    if not tokens_with_spans:
        continue
    current_item_tokens_text = [t['text'] for t in tokens_with_spans]
    current_item_iob_tags = create_iob_tags(tokens_with_spans, entities)
    sents_tokens_text.append(current_item_tokens_text)
    sents_iob_tags.append(current_item_iob_tags)
    sents_tokens_info.append(tokens_with_spans)

print(f"Обработано {len(sents_tokens_text)} предложений для IOB-разметки.")

# --- 3. Разделение данных ---
X_train_sents, X_test_sents, \
y_train_sents, y_test_sents, \
X_train_sents_tokens_info, X_test_sents_tokens_info = train_test_split(
    sents_tokens_text, sents_iob_tags, sents_tokens_info,
    test_size=0.2, random_state=42
)

print(f"Обучающая выборка: {len(X_train_sents)} предложений.")
print(f"Тестовая выборка: {len(X_test_sents)} предложений (с сохраненными tokens_info).")

print("\nПример обработанных данных (первые 2 предложения из обучающей выборки):")
for i in range(min(2, len(X_train_sents))):
    print(f"Предложение {i+1} токены: {X_train_sents[i]}")
    print(f"Предложение {i+1} теги:   {y_train_sents[i]}")

print("\n--- Подготовка данных завершена (токенизация и IOB-теги) ---")

# --- 4. Векторизация и Обучение ---
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

print("\nИзвлечение признаков из обучающей выборки...")
X_train_features = [sent2features(s) for s in X_train_sents]
print("Извлечение признаков из тестовой выборки...")
X_test_features = [sent2features(s) for s in X_test_sents]

X_train_flat = [item for sublist in X_train_features for item in sublist]
y_train_flat = [item for sublist in y_train_sents for item in sublist]
X_test_flat = [item for sublist in X_test_features for item in sublist]
y_test_flat = [item for sublist in y_test_sents for item in sublist]

print(f"Количество токенов в обучающей выборке (для DictVectorizer): {len(X_train_flat)}")
print(f"Количество токенов в тестовой выборке: {len(X_test_flat)}")

vectorizer = None
model = None
y_pred_flat = None

if not X_train_flat:
    print("ОШИБКА: Нет признаков для обучения после обработки.")
else:
    print("\nОбучение DictVectorizer...")
    vectorizer = DictVectorizer(sparse=True)
    X_train_vectorized = vectorizer.fit_transform(X_train_flat)
    print("Векторизация тестовых данных...")
    X_test_vectorized = vectorizer.transform(X_test_flat)
    print(f"Размерность векторизованных обучающих данных: {X_train_vectorized.shape}")
    print(f"Количество признаков (размер словаря DictVectorizer): {len(vectorizer.feature_names_)}")

    print("\nОбучение модели LogisticRegression...")
    model = LogisticRegression(solver='liblinear', multi_class='auto', random_state=42, C=0.1, max_iter=100)
    try:
        model.fit(X_train_vectorized, y_train_flat)
        print("Модель успешно обучена.")
        print("\nПредсказание на тестовой выборке...")
        y_pred_flat = model.predict(X_test_vectorized)
        print(f"Сделано {len(y_pred_flat)} предсказаний для токенов тестовой выборки.")
    except Exception as e:
        print(f"ОШИБКА при обучении или предсказании модели: {e}")

# --- Функции для использования модели и оценки на уровне сущностей ---
def iob_tags_to_entities(tokens_info_list, predicted_tags_list):
    entities = []
    current_entity_tokens = []
    current_entity_start_char = -1
    current_entity_label = None
    # Убедимся, что tokens_info_list и predicted_tags_list имеют одинаковую длину
    if len(tokens_info_list) != len(predicted_tags_list):
        # print(f"Предупреждение: несоответствие длин tokens_info ({len(tokens_info_list)}) и predicted_tags ({len(predicted_tags_list)})")
        return entities # Возвращаем пустой список, если длины не совпадают

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

# --- Основной блок выполнения ---
if __name__ == '__main__':
    if not sents_tokens_text or not sents_iob_tags:
        print("ОШИБКА: Начальная подготовка данных не удалась или данные пусты.")
    elif not X_train_flat:
        print("ОШИБКА: Не удалось извлечь признаки для обучающей выборки.")
    elif model is None or y_pred_flat is None:
         print("ОШИБКА: Модель не была успешно обучена или предсказания не были сделаны.")
    else:
        print("\n--- Векторизация и обучение модели завершены успешно (внутри if __name__ == '__main__') ---")
        
        print("\nОтчет по классификации IOB-тегов на тестовой выборке:")
        if y_test_flat and len(y_pred_flat) > 0:
            labels = sorted(list(set(y_test_flat) | set(y_pred_flat)))
            if not labels:
                print("Нет меток для оценки.")
            else:
                print(classification_report(y_test_flat, y_pred_flat, labels=labels, zero_division=0))
        else:
            print("Нет данных для генерации отчета по классификации.")

        print("\n--- Анализ предсказаний на нескольких тестовых примерах ---")
        num_samples_to_check = 3
        for i in range(min(num_samples_to_check, len(X_test_sents))):
            print(f"\nПример {i+1}:")
            current_tokens_info_for_sample = X_test_sents_tokens_info[i]
            current_token_features_for_sample = X_test_features[i] # Это список словарей признаков

            # Предсказываем теги для текущего тестового предложения
            current_vectorized_features_for_sample = vectorizer.transform(current_token_features_for_sample)
            current_predicted_tags_for_sample = model.predict(current_vectorized_features_for_sample)

            print(f"  Токены: {[t['text'] for t in current_tokens_info_for_sample]}")
            print(f"  Эталонные IOB: {y_test_sents[i]}")
            print(f"  Предсказанные IOB: {list(current_predicted_tags_for_sample)}")

            predicted_entities = iob_tags_to_entities(current_tokens_info_for_sample, list(current_predicted_tags_for_sample))
            print(f"  Предсказанные сущности (start/end из токенизатора): {predicted_entities}")

            reference_entities_from_iob = iob_tags_to_entities(current_tokens_info_for_sample, y_test_sents[i])
            print(f"  Эталонные сущности (из IOB, start/end из токенизатора): {reference_entities_from_iob}")

        output_model_dir = os.path.join(script_dir, "models_sklearn")
        if not os.path.exists(output_model_dir):
            os.makedirs(output_model_dir)
            print(f"Создана директория: {output_model_dir}")

        vectorizer_path = os.path.join(output_model_dir, "vectorizer.pkl")
        model_path = os.path.join(output_model_dir, "ner_model.pkl")

        try:
            if vectorizer: joblib.dump(vectorizer, vectorizer_path)
            print(f"Векторизатор сохранен в: {vectorizer_path}")
            if model: joblib.dump(model, model_path)
            print(f"Модель сохранена в: {model_path}")
        except Exception as e:
            print(f"ОШИБКА при сохранении модели или векторизатора: {e}")

        print("\nСкрипт model_training.py завершен.")
