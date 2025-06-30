import json
import os
import joblib
import nltk
import numpy as np
from sklearn.metrics import classification_report, mean_squared_error, mean_absolute_error
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pymorphy3
import math


morph = pymorphy3.MorphAnalyzer()


script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(script_dir)
project_root = os.path.dirname(base_dir)


MODELS_DIR = os.path.join(base_dir, "models_sklearn")
DATA_FILE_PATH = os.path.join(base_dir, "freeform_task_dataset.json")

NER_VECTORIZER_PATH = os.path.join(MODELS_DIR, "ner_vectorizer.pkl")
NER_MODEL_PATH = os.path.join(MODELS_DIR, "ner_model.pkl")
DURATION_MODEL_PATH = os.path.join(MODELS_DIR, "duration_model.pkl")
PRIORITY_MODEL_PATH = os.path.join(MODELS_DIR, "priority_model.pkl")


def download_nltk_resource_if_needed(resource_name, resource_path_to_check):
    try:
        nltk.data.find(resource_path_to_check)
        print(f"NLTK resource '{resource_path_to_check}' found.")
    except LookupError:
        print(f"Downloading NLTK resource '{resource_name}' (for '{resource_path_to_check}')...")
        nltk.download(resource_name, quiet=True)
        try:
            nltk.data.find(resource_path_to_check)
            print(f"NLTK resource '{resource_name}' successfully downloaded and found at '{resource_path_to_check}'.")
        except LookupError:
            print(f"WARNING: NLTK resource '{resource_name}' was downloaded but still not found at '{resource_path_to_check}'.")

download_nltk_resource_if_needed('punkt', 'tokenizers/punkt')



def load_dataset(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        print(f"Loaded {len(dataset)} records from '{file_path}'")
        return dataset
    except FileNotFoundError:
        print(f"ERROR: Data file '{file_path}' not found.")
        return []
    except json.JSONDecodeError:
        print(f"ERROR: Could not decode JSON from file '{file_path}'.")
        return []


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

def iob_tags_to_entities(tokens_info_list, predicted_tags_list):
    entities = []
    current_entity_tokens = []
    current_entity_start_char = -1
    current_entity_label = None
    if len(tokens_info_list) != len(predicted_tags_list):
        print("Warning: Mismatch between token count and predicted tags count.")
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


def lemmatize_text_for_model(text: str) -> str:
    """Лемматизирует текст для подачи в ML модель."""
    words = text.split()
    lemmatized_words = [morph.parse(word)[0].normal_form for word in words]
    return " ".join(lemmatized_words)

def extract_duration_features(task_text: str) -> list:
    """Извлекает признаки из текста задачи для модели предсказания длительности."""
    lemmatized_text = lemmatize_text_for_model(task_text)
    text_length = len(task_text.split())

    has_explicit_duration = 0
    explicit_duration_parsed_minutes = 0.0

    return [lemmatized_text, has_explicit_duration, float(text_length), explicit_duration_parsed_minutes]


def load_models():
    models = {}
    try:
        models['ner_vectorizer'] = joblib.load(NER_VECTORIZER_PATH)
        models['ner_model'] = joblib.load(NER_MODEL_PATH)
        models['duration_model'] = joblib.load(DURATION_MODEL_PATH)
        models['priority_model'] = joblib.load(PRIORITY_MODEL_PATH)
        print("All models loaded successfully.")
    except FileNotFoundError as e:
        print(f"ERROR: Could not load model - {e}. Ensure models are trained and paths are correct.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while loading models: {e}")
        return None
    return models


def test_ner_model(ner_model, ner_vectorizer, dataset):
    print("\n--- Testing NER Model ---")
    if not dataset:
        print("NER test skipped: No data.")
        return

    all_true_iob_tags_flat = []
    all_pred_iob_tags_flat = []
    all_tokens_info_sents = []

    for item_idx, item in enumerate(dataset):
        text = item['text']
        true_entities = item['entities']

        tokens_with_spans = get_tokens_with_char_spans(text)
        if not tokens_with_spans:
            continue
        all_tokens_info_sents.append(tokens_with_spans)

        true_iob_tags = create_iob_tags(tokens_with_spans, true_entities)
        all_true_iob_tags_flat.extend(true_iob_tags)

        token_texts = [t['text'] for t in tokens_with_spans]
        features = sent2features(token_texts)

        if not features:
            all_pred_iob_tags_flat.extend(['O'] * len(true_iob_tags))
            continue

        try:
            vectorized_features = ner_vectorizer.transform(features)
            pred_iob_tags = ner_model.predict(vectorized_features)
            all_pred_iob_tags_flat.extend(pred_iob_tags)
        except Exception as e:
            print(f"Error during NER prediction for item {item_idx}: {e}")

            all_pred_iob_tags_flat.extend(['O'] * len(true_iob_tags))


    if not all_true_iob_tags_flat or not all_pred_iob_tags_flat:
        print("No IOB tags generated for NER evaluation.")
        return

    if len(all_true_iob_tags_flat) != len(all_pred_iob_tags_flat):
        print(f"Warning: Mismatch in length of true IOB tags ({len(all_true_iob_tags_flat)}) and predicted IOB tags ({len(all_pred_iob_tags_flat)}). NER report might be inaccurate.")

        min_len = min(len(all_true_iob_tags_flat), len(all_pred_iob_tags_flat))
        all_true_iob_tags_flat = all_true_iob_tags_flat[:min_len]
        all_pred_iob_tags_flat = all_pred_iob_tags_flat[:min_len]
        if not min_len:
             print("Cannot generate NER classification report due to length mismatch leading to zero length.")
             return


    print("\nNER IOB Tags Classification Report (on full dataset):")
    labels_ner = sorted(list(set(all_true_iob_tags_flat) | set(all_pred_iob_tags_flat)))
    if not labels_ner:
        print("No NER labels found for classification report.")
    else:
        print(classification_report(all_true_iob_tags_flat, all_pred_iob_tags_flat, labels=labels_ner, zero_division=0))

    print("\nNER: Example of predicted entities (first few dataset entries):")


    start_idx = 0
    for i, tokens_info_sent in enumerate(all_tokens_info_sents[:min(3, len(all_tokens_info_sents))]):
        num_tokens_in_sent = len(tokens_info_sent)
        pred_tags_for_sent = all_pred_iob_tags_flat[start_idx : start_idx + num_tokens_in_sent]
        start_idx += num_tokens_in_sent

        true_tags_for_sent = create_iob_tags(tokens_info_sent, dataset[i]['entities'])

        predicted_entities = iob_tags_to_entities(tokens_info_sent, pred_tags_for_sent)
        true_entities_text = [f"'{e['text']}' ({e['label']})" for e in dataset[i]['entities']]

        print(f"\nExample {i+1}:")
        print(f"  Text: {dataset[i]['text']}")
        print(f"  Tokens: {[t['text'] for t in tokens_info_sent]}")
        print(f"  True IOB: {true_tags_for_sent}")
        print(f"  Pred IOB: {list(pred_tags_for_sent)}")
        print(f"  True Entities: {', '.join(true_entities_text) if true_entities_text else 'None'}")
        print(f"  Predicted Entities: {predicted_entities if predicted_entities else 'None'}")

def test_duration_model(duration_model, dataset):
    print("\n--- Testing Duration Model ---")
    if not dataset:
        print("Duration test skipped: No data.")
        return

    task_texts = []
    true_durations = []

    for item in dataset:
        for entity in item['entities']:
            if entity['label'] == 'TASK':
                task_texts.append(entity['text'])
                true_durations.append(entity['duration_minutes'])

    if not task_texts:
        print("No 'TASK' entities found in the dataset for duration model testing.")
        return


    features_for_duration_model = []
    for task_text in task_texts:
        features_for_duration_model.append(extract_duration_features(task_text))

    if not features_for_duration_model:
        print("No features extracted for duration model testing.")
        return

    try:

        pred_durations = duration_model.predict(features_for_duration_model)
        print("\nDuration Model Evaluation (on all tasks from dataset):")
        print(f"  Mean Squared Error (MSE): {mean_squared_error(true_durations, pred_durations):.2f}")
        print(f"  Mean Absolute Error (MAE): {mean_absolute_error(true_durations, pred_durations):.2f}")
        print(f"  Root Mean Squared Error (RMSE): {np.sqrt(mean_squared_error(true_durations, pred_durations)):.2f}")

        print("\nDuration Model: Example predictions (first few tasks):")
        for i in range(min(5, len(task_texts))):
            print(f"  Task: \"{task_texts[i]}\", True Duration: {true_durations[i]}, Predicted Duration: {pred_durations[i]:.1f}")

    except Exception as e:
        print(f"Error during duration model prediction or evaluation: {e}")


def test_priority_model(priority_model, dataset):
    print("\n--- Testing Priority Model ---")
    if not dataset:
        print("Priority test skipped: No data.")
        return

    task_texts = []
    true_priorities = []

    for item in dataset:
        for entity in item['entities']:
            if entity['label'] == 'TASK' and 'priority' in entity:
                task_texts.append(entity['text'])
                priority_val = entity['priority']

                if isinstance(priority_val, str) and priority_val.lower() in ["low", "medium", "high"]:
                    true_priorities.append(priority_val.lower())
                else:

                    print(f"Warning: Unexpected priority value '{priority_val}' for task '{entity['text']}'. Skipping this task for priority test.")
                    task_texts.pop()
                    continue


    if not task_texts:
        print("No 'TASK' entities with valid priorities found in the dataset for priority model testing.")
        return

    try:

        pred_priorities = priority_model.predict(task_texts)


        if not all(isinstance(p, str) for p in pred_priorities):
            print("Warning: Predicted priorities are not all strings. This might indicate an issue.")

            pred_priorities = [str(p).lower() if isinstance(p, (str, int, float)) else "unknown" for p in pred_priorities]


        print("\nPriority Model Classification Report (on all tasks from dataset):")


        defined_labels = ["low", "medium", "high"]
        actual_labels_in_data = sorted(list(set(true_priorities) | set(pred_priorities)))


        report_labels = [l for l in defined_labels if l in actual_labels_in_data]
        if not report_labels:
            report_labels = actual_labels_in_data
        if not report_labels:
             print("No valid priority labels available for classification report.")
             return

        print(classification_report(true_priorities, pred_priorities, labels=report_labels, zero_division=0))

        print("\nPriority Model: Example predictions (first few tasks):")
        for i in range(min(5, len(task_texts))):

            true_p_text = true_priorities[i]
            pred_p_text = pred_priorities[i] if i < len(pred_priorities) else "N/A"
            print(f"  Task: \"{task_texts[i]}\", True Priority: {true_p_text}, Predicted Priority: {pred_p_text}")
    except Exception as e:
        print(f"Error during priority model prediction or evaluation: {e}")
        import traceback
        traceback.print_exc()



if __name__ == "__main__":
    print("Starting model testing script...")

    dataset = load_dataset(DATA_FILE_PATH)
    if not dataset:
        print("No dataset loaded. Exiting.")
        exit()

    models = load_models()
    if not models:
        print("Failed to load models. Exiting.")
        exit()


    if models.get('ner_model') and models.get('ner_vectorizer'):
        test_ner_model(models['ner_model'], models['ner_vectorizer'], dataset)
    else:
        print("NER model or vectorizer not loaded. Skipping NER tests.")


    if models.get('duration_model'):
        test_duration_model(models['duration_model'], dataset)
    else:
        print("Duration model not loaded. Skipping duration tests.")


    if models.get('priority_model'):
        test_priority_model(models['priority_model'], dataset)
    else:
        print("Priority model not loaded. Skipping priority tests.")

    print("\nModel testing script finished.")
