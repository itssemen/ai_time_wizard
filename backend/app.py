
import os
from flask import Flask, jsonify, request, send_from_directory
from dotenv import load_dotenv
import caldav
from datetime import datetime, timedelta
import re
import logging
import secrets
import joblib
import nltk
import math
import pymorphy3


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


load_dotenv()


MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'ml', 'models_sklearn')
NER_VECTORIZER_PATH = os.path.join(MODEL_DIR, "ner_vectorizer.pkl")
NER_MODEL_PATH = os.path.join(MODEL_DIR, "ner_model.pkl")
DURATION_MODEL_PATH = os.path.join(MODEL_DIR, "duration_model.pkl")
PRIORITY_MODEL_PATH = os.path.join(MODEL_DIR, "priority_model.pkl")

ner_vectorizer = None
ner_model = None
duration_model = None
priority_model = None


morph = pymorphy3.MorphAnalyzer()


from nltk.tokenize.util import align_tokens

try:
    nltk.data.find('tokenizers/punkt')
    logger.info("NLTK resource 'punkt' found.")
except LookupError:
    logger.info("NLTK resource 'punkt' not found. Downloading...")
    try:
        nltk.download('punkt', quiet=True)
        nltk.data.find('tokenizers/punkt')
        logger.info("NLTK resource 'punkt' downloaded successfully.")
    except Exception as e:
        logger.error(f"Failed to download NLTK 'punkt': {e}")

try:
    nltk.data.find('tokenizers/punkt_tab')
    logger.info("NLTK resource 'punkt_tab' found.")
except LookupError:
    logger.info("NLTK resource 'punkt_tab' not found. Downloading...")
    try:
        nltk.download('punkt_tab', quiet=True)
        nltk.data.find('tokenizers/punkt_tab')
        logger.info("NLTK resource 'punkt_tab' downloaded successfully.")
    except Exception as e:
        logger.error(f"Failed to download NLTK 'punkt_tab': {e}")

def load_ml_models():
    global ner_vectorizer, ner_model, duration_model, priority_model
    all_models_loaded_successfully = True
    try:

        abs_ner_vectorizer_path = os.path.abspath(NER_VECTORIZER_PATH)
        logger.info(f"Attempting to load NER Vectorizer from: {abs_ner_vectorizer_path}")
        if not os.path.exists(NER_VECTORIZER_PATH):
            logger.error(f"NER Vectorizer file not found at resolved path: {abs_ner_vectorizer_path}")
            all_models_loaded_successfully = False
        else:
            ner_vectorizer = joblib.load(NER_VECTORIZER_PATH)
            logger.info(f"NER Vectorizer loaded successfully from: {abs_ner_vectorizer_path}")


        abs_ner_model_path = os.path.abspath(NER_MODEL_PATH)
        logger.info(f"Attempting to load NER model from: {abs_ner_model_path}")
        if not os.path.exists(NER_MODEL_PATH):
            logger.error(f"NER model file not found at resolved path: {abs_ner_model_path}")
            all_models_loaded_successfully = False
        else:
            ner_model = joblib.load(NER_MODEL_PATH)
            logger.info(f"NER model loaded successfully from: {abs_ner_model_path}")


        abs_duration_model_path = os.path.abspath(DURATION_MODEL_PATH)
        logger.info(f"Attempting to load Duration model from: {abs_duration_model_path}")

        if not os.path.exists(abs_duration_model_path):
            logger.warning(f"Duration model file not found at resolved path: {abs_duration_model_path}. Duration prediction will use fallback.")
            duration_model = None
        else:
            duration_model = joblib.load(abs_duration_model_path)
            logger.info(f"Duration model loaded successfully from: {abs_duration_model_path}")


        abs_priority_model_path = os.path.abspath(PRIORITY_MODEL_PATH)
        logger.info(f"Attempting to load Priority model from: {abs_priority_model_path}")
        if not os.path.exists(PRIORITY_MODEL_PATH):
            logger.warning(f"Priority model file not found at resolved path: {abs_priority_model_path}. Priority prediction will use fallback.")
            priority_model = None
        else:
            priority_model = joblib.load(PRIORITY_MODEL_PATH)
            logger.info("Priority model loaded successfully.")

        if ner_vectorizer and ner_model and duration_model and priority_model:
             logger.info("All ML models (NER, Duration, Priority) loaded successfully.")
        elif ner_vectorizer and ner_model:
            logger.warning("Only NER models loaded. Duration and/or Priority models are missing. Fallbacks will be used.")
        else:
            logger.error("Critical NER models failed to load. Task processing will likely fail.")
            all_models_loaded_successfully = False

        return all_models_loaded_successfully
    except Exception as e:
        logger.error(f"General error loading ML models: {e}")
        ner_vectorizer, ner_model, duration_model, priority_model = None, None, None, None
        return False


def tokenize_text_for_ner(text: str) -> list[dict]:
    tokens_text = nltk.word_tokenize(text, language='russian')
    tokens_info = []
    try:
        aligned_spans = list(align_tokens(tokens_text, text))
        if len(tokens_text) == len(aligned_spans):
            for i, token_str in enumerate(tokens_text):
                start, end = aligned_spans[i]
                tokens_info.append({'text': token_str, 'start': start, 'end': end})
        else:
            logger.warning(f"Mismatch token/span count. Fallback: basic token info.")
            for token_str in tokens_text:
                 tokens_info.append({'text': token_str, 'start': -1, 'end': -1})
    except Exception as e:
        logger.error(f"Error using align_tokens: {e}. Fallback: basic token info.")
        for token_str in tokens_text:
             tokens_info.append({'text': token_str, 'start': -1, 'end': -1})
    return tokens_info

def word2features(sent_tokens_text, i):
    word = sent_tokens_text[i]
    features = {
        'bias': 1.0, 'word_lower': word.lower(), 'word_istitle': word.istitle(),
        'word_isupper': word.isupper(), 'word_isdigit': word.isdigit(),
        'word_suffix_2': word[-2:], 'word_prefix_2': word[:2],
    }
    if i > 0:
        prev_word = sent_tokens_text[i-1]
        features.update({
            'prev_word_lower': prev_word.lower(), 'prev_word_istitle': prev_word.istitle(),
            'prev_word_isupper': prev_word.isupper(), 'prev_word_isdigit': prev_word.isdigit(),
        })
    else:
        features['BOS'] = True
    if i < len(sent_tokens_text)-1:
        next_word = sent_tokens_text[i+1]
        features.update({
            'next_word_lower': next_word.lower(), 'next_word_istitle': next_word.istitle(),
            'next_word_isupper': next_word.isupper(), 'next_word_isdigit': next_word.isdigit(),
        })
    else:
        features['EOS'] = True
    return features

def sent2features(sent_tokens_text):
    return [word2features(sent_tokens_text, i) for i in range(len(sent_tokens_text))]

def iob_tags_to_extracted_tasks(raw_text: str, tokens_info_list: list[dict], predicted_tags_list: list[str]) -> list[dict]:
    entities = []
    current_entity_start_char = -1
    if len(tokens_info_list) != len(predicted_tags_list):
        logger.warning(f"Mismatch token_info/predicted_tags lengths.")
        return entities
    for i, token_info in enumerate(tokens_info_list):
        tag = predicted_tags_list[i]
        if token_info['start'] == -1 or token_info['end'] == -1:
            logger.warning(f"Token '{token_info['text']}' at index {i} has invalid span.")
            if tag == 'O' and current_entity_start_char != -1:
                last_valid_token_end = tokens_info_list[i-1]['end'] if i > 0 and tokens_info_list[i-1]['end'] != -1 else -1
                if last_valid_token_end != -1:
                     entities.append({"text": raw_text[current_entity_start_char : last_valid_token_end], "start_char": current_entity_start_char, "end_char": last_valid_token_end})
                current_entity_start_char = -1
            continue
        if tag.startswith('B-'):
            if current_entity_start_char != -1:
                entities.append({"text": raw_text[current_entity_start_char : tokens_info_list[i-1]['end']], "start_char": current_entity_start_char, "end_char": tokens_info_list[i-1]['end']})
            current_entity_start_char = token_info['start']
        elif tag.startswith('I-'):
            if current_entity_start_char == -1:
                current_entity_start_char = token_info['start']
        elif tag == 'O':
            if current_entity_start_char != -1:
                entities.append({"text": raw_text[current_entity_start_char : tokens_info_list[i-1]['end']], "start_char": current_entity_start_char, "end_char": tokens_info_list[i-1]['end']})
                current_entity_start_char = -1
    if current_entity_start_char != -1:
        last_token_end = -1
        for k in range(len(tokens_info_list) - 1, -1, -1):
            if tokens_info_list[k]['end'] != -1:
                last_token_end = tokens_info_list[k]['end']
                break
        if last_token_end != -1:
            entities.append({"text": raw_text[current_entity_start_char : last_token_end], "start_char": current_entity_start_char, "end_char": last_token_end})
    return entities

def lemmatize_text_for_model(text: str) -> str:

    words = text.split()
    lemmatized_words = [morph.parse(word)[0].normal_form for word in words]
    return " ".join(lemmatized_words)

def extract_duration_features(task_text: str) -> list:

    lemmatized_text = lemmatize_text_for_model(task_text)
    text_length = len(task_text.split())

    has_explicit_duration = 0
    explicit_duration_parsed_minutes = 0.0

    return [lemmatized_text, has_explicit_duration, float(text_length), explicit_duration_parsed_minutes]

def predict_duration_ml(task_text: str) -> tuple[int, bool]:

    default_duration = 60
    is_default = False
    if not duration_model:
        logger.info(f"Duration model not loaded for '{task_text}'. Using default duration ({default_duration} min).")
        is_default = True
        return default_duration, is_default
    try:

        features = extract_duration_features(task_text)
        predicted_duration = duration_model.predict([features])[0]

        if isinstance(predicted_duration, (int, float)) and predicted_duration > 0:
            return math.ceil(predicted_duration), is_default
        else:

            logger.warning(f"Predicted duration for '{task_text}' (features: {features}) is not a positive number: {predicted_duration}. Using default.")
            is_default = True
            return default_duration, is_default
    except Exception as e:
        logger.error(f"Error predicting duration for '{task_text}': {e}. Using default.")
        is_default = True
        return default_duration, is_default

def predict_priority_ml(task_text: str) -> tuple[str, bool]:

    default_priority = "medium"
    is_default = False
    if not priority_model:
        logger.info(f"Priority model not loaded for '{task_text}'. Using default priority ({default_priority}).")
        is_default = True
        return default_priority, is_default
    try:

        predicted_priority_str = priority_model.predict([task_text])[0]

        if isinstance(predicted_priority_str, str) and predicted_priority_str.lower() in ["low", "medium", "high"]:
            return predicted_priority_str.lower(), is_default
        else:
            logger.warning(f"Unexpected priority value or type from model for '{task_text}': '{predicted_priority_str}'. Using default ({default_priority}).")
            is_default = True
            return default_priority, is_default

    except Exception as e:
        logger.error(f"Error predicting priority for '{task_text}': {e}. Using default ({default_priority}).")
        import traceback
        logger.error(traceback.format_exc())
        is_default = True
        return default_priority, is_default


def process_text_with_ml(raw_text: str) -> list[dict]:
    if not ner_vectorizer or not ner_model:
        logger.error("NER ML models not loaded. Cannot process text.")
        raise ValueError("NER ML models are not available.")

    tokens_info = tokenize_text_for_ner(raw_text)
    if not tokens_info: return []
    sent_tokens_text = [t['text'] for t in tokens_info]
    if not sent_tokens_text: return []
    features = sent2features(sent_tokens_text)
    try:
        vectorized_features = ner_vectorizer.transform(features)
        predicted_tags = ner_model.predict(vectorized_features)
    except Exception as e:
        logger.error(f"Error during NER prediction: {e}")
        raise ValueError(f"NER prediction failed: {e}")
    extracted_raw_tasks = iob_tags_to_extracted_tasks(raw_text, tokens_info, predicted_tags)

    processed_tasks = []
    for raw_task in extracted_raw_tasks:
        task_text_from_ner = raw_task['text']
        if not task_text_from_ner or len(task_text_from_ner.split()) < 1: continue

        predicted_duration_minutes, duration_defaulted = predict_duration_ml(task_text_from_ner)
        predicted_priority_str, priority_defaulted = predict_priority_ml(task_text_from_ner)

        final_task_text = f"{task_text_from_ner} ({predicted_priority_str})"

        processed_tasks.append({
            "text": final_task_text,
            "duration": predicted_duration_minutes,
            "priority": predicted_priority_str,
            "duration_defaulted": duration_defaulted,
            "priority_defaulted": priority_defaulted,
            "original_order": len(processed_tasks)
        })

    if not processed_tasks: return []
    priority_map = {"high": 0, "medium": 1, "low": 2}

    processed_tasks.sort(key=lambda x: (priority_map.get(x['priority'], 2), x['original_order']))

    schedule_items = []
    current_time = datetime.now().replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)

    for task_details in processed_tasks:
        start_time = current_time
        end_time = start_time + timedelta(minutes=task_details['duration'])
        task_display_text = task_details['text']
        schedule_items.append({
            "task": task_display_text,
            "time": f"{start_time.strftime('%H:%M')}-{end_time.strftime('%H:%M')}",
            "start": start_time.isoformat(),
            "end": end_time.isoformat(),
            "duration_defaulted": task_details['duration_defaulted'],
            "priority_defaulted": task_details['priority_defaulted']
        })
        current_time = end_time
    return schedule_items


app = Flask(__name__, static_folder='static')
app.secret_key = os.getenv('FLASK_SECRET_KEY', secrets.token_hex(32))

class CalendarManager:
    def __init__(self):
        self.client = None
        self.calendar = None
    def connect(self):
        try:
            yandex_login = os.getenv('YANDEX_LOGIN')
            yandex_password = os.getenv('YANDEX_APP_PASSWORD')
            if not yandex_login or not yandex_password:
                logger.error("Yandex credentials not configured.")
                return False
            self.client = caldav.DAVClient(url='https://caldav.yandex.ru', username=yandex_login, password=yandex_password)
            principal = self.client.principal()
            calendars = principal.calendars()
            if not calendars:
                logger.error("No calendars found.")
                return False
            self.calendar = calendars[0]
            logger.info(f"Connected to calendar: {self.calendar.name}")
            return True
        except Exception as e:
            logger.error(f"Calendar connection error: {str(e)}")
            return False

calendar_manager = CalendarManager()

@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/calendar_status')
def calendar_status():
    is_connected = calendar_manager.calendar is not None
    return jsonify({"connected": is_connected, "message": "Connected" if is_connected else "Not connected"})

@app.route('/api/optimize', methods=['POST'])
def optimize_tasks_endpoint():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "No text provided"}), 400
    try:
        schedule = process_text_with_ml(data['text'])
        return jsonify({"schedule": schedule})
    except ValueError as ve:
        logger.error(f"Value error during optimization: {str(ve)}")
        return jsonify({"error": str(ve)}), 500
    except Exception as e:
        logger.error(f"Unexpected optimization error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({"error": "Server error during task processing"}), 500

@app.route('/api/add_to_calendar', methods=['POST'])
def add_to_calendar():
    if not calendar_manager.calendar and not calendar_manager.connect():
        return jsonify({"error": "Calendar connection failed"}), 500
    data = request.get_json()
    if not data or 'schedule' not in data:
        return jsonify({"error": "No schedule provided"}), 400
    results = []
    for event in data['schedule']:
        try:
            start = datetime.fromisoformat(event['start'])
            end = datetime.fromisoformat(event['end'])
            calendar_manager.calendar.save_event(dtstart=start, dtend=end, summary=event['task'], description="Created by TimeWizard")
            results.append({"status": "success", "task": event['task']})
        except Exception as e:
            logger.error(f"Failed to add event: {str(e)}")
            results.append({"status": "error", "task": event['task'], "error": str(e)})
    return jsonify({"results": results})

if __name__ == '__main__':
    load_ml_models()
    calendar_manager.connect()
    app.run(host='0.0.0.0', port=5000, debug=True)
