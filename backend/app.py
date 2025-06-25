# import os
# from flask import Flask, jsonify, request, send_from_directory
# from dotenv import load_dotenv
# import caldav
# from datetime import datetime, timedelta
# import re
# import logging

# # Настройка логирования
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Загрузка переменных окружения
# load_dotenv()

# app = Flask(__name__, static_folder='static')
# app.secret_key = os.getenv('FLASK_SECRET_KEY', 'secret-fallback-key')

# class CalendarManager:
#     def __init__(self):
#         self.client = None
#         self.calendar = None
        
#     def connect(self):
#         """Подключение к Яндекс.Календарю через CalDAV"""
#         try:
#             self.client = caldav.DAVClient(
#                 url='https://caldav.yandex.ru',
#                 username=os.getenv('YANDEX_LOGIN'),
#                 password=os.getenv('YANDEX_APP_PASSWORD')
#             )
#             principal = self.client.principal()
#             calendars = principal.calendars()
            
#             if not calendars:
#                 logger.error("No calendars found in Yandex account")
#                 return False
                
#             self.calendar = calendars[0]  # Используем первый календарь
#             logger.info("Successfully connected to Yandex Calendar")
#             return True
            
#         except Exception as e:
#             logger.error(f"Calendar connection error: {str(e)}")
#             return False

# calendar_manager = CalendarManager()

# @app.route('/')
# def serve_index():
#     return send_from_directory(app.static_folder, 'index.html')

# @app.route('/api/status')
# def service_status():
#     """Проверка статуса подключения к календарю"""
#     return jsonify({
#         "calendar_connected": calendar_manager.calendar is not None
#     })

# @app.route('/api/optimize', methods=['POST'])
# def optimize_tasks():
#     """API для оптимизации расписания"""
#     data = request.get_json()
#     if not data or 'text' not in data:
#         return jsonify({"error": "No text provided"}), 400

#     try:
#         parsed_schedule = parse_tasks(data['text'])
#         return jsonify({"schedule": parsed_schedule})
#     except Exception as e:
#         logger.error(f"Optimization error: {str(e)}")
#         return jsonify({"error": str(e)}), 500

# @app.route('/api/add_to_calendar', methods=['POST'])
# def add_to_calendar():
#     """Добавление событий в Яндекс.Календарь"""
#     if not calendar_manager.calendar:
#         if not calendar_manager.connect():
#             return jsonify({"error": "Failed to connect to calendar"}), 500

#     data = request.get_json()
#     if not data or 'schedule' not in data:
#         return jsonify({"error": "No schedule provided"}), 400

#     success_count = 0
#     errors = []
    
#     for event in data['schedule']:
#         try:
#             start = datetime.fromisoformat(event['start'])
#             end = datetime.fromisoformat(event['end'])
            
#             calendar_manager.calendar.save_event(
#                 dtstart=start,
#                 dtend=end,
#                 summary=event['task'],
#                 description="Created by TimeWizard"
#             )
#             success_count += 1
#         except Exception as e:
#             errors.append(f"Failed to add event '{event['task']}': {str(e)}")
#             logger.error(errors[-1])

#     result = {
#         "status": "success" if success_count > 0 else "partial",
#         "added": success_count,
#         "total": len(data['schedule'])
#     }
    
#     if errors:
#         result["errors"] = errors
        
#     return jsonify(result)

# def parse_tasks(text):
#     """Парсинг текста с задачами и генерация расписания"""
#     lines = re.split(r'[\n,]', text)
#     lines = [line.strip() for line in lines if line.strip()]
#     schedule_items = []
#     current_time = datetime.now().replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)

#     for i, line in enumerate(lines):
#         task_name = line
#         duration_minutes = 60  # По умолчанию 1 час

#         # Поиск указания времени
#         time_match = re.search(r'(\d+)\s*(час|часа|часов|мин|минут)', line, re.IGNORECASE)
#         if time_match:
#             value = int(time_match.group(1))
#             unit = time_match.group(2).lower()
#             duration_minutes = value * 60 if unit.startswith('час') else value
#             task_name = re.sub(r'(\d+)\s*(час|часа|часов|мин|минут)', '', line).strip()

#         # Очистка названия задачи
#         task_name = re.sub(r'^(с|в|на|и)\s+', '', task_name, flags=re.IGNORECASE).strip()
#         if not task_name:
#             task_name = f"Задача {i+1}"

#         # Расчет времени
#         start_time = current_time
#         end_time = start_time + timedelta(minutes=duration_minutes)

#         schedule_items.append({
#             "task": task_name,
#             "time": f"{start_time.strftime('%H:%M')}-{end_time.strftime('%H:%M')}",
#             "start": start_time.isoformat(),
#             "end": end_time.isoformat()
#         })

#         current_time = end_time + timedelta(minutes=15)  # Буфер между задачами

#     return schedule_items

# if __name__ == '__main__':
#     # Проверка подключения к календарю при старте
#     calendar_manager.connect()
#     app.run(host='0.0.0.0', port=5000, debug=True)

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

# --- Настройка логирования ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Загрузка переменных окружения ---
load_dotenv()

# --- Глобальные переменные и константы для ML ---
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'ml', 'models_sklearn')
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")
MODEL_PATH = os.path.join(MODEL_DIR, "ner_model.pkl")

vectorizer = None
ner_model = None

# Константы для определения длительности и приоритета (из ml/generate_dataset.py)
DEFAULT_TASK_DURATION_MAP = {
    "позвонить": 15, "написать": 20, "встреча": 60, "совещание": 90,
    "поработать над": 120, "сходить в магазин": 45, "приготовить ужин": 60,
    "почитать книгу": 60, "посмотреть фильм": 120, "сделать уборку": 90,
    "заняться спортом": 60, "погулять": 45, "проверить почту": 10,
    "запланировать": 20, "подготовить отчет": 180, "провести исследование": 240,
    "обсудить проект": 60, "купить билеты": 30, "записаться к врачу": 15,
    "починить": 60, "изучить": 90,
}
HIGH_PRIORITY_KEYWORDS = ["срочно", "важно", "немедленно", "в первую очередь", "обязательно"]
LOW_PRIORITY_KEYWORDS = ["если будет время", "потом", "не горит", "можно отложить"]

# --- Загрузка NLTK ресурсов ---
from nltk.tokenize.util import align_tokens # Added for robust token span calculation

try:
    nltk.data.find('tokenizers/punkt')
    logger.info("NLTK resource 'punkt' found.")
except LookupError:
    logger.info("NLTK resource 'punkt' not found. Downloading...")
    try:
        nltk.download('punkt', quiet=True)
        nltk.data.find('tokenizers/punkt') # Verify download
        logger.info("NLTK resource 'punkt' downloaded successfully.")
    except Exception as e:
        logger.error(f"Failed to download NLTK 'punkt': {e}")
        # Consider how to handle this error - app might not function correctly.

def load_ml_models():
    global vectorizer, ner_model
    try:
        if not os.path.exists(VECTORIZER_PATH):
            logger.error(f"Vectorizer file not found at {VECTORIZER_PATH}")
            return False
        if not os.path.exists(MODEL_PATH):
            logger.error(f"NER model file not found at {MODEL_PATH}")
            return False

        vectorizer = joblib.load(VECTORIZER_PATH)
        ner_model = joblib.load(MODEL_PATH)
        logger.info("ML models loaded successfully.")
        return True
    except Exception as e:
        logger.error(f"Error loading ML models: {e}")
        return False

# --- Функции для ML (адаптированные из ml/model_training.py) ---
def tokenize_text_for_ner(text: str) -> list[dict]:
    """
    Tokenizes text using nltk.word_tokenize for Russian and aligns tokens to get start/end spans.
    """
    tokens_text = nltk.word_tokenize(text, language='russian')
    tokens_info = []

    try:
        # align_tokens expects a list of strings, which is what word_tokenize provides.
        # It returns a list of (start, end) tuples.
        aligned_spans = list(align_tokens(tokens_text, text))

        if len(tokens_text) == len(aligned_spans):
            for i, token_str in enumerate(tokens_text):
                start, end = aligned_spans[i]
                tokens_info.append({'text': token_str, 'start': start, 'end': end})
        else:
            logger.warning(f"Mismatch between token count ({len(tokens_text)}) and span count ({len(aligned_spans)}) after align_tokens. Falling back to basic token info.")
            for token_str in tokens_text:
                 tokens_info.append({'text': token_str, 'start': -1, 'end': -1}) # Indicate invalid span

    except Exception as e:
        logger.error(f"Error using align_tokens: {e}. Falling back to basic token info (no spans).")
        # Fallback: create token_info without proper spans if alignment failed badly
        for token_str in tokens_text:
             tokens_info.append({'text': token_str, 'start': -1, 'end': -1}) # Indicate invalid span

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
    # current_entity_label = None # Not strictly needed if only one label type (TASK)
    # current_entity_end_char will be determined by the last token of the entity

    if len(tokens_info_list) != len(predicted_tags_list):
        logger.warning(f"Mismatch in token_info ({len(tokens_info_list)}) and predicted_tags ({len(predicted_tags_list)}) lengths.")
        return entities

    for i, token_info in enumerate(tokens_info_list):
        tag = predicted_tags_list[i]

        # Ensure token_info has valid start and end, otherwise we can't use it for slicing
        if token_info['start'] == -1 or token_info['end'] == -1:
            logger.warning(f"Token '{token_info['text']}' at index {i} has invalid span, skipping its effect on entity boundaries.")
            # If we are in an entity, and this token is O, we should close the entity
            # based on the *previous* valid token.
            if tag == 'O' and current_entity_start_char != -1:
                # Try to find the end of the previous valid token
                last_valid_token_end = -1
                if i > 0 and tokens_info_list[i-1]['end'] != -1:
                    last_valid_token_end = tokens_info_list[i-1]['end']

                if last_valid_token_end != -1:
                     entities.append({
                        "text": raw_text[current_entity_start_char : last_valid_token_end],
                        "start_char": current_entity_start_char,
                        "end_char": last_valid_token_end
                    })
                else:
                    logger.warning(f"Could not determine end for entity before invalid token at index {i}. Entity may be lost or malformed.")
                current_entity_start_char = -1
            continue # Skip this token for boundary decisions if its own span is invalid

        if tag.startswith('B-'): # B-TASK
            if current_entity_start_char != -1: # Finalize previous entity
                # Previous entity ended at the end of the previous token
                entities.append({
                    "text": raw_text[current_entity_start_char : tokens_info_list[i-1]['end']],
                    "start_char": current_entity_start_char,
                    "end_char": tokens_info_list[i-1]['end']
                })
            current_entity_start_char = token_info['start']
            # current_entity_label = tag[2:]
        elif tag.startswith('I-'): # I-TASK
            if current_entity_start_char == -1: # I-tag without B-tag
                # Treat as a new B-tag if its span is valid
                current_entity_start_char = token_info['start']
                # current_entity_label = tag[2:]
            # If already in an entity, current_entity_start_char is set, and this token just extends it.
            # The end_char will be updated when the entity finishes.
        elif tag == 'O':
            if current_entity_start_char != -1: # Finalize current entity
                # Entity ended at the end of the previous token
                entities.append({
                    "text": raw_text[current_entity_start_char : tokens_info_list[i-1]['end']],
                    "start_char": current_entity_start_char,
                    "end_char": tokens_info_list[i-1]['end']
                })
                current_entity_start_char = -1
                # current_entity_label = None

    if current_entity_start_char != -1: # Finalize any remaining entity
        # Entity ends at the end of the last token processed (which should be valid)
        last_token_end = -1
        # Find the last valid token's end to define the entity boundary
        for k in range(len(tokens_info_list) - 1, -1, -1):
            if tokens_info_list[k]['end'] != -1:
                last_token_end = tokens_info_list[k]['end']
                break

        if last_token_end != -1:
            entities.append({
                "text": raw_text[current_entity_start_char : last_token_end],
                "start_char": current_entity_start_char,
                "end_char": last_token_end
            })
        else:
            logger.warning(f"Could not determine end for final entity starting at {current_entity_start_char}. Entity may be lost or malformed.")

    return entities

# --- Функции для определения длительности и приоритета ---
def get_task_duration(task_text: str) -> tuple[int, str]:
    """
    Determines task duration from text.
    Returns a tuple: (duration_in_minutes, cleaned_task_text).
    """
    task_text_lower = task_text.lower()
    cleaned_task_text = task_text # Default to original text

    # 1. Check for explicit "N час/мин" (e.g., "1 час", "30 минут")
    explicit_time_pattern = r'\b(\d+)\s*(час(?:а|ов)?|ч|мин(?:ут|уты)?|м)\b'
    explicit_time_match = re.search(explicit_time_pattern, task_text, re.IGNORECASE)

    if explicit_time_match:
        value_str = explicit_time_match.group(1)
        unit_str = explicit_time_match.group(2).lower()
        try:
            value = int(value_str)
            duration_minutes = value * 60 if unit_str.startswith('ч') else value

            start_span, end_span = explicit_time_match.span()
            temp_cleaned_text = task_text[:start_span] + task_text[end_span:]
            cleaned_task_text = re.sub(r'\s\s+', ' ', temp_cleaned_text).strip()
            if not cleaned_task_text:
                cleaned_task_text = f"Задача ({explicit_time_match.group(0)})"
            return duration_minutes, cleaned_task_text
        except ValueError:
            logger.warning(f"Could not parse value '{value_str}' as integer from time pattern in '{task_text}'. Proceeding to other patterns.")

    # 2. Check for standalone "час" (meaning 1 hour) or "полчаса" (30 minutes)
    # These are checked after explicit "N час/мин" to avoid conflict.
    standalone_hour_match = re.search(r'\b(час(?:а|ов)?)\b', task_text, re.IGNORECASE)
    if standalone_hour_match:
        # Ensure it's not preceded by a digit (already handled by explicit_time_pattern check order)
        # Check if the found "час" is part of a larger construct that explicit_time_pattern should have caught.
        # This is implicitly handled because explicit_time_pattern is more specific and checked first.
        duration_minutes = 60 # "час" alone means 1 hour
        start_span, end_span = standalone_hour_match.span()
        temp_cleaned_text = task_text[:start_span] + task_text[end_span:]
        cleaned_task_text = re.sub(r'\s\s+', ' ', temp_cleaned_text).strip()
        if not cleaned_task_text:
            cleaned_task_text = f"Задача ({standalone_hour_match.group(0)})"
        return duration_minutes, cleaned_task_text

    standalone_half_hour_match = re.search(r'\b(полчаса)\b', task_text, re.IGNORECASE)
    if standalone_half_hour_match:
        duration_minutes = 30
        start_span, end_span = standalone_half_hour_match.span()
        temp_cleaned_text = task_text[:start_span] + task_text[end_span:]
        cleaned_task_text = re.sub(r'\s\s+', ' ', temp_cleaned_text).strip()
        if not cleaned_task_text:
            cleaned_task_text = f"Задача ({standalone_half_hour_match.group(0)})"
        return duration_minutes, cleaned_task_text

    # 3. Check DEFAULT_TASK_DURATION_MAP
    # This check is on task_text_lower of the original task_text.
    for keyword, duration in DEFAULT_TASK_DURATION_MAP.items():
        if keyword in task_text_lower:
            return duration, task_text

    # 4. Default duration if no pattern or keyword matched
    return 60, task_text

def get_task_priority(task_text: str) -> str:
    task_text_lower = task_text.lower()
    if any(k in task_text_lower for k in HIGH_PRIORITY_KEYWORDS):
        return "high"
    if any(k in task_text_lower for k in LOW_PRIORITY_KEYWORDS):
        return "low"

    # Simple heuristics from generate_dataset.py (optional, can be expanded)
    if "отчет" in task_text_lower or "презентац" in task_text_lower or "документ" in task_text_lower:
        return "high"
    if "посмотреть фильм" in task_text_lower or "почитать книгу" in task_text_lower or "погулять" in task_text_lower:
        return "low"

    return "medium"

# --- Основная функция обработки текста с ML ---
def process_text_with_ml(raw_text: str) -> list[dict]:
    if not vectorizer or not ner_model:
        logger.error("ML models not loaded. Cannot process text.")
        # Fallback to old parser or return error
        # For now, let's try to use the old parser if ML fails.
        # Or raise an exception to be caught by the endpoint.
        raise ValueError("ML models are not available.")

    # 1. NER: Extract task texts
    tokens_info = tokenize_text_for_ner(raw_text)
    if not tokens_info:
        logger.info("No tokens were generated from the raw text.")
        return []

    # Filter out tokens with invalid spans if they occurred, as they can't be used for feature extraction
    # or reliable task extraction. However, word2features expects a list of token texts.
    # If a token has an invalid span, it might be better to log it and decide if it should be excluded
    # from sent_tokens_text, which could affect feature generation for adjacent tokens.
    # For now, we'll pass all token texts to feature generation, assuming word2features can handle them.
    # The iob_tags_to_extracted_tasks function will handle invalid spans more directly.

    sent_tokens_text = [t['text'] for t in tokens_info]
    if not sent_tokens_text: # Should not happen if tokens_info is not empty, but as a safeguard
        logger.info("Token texts list is empty after tokenization.")
        return []

    features = sent2features(sent_tokens_text)

    try:
        vectorized_features = vectorizer.transform(features)
        predicted_tags = ner_model.predict(vectorized_features)
    except Exception as e:
        logger.error(f"Error during NER prediction: {e}")
        raise ValueError(f"NER prediction failed: {e}")

    extracted_raw_tasks = iob_tags_to_extracted_tasks(raw_text, tokens_info, predicted_tags)

    # 2. Determine attributes (duration, priority) for each task
    processed_tasks = []
    for raw_task in extracted_raw_tasks:
        task_text_from_ner = raw_task['text']
        # It's possible NER extracts very short/irrelevant text. Add a simple filter.
        if not task_text_from_ner or len(task_text_from_ner.split()) < 1 : # e.g. ignore single punctuation if extracted
            continue

        # Determine priority based on the text as extracted by NER (before time removal)
        priority_str = get_task_priority(task_text_from_ner)

        # Determine duration and potentially clean the task text
        duration_minutes, final_task_text = get_task_duration(task_text_from_ner)

        # If after cleaning, the task text becomes empty (e.g. it was only "1 час"),
        # and get_task_duration returned a placeholder, we use that.
        # If it's genuinely empty for other reasons, we might skip it or use NER text.
        if not final_task_text and task_text_from_ner: # If cleaning resulted in empty but original was not
            final_task_text = task_text_from_ner # Revert to original NER text if cleaning made it empty and no placeholder was made
            logger.info(f"Task text for '{task_text_from_ner}' became empty after duration removal, using original NER text or placeholder if generated.")


        processed_tasks.append({
            "text": final_task_text, # Use cleaned text
            "duration": duration_minutes,
            "priority": priority_str,
            "original_order": len(processed_tasks) # Keep original order for tie-breaking
        })

    if not processed_tasks:
        return []

    # 3. Sort tasks: high -> medium -> low. Within priority, by original order.
    priority_map = {"high": 0, "medium": 1, "low": 2}
    processed_tasks.sort(key=lambda x: (priority_map.get(x['priority'], 2), x['original_order']))

    # 4. Schedule tasks
    schedule_items = []
    current_time = datetime.now().replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)

    for task_details in processed_tasks:
        start_time = current_time
        end_time = start_time + timedelta(minutes=task_details['duration'])

        schedule_items.append({
            "task": task_details['text'],
            "time": f"{start_time.strftime('%H:%M')}-{end_time.strftime('%H:%M')}",
            "start": start_time.isoformat(),
            "end": end_time.isoformat()
        })
        current_time = end_time + timedelta(minutes=15) # Buffer between tasks

    return schedule_items

# --- Flask App and CalendarManager (largely unchanged below this line initially) ---
app = Flask(__name__, static_folder='static')
app.secret_key = os.getenv('FLASK_SECRET_KEY', secrets.token_hex(32))

class CalendarManager:
    def __init__(self):
        self.client = None
        self.calendar = None
        
    def connect(self):
        """Подключение к Яндекс.Календарю через CalDAV"""
        try:
            yandex_login = os.getenv('YANDEX_LOGIN')
            yandex_password = os.getenv('YANDEX_APP_PASSWORD')
            
            if not yandex_login or not yandex_password:
                logger.error("Yandex credentials not configured in .env")
                return False
                
            self.client = caldav.DAVClient(
                url='https://caldav.yandex.ru',
                username=yandex_login,
                password=yandex_password
            )
            principal = self.client.principal()
            calendars = principal.calendars()
            
            if not calendars:
                logger.error("No calendars found in Yandex account")
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
    """Проверка статуса подключения к календарю"""
    is_connected = calendar_manager.calendar is not None
    return jsonify({
        "connected": is_connected,
        "message": "Successfully connected to Yandex Calendar" if is_connected 
                  else "Not connected to calendar"
    })

@app.route('/api/optimize', methods=['POST'])
def optimize_tasks_endpoint(): # Renamed from optimize_tasks to avoid conflict
    """Оптимизация расписания из текста с использованием ML"""
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "No text provided"}), 400

    try:
        # schedule = parse_tasks(data['text']) # Old method
        schedule = process_text_with_ml(data['text']) # New ML-based method
        return jsonify({"schedule": schedule})
    except ValueError as ve: # Specific error for known issues like model loading or NER failure
        logger.error(f"Value error during optimization: {str(ve)}")
        return jsonify({"error": str(ve)}), 500
    except Exception as e:
        logger.error(f"Unexpected optimization error: {str(e)}")
        # Log full traceback for unexpected errors
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({"error": "Failed to process tasks due to an unexpected server error"}), 500

@app.route('/api/add_to_calendar', methods=['POST'])
def add_to_calendar():
    """Добавление событий в календарь"""
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
            
            calendar_manager.calendar.save_event(
                dtstart=start,
                dtend=end,
                summary=event['task'],
                description="Created by TimeWizard"
            )
            results.append({"status": "success", "task": event['task']})
        except Exception as e:
            logger.error(f"Failed to add event: {str(e)}")
            results.append({
                "status": "error",
                "task": event['task'],
                "error": str(e)
            })

    return jsonify({"results": results})

# Old parse_tasks function is removed as its logic is now within process_text_with_ml or get_task_duration

if __name__ == '__main__':
    # Загружаем ML модели перед запуском Flask приложения
    if not load_ml_models():
        logger.error("ML Models failed to load. The /api/optimize endpoint might not work correctly.")
        # Depending on desired behavior, you might want to exit or prevent app run:
        # exit(1)

    # Предварительное подключение к календарю
    calendar_manager.connect()
    app.run(host='0.0.0.0', port=5000, debug=True)
