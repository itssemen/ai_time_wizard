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
def tokenize_text_for_ner(text): # Renamed to avoid conflict if other tokenizers are used
    tokens_info = []
    # Using WhitespaceTokenizer as in model_training.py
    tokenizer = nltk.tokenize.WhitespaceTokenizer()
    for start, end in tokenizer.span_tokenize(text):
        token_text = text[start:end]
        tokens_info.append({'text': token_text, 'start': start, 'end': end})
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

def iob_tags_to_extracted_tasks(tokens_info_list, predicted_tags_list):
    entities = []
    current_entity_tokens = []
    current_entity_start_char = -1
    # current_entity_label = None # Not strictly needed if only one label type (TASK)

    if len(tokens_info_list) != len(predicted_tags_list):
        logger.warning(f"Mismatch in token_info ({len(tokens_info_list)}) and predicted_tags ({len(predicted_tags_list)}) lengths.")
        return entities

    for i, token_info in enumerate(tokens_info_list):
        tag = predicted_tags_list[i]
        token_text = token_info['text']

        if tag.startswith('B-'): # B-TASK
            if current_entity_tokens: # Finalize previous entity
                entities.append({
                    "text": " ".join(current_entity_tokens),
                    "start_char": current_entity_start_char,
                    "end_char": tokens_info_list[i-1]['end'] # end of the last token of previous entity
                })
            current_entity_tokens = [token_text]
            current_entity_start_char = token_info['start']
            # current_entity_label = tag[2:]
        elif tag.startswith('I-'): # I-TASK
            # if current_entity_tokens and tag[2:] == current_entity_label:
            if current_entity_tokens: # Simpler check if only one entity type
                current_entity_tokens.append(token_text)
            else: # I-tag without B-tag, treat as a new B-tag or ignore
                  # For simplicity, let's start a new entity if current_entity_tokens is empty
                if not current_entity_tokens:
                    current_entity_tokens = [token_text]
                    current_entity_start_char = token_info['start']
                    # current_entity_label = tag[2:]
        elif tag == 'O':
            if current_entity_tokens: # Finalize current entity
                entities.append({
                    "text": " ".join(current_entity_tokens),
                    "start_char": current_entity_start_char,
                    "end_char": tokens_info_list[i-1]['end']
                })
                current_entity_tokens = []
                current_entity_start_char = -1
                # current_entity_label = None

    if current_entity_tokens: # Finalize any remaining entity
        entities.append({
            "text": " ".join(current_entity_tokens),
            "start_char": current_entity_start_char,
            "end_char": tokens_info_list[-1]['end']
        })
    return entities

# --- Функции для определения длительности и приоритета ---
def get_task_duration(task_text: str) -> int:
    task_text_lower = task_text.lower()
    # 1. Check for explicit time mentions (e.g., "1 час", "30 минут")
    time_match = re.search(r'(\d+)\s*(час|часа|часов|ч|мин|минут|м)', task_text_lower)
    if time_match:
        value = int(time_match.group(1))
        unit = time_match.group(2).lower()
        if unit.startswith('ч'):
            return value * 60
        elif unit.startswith('м'):
            return value

    # 2. Check DEFAULT_TASK_DURATION_MAP
    for keyword, duration in DEFAULT_TASK_DURATION_MAP.items():
        if keyword in task_text_lower:
            return duration

    return 60 # Default duration if not found

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
        return []

    sent_tokens_text = [t['text'] for t in tokens_info]
    features = sent2features(sent_tokens_text)

    try:
        vectorized_features = vectorizer.transform(features)
        predicted_tags = ner_model.predict(vectorized_features)
    except Exception as e:
        logger.error(f"Error during NER prediction: {e}")
        raise ValueError(f"NER prediction failed: {e}")

    extracted_raw_tasks = iob_tags_to_extracted_tasks(tokens_info, predicted_tags)

    # 2. Determine attributes (duration, priority) for each task
    processed_tasks = []
    for raw_task in extracted_raw_tasks:
        task_text = raw_task['text']
        # It's possible NER extracts very short/irrelevant text. Add a simple filter.
        if not task_text or len(task_text.split()) < 1 : # e.g. ignore single punctuation if extracted
            continue

        duration_minutes = get_task_duration(task_text)
        priority_str = get_task_priority(task_text)
        processed_tasks.append({
            "text": task_text,
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
