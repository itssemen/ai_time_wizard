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

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Загрузка переменных окружения
load_dotenv()

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
def optimize_tasks():
    """Оптимизация расписания из текста"""
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "No text provided"}), 400

    try:
        schedule = parse_tasks(data['text'])
        return jsonify({"schedule": schedule})
    except Exception as e:
        logger.error(f"Optimization error: {str(e)}")
        return jsonify({"error": "Failed to parse tasks"}), 500

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

def parse_tasks(text):
    """Парсинг текста с задачами"""
    tasks = []
    current_time = datetime.now().replace(second=0, microsecond=0) + timedelta(hours=1)
    
    # Разделяем по переносам строк или запятым
    for line in re.split(r'[\n,]', text):
        line = line.strip()
        if not line:
            continue
            
        # Извлекаем длительность
        duration = 60  # По умолчанию 1 час
        time_match = re.search(r'(\d+)\s*(час|часа|часов|мин|минут)', line, re.IGNORECASE)
        if time_match:
            duration = int(time_match.group(1))
            if time_match.group(2).startswith('час'):
                duration *= 60
            line = re.sub(r'(\d+)\s*(час|часа|часов|мин|минут)', '', line).strip()
        
        # Очистка названия задачи
        task_name = re.sub(r'^(с|в|на|и)\s+', '', line, flags=re.IGNORECASE).strip()
        if not task_name:
            task_name = f"Задача {len(tasks) + 1}"
        
        # Расчет времени
        start_time = current_time
        end_time = start_time + timedelta(minutes=duration)
        
        tasks.append({
            "task": task_name,
            "time": f"{start_time.strftime('%H:%M')}-{end_time.strftime('%H:%M')}",
            "start": start_time.isoformat(),
            "end": end_time.isoformat()
        })
        
        current_time = end_time + timedelta(minutes=15)  # Буфер
    
    return tasks

if __name__ == '__main__':
    # Предварительное подключение к календарю
    calendar_manager.connect()
    app.run(host='0.0.0.0', port=5000, debug=True)
