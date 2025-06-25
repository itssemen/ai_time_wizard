# from flask import Flask, jsonify, request, send_from_directory
# import re
# from datetime import datetime, timedelta

# app = Flask(__name__, static_folder='static')

# @app.route('/')
# def serve_index():
#     return send_from_directory(app.static_folder, 'index.html')

# # Удаляем старый /api/data, так как он больше не нужен
# # @app.route('/api/data')
# # def get_data():
# #     return jsonify({'message': 'Hello from Backend!'})

# def parse_tasks_for_backend(text):
#     """
#     Простой парсер для MVP на стороне бэкенда.
#     Извлекает задачи и их предполагаемую длительность.
#     """
#     # Используем re.split для разделения по новой строке или запятой
#     lines = re.split(r'[\n,]', text)
#     lines = [line.strip() for line in lines if line.strip()]

#     schedule_items = []
#     current_time = datetime.now().replace(hour=9, minute=0, second=0, microsecond=0)

#     for i, line in enumerate(lines):
#         task_name = line
#         duration_minutes = 60  # Длительность по умолчанию 1 час

#         time_match = re.search(r'(\d+)\s*(час|часа|часов|мин|минут)', line, re.IGNORECASE)
#         if time_match:
#             value = int(time_match.group(1))
#             unit = time_match.group(2).lower()
#             if unit.startswith('час'):
#                 duration_minutes = value * 60
#             elif unit.startswith('мин'):
#                 duration_minutes = value

#             # Удаляем упоминание времени из названия задачи
#             task_name = line[:time_match.start()].strip() + line[time_match.end():].strip()
#             task_name = task_name.strip(" ,.-")


#         # Убираем лишние предлоги/союзы в начале, если они остались
#         task_name = re.sub(r'^(с|в|на|и)\s+', '', task_name, flags=re.IGNORECASE).strip()

#         if not task_name: # Если после удаления времени название стало пустым
#              task_name = f"Задача {i+1}"


#         start_time_dt = current_time
#         end_time_dt = start_time_dt + timedelta(minutes=duration_minutes)

#         schedule_items.append({
#             "time": f"{start_time_dt.strftime('%H:%M')}-{end_time_dt.strftime('%H:%M')}",
#             "task": task_name
#         })

#         # Обновляем current_time для следующей задачи, добавляя 15 минут перерыва
#         current_time = end_time_dt + timedelta(minutes=15)

#     return schedule_items

# @app.route('/api/optimize', methods=['POST'])
# def optimize_tasks():
#     data = request.get_json()
#     if not data or 'text' not in data:
#         return jsonify({"error": "No text provided"}), 400

#     input_text = data['text']

#     # Временно (MVP): Используем простой парсер на бэкенде
#     # В будущем здесь будет логика ИИ
#     parsed_schedule = parse_tasks_for_backend(input_text)

#     if not parsed_schedule:
#         # Возвращаем пустой список, если ничего не распознано,
#         # чтобы фронтенд мог это обработать
#         return jsonify({"schedule": []})

#     return jsonify({"schedule": parsed_schedule})

# @app.route('/api/add_to_calendar', methods=['POST'])
# def add_to_calendar():
#     data = request.get_json()
#     if not data or 'schedule' not in data:
#         return jsonify({"error": "No schedule provided"}), 400

#     # schedule_data = data['schedule']
#     # Здесь в будущем будет логика взаимодействия с API Яндекс.Календаря

#     # Временно (MVP): Просто возвращаем успех
#     return jsonify({"status": "success", "message": "Задачи (теоретически) добавлены в ваш календарь 🎉"})

# if __name__ == '__main__':
#     app.run(debug=True, port=5000)

import os
import re
from datetime import datetime, timedelta
from flask import Flask, jsonify, request, send_from_directory, redirect, session, url_for
import requests
from dotenv import load_dotenv

# Загрузка переменных окружения
load_dotenv()

app = Flask(__name__, static_folder='static')
app.secret_key = os.getenv('FLASK_SECRET_KEY')

# Конфигурация Яндекс.OAuth
YANDEX_OAUTH_CONFIG = {
    'client_id': os.getenv('YANDEX_CLIENT_ID'),
    'client_secret': os.getenv('YANDEX_CLIENT_SECRET'),
    'auth_url': 'https://oauth.yandex.ru/authorize',
    'token_url': 'https://oauth.yandex.ru/token',
    'redirect_uri': os.getenv('YANDEX_REDIRECT_URI'),
    'calendar_api': 'https://api.calendar.yandex.net/v3'
}

@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/auth/yandex')
def auth_yandex():
    """Перенаправление на авторизацию Яндекс"""
    auth_url = (
        f"{YANDEX_OAUTH_CONFIG['auth_url']}?"
        f"response_type=code&"
        f"client_id={YANDEX_OAUTH_CONFIG['client_id']}&"
        f"redirect_uri={YANDEX_OAUTH_CONFIG['redirect_uri']}"
    )
    return redirect(auth_url)

@app.route('/auth/yandex/callback')
def yandex_callback():
    """Обработчик callback от Яндекс.OAuth"""
    code = request.args.get('code')
    if not code:
        return jsonify({"error": "Authorization failed"}), 400
    
    # Получаем токен доступа
    token_data = {
        'grant_type': 'authorization_code',
        'code': code,
        'client_id': YANDEX_OAUTH_CONFIG['client_id'],
        'client_secret': YANDEX_OAUTH_CONFIG['client_secret']
    }
    
    try:
        response = requests.post(YANDEX_OAUTH_CONFIG['token_url'], data=token_data)
        response.raise_for_status()
        session['yandex_token'] = response.json()['access_token']
        return redirect(url_for('serve_index'))
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/auth/status')
def auth_status():
    """Проверка статуса авторизации"""
    return jsonify({
        "authenticated": 'yandex_token' in session
    })

def parse_tasks(text):
    """Парсинг текста с задачами"""
    lines = re.split(r'[\n,]', text)
    lines = [line.strip() for line in lines if line.strip()]
    schedule_items = []
    current_time = datetime.now().replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)

    for i, line in enumerate(lines):
        task_name = line
        duration_minutes = 60  # По умолчанию 1 час

        # Поиск указания времени
        time_match = re.search(r'(\d+)\s*(час|часа|часов|мин|минут)', line, re.IGNORECASE)
        if time_match:
            value = int(time_match.group(1))
            unit = time_match.group(2).lower()
            duration_minutes = value * 60 if unit.startswith('час') else value
            task_name = re.sub(r'(\d+)\s*(час|часа|часов|мин|минут)', '', line).strip()

        # Очистка названия задачи
        task_name = re.sub(r'^(с|в|на|и)\s+', '', task_name, flags=re.IGNORECASE).strip()
        if not task_name:
            task_name = f"Задача {i+1}"

        # Расчет времени
        start_time = current_time
        end_time = start_time + timedelta(minutes=duration_minutes)

        schedule_items.append({
            "task": task_name,
            "time": f"{start_time.strftime('%H:%M')}-{end_time.strftime('%H:%M')}",
            "start": start_time.isoformat(),
            "end": end_time.isoformat()
        })

        current_time = end_time + timedelta(minutes=15)  # Буфер между задачами

    return schedule_items

@app.route('/api/optimize', methods=['POST'])
def optimize_tasks():
    """API для оптимизации расписания"""
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "No text provided"}), 400

    try:
        parsed_schedule = parse_tasks(data['text'])
        return jsonify({"schedule": parsed_schedule})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/add_to_calendar', methods=['POST'])
def add_to_calendar():
    """Добавление событий в Яндекс.Календарь"""
    if 'yandex_token' not in session:
        return jsonify({"error": "Not authorized with Yandex"}), 401
        
    data = request.get_json()
    if not data or 'schedule' not in data:
        return jsonify({"error": "No schedule provided"}), 400

    success_count = 0
    for event in data['schedule']:
        event_data = {
            "summary": event['task'],
            "start": {"dateTime": event['start'], "timeZone": "Europe/Moscow"},
            "end": {"dateTime": event['end'], "timeZone": "Europe/Moscow"}
        }
        
        try:
            headers = {
                'Authorization': f'OAuth {session["yandex_token"]}',
                'Content-Type': 'application/json'
            }
            
            response = requests.post(
                f"{YANDEX_OAUTH_CONFIG['calendar_api']}/events",
                headers=headers,
                json=event_data
            )
            
            if response.status_code == 201:
                success_count += 1
        except:
            continue

    return jsonify({
        "status": "success",
        "message": f"Успешно добавлено {success_count} из {len(data['schedule'])} задач в календарь"
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
