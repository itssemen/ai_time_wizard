from flask import Flask, jsonify, request, send_from_directory
import re
from datetime import datetime, timedelta

app = Flask(__name__, static_folder='static')

@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

# Удаляем старый /api/data, так как он больше не нужен
# @app.route('/api/data')
# def get_data():
#     return jsonify({'message': 'Hello from Backend!'})

def parse_tasks_for_backend(text):
    """
    Простой парсер для MVP на стороне бэкенда.
    Извлекает задачи и их предполагаемую длительность.
    """
    # Используем re.split для разделения по новой строке или запятой
    lines = re.split(r'[\n,]', text)
    lines = [line.strip() for line in lines if line.strip()]

    schedule_items = []
    current_time = datetime.now().replace(hour=9, minute=0, second=0, microsecond=0)

    for i, line in enumerate(lines):
        task_name = line
        duration_minutes = 60  # Длительность по умолчанию 1 час

        time_match = re.search(r'(\d+)\s*(час|часа|часов|мин|минут)', line, re.IGNORECASE)
        if time_match:
            value = int(time_match.group(1))
            unit = time_match.group(2).lower()
            if unit.startswith('час'):
                duration_minutes = value * 60
            elif unit.startswith('мин'):
                duration_minutes = value

            # Удаляем упоминание времени из названия задачи
            task_name = line[:time_match.start()].strip() + line[time_match.end():].strip()
            task_name = task_name.strip(" ,.-")


        # Убираем лишние предлоги/союзы в начале, если они остались
        task_name = re.sub(r'^(с|в|на|и)\s+', '', task_name, flags=re.IGNORECASE).strip()

        if not task_name: # Если после удаления времени название стало пустым
             task_name = f"Задача {i+1}"


        start_time_dt = current_time
        end_time_dt = start_time_dt + timedelta(minutes=duration_minutes)

        schedule_items.append({
            "time": f"{start_time_dt.strftime('%H:%M')}-{end_time_dt.strftime('%H:%M')}",
            "task": task_name
        })

        # Обновляем current_time для следующей задачи, добавляя 15 минут перерыва
        current_time = end_time_dt + timedelta(minutes=15)

    return schedule_items

@app.route('/api/optimize', methods=['POST'])
def optimize_tasks():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "No text provided"}), 400

    input_text = data['text']

    # Временно (MVP): Используем простой парсер на бэкенде
    # В будущем здесь будет логика ИИ
    parsed_schedule = parse_tasks_for_backend(input_text)

    if not parsed_schedule:
        # Возвращаем пустой список, если ничего не распознано,
        # чтобы фронтенд мог это обработать
        return jsonify({"schedule": []})

    return jsonify({"schedule": parsed_schedule})

@app.route('/api/add_to_calendar', methods=['POST'])
def add_to_calendar():
    data = request.get_json()
    if not data or 'schedule' not in data:
        return jsonify({"error": "No schedule provided"}), 400

    # schedule_data = data['schedule']
    # Здесь в будущем будет логика взаимодействия с API Яндекс.Календаря

    # Временно (MVP): Просто возвращаем успех
    return jsonify({"status": "success", "message": "Задачи (теоретически) добавлены в ваш календарь 🎉"})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
