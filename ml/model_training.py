import spacy
from spacy.training import Example
from spacy.util import minibatch, compounding
import random
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 1. Загрузка данных
with open('freeform_task_dataset.json', 'r', encoding='utf-8') as f:
    dataset = json.load(f)

# Пример из датасета
print("Пример текста:", dataset[0]["text"])
print("Сущности:", dataset[0]["entities"])

# 2. Подготовка данных для NER
def convert_to_spacy_format(data):
    spacy_data = []
    for item in data:
        entities = []
        for ent in item["entities"]:
            entities.append((ent["start"], ent["end"], ent["label"]))
        
        spacy_data.append((item["text"], {"entities": entities}))
    return spacy_data

# Разделение данных
random.shuffle(dataset)
split = int(0.8 * len(dataset))
train_data = convert_to_spacy_format(dataset[:split])
test_data = convert_to_spacy_format(dataset[split:]))

# 3. Создание и обучение NER модели
nlp = spacy.blank("ru")
ner = nlp.add_pipe("ner")

# Добавление лейблов
ner.add_label("TASK")
ner.add_label("DURATION")

# Инициализация модели
nlp.initialize()

# Функция для оценки
def evaluate_model(ner_model, examples):
    correct = 0
    total = 0
    for text, annotations in examples:
        doc = ner_model.make_doc(text)
        example = Example.from_dict(doc, annotations)
        ner_model.update([example])
        predicted = ner_model(text)
        
        # Сравнение с истинными значениями
        true_ents = annotations["entities"]
        pred_ents = [(ent.start_char, ent.end_char, ent.label_) for ent in predicted.ents]
        
        for true_ent in true_ents:
            total += 1
            if true_ent in pred_ents:
                correct += 1
    
    return correct / total if total > 0 else 0

# Обучение с прогрессом
print("Начало обучения NER модели...")
for epoch in range(30):
    random.shuffle(train_data)
    losses = {}
    batches = minibatch(train_data, size=compounding(4.0, 32.0, 1.001))
    
    for batch in batches:
        examples = []
        for text, annotations in batch:
            doc = nlp.make_doc(text)
            examples.append(Example.from_dict(doc, annotations))
        
        nlp.update(examples, losses=losses, drop=0.3)
    
    # Оценка
    train_acc = evaluate_model(nlp, train_data[:50])  # Оценка на части тренировочных данных
    test_acc = evaluate_model(nlp, test_data[:50])    # Оценка на части тестовых данных
    
    print(f"Epoch {epoch + 1}, Loss: {losses['ner']:.2f}, Train Acc: {train_acc:.2f}, Test Acc: {test_acc:.2f}")

# 4. Обучение классификатора приоритетов
# Создаем данные для классификации
priority_data = []
priority_keywords = {
    0: ["фильм", "гулять", "книга", "магазин", "отдых"],
    1: ["домашка", "проект", "отчет", "документы", "презентация"],
    2: ["срочно", "быстро", "сейчас", "немедленно", "к сроку", "горящее"]
}

for item in dataset:
    for ent in item["entities"]:
        if ent["label"] == "TASK":
            text = ent["text"].lower()
            priority = 0  # По умолчанию
            
            # Проверка ключевых слов
            for level, keywords in priority_keywords.items():
                if any(keyword in text for keyword in keywords):
                    priority = max(priority, level)  # Берем максимальный приоритет
            
            # Учитываем длительность (дольше = важнее)
            duration = ent["duration"]
            if duration > 120:
                priority = min(priority + 1, 2)
            
            priority_data.append((ent["text"], priority))

# Векторизация и обучение
X = [text for text, _ in priority_data]
y = [priority for _, priority in priority_data]

vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
X_vec = vectorizer.fit_transform(X)

clf = RandomForestClassifier(n_estimators=100, class_weight='balanced')
clf.fit(X_vec, y)

# Проверка точности
print(f"Priority Classifier Accuracy: {clf.score(X_vec, y):.2f}")

# 5. Сохранение моделей
nlp.to_disk("models/ner_model")
joblib.dump(clf, "models/priority_model.pkl")
joblib.dump(vectorizer, "models/priority_vectorizer.pkl")

print("Все модели успешно сохранены")
