# ML Модуль для извлечения задач (NER) с использованием Scikit-learn

Этот модуль отвечает за обучение модели для извлечения именованных сущностей (NER), конкретно для задач (TASK), из текстовых описаний. Вместо SpaCy, данный подход использует библиотеки `scikit-learn`, `nltk` и `joblib`.

## Основной принцип

1.  **Подготовка данных**:
    *   Тексты токенизируются (разбиваются на слова/токены).
    *   Для каждого токена определяется IOB-тег (`B-TASK`, `I-TASK`, `O`):
        *   `B-TASK`: Начало задачи.
        *   `I-TASK`: Внутри задачи.
        *   `O`: Вне задачи.
2.  **Извлечение признаков**:
    *   Для каждого токена генерируется набор признаков (например, само слово, его префиксы/суффиксы, информация о регистре, а также признаки для соседних слов).
3.  **Векторизация**:
    *   Полученные признаки преобразуются в числовые векторы с помощью `DictVectorizer` из `scikit-learn`.
4.  **Обучение модели**:
    *   На векторизованных признаках и IOB-тегах обучается классификатор из `scikit-learn` (например, `LogisticRegression`). Модель учится предсказывать IOB-тег для каждого токена.
5.  **Сохранение модели**:
    *   Обученная модель и векторизатор сохраняются в файлы `.pkl` с помощью `joblib`.

## Файлы

*   `model_training.py`: Основной скрипт для обучения модели, включая подготовку данных, векторизацию, обучение, оценку и сохранение.
*   `generate_dataset.py`: Скрипт для генерации синтетического датасета `freeform_task_dataset.json`. (Предполагается, что он остается без изменений от предыдущей версии).
*   `freeform_task_dataset.json`: Датасет в формате JSON, содержащий тексты и аннотации сущностей.
*   `requirements.txt`: Список Python-зависимостей.
*   `models_sklearn/`: Директория, куда сохраняются обученные модели (`ner_model.pkl`) и векторизаторы (`vectorizer.pkl`).

## Зависимости

Основные зависимости перечислены в `ml/requirements.txt`:

*   `scikit-learn`
*   `joblib`
*   `nltk`

Для установки выполните:
```bash
pip install -r ml/requirements.txt
```
При первом запуске `model_training.py` также могут быть автоматически загружены необходимые ресурсы `nltk` (например, `punkt`).

## Обучение модели

Для обучения модели запустите скрипт `model_training.py` из корневой директории проекта:

```bash
python ml/model_training.py
```

Скрипт выполнит следующие шаги:
1.  Загрузит данные из `freeform_task_dataset.json`.
2.  Подготовит данные (токенизация, IOB-тегирование).
3.  Извлечет признаки и векторизует их.
4.  Обучит модель логистической регрессии.
5.  Выведет отчет по качеству классификации IOB-тегов на тестовой выборке.
6.  Сохранит обученный векторизатор в `ml/models_sklearn/vectorizer.pkl` и модель в `ml/models_sklearn/ner_model.pkl`.

## Использование обученной модели (Пример)

Для использования обученной модели вам понадобятся сохраненные `vectorizer.pkl` и `ner_model.pkl`.

```python
import joblib
import nltk # для токенизации и, возможно, извлечения признаков

# Загрузка моделей
# script_dir = os.path.dirname(__file__) # Определите script_dir или укажите полные пути
# output_model_dir = os.path.join(script_dir, "models_sklearn")
# vectorizer_path = os.path.join(output_model_dir, "vectorizer.pkl")
# model_path = os.path.join(output_model_dir, "ner_model.pkl")

# vectorizer = joblib.load(vectorizer_path)
# model = joblib.load(model_path)

# # Функция токенизации (должна быть такой же, как при обучении)
# def tokenize_text(text):
#     # Пример с WhitespaceTokenizer
#     tokenizer = nltk.tokenize.WhitespaceTokenizer()
#     tokens_info = []
#     for start, end in tokenizer.span_tokenize(text):
#         tokens_info.append({'text': text[start:end], 'start': start, 'end': end})
#     return tokens_info

# # Функция извлечения признаков (должна быть такой же, как при обучении)
# def word2features(sent_tokens_text, i):
#     # ... (реализация из model_training.py) ...
#     pass

# def sent2features(sent_tokens_text):
#     return [word2features(sent_tokens_text, i) for i in range(len(sent_tokens_text))]

# def extract_tasks_from_text(text, vectorizer, model):
#     tokens_info = tokenize_text(text)
#     if not tokens_info:
#         return []

#     sent_tokens_text = [t['text'] for t in tokens_info]
#     features = sent2features(sent_tokens_text)
#     vectorized_features = vectorizer.transform(features)
#     predicted_tags = model.predict(vectorized_features)

#     tasks = []
#     current_task_tokens = []
#     current_task_start_char = -1

#     for i, token_info in enumerate(tokens_info):
#         tag = predicted_tags[i]
#         token_text = token_info['text']
#         start_char = token_info['start']
#         end_char = token_info['end']

#         if tag == 'B-TASK':
#             if current_task_tokens: # Завершить предыдущую задачу, если она была
#                 tasks.append({
#                     "text": " ".join(current_task_tokens),
#                     "start": current_task_start_char,
#                     # Энд нужно будет вычислить по последнему токену предыдущей задачи
#                 })
#             current_task_tokens = [token_text]
#             current_task_start_char = start_char
#         elif tag == 'I-TASK':
#             if current_task_tokens:
#                 current_task_tokens.append(token_text)
#             # else: # I-TASK без B-TASK - можно игнорировать или обрабатывать как ошибку
#         elif tag == 'O':
#             if current_task_tokens: # Завершить текущую задачу
#                 # Конец предыдущего токена (или текущего, если он последний в задаче)
#                 # Это упрощение, нужно аккуратнее определять end_char для всей задачи
#                 tasks.append({
#                     "text": " ".join(current_task_tokens),
#                     "start": current_task_start_char,
#                     "end": tokens_info[i-1]['end'] if i > 0 and current_task_tokens else end_char
#                 })
#                 current_task_tokens = []
#                 current_task_start_char = -1

#     # Если последняя задача не была закрыта тегом 'O'
#     if current_task_tokens:
#          tasks.append({
#             "text": " ".join(current_task_tokens),
#             "start": current_task_start_char,
#             "end": tokens_info[-1]['end']
#         })
#     return tasks

# # Пример использования
# # text_to_analyze = "Мне нужно купить молоко и позвонить маме завтра"
# # extracted_tasks = extract_tasks_from_text(text_to_analyze, vectorizer, model)
# # print(extracted_tasks)
```
**Примечание**: Пример кода для использования модели (извлечения задач) дан в общих чертах. Функция `extract_tasks_from_text` потребует аккуратной реализации для корректного восстановления границ сущностей из IOB-тегов и позиций токенов. Это нетривиальная задача, так как токенизация может изменять исходный текст, и нужно точно сопоставлять токены с их позициями в оригинальном тексте.

## Дальнейшие улучшения (TODO)

*   **Улучшение токенизации**: `WhitespaceTokenizer` очень прост. Для русского языка лучше использовать более продвинутые токенизаторы (например, `nltk.RegexpTokenizer` с настроенным паттерном или попробовать решить проблему с `nltk.word_tokenize(language='russian')`).
*   **Более сложные признаки**: Добавить POS-теги, леммы, информацию о форме слова и т.д.
*   **Подбор модели и гиперпараметров**: Экспериментировать с другими классификаторами `sklearn` (например, `RandomForestClassifier`, `SGDClassifier`, `Perceptron`) и их параметрами.
*   **Оценка на уровне сущностей**: Реализовать или использовать готовую метрику для оценки качества извлечения полных сущностей (а не только IOB-тегов).
*   **Использование CRF**: Для последовательного тегирования, каким является NER, модели Conditional Random Fields (CRF) часто показывают лучшие результаты. Библиотека `sklearn-crfsuite` предоставляет такую возможность.
```
