# ML Модуль для извлечения задач (NER) с использованием Scikit-learn

Этот модуль отвечает за обучение модели для извлечения именованных сущностей (NER), конкретно для задач (TASK), из текстовых описаний. Вместо SpaCy, данный подход использует библиотеки `scikit-learn`, `nltk` и `joblib`.

## Зависимости

Для установки выполните:
```bash
pip install -r ml/requirements.txt
```

## Обучение модели

```bash
python ml/generate_dataset.py
python ml/model_training.py
```