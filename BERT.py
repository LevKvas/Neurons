import numpy as np
import torch
import evaluate
from datasets import load_dataset
from sklearn.metrics import classification_report

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    pipeline
)



def compute_metrics(eval_pred):
    """Расчет F1-macro для оценки модели."""
    cls_metric = evaluate.load('f1')
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return cls_metric.compute(predictions=predictions, references=labels, average='macro')


def get_label_mappings(dataset):
    """Создает словари для перевода ID в текст и обратно."""
    # Собираем все уникальные категории из всех сплитов
    all_categories = sorted(list(set(dataset['train']['category'])))
    id2label = {i: label for i, label in enumerate(all_categories)}
    label2id = {label: i for i, label in enumerate(all_categories)}
    return id2label, label2id


def load_and_prepare_data(model_name, dataset_name, language):
    """Загрузка данных и токенизация."""
    raw_datasets = load_dataset(dataset_name, language)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    id2label, label2id = get_label_mappings(raw_datasets)

    def preprocess_function(examples):
        # Убираем лишние пробелы и переносы
        texts = [text.replace("\n", " ").strip() for text in examples['text']]
        # Токенизация текста
        result = tokenizer(texts, truncation=True, max_length=128)
        # Превращение текстовой категории в числовой ID (label)
        result['label'] = [label2id[cat] for cat in examples['category']]
        return result

    tokenized_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=raw_datasets['train'].column_names
    )

    return tokenized_datasets, tokenizer, id2label, label2id


def train_model(model_name, tokenized_datasets, tokenizer, id2label, label2id):
    """Настройка и запуск обучения."""
    num_labels = len(id2label)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )

    # Если есть GPU, переводим модель туда
    if torch.cuda.is_available():
        model.cuda()

    training_args = TrainingArguments(
        output_dir='rubert_sib200_results',
        learning_rate=3e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=7,
        weight_decay=0.01,
        warmup_ratio=0.1,  # Плавный старт (10% шагов)
        lr_scheduler_type='cosine',  # Мягкое затухание скорости
        eval_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model='f1',  # Будем сохранять лучшую модель по F1
        report_to="none",
        data_seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    return model


def evaluate_model(model, tokenizer, test_dataset, id2label):
    """Финальная проверка на тестовых данных."""
    device = 0 if torch.cuda.is_available() else -1
    clf_pipeline = pipeline(
        'text-classification',
        model=model,
        tokenizer=tokenizer,
        device=device
    )

    texts = list(test_dataset['text'])
    y_true = list(test_dataset['category'])

    # Инференс
    results = clf_pipeline(texts, batch_size=32)
    y_pred = [x['label'] for x in results]

    print("\nClassification Report on Test Set:")
    print(classification_report(y_true, y_pred))


def main():
    MODEL_NAME = 'DeepPavlov/rubert-base-cased'
    DATASET_NAME = 'Davlan/sib200'
    LANGUAGE = 'rus_Cyrl'

    # 1. Данные
    tokenized_datasets, tokenizer, id2label, label2id = load_and_prepare_data(
        MODEL_NAME, DATASET_NAME, LANGUAGE
    )

    # 2. Обучение
    trained_model = train_model(
        MODEL_NAME, tokenized_datasets, tokenizer, id2label, label2id
    )

    # estimate
    raw_test_set = load_dataset(DATASET_NAME, LANGUAGE, split='test')
    evaluate_model(trained_model, tokenizer, raw_test_set, id2label)


if __name__ == '__main__':
    main()


# Нижние слои (1–4): Учат базовую грамматику, распознают части речи, точки, запятые.
# Средние слои (5–8): Начинают понимать синтаксис и простые связи
# Верхние слои (9–12): Формируют глубокий семантический смысл и абстрактные понятия (то, что нужно для классификации).