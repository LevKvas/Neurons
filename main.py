from tqdm import tqdm

from transformers import pipeline
from config import Config
from Classifier import classify
from PromptBuilder import PromptTemplates
from Datasets import get_train_dataset, get_test_dataset, get_val_dataset
from Metrics import calc_metrics


def get_examples() -> dict:
    train = get_train_dataset()

    features = train.features['category']

    if hasattr(features, 'names'):
        category_names = features.names
    else:
        train = train.class_encode_column("category")
        category_names = train.features['category'].names

    examples = {}
    for i, name in enumerate(category_names):
        # Find example
        example_row = next(item for item in train if item['category'] == i)
        examples[name] = example_row['text']

    return examples


def run_test(classifier, test_set, examples):
    y_true = []
    y_pred = []

    print("Запуск классификации тестовой выборки...")
    for item in tqdm(test_set):
        text = item['text']
        # get name of category
        label_name = test_set.features['category'].names[item['category']]

        messages = PromptTemplates.build_messages(text, examples)

        allowed_labels = test_set.features['category'].names
        prediction = classify(messages, classifier, allowed_labels)

        y_true.append(label_name)
        y_pred.append(prediction)

    return y_true, y_pred


def main():
    config = Config()
    classifier = pipeline(
        "text-generation",
        model=config.model_id,
        device_map="cpu"
    )

    examples = get_examples()
    val_set = get_val_dataset().class_encode_column("category")

    for i in range(5):
        item = val_set[i]
        text_to_classify = item['text']
        true_label = val_set.features['category'].names[item['category']]

        messages = PromptTemplates.build_messages(text_to_classify, examples)

        allowed_labels = val_set.features['category'].names
        result = classify(messages, classifier, allowed_labels)

        print(f"Текст: {text_to_classify[:50]}...")
        print(f"Предсказано: {result} | На самом деле: {true_label}\n")

    # Check on test dataset
    test_set = get_test_dataset().class_encode_column("category")
    y_true, y_pred = run_test(classifier, test_set, examples)

    # Output metrics
    calc_metrics(y_true, y_pred)


if __name__ == "__main__":
    main()