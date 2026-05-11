from datasets import load_dataset
from config import Config

cfg = Config()


def get_test_dataset():
    test_set = load_dataset(cfg.DATASET_NAME, cfg.DATASET_LANGUAGE, split='test')
    return test_set


def get_train_dataset():
    train_set = load_dataset(cfg.DATASET_NAME, cfg.DATASET_LANGUAGE, split='train')
    return train_set


def get_val_dataset():
    validation_set = load_dataset(cfg.DATASET_NAME, cfg.DATASET_LANGUAGE, split='validation')
    return validation_set





