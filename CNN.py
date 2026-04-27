import numpy as np
import spacy
from tqdm import tqdm
from sklearn.metrics import classification_report
from typing import Tuple, List
from datasets import load_dataset

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout

import gensim
import logging
import wget
import zipfile


def load_sib200_ru() -> Tuple[Tuple[List[str], List[int]], Tuple[List[str], List[int]], Tuple[List[str], List[int]], List[str]]:
    trainset = load_dataset('Davlan/sib200', 'rus_Cyrl', split='train')
    X_train = trainset['text']
    y_train = trainset['category']
    valset = load_dataset('Davlan/sib200', 'rus_Cyrl', split='validation')
    X_val = valset['text']
    y_val = valset['category']
    testset = load_dataset('Davlan/sib200', 'rus_Cyrl', split='test')
    X_test = testset['text']
    y_test = testset['category']
    categories = set(y_train)
    unknown_categories = set(y_val) - categories
    if len(unknown_categories) > 0:
        err_msg = f'The categories {unknown_categories} are represented in the validation set, but they are not represented in the training set.'
        raise RuntimeError(err_msg)
    unknown_categories = set(y_test) - categories
    if len(unknown_categories) > 0:
        err_msg = f'The categories {unknown_categories} are represented in the test set, but they are not represented in the training set.'
        raise RuntimeError(err_msg)
    categories = sorted(list(categories))
    y_train = [categories.index(it) for it in y_train]
    y_val = [categories.index(it) for it in y_val]
    y_test = [categories.index(it) for it in y_test]
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), categories


def normalize_text(s: str, nlp_pipeline: spacy.Language) -> str:
    doc = nlp_pipeline(s)
    lemmas = [
        ('<NUM>' if token.like_num
         else token.lemma_.lower() + '_' + token.pos_.upper())
        for token in filter(lambda it1: not it1.is_punct and not it1.like_num, doc)
    ]
    return lemmas


def pre_processing(train_data, val_data, test_data, nlp, model):
    train_data_norm = []
    for sent in tqdm(train_data[0]):
        sent_norm = normalize_text(sent, nlp)
        sent_norm = [w for w in sent_norm if w in model.key_to_index]
        train_data_norm.append(sent_norm)
    val_data_norm = []
    for sent in tqdm(val_data[0]):
        sent_norm = normalize_text(sent, nlp)
        sent_norm = [w for w in sent_norm if w in model.key_to_index]
        val_data_norm.append(sent_norm)
    test_data_norm = []
    for sent in tqdm(test_data[0]):
        sent_norm = normalize_text(sent, nlp)
        sent_norm = [w for w in sent_norm if w in model.key_to_index]
        test_data_norm.append(sent_norm)

    tokenizer = Tokenizer(num_words=10000, filters='', lower=False)  # берем топ 10к слов
    tokenizer.fit_on_texts(train_data_norm)
    X_train_seq = tokenizer.texts_to_sequences(train_data_norm)
    X_test_seq = tokenizer.texts_to_sequences(test_data_norm)

    # 2. Делаем все тексты одной длины (например, 100 слов), обрезая или добавляя нули
    X_train_pad = pad_sequences(X_train_seq, maxlen=100)
    X_test_pad = pad_sequences(X_test_seq, maxlen=100)

    return X_train_pad, X_test_pad, tokenizer


def start_CNN(model, tokenizer, classes_list):
    embedding_dim = model.vector_size
    word_index = tokenizer.word_index
    num_words = min(10000, len(word_index) + 1)

    embedding_matrix = np.zeros((num_words, embedding_dim))
    count = 0
    #  заполнение матрицы embedding
    for word, i in word_index.items():
        if i >= num_words:
            continue
        if word in model:
            embedding_matrix[i] = model[word]
            count += 1

    print(f"\n[DEBUG] Из {num_words} слов в матрицу попало: {count}")

    if count == 0:
        print("[DEBUG] Примеры слов из токенайзера:", list(word_index.keys())[:5])
        print("[DEBUG] Пример слова из Word2Vec:", model.index_to_key[0])

    # Описание
    cnn_model = Sequential([
        Embedding(input_dim=num_words,
                  output_dim=embedding_dim,
                  weights=[embedding_matrix],
                  input_length=100,
                  trainable=True),

        Conv1D(filters=128, kernel_size=3, activation='relu'),
        GlobalMaxPooling1D(),

        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(len(classes_list), activation='softmax')
    ])

    cnn_model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

    return cnn_model


def main():
    train_data, val_data, test_data, classes_list = load_sib200_ru()

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    model_url = 'http://vectors.nlpl.eu/repository/20/180.zip'
    # Download the model zip file
    zip_filename = wget.download(model_url)

    # Extract the 'model.bin' file from the zip archive
    with zipfile.ZipFile(zip_filename, 'r') as archive:
        archive.extract('model.bin')

    # Load the model using the extracted file
    model = gensim.models.KeyedVectors.load_word2vec_format('model.bin', binary=True)
    nlp = spacy.load('ru_core_news_sm')

    X_train_pad, X_test_pad, tokenizer = pre_processing(train_data, val_data, test_data, nlp, model)

    cnn_model = start_CNN(model, tokenizer, classes_list)

    y_train = np.array(train_data[1])
    y_test = np.array(test_data[1])

    cnn_model.fit(X_train_pad, y_train, epochs=8, batch_size=32, validation_data=(X_test_pad, y_test))

    # Result of learning
    y_pred_probs = cnn_model.predict(X_test_pad)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print(classification_report(y_test, y_pred, target_names=classes_list))


if __name__ == '__main__':
    main()