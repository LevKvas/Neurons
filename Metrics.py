from sklearn.metrics import classification_report


def calc_metrics(y_true, y_pred):
    print("\nОтчет о классификации:")
    # output_dict=False выведет красивую таблицу в консоль
    report = classification_report(y_true, y_pred)
    print(report)