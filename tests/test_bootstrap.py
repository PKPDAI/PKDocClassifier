from pk_classifier.bootstrap import split_train_val_test

FEATURES_SAMPLE = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
LABELS_SAMPLE = {'label': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1], 'pmid': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]}


def test_split_train_val_test():
    x_train, x_dev, x_test, y_train, y_dev, \
    y_test, pmids_train, pmids_val, pmids_test = split_train_val_test(features=FEATURES_SAMPLE,
                                                                      labels=LABELS_SAMPLE, test_size=0.2, seed=123)

    # test sizes
    assert len(x_train) == len(y_train) == 6
    assert len(x_dev) == len(y_dev) == 2
    assert len(x_test) == len(y_test) == 2
    # test proportions (stratification)
    initial_proportion = sum(LABELS_SAMPLE['label']) / len(LABELS_SAMPLE['label'])  # proportion of samples with label 1
    assert sum(y_train) / len(y_train) == initial_proportion
    assert sum(y_dev) / len(y_dev) == initial_proportion
    assert sum(y_dev) / len(y_dev) == initial_proportion


def test_split_train_val_test_2():
    x_train, x_dev, x_test, y_train, y_dev, \
    y_test, pmids_train, pmids_val, pmids_test = split_train_val_test(features=FEATURES_SAMPLE,
                                                                      labels=LABELS_SAMPLE, test_size=0.4, seed=123)

    # test sizes
    assert len(x_train) == len(y_train) == 2
    assert len(x_dev) == len(y_dev) == 4
    assert len(x_test) == len(y_test) == 4
    # test proportions (stratification)
    initial_proportion = sum(LABELS_SAMPLE['label']) / len(LABELS_SAMPLE['label'])  # proportion of samples with label 1
    assert sum(y_train) / len(y_train) == initial_proportion
    assert sum(y_dev) / len(y_dev) == initial_proportion
    assert sum(y_dev) / len(y_dev) == initial_proportion
