from typing import List


def get_confusion_matrix(
    actual: List[int], predicted: List[int]
) -> List[List[int]]:
    """Computes confusion matrix from lists of actual or predicted labels.

    Args:
        actual: List of integers (0 or 1) representing the actual classes of
            some instances.
        predicted: List of integers (0 or 1) representing the predicted classes
            of the corresponding instances.

    Returns:
        List of two lists of length 2 each, representing the confusion matrix.
    """
    confusion_matrix = [[0, 0], [0, 0]]
    try:
        for i in range(len(predicted)):
            if predicted[i] == 1 and actual[i] == 1:
                confusion_matrix[1][1] += 1
            elif predicted[i] == 0 and actual[i] == 0:
                confusion_matrix[0][0] += 1 
            elif predicted[i] == 1 and actual[i] == 0:
                confusion_matrix[0][1] += 1
            elif predicted[i] == 0 and actual[i] == 1:
                confusion_matrix[1][0] += 1
            
        return confusion_matrix
    except:
        return [[0, 0], [0, 0]]


def accuracy(actual: List[int], predicted: List[int]) -> float:
    """Computes the accuracy from lists of actual or predicted labels.

    Args:
        actual: List of integers (0 or 1) representing the actual classes of
            some instances.
        predicted: List of integers (0 or 1) representing the predicted classes
            of the corresponding instances.

    Returns:
        Accuracy as a float.
    """
    try:
        confusion_matrix = get_confusion_matrix(actual, predicted)
        tp = confusion_matrix[1][1]
        tn = confusion_matrix[0][0]
        fp = confusion_matrix[0][1]
        fn = confusion_matrix[1][0]
        return (tp + tn) / (tp + tn + fp + fn)
    except:
        return 0


def precision(actual: List[int], predicted: List[int]) -> float:
    """Computes the precision from lists of actual or predicted labels.

    Args:
        actual: List of integers (0 or 1) representing the actual classes of
            some instances.
        predicted: List of integers (0 or 1) representing the predicted classes
            of the corresponding instances.

    Returns:
        Precision as a float.
    """
    try:
        confusion_matrix = get_confusion_matrix(actual, predicted)
        tp = confusion_matrix[1][1]
        tn = confusion_matrix[0][0]
        fp = confusion_matrix[0][1]
        fn = confusion_matrix[1][0]
        return tp / (tp + fp)
    except:
        return 0


def recall(actual: List[int], predicted: List[int]) -> float:
    """Computes the recall from lists of actual or predicted labels.

    Args:
        actual: List of integers (0 or 1) representing the actual classes of
            some instances.
        predicted: List of integers (0 or 1) representing the predicted classes
            of the corresponding instances.

    Returns:
        Recall as a float.
    """
    try:
        confusion_matrix = get_confusion_matrix(actual, predicted)
        tp = confusion_matrix[1][1]
        tn = confusion_matrix[0][0]
        fp = confusion_matrix[0][1]
        fn = confusion_matrix[1][0]
        return tp / (tp + fn)
    except:
        return 0


def f1(actual: List[int], predicted: List[int]) -> float:
    """Computes the F1-score from lists of actual or predicted labels.

    Args:
        actual: List of integers (0 or 1) representing the actual classes of
            some instances.
        predicted: List of integers (0 or 1) representing the predicted classes
            of the corresponding instances.

    Returns:
        float of harmonic mean of precision and recall.
    """
    try:
        confusion_matrix = get_confusion_matrix(actual, predicted)
        p = precision(actual, predicted)
        r = recall(actual, predicted)
        return (2 * p * r) / (p + r)
    except:
        return 0


def false_positive_rate(actual: List[int], predicted: List[int]) -> float:
    """Computes the false positive rate from lists of actual or predicted
        labels.

    Args:
        actual: List of integers (0 or 1) representing the actual classes of
            some instances.
        predicted: List of integers (0 or 1) representing the predicted classes
            of the corresponding instances.

    Returns:
        float of number of instances incorrectly classified as positive divided
            by number of actually negative instances.
    """
    try:
        confusion_matrix = get_confusion_matrix(actual, predicted)
        tp = confusion_matrix[1][1]
        tn = confusion_matrix[0][0]
        fp = confusion_matrix[0][1]
        fn = confusion_matrix[1][0]
        return fp / (fp + tn)
    except:
        return 0


def false_negative_rate(actual: List[int], predicted: List[int]) -> float:
    """Computes the false negative rate from lists of actual or predicted
        labels.

    Args:
        actual: List of integers (0 or 1) representing the actual classes of
            some instances.
        predicted: List of integers (0 or 1) representing the predicted classes
            of the corresponding instances.

    Returns:
        float of number of instances incorrectly classified as negative divided
            by number of actually positive instances.
    """
    try:
        confusion_matrix = get_confusion_matrix(actual, predicted)
        tp = confusion_matrix[1][1]
        tn = confusion_matrix[0][0]
        fp = confusion_matrix[0][1]
        fn = confusion_matrix[1][0]
        return fn / (fn + tp)
    except:
        return 0
