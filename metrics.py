import numpy as np


def binary_classification_metrics(timing_pred, timing_truth, eps):

    true_positive = 0
    i = 0
    for a in timing_truth:
        for b in range(i, len(timing_pred)):
            if abs(a - timing_pred[b]) < eps:
                print("Matched blinks ", a, timing_pred[b])
                true_positive +=1
                i +=1
                break

    print('Count Matched Blinks ', true_positive)
    false_positive = max(0, len(timing_pred) - true_positive)
    false_negative = max(0,len(timing_truth) - true_positive)

    try:
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        f1 = 2 * (precision * recall) / (precision + recall)
    except Exception as e:
        print("Metrics error")

        pass

    return precision, recall, f1


