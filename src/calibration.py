"""Calibration on API based method."""
import numpy as np
import pickle
from typing import List, Dict


def select_from_samples(sample_file: str) -> List:
    """Select the samples that can be used for calibration.

    Args:
        sample_file (str): The name of the samples stored file.
    """
    with open(sample_file, 'rb') as f:
        samples = pickle.load(f)
    for sample in samples:
        for i in range(len(sample['responses'])):
            if sample['responses'][i] != '':
                sample['responses'][i] = sample['responses'][i][0]
    calibration_set = []
    for i in range(len(samples)//2):
        if len(set(samples[i]['responses'])) >= 3:
            calibration_set.append(samples[i])
    return calibration_set


def LAC_CP(sample: Dict, label: str) -> float:
    """The Least Ambiguous set-valued Classifiers of conformal prediction on single sample.

    Args:
        sample (Dict): A dict object which is the sample of the point.
        label (str): The label for calculation.

    Returns:
        A float which is the conformal score of the true label.
    """
    return 1 - sample['responses'].count(label) / len(sample['responses'])


def calibration(calibration_set: List, error_rate: float) -> float:
    """Using conformal prediction on calibration data.

    Args:
        calibration_set (List): The samples list from the selection of samples from the sampled question.
        error_rate (float): The error rate for calibration on data.

    Returns:
        A float which is the threshold for prediction set construction.
    """
    calibrated_score = []
    for sample in calibration_set:
        calibrated_score.append(LAC_CP(sample, sample['answer']))
    n = len(calibration_set)
    q_level = np.ceil((n+1) * (1-error_rate)) / n
    qhat = np.quantile(calibrated_score, q_level, method='higher')
    return qhat


def estimation(sample_file: str, threshold: float) -> float:
    """Returned the uncertainty estimation on the sampled result.

    Args:
        sample_file (str): The sampled result of questions.
        threshold (float): The threshold on calibration setting.

    Returns:
        A float that contained the final prediction set's size.
    """
    with open(sample_file, 'rb') as f:
        samples = pickle.load(f)
    for sample in samples:
        for i in range(len(sample['responses'])):
            if sample['responses'][i] != '':
                sample['responses'][i] = sample['responses'][i][0]
    prediction_sets = dict()
    coverate = 0
    for i in range(len(samples)//2, len(samples)):
        prediction_sets[samples[i]['id']] = []
        answer_set = set(samples[i]['responses'])
        for answer in answer_set:
            if LAC_CP(samples[i], answer) < threshold:
                if answer == samples[i]['answer']:
                    coverate += 1
                prediction_sets[samples[i]['id']].append(answer)
    return sum(len(value) for value in prediction_sets.values()) / len(prediction_sets), coverate / len(prediction_sets)


calibration_set = select_from_samples('../output/mmlu_llama3.pkl')
threshold = calibration(calibration_set, 0.05)
avg_prediction_set_size, coverate = estimation('../output/mmlu_llama3.pkl', threshold)
print(avg_prediction_set_size, coverate)
