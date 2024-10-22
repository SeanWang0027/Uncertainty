"""Calibration on API based method."""
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from typing import List, Dict
import argparse


def select_from_samples(sample_file: str, division: float) -> List:
    """Select the samples that can be used for calibration.

    Args:
        sample_file (str): The name of the samples stored file.
        division (float): The division factor for dividing calibration set and estimation set.

    Returns:
        Three lists, in corresponding to whole data, calibration data and estimation data.
    """
    data = []
    with open(sample_file, 'rb') as f:
        while True:
            try:
                data.append(pickle.load(f))
            except EOFError:
                break
    calibration_set = data[:int(len(data) * division)]
    estimation_set = data[int(len(data) * division):]
    return data, calibration_set, estimation_set


def LAC_CP(sample: Dict) -> float:
    """The Least Ambiguous set-valued Classifiers of conformal prediction on single sample.

    Args:
        sample (Dict): A dict object which is the sample of the point.

    Returns:
        A float which is the conformal score of the true label.
    """
    gt_answer = sample['answer'][0] if isinstance(sample['answer'], list) else sample['answer']
    non_conformity_value = 1
    for response, prob in sample['candidates_logit'].items():
        if gt_answer in response:
            non_conformity_value -= prob
    return non_conformity_value


def APS_CP(sample: Dict, label: str) -> float:
    """The Adaptive Prediction Sets of conformal prediction on single sample.

    Args:
        sample (Dict): A dict object which is the sample of the point.
        label (str): The label for APS nonconformity function.

    Returns:
        A float which is the conformal score of the true label.
    """
    non_conformity_value = 0
    for response, prob in sample['candidates_logit'].items():
        if prob >= sample['candidates_logit'][label]:
            non_conformity_value += prob
    return non_conformity_value


def calibration(calibration_set: List, error_rate: float, nonconformity_method='LAC') -> float:
    """Using conformal prediction on calibration data.

    Args:
        calibration_set (List): The samples list from the selection of samples from the sampled question.
        error_rate (float): The error rate for calibration on data.
        nonconformity_method (str): The method for nonconformity function.

    Returns:
        A float which is the threshold for prediction set construction.
    """
    calibrated_score = []
    for sample in calibration_set:
        if nonconformity_method == 'LAC':
            calibrated_score.append(LAC_CP(sample))
        else:
            gt_answer = sample['answer'][0] if isinstance(sample['answer'], list) else sample['answer']
            calibrated_score.append(APS_CP(sample, gt_answer))
    n = len(calibration_set)
    q_level = np.ceil((n+1) * (1-error_rate)) / n
    print('q_level:', q_level)
    threshold = np.quantile(calibrated_score, q_level, method='higher')
    return threshold


def estimation(estimation_set: List, threshold: float, error_rate: float, nonconformity_method='LAC') -> float:
    """Returned the uncertainty estimation on the sampled result.

    Args:
        estimation_set (List): A list that contains the points for estimation.
        threshold (float): The threshold on calibration setting.
        error_rate (float): The error rate for calibration on data.
        nonconformity_method (str): The method for nonconformity function.

    Returns:
        Three floats, that contained the final prediction expected coverrate, actual coverrate and avg set size for prediction set.
    """
    prediction_sets = dict()
    coverate = 0
    set_size = 0
    for item in estimation_set:
        prediction_sets[item['id']] = []
        for answer, logit in item['candidates_logit'].items():
            if nonconformity_method == 'LAC':
                if 1 - logit <= threshold:
                    prediction_sets[item['id']].append(answer)
            else:
                if APS_CP(item, answer) <= threshold:
                    prediction_sets[item['id']].append(answer)
        for sequence in prediction_sets[item['id']]:
            gt_answer = item['answer'][0] if isinstance(item['answer'], list) else item['answer']
            if gt_answer in sequence:
                coverate += 1
                break
        if item['exist_answer']:
            mark = -1
            for sequence in prediction_sets[item['id']]:
                if gt_answer in sequence:
                    mark += 1
            set_size -= mark
        set_size += len(prediction_sets[item['id']])
    return 1-error_rate, coverate/len(estimation_set), set_size/len(estimation_set)


def plot_calibration(expected_cover_rates: List, coverates: List, dataset='trivia_qa', start=0, end=1000, division=0.5) -> None:
    """Plot the calibration figure.

    Args:
        expected_cover_rates (List): The expected cover rate list.
        coverates (List): The actual cover rate.
        dataset (str): The current estimation's dataset.
        start (int): The start of the data points.
        end (int): The end of the data points.
        division (float): The division factor.
    """
    ground_truth = expected_cover_rates
    plt.figure(figsize=(8, 6))
    plt.plot(expected_cover_rates, ground_truth, linestyle='--', color='grey', label='Best Calibration line.')
    plt.plot(expected_cover_rates, coverates, marker='o', color='blue', label='Empirical Coverage Rate')
    plt.xlabel("Target Correctness Coverage Rate")
    plt.ylabel("Empirical Correctness Coverage Rate")
    plt.legend()
    plt.grid(True)
    save_path = f'../pics/mistral/{dataset}/{dataset}_{division}_5.png'
    directory = os.path.dirname(save_path)
    os.makedirs(directory, exist_ok=True)
    plt.savefig(save_path)


# division = 0.50
# start = 0
# end = 6000
# data, calibration_set, estimation_set = select_from_samples('../output/mistral/trivia_qa/trivia_qa_0_2000_5.pkl', division=division)
# error_rates = list(np.arange(0.05, 1.05, 0.05))
# target_coverates = []
# coverates = []
# for error_rate in error_rates:
#     target_coverates.append(1-error_rate)
#     threshold = calibration(calibration_set, error_rate, 'APS')
#     print(threshold)
#     expected_cover_rate, coverate, set_size = estimation(estimation_set, threshold, error_rate, 'APS')
#     coverates.append(coverate)
#     print(expected_cover_rate, coverate, set_size)
# plot_calibration(expected_cover_rates=target_coverates, coverates=coverates, dataset='trivia_qa', start=start, end=end, division=division)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calibration on API based method')
    parser.add_argument('--division', type=float, default=0.5, help='Division factor for calibration and estimation sets')
    parser.add_argument('--start', type=int, default=0, help='Start index of the dataset')
    parser.add_argument('--end', type=int, default=6000, help='End index of the dataset')
    parser.add_argument('--input_data', type=str, required=True, help='Path to the input data file')
    parser.add_argument('--dataset', type=str, default='trivia_qa', help='Name of the dataset')
    parser.add_argument('--nonconformity_method', type=str, default='LAC', help='Nonconformity method to use (LAC or APS)')

    args = parser.parse_args()
    data, calibration_set, estimation_set = select_from_samples(args.input_data, division=args.division)

    error_rates = list(np.arange(0.05, 1.05, 0.05))
    target_coverates = []
    coverates = []

    for error_rate in error_rates:
        target_coverates.append(1-error_rate)
        threshold = calibration(calibration_set, error_rate, args.nonconformity_method)
        print(f"Threshold: {threshold}")
        expected_cover_rate, coverate, set_size = estimation(estimation_set, threshold, error_rate, args.nonconformity_method)
        coverates.append(coverate)
        print(f"Expected Coverage Rate: {expected_cover_rate}, Empirical Coverage Rate: {coverate}, Set Size: {set_size}")

    plot_calibration(expected_cover_rates=target_coverates, coverates=coverates, dataset=args.dataset, start=args.start, end=args.end, division=args.division)
