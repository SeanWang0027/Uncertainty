import pickle
import matplotlib.pyplot as plt
import numpy as np
data = []
with open('../output/trivia_qa_7000_8000.pkl', 'rb') as f:
    while True:
        try:
            data.append(pickle.load(f))
        except EOFError:
            break
def LAC_CP(exp):
    non_conformity_value = 1
    for response, prob in exp['candidates_logit'].items():
        if exp['answer'] in response:
            non_conformity_value -= prob
    return non_conformity_value
calibration_set = data[:200]
estimation_set = data[200:]
calibrated_score = []
error_rates = list(np.arange(0.05, 1.05, 0.05))
target_coverates = []
coverates = []
for error_rate in error_rates:
    error_rate = 0.2
    target_coverates.append(1-error_rate)
    for sample in calibration_set:
        calibrated_score.append(LAC_CP(sample))
    n = len(calibration_set)
    q_level = np.ceil((n+1) * (1-error_rate)) / n
    threshold = np.quantile(calibrated_score, q_level, method='higher')
    print(threshold)
    prediction_sets = dict()
    coverate = 0
    set_size = 0
    total = 0
    for item in estimation_set:
        prediction_sets[item['id']] = []
        for answer, logit in item['candidates_logit'].items():
            if 1 - logit <= threshold:
                prediction_sets[item['id']].append(answer)
        for sequence in prediction_sets[item['id']]:
            if item['answer'] in sequence:
                coverate += 1
                break
        if item['exist_answer']:
            mark = -1
            for sequence in prediction_sets[item['id']]:
                if item['answer'] in sequence:
                    mark += 1
            set_size -= mark
        set_size += len(prediction_sets[item['id']])
    coverates.append(coverate/len(estimation_set))
    print(1-error_rate, coverate/len(estimation_set), set_size/len(estimation_set))
    break
# ground_truth = target_coverates

# plt.figure(figsize=(8, 6))
# plt.plot(target_coverates, ground_truth, linestyle='--', color='grey', label='Best Calibration line.')
# plt.plot(target_coverates, coverates, marker='o', color='blue', label='Empirical Coverage Rate')

# # Labels and legend
# plt.xlabel("Target Correctness Coverage Rate")
# plt.ylabel("Empirical Correctness Coverage Rate")
# plt.legend()
# plt.grid(True)
# plt.savefig('./test.png')