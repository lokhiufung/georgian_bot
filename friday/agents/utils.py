import numpy as np


def gap_mean_filter(answers, n_std: float=1.0):
    if len(answers) > 1:
        scores = [answer['score'] for answer in answers]
        gaps = [next_score - score for next_score, score in zip(scores[1:], scores[:-1])]
        mean_gap = np.mean(gaps)
        std_gap = np.std(gaps, ddof=1)

        if gaps[0] > mean_gap + n_std * std_gap:
            return [answers[0]]        
    return answers  


def gap_iqr_filter(answers, n_iqr: float=1.5):
    if len(answers) > 1:
        scores = [answer['score'] for answer in answers]
        gaps = [next_score - score for next_score, score in zip(scores[1:], scores[:-1])]
        q75 = np.quantile(gaps, q=0.75)
        q25 = np.quantile(gaps, q=0.25)
        iqr = q75 - q25

        if gaps[0] > q75 + n_iqr * iqr:
            return [answers[0]]        
    return answers  
