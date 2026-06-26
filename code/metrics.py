from sklearn.metrics import cohen_kappa_score

LABEL_OFFSET = 1

def quadratic_weighted_kappa(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def label_to_score(label):
    return label + LABEL_OFFSET

def score_to_label(score):
    return score - LABEL_OFFSET
