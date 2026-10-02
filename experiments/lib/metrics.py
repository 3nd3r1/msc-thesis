def score(truth, guess):
    tp = (truth & guess).sum()
    fp = (~truth & guess).sum()
    fn = (truth & ~guess).sum()
    f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    return (truth == guess).mean(), f1
