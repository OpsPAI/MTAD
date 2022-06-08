import copy


def adjust_pred(pred, label):
    """
    Borrow from https://github.com/NetManAIOps/OmniAnomaly/blob/master/omni_anomaly/eval_methods.py
    """
    adjusted_pred = copy.deepcopy(pred)

    anomaly_state = False
    anomaly_count = 0
    latency = 0
    for i in range(len(adjusted_pred)):
        if label[i] and adjusted_pred[i] and not anomaly_state:
            anomaly_state = True
            anomaly_count += 1
            for j in range(i, 0, -1):
                if not label[j]:
                    break
                else:
                    if not adjusted_pred[j]:
                        adjusted_pred[j] = True
                        latency += 1
        elif not label[i]:
            anomaly_state = False
        if anomaly_state:
            adjusted_pred[i] = True
    return adjusted_pred
