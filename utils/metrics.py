import copy

def PA(y, y_pred):
    '''
    Point-Adjust Algorithm
    https://github.com/thuml/Anomaly-Transformer/blob/main/solver.py
    '''
    anomaly_state = False
    y_pred_pa = copy.deepcopy(y_pred)
    for i in range(len(y)):
        if y[i] == 1 and y_pred_pa[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if y[j] == 0:
                    break
                else:
                    if y_pred_pa[j] == 0:
                        y_pred_pa[j] = 1
            for j in range(i, len(y)):
                if y[j] == 0:
                    break
                else:
                    if y_pred_pa[j] == 0:
                        y_pred_pa[j] = 1
        elif y[i] == 0:
            anomaly_state = False
        if anomaly_state:
            y_pred_pa[i] = 1
    return