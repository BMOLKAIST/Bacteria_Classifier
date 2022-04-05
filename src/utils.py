# utils.py
import os
import numpy as np
import scipy
from sklearn.metrics import confusion_matrix

def get_confusion(preds, labels):
    confusion = confusion_matrix(labels, preds)
    return confusion
    
def get_sen_spec(confusion, confusion_label):
    idx_to_cls = {v:k for k,v in confusion_label.items()}
    
    confusion_sum = confusion.sum()
    zz = {k:[] for k, _ in confusion_label.items()}

    for i, k in enumerate(confusion_label.keys()):
        tp = confusion[i, i]
        fp = confusion[:, i].sum() - tp
        fn = confusion[i, :].sum() - tp
        tn = confusion_sum - (tp + fp + fn)
        zz[k] += [tp, fp, fn, tn]

    tp, fp, fn, tn = np.array(list(zz.values())).sum(axis=0)    
    print("get_sen_spec | sum of tp, fp, fn, tn : ", tp, fp, fn, tn)
    acc = (tp + tn) / (tp + fp + tn + fn) * 100
    spec = [((tn * 100) / (tn + fp)) for k, (tp, fp, fn, tn) in zz.items()] 
    sens = [((tp * 100) / (tp + fn)) for k, (tp, fp, fn, tn) in zz.items()] + [acc]

    confusion = np.vstack([confusion, np.array(spec)])
    confusion = np.hstack([confusion, np.array(sens).reshape(-1, 1)])
    return confusion


if __name__=="__main__":
    from sklearn.metrics import jaccard_similarity_score, f1_score
    y_true = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    y_pred = np.array([[1, 1, 0], [0, 0, 0], [1, 0, 0]])

    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()

    sk_jss   = jaccard_similarity_score(y_true, y_pred)
    sk_jss_f = jaccard_similarity_score(y_true_f, y_pred_f)

    confusion = confusion_matrix(y_true_f, y_pred_f).ravel()
    print(confusion)
    sensitivity, specificity, precision, recall, f1, jaccard, dice = get_roc_pr(*confusion)
    print("sk_jss : ", sk_jss, "sk_jss_f : ", sk_jss_f, "jss : ", jaccard)


