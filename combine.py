import csv
from sklearn.metrics import average_precision_score, auc, precision_recall_curve, roc_curve, roc_auc_score, \
    accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, precision_recall_fscore_support
import numpy as np

np.random.seed(10)

def read_csv_1(fname):
    label = []
    la = []
    pred = []
    with open(fname, 'r') as f:
        reader = csv.reader(f)
        i = 0
        for line in reader:
            i += 1
            if i == 1:
                continue
            label.append(int(line[0]))
            la.append(int(line[1]))
            pred.append(float(line[2]))

    # print(len(pred), len(label), len(la))
    return pred, label, la


def read_csv_2(fname):
    label = []
    la = []
    pred = []
    with open(fname, 'r') as f:
        reader = csv.reader(f)
        i = 0
        for line in reader:
            i += 1
            if i == 1:
                continue
            label.append(line[0])
            pred.append(float(line[1]))

    # print(len(pred), len(label))
    return pred, label


## AUC-PC
# predict class values
def auc_pc(label, pred):
    lr_probs = np.array(pred)
    testy = np.array([float(l) for l in label])
    no_skill = len(testy[testy == 1]) / len(testy)
    # yhat = np.array(pred)

    lr_precision, lr_recall, _ = precision_recall_curve(testy, lr_probs)
    # lr_f1 = f1_score(testy, yhat)
    # print(type(lr_precision), type(lr_recall))
    # print(np.shape(lr_precision), np.shape(lr_recall))
    lr_auc = auc(lr_recall, lr_precision)
    # summarize scores
    # print('AUC-PR:  auc=%.3f' % ( lr_auc))
    # plot the precision-recall curves

    # pyplot.plot(lr_recall, lr_precision, marker='.', label='Logistic')
    # pyplot.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    # # axis labels
    # pyplot.xlabel('Recall')
    # pyplot.ylabel('Precision')
    # # show the legend
    # pyplot.legend()
    # # show the plot
    # pyplot.show()

    return lr_auc



project = "qt"      # "gerrit" "go" "jdt" "openstack" "platform" "qt"

com_ = f"/home/fanguisheng/Liang/DP/cct5/outputs/models/fine-tuning/JITDefectPrediction/SF/LApredict_filtered_inconsistent_preprocessed/inherit_clone_with_IL_modified0/{project}/all_metrics/bs32_gas4_1500step/checkpoint-best-f1/test_com.csv"

sim_ = f"/home/fanguisheng/Ye/jitdp-test/baselines/SimCom_JIT-main/Sim/pred_scores/test_sim_{project}.csv"

# sim_res
pred, label = read_csv_2(sim_)

# com_res
pred_, label_ = read_csv_2(com_)


## Simple add
pred2 = [pred_[i] + pred[i] for i in range(len(pred_))]
print(pred2[0])
print(type(pred2[0]))
# print(len(pred2), len(label_))
auc2 = roc_auc_score(y_true=np.array(label_), y_score=np.array(pred2))
# print('\n SimCom: ')
# mean_pred = float(sum(pred2)/len(pred2))
# eval_(y_true=np.array(label_),  y_pred=np.array(pred2), thresh = mean_pred )
pc_ = auc_pc(label_, pred2)

t = 1
real_label = [float(l) for l in label_]
real_pred = [1 if p > t else 0 for p in pred2]
f1_ = f1_score(y_true=real_label, y_pred=real_pred)

from sklearn.metrics import recall_score

recall = recall_score(real_label, real_pred, average='binary')
from sklearn.metrics import precision_score

precision = precision_score(real_label, real_pred, average='binary')
acc = accuracy_score(y_true=real_label, y_pred=real_pred)
mcc = matthews_corrcoef(real_label, real_pred)

precision_2, recall_2, f1_2, _ = precision_recall_fscore_support(
        real_label, real_pred, average='binary')

print("AUC-ROC:{}  AUC-PR:{} F1-Score:{} percison:{} recall:{} MCC:{}".format(auc2, pc_, f1_, precision, recall, mcc))
print("done!")

