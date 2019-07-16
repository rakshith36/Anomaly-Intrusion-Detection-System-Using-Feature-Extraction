from sklearn import svm
from sklearn.metrics import confusion_matrix, f1_score, recall_score, roc_curve
from sklearn.preprocessing import binarize
import numpy as np
import datetime
from datasets.kdd99 import input_data
from myae import MyAutoEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import  matplotlib.pyplot as plt

fprArray = []
tprArray = []

def Calculate(attackType):
    # This is for KDDKup
    TRAIN_FILE = 'kddcup.data_10_percent.gz'
    TEST_FILE = 'corrected.gz'

    #This is for NSL
    #TRAIN_FILE = 'KDDTrain.gz'
    #TEST_FILE = 'KDDTest.gz'

    ninput = 41

    kdd99 = input_data.read_data_sets(attackType, TRAIN_FILE, TEST_FILE)
    print(datetime.datetime.now(), ': data loaded.train examples num:%s,test examples num:%s' % (
    kdd99.train.num_examples, kdd99.test.num_examples))

    Xs, Ys = kdd99.train.random_select()
    tXs, tYs = kdd99.test.random_select()

    #------------- SVM Start-------------
    # ae = MyAutoEncoder(ninput, [32, 16])
    # ae.fit(kdd99.train)
    #
    # ae_Xs = ae.transform(Xs)
    # ae_tXs = ae.transform(tXs)
    # # Uncomment while need Using SVM
    # #clf_ae = svm.SVC(probability=True)
    # clf_ae = RandomForestClassifier(n_jobs=2)
    #
    # clf_ae.fit(ae_Xs, Ys)
    # result_ae = clf_ae.predict(ae_tXs)
    # ae_cm = confusion_matrix(tYs, result_ae)
    #--------------SVM End--------------------

    #--------------PCA Start---------------------
    pca = PCA(n_components=16)
    pca.fit(Xs)

    ae_Xs = pca.transform(Xs)
    ae_tXs = pca.transform(tXs)

    clf_ae = svm.SVC(probability=True)
    clf_ae.fit(ae_Xs, Ys)
    result_ae = clf_ae.predict(ae_tXs)
    ae_cm = confusion_matrix(tYs, result_ae)
    #--------------PCA END---------------------

    # ----------------ROC Start---------------------------

    y_pred_prob = clf_ae.predict_proba(ae_tXs)[:, 1]

    line = np.linspace(0, 1, 2)
    plt.plot(line, 'r')

    print('Before', tYs)
    y_true = np.array(tYs)
    print('After', y_true)

    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)

    fprArray.append(fpr)
    tprArray.append(tpr)

    # ----------------ROC End---------------------------

    print(attackType,' :---' )
    print(datetime.datetime.now(), ': start..')
    print('confusion matrix:---', ae_cm)
    print(ae_cm)
    precision = np.mean(np.equal(result_ae, tYs))
    recall = recall_score(tYs, result_ae, average="macro")
    f = f1_score(tYs, result_ae, average="macro")
    print(attackType," :-\t%0.5f\t%0.5f\t%0.5f\t" % (precision, recall, f))
    print(datetime.datetime.now(), ': End...')

# to print the over all accuracy
'''
attackType = 'All'
Calculate(attackType)
'''

# to print the individual accuracy
attackType = 'DoS'
Calculate(attackType)

attackType = 'Probe'
Calculate(attackType)

attackType = 'R2L'
Calculate(attackType)

attackType = 'U2R'
Calculate(attackType)

index=0
for eachfpr in fprArray:
    plt.plot(fprArray[index], tprArray[index])
    index += 1

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC Curve for Network Attack Classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)

plt.show()