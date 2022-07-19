import numpy
import scipy.stats
import library.LogisticRegression as LR
import matplotlib.pyplot as plt



def load_dataset_shuffle(filename1, filename2, features):
    dList = []
    lList = []

    with(open(filename1, 'r')) as f:
        for line in f:
            attr = line.split(',')[0:features]
            attr = numpy.array([i for i in attr])
            attr = vcol(attr)
            clss = line.split(',')[-1].strip()
            dList.append(attr)
            lList.append(clss)
    DTR = numpy.hstack(numpy.array(dList, dtype=numpy.float32))
    DTRmean = empirical_mean(DTR)
    DTRstd = vcol(numpy.std(DTR, axis=1))
    DTR = (DTR - DTRmean) / DTRstd
    LTR = numpy.array(lList, dtype=numpy.int32)

    with(open(filename2, 'r')) as f:
        for line in f:
            attr = line.split(',')[0:features]
            attr = numpy.array([i for i in attr])
            attr = vcol(attr)
            clss = line.split(',')[-1].strip()
            dList.append(attr)
            lList.append(clss)
    DTE = numpy.hstack(numpy.array(dList, dtype=numpy.float32))
    DTE = (DTE - DTRmean) / DTRstd
    LTE = numpy.array(lList, dtype=numpy.int32)

    return shuffle_dataset(DTR, LTR), shuffle_dataset(DTE, LTE)


def shuffle_dataset(D, L):
    numpy.random.seed(0)
    idx = numpy.random.permutation(D.shape[1])
    return D[:, idx], L[idx]


def split_db_2to1(D, L):
    nTrain = int(D.shape[1] * 2./3.)
    numpy.random.seed(0)
    index = numpy.random.permutation(D.shape[1])
    iTrain = index[0:nTrain]
    iTest = index[nTrain:]
    DTR = D[:, iTrain]
    DTE = D[:, iTest]
    LTR = L[iTrain]
    LTE = L[iTest]
    return (DTR, LTR), (DTE, LTE)


def features_gaussianization(DTR, DTE):
    rankDTR = numpy.zeros(DTR.shape)
    for i in range(DTR.shape[0]):
        for j in range(DTR.shape[1]):
            rankDTR[i][j] = (DTR[i] < DTR[i][j]).sum()
    rankDTR = (rankDTR + 1) / (DTR.shape[1] + 2)
    if(DTE is not None):
        rankDTE = numpy.zeros(DTE.shape)
        for i in range(DTE.shape[0]):
            for j in range(DTE.shape[1]):
                rankDTE[i][j] = (DTR[i] < DTE[i][j]).sum() + 1
        rankDTE /= (DTR.shape[1] + 2)
        return scipy.stats.norm.ppf(rankDTR), scipy.stats.norm.ppf(rankDTE)
    return scipy.stats.norm.ppf(rankDTR)


def empirical_withinclass_cov(D, labels):
    SW = 0
    for i in set(list(labels)):
        X = D[:, labels == i]
        SW += X.shape[1] * empirical_covariance(X)
    return SW / D.shape[1]


def empirical_betweenclass_cov(D, labels):
    SB = 0
    muGlob = empirical_mean(D)  # mean of the dataset
    for i in set(list(labels)):
        X = D[:, labels == i]
        mu = empirical_mean(X)  # mean of the class
        SB += X.shape[1] * numpy.dot((mu - muGlob), (mu - muGlob).T)
    return SB / D.shape[1]


def vrow(v):
    return v.reshape(1, v.size)


def vcol(v):
    return v.reshape(v.size, 1)


def empirical_mean(X):
    return vcol(X.mean(1))


def empirical_covariance(X):
    mu = empirical_mean(X)
    C = numpy.dot((X - mu), (X - mu).T) / X.shape[1]
    return C


def PCA(D, m):
    DC = (D - empirical_mean(D))
    C = (1 / DC.shape[1]) * numpy.dot(DC, DC.T)
    s, U = numpy.linalg.eigh(C)
    P = U[:, ::-1][:, 0:m]
    return numpy.dot(P.T, D), P


def LDA(D, L, m):
    SW = empirical_withinclass_cov(D, L)
    SB = empirical_betweenclass_cov(D, L)
    s, U = scipy.linalg.eigh(SB, SW)
    W = U[:, ::-1][:, 0:m]
    return numpy.dot(W.T, D), W


def assign_labels(scores, pi, Cfn, Cfp, th=None):
    if th is None:
        th = - numpy.log(pi * Cfn) + numpy.log((1 - pi) * Cfp)
    P = scores > th
    return numpy.int32(P)


def conf_matrix(Pred, labels):
    C = numpy.zeros((2, 2))
    C[0, 0] = ((Pred == 0) * (labels == 0)).sum()
    C[0, 1] = ((Pred == 0) * (labels == 1)).sum()
    C[1, 0] = ((Pred == 1) * (labels == 0)).sum()
    C[1, 1] = ((Pred == 1) * (labels == 1)).sum()
    return C


def DCFu(Conf, pi, Cfn, Cfp):
    FNR = Conf[0, 1]/(Conf[0, 1] + Conf[1, 1])
    FPR = Conf[1, 0]/(Conf[0, 0] + Conf[1, 0])
    return pi * Cfn * FNR + (1 - pi) * Cfp * FPR


def DCF(Conf, pi, Cfn, Cfp):
    _DCFu = DCFu(Conf, pi, Cfn, Cfp)
    return _DCFu / min(pi * Cfn, (1 - pi) * Cfp)


def minDCF(scores, labels, pi, Cfn, Cfp):
    t = numpy.array(scores)
    t.sort()

    dcfList = []
    for _th in t:
        dcfList.append(actDCF(scores, labels, pi, Cfn, Cfp, th=_th))
    return numpy.array(dcfList).min()


def actDCF(scores, labels, pi, Cfn, Cfp, th=None):
    Pred = assign_labels(scores, pi, Cfn, Cfp, th=th)
    CM = conf_matrix(Pred, labels)
    return DCF(CM, pi, Cfn, Cfp)


def compute_rates_values(scores, labels):
    t = numpy.array(scores)
    t.sort()
    t = numpy.concatenate([numpy.array([-numpy.inf]), t, numpy.array([numpy.inf])])
    FPR = []
    FNR = []
    for threshold in t:
        Pred = numpy.int32(scores > threshold)
        Conf = conf_matrix(Pred, labels)
        FPR.append(Conf[1, 0]/(Conf[1, 0] + Conf[0, 0]))
        FNR.append(Conf[0, 1]/(Conf[0, 1] + Conf[1, 1]))
    return numpy.array(FPR), numpy.array(FNR), 1 - numpy.array(FPR), 1 - numpy.array(FNR)


def compute_calibrated_scores_param(scores, labels):
    scores = vrow(scores)
    model = LR.LogisticRegression().trainClassifier(scores, labels, 1e-4, 0.5)
    alpha = model.w
    beta = model.b
    return alpha, beta


def plot_features(DTR, LTR, name, defPath = ''):
    D0 = DTR[:, LTR == 0]
    D1 = DTR[:, LTR == 1]
    labels = {
        0: 'Mean of the integrated profile',
        1: 'Standard deviation of the integrated profile',
        2: 'Excess kurtosis of the integrated profile',
        3: 'Skewness of the integrated profile',
        4: 'Mean of the DM-SNR curve',
        5: 'Standard deviation of the DM-SNR curve',
        6: 'Excess kurtosis of the DM-SNR curve',
        7: 'Skewness of the DM-SNR curve',
    }
    for i in range(DTR.shape[0]):
        fig = plt.figure()
        plt.title(labels[i])
        plt.hist(D0[i, :], bins=70, density=True, alpha=0.7, facecolor='orange', label='Negative pulsar signal', edgecolor='darkorange')
        plt.hist(D1[i, :], bins=70, density=True, alpha=0.7, facecolor='cornflowerblue', label='Positive pulsar signal', edgecolor='royalblue')
        plt.legend(loc='best')
        plt.savefig(defPath + 'img/features/%s_%d.jpg' % (name, i), dpi=300, bbox_inches='tight')
        plt.close(fig)


def plot_correlations(DTR, LTR, defPath = ''):
    cmap = ['Greys', 'Reds', 'Blues']
    labels = {
        0: 'Whole dataset (absolute Pearson coeff.)',
        1: 'Negative pulsar signal (absolute Pearson coeff.)',
        2: 'Positive pulsar signal (absolute Pearson coeff.)'
    }

    CorrCoeff = {
        0: numpy.abs(numpy.corrcoef(DTR)),
        1: numpy.abs(numpy.corrcoef(DTR[:, LTR == 0])),
        2: numpy.abs(numpy.corrcoef(DTR[:, LTR == 1]))
    }
    for i in range(3):
        fig = plt.figure()
        plt.title(labels[i])
        plt.imshow(CorrCoeff[i], cmap=cmap[i], interpolation='nearest')
        plt.savefig(defPath + 'img/heatmaps/heatmap_%d.jpg' % i, dpi=300, bbox_inches='tight')
        plt.close(fig)


def kfolds(D, L, pi, model, args, calibrated=False, folds = 5, Cfn = 1, Cfp = 1):
    scores = []
    Ds = numpy.array_split(D, folds, axis=1)
    Ls = numpy.array_split(L, folds)

    for i in range(folds):
        DTRk, LTRk = numpy.hstack(
            Ds[:i] + Ds[i+1:]), numpy.hstack(Ls[:i] + Ls[i+1:])
        DTEk, LTEk = numpy.asanyarray(Ds[i]), numpy.asanyarray(Ls[i])
        if calibrated:
            scoresTrain = model.trainClassifier(DTRk, LTRk, *args).computeLLR(DTRk)
            alpha, beta = compute_calibrated_scores_param(scoresTrain, LTRk)
            scoresEval = model.computeLLR(DTEk)
            computeLLR = alpha * scoresEval + beta - numpy.log(0.5/(1 - 0.5))
        else:
            computeLLR = model.trainClassifier(DTRk, LTRk, *args).computeLLR(DTEk)
        scores.append(computeLLR)
    minDCFtmp = minDCF(numpy.hstack(scores), L, pi, Cfn, Cfp)
    actDCFtmp = actDCF(numpy.hstack(scores), L, pi, Cfn, Cfp)
    return minDCFtmp, actDCFtmp


def single_split(D, L, pi, model, args, calibrated=False, Cfn = 1, Cfp = 1):
    (DTRk, LTRk), (DTEk, LTEk) = split_db_2to1(D, L)
    scores = []
    if calibrated:
        scoresTrain = model.trainClassifier(DTRk, LTRk, *args).computeLLR(DTRk)
        alpha, beta = compute_calibrated_scores_param(scoresTrain, LTRk)
        scoresEval = model.computeLLR(DTEk)
        scores = alpha * scoresEval + beta - numpy.log(0.5/(1 - 0.5))
    else:
        scores = model.trainClassifier(DTRk, LTRk, *args).computeLLR(DTEk)
    minDCFtmp = minDCF(numpy.hstack(scores), LTEk, pi, Cfn, Cfp)
    actDCFtmp = actDCF(numpy.hstack(scores), LTEk, pi, Cfn, Cfp)
    return minDCFtmp, actDCFtmp


def plot_minDCF_lr(l, y5, y1, y9, filename, title, defPath = ''):
    fig = plt.figure()
    plt.title(title)
    plt.plot(l, numpy.array(y5), label='minDCF(π~ = 0.5)', color='r')
    plt.plot(l, numpy.array(y1), label='minDCF(π~ = 0.1)', color='b')
    plt.plot(l, numpy.array(y9), label='minDCF(π~ = 0.9)', color='g')
    plt.xscale('log')
    plt.ylim([0, 1])
    plt.xlabel('λ')
    plt.ylabel('minDCF')
    plt.legend(loc='best')
    plt.savefig(defPath + 'img/minDCF/lr_minDCF_%s.jpg' % filename, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_minDCF_svm(C, y5, y1, y9, filename, title, type='linear', defPath = ''):
    labels = {
        0: 'minDCF(π~ = 0.5)' if type == 'linear' else ('minDCF(π~ = 0.5, γ = 1e-3)' if type == 'RBF' else 'minDCF(π~ = 0.5, c = 1)'),
        1: 'minDCF(π~ = 0.1)' if type == 'linear' else ('minDCF(π~ = 0.5, γ = 1e-2)' if type == 'RBF' else 'minDCF(π~ = 0.5, c = 10)'),
        2: 'minDCF(π~ = 0.9)' if type == 'linear' else ('minDCF(π~ = 0.5, γ = 1e-1)' if type == 'RBF' else 'minDCF(π~ = 0.5, c = 100)'),
    }
    fig = plt.figure()
    plt.title(title)
    plt.plot(C, numpy.array(y5), label=labels[0], color='r')
    plt.plot(C, numpy.array(y1), label=labels[1], color='b')
    plt.plot(C, numpy.array(y9), label=labels[2], color='g')
    plt.xscale('log')
    plt.ylim([0, 1])
    plt.xlabel('C')
    plt.ylabel('minDCF')
    plt.legend(loc='best')
    plt.savefig(defPath + 'img/minDCF/svm_minDCF_%s.jpg' % filename, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_minDCF_gmm(components, y5, y1, y9, filename, title, defPath = ''):
    fig = plt.figure()
    plt.title(title)
    plt.plot(components, numpy.array(y5), label = 'minDCF(π~ = 0.5)', color='r')
    plt.plot(components, numpy.array(y1), label = 'minDCF(π~ = 0.1)', color='b')
    plt.plot(components, numpy.array(y9), label = 'minDCF(π~ = 0.9)', color='g')
    plt.ylim([0, 1])
    plt.xlabel('components')
    plt.ylabel('minDCF')
    plt.legend(loc='best')
    plt.savefig(defPath + 'img/minDCF/gmm_minDCF_%s.jpg' % filename, dpi=300, bbox_inches='tight')
    plt.close(fig)


def bayes_error_plot(p, minDCF, actDCF, filename, title, defPath = ''):
    fig = plt.figure()
    plt.title(title)
    plt.plot(p, numpy.array(actDCF), label = 'actDCF', color='r')
    plt.plot(p, numpy.array(minDCF), label = 'minDCF', color='b', linestyle='--')
    plt.ylim([0, 1])
    plt.xlim([-3, 3])
    plt.xlabel('prior')
    plt.ylabel('minDCF')
    plt.legend(loc='best')
    plt.savefig(defPath + 'img/bep/bep_%s.jpg' % filename, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_ROC(results, LTE, filename, title, defPath = ''):
    fig = plt.figure()
    plt.title(title)
    for result in results:
        FPR, FNR, TNR, TPR = compute_rates_values(result[0], LTE)
        plt.plot(FPR, TPR, label = result[1], color=result[2])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='best')
    plt.savefig(defPath + 'img/eval/roc_%s.jpg' % filename, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_DET(results, LTE, filename, title, defPath = ''):
    fig = plt.figure()
    plt.title(title)
    for result in results:
        FPR, FNR, TNR, TPR = compute_rates_values(result[0], LTE)
        plt.plot(FPR, FNR, label = result[1], color=result[2])
    plt.xlabel('False Positive Rate')
    plt.ylabel('False Negative Rate')
    plt.legend(loc='best')
    plt.savefig(defPath + 'img/eval/det_%s.jpg' % filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
