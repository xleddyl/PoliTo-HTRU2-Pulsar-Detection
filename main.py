import numpy
import matplotlib.pyplot as plt
import importlib
import sys
import os
from library import utils, GaussianClassifier as GC, LogisticRegression as LR, SVM, GMM


# # # # # # # FUNCTIONS # # # # # # #


# # Load data
def load_data(defPath = ''):
    print('Loading data ...')
    # # class 1 -> Positive pulsar signal
    # # class 0 -> Negative pulsar signal
    (DTR, LTR), (DTE, LTE) = utils.load_dataset_shuffle(defPath + 'data/Train.txt', defPath + 'data/Test.txt', 8)
    # DTRg, DTEg = utils.features_gaussianization(DTR, DTE)
    print('Done.\n\n')
    return (DTR, LTR), (DTE, LTE)


# # Plot of the features
def plot_features(DTR, LTR):
    print('Plotting features ...')
    utils.plot_features(DTR, LTR, 'plot_raw_features')
    utils.plot_correlations(DTR, LTR)
    print('Done.\n\n')


# # Gaussian classifiers
def gaussian_classifier_report(DTR, LTR):
    print('Gaussian Classifiers report:')
    model = GC.GaussianClassifier()
    DTRpca = DTR
    print('# # 5-folds')
    for i in range(4):  # raw, pca7, pca6, pca5
        print(f'# PCA m = {DTR.shape[0] - i}' if i > 0 else '# RAW')
        if(i > 0):
            PCA_ = utils.PCA(DTR, DTR.shape[0] - i)
            DTRpca = PCA_[0]
        print('Full-Cov')
        for pi in priors:
            minDCF = utils.kfolds(DTRpca, LTR, pi, model, ([pi, 1 - pi]))[0]
            print(f'- with prior = {pi} -> minDCF = %.3f' % minDCF)
        print('Diag-Cov')
        for pi in priors:
            minDCF = utils.kfolds(DTRpca, LTR, pi, model,
                            ([pi, 1 - pi], 'NBG'))[0]
            print(f'- with prior = {pi} -> minDCF = %.3f' % minDCF)
        print('Tied Full-Cov')
        for pi in priors:
            minDCF = utils.kfolds(DTRpca, LTR, pi, model,
                            ([pi, 1 - pi], 'MVG', True))[0]
            print(f'- with prior = {pi} -> minDCF = %.3f' % minDCF)
        print('Tied Diag-Cov')
        for pi in priors:
            minDCF = utils.kfolds(DTRpca, LTR, pi, model,
                            ([pi, 1 - pi], 'NBG', True))[0]
            print(f'- with prior = {pi} -> minDCF = %.3f' % minDCF)
    print('\n')
    print('# # single-split')
    for i in range(4):  # raw, pca7, pca6, pca5
        print(f'# PCA m = {DTR.shape[0] - i}' if i > 0 else '# RAW')
        if(i > 0):
            PCA_ = utils.PCA(DTR, DTR.shape[0] - i)
            DTRpca = PCA_[0]
        print('Full-Cov')
        for pi in priors:
            minDCF = utils.single_split(
                DTRpca, LTR, pi, model, ([pi, 1 - pi]))[0]
            print(f'- with prior = {pi} -> minDCF = %.3f' % minDCF)
        print('Diag-Cov')
        for pi in priors:
            minDCF = utils.single_split(
                DTRpca, LTR, pi, model, ([pi, 1 - pi], 'NBG'))[0]
            print(f'- with prior = {pi} -> minDCF = %.3f' % minDCF)
        print('Tied Full-Cov')
        for pi in priors:
            minDCF = utils.single_split(
                DTRpca, LTR, pi, model,  ([pi, 1 - pi], 'MVG', True))[0]
            print(f'- with prior = {pi} -> minDCF = %.3f' % minDCF)
        print('Tied Diag-Cov')
        for pi in priors:
            minDCF = utils.single_split(
                DTRpca, LTR, pi, model, ([pi, 1 - pi], 'NBG', True))[0]
            print(f'- with prior = {pi} -> minDCF = %.3f' % minDCF)
    print('\n\n')


# # Logistic Regression
def logistic_regression_report(DTR, LTR):
    print('Logistic Regression report:')
    model = LR.LogisticRegression()
    DTRpca = DTR
    print('Plotting minDCF graphs ...')
    l = numpy.logspace(-5, 1, 10)
    for i in range(3):  # raw, pca7, pca6
        y5, y1, y9 = [], [], []
        title = 'raw'
        if(i > 0):
            PCA_ = utils.PCA(DTR, DTR.shape[0] - i)
            DTRpca = PCA_[0]
            title = f'pca{DTR.shape[0] - i}'
        for il in l:
            y5.append( utils.kfolds(DTRpca, LTR, priors[0], model, (il, priors[0]))[0])
            y1.append(utils.kfolds(DTRpca, LTR, priors[1], model, (il, priors[0]))[0])
            y9.append(utils.kfolds(DTRpca, LTR, priors[2], model, (il, priors[0]))[0])
        utils.plot_minDCF_lr(l, y5, y1, y9, f'{title}_5-folds', f'5-folds / {title} / πT = 0.5')
    print('Done.')
    print('# # 5-folds')
    for i in range(3):  # raw, pca7, pca6
        print(f'# PCA m = {DTR.shape[0] - i}' if i > 0 else '# RAW')
        if(i > 0):
            PCA_ = utils.PCA(DTR, DTR.shape[0] - i)
            DTRpca = PCA_[0]
        print('LogReg(λ = 1e-5, πT = 0.5)')
        for pi in priors:
            minDCF = utils.kfolds(DTRpca, LTR, pi, model, (1e-5, priors[0]))[0]
            print(f'- with prior = {pi} -> minDCF = %.3f' % minDCF)
        print('LogReg(λ = 1e-5, πT = 0.1)')
        for pi in priors:
            minDCF = utils.kfolds(DTRpca, LTR, pi, model, (1e-5, priors[1]))[0]
            print(f'- with prior = {pi} -> minDCF = %.3f' % minDCF)
        print('LogReg(λ = 1e-5, πT = 0.9)')
        for pi in priors:
            minDCF = utils.kfolds(DTRpca, LTR, pi, model, (1e-5, priors[2]))[0]
            print(f'- with prior = {pi} -> minDCF = %.3f' % minDCF)
    print('\n\n')


# # Linear SVM
def linear_svm_report(DTR, LTR):
    print('Support Vector Machine report:')
    model = SVM.SVM()
    DTRpca = DTR
    print('Plotting minDCF graphs ...')
    C = numpy.logspace(-4, 1, 10)
    for mode in ['unbalanced', 'balanced']:
        for i in priors:
            y5, y1, y9 = [], [], []
            PCA_ = utils.PCA(DTR, 7)
            DTRpca = PCA_[0]
            title = f'pca7'
            for iC in C:
                y5.append(utils.kfolds(DTRpca, LTR, priors[0], model, ('linear', i, mode == 'balanced', 1, iC))[0])
                y1.append(utils.kfolds(DTRpca, LTR, priors[1], model, ('linear', i, mode == 'balanced', 1, iC))[0])
                y9.append(utils.kfolds(DTRpca, LTR, priors[2], model, ('linear', i, mode == 'balanced', 1, iC))[0])
            utils.plot_minDCF_svm(C, y5, y1, y9, f'linear_{title}_{mode}{i}_5-folds', f'5-folds / {title} / {f"πT = {i}" if mode == "balanced" else "unbalanced"}')
            if(mode == 'unbalanced'):
                break
    print('Done.')
    print('# # 5-folds')
    for i in range(2):  # raw, pca7
        print(f'# PCA m = {DTR.shape[0] - i}' if i > 0 else '# RAW')
        if(i > 0):
            PCA_ = utils.PCA(DTR, DTR.shape[0] - i)
            DTRpca = PCA_[0]
        print('Linear SVM(C = 1e-2)')
        for pi in priors:
            minDCF = utils.kfolds(DTRpca, LTR, pi, model,
                            ('linear', priors[0], False, 1, 1e-2))[0]
            print(f'- with prior = {pi} -> minDCF = %.3f' % minDCF)
        print('Linear SVM(C = 1e-2, πT = 0.5)')
        for pi in priors:
            minDCF = utils.kfolds(DTRpca, LTR, pi, model,
                            ('linear', priors[0], True, 1, 1e-2))[0]
            print(f'- with prior = {pi} -> minDCF = %.3f' % minDCF)
        print('Linear SVM(C = 1e-2, πT = 0.1)')
        for pi in priors:
            minDCF = utils.kfolds(DTRpca, LTR, pi, model,
                            ('linear', priors[1], True, 1, 1e-2))[0]
            print(f'- with prior = {pi} -> minDCF = %.3f' % minDCF)
        print('Linear SVM(C = 1e-2, πT = 0.9)')
        for pi in priors:
            minDCF = utils.kfolds(DTRpca, LTR, pi, model,
                            ('linear', priors[2], True, 1, 1e-2))[0]
            print(f'- with prior = {pi} -> minDCF = %.3f' % minDCF)
    print('\n\n')


# # RBF SVM, Poly SVM
def quadratic_svm_report(DTR, LTR):
    print('RBF SVM, Poly SVM report:')
    model = SVM.SVM()
    DTRpca = DTR
    print('Plotting minDCF graphs ...')
    C = numpy.logspace(-4, 1, 10)
    for i in range(2):  # raw, pca7
        y5, y1, y9 = [], [], []
        title = 'raw'
        if(i > 0):
            PCA_ = utils.PCA(DTR, DTR.shape[0] - i)
            DTRpca = PCA_[0]
            title = f'pca{DTR.shape[0] - i}'
        for iC in C:
            y5.append(utils.kfolds(
                DTRpca, LTR, priors[0], model, ('poly', priors[0], False, 1, iC, 1, 2))[0])
            y1.append(utils.kfolds(
                DTRpca, LTR, priors[1], model, ('poly', priors[0], False, 1, iC, 10, 2))[0])
            y9.append(utils.kfolds(
                DTRpca, LTR, priors[2], model, ('poly', priors[0], False, 1, iC, 100, 2))[0])
        utils.plot_minDCF_svm(C, y5, y1, y9, f'poly_{title}_unbalanced_5-folds',
                        f'5-folds / {title} / unbalanced', type='poly')
    for i in range(2):  # raw, pca7
        y5, y1, y9 = [], [], []
        title = 'raw'
        if(i > 0):
            PCA_ = utils.PCA(DTR, DTR.shape[0] - i)
            DTRpca = PCA_[0]
            title = f'pca{DTR.shape[0] - i}'
        for iC in C:
            y5.append(utils.kfolds(
                DTRpca, LTR, priors[0], model, ('RBF', priors[0], False, 1, iC, 0, 0, 1e-3))[0])
            y1.append(utils.kfolds(
                DTRpca, LTR, priors[1], model, ('RBF', priors[0], False, 1, iC, 0, 0, 1e-2))[0])
            y9.append(utils.kfolds(
                DTRpca, LTR, priors[2], model, ('RBF', priors[0], False, 1, iC, 0, 0, 1e-1))[0])
        utils.plot_minDCF_svm(C, y5, y1, y9, f'rbf_{title}_unbalanced_5-folds',
                        f'5-folds / {title} / unbalanced', type='RBF')
    print('Done.')
    print('# # 5-folds')
    for i in range(2):  # raw, pca7
        print(f'# PCA m = {DTR.shape[0] - i}' if i > 0 else '# RAW')
        if(i > 0):
            PCA_ = utils.PCA(DTR, DTR.shape[0] - i)
            DTRpca = PCA_[0]
        print('RBF SVM(C = 1e-1, γ = 1e-3)')
        for pi in priors:
            minDCF = utils.kfolds(DTRpca, LTR, pi, model, ('RBF',
                            priors[0], False, 1, 1e-1, 0, 0, 1e-3))[0]
            print(f'- with prior = {pi} -> minDCF = %.3f' % minDCF)
        print('Poly SVM(C = 1e-3, c = 1, d = 2)')
        for pi in priors:
            minDCF = utils.kfolds(DTRpca, LTR, pi, model,
                            ('poly', priors[0], False, 1, 1e-3, 1, 2, 0))[0]
            print(f'- with prior = {pi} -> minDCF = %.3f' % minDCF)
    print('\n\n')


# # GMM
def gmm_report(DTR, LTR):
    print('GMM report:')
    model = GMM.GMM()
    DTRpca = DTR
    print('Plotting minDCF graphs ...')
    components = [2, 4, 8, 16, 32]
    for type in ['full', 'tied', 'diag']:
        for i in range(2):  # raw, pca7
            y5, y1, y9 = [], [], []
            title = 'raw'
            if(i > 0):
                PCA_ = utils.PCA(DTR, DTR.shape[0] - i)
                DTRpca = PCA_[0]
                title = f'pca{DTR.shape[0] - i}'
            for c in components:
                y5.append(
                    utils.kfolds(DTRpca, LTR, priors[0], model, (c, type))[0])
                y1.append(
                    utils.kfolds(DTRpca, LTR, priors[1], model, (c, type))[0])
                y9.append(
                    utils.kfolds(DTRpca, LTR, priors[2], model, (c, type))[0])
            utils.plot_minDCF_gmm(components, y5, y1, y9,
                            f'{type}_{title}', f'gmm {type}-cov / {title}')
    print('Done.')
    print('# # 5-folds')
    for i in range(2):  # raw, pca7
        print(f'# PCA m = {DTR.shape[0] - i}' if i > 0 else '# RAW')
        if(i > 0):
            PCA_ = utils.PCA(DTR, DTR.shape[0] - i)
            DTRpca = PCA_[0]
        print('GMM Full (8 components)')
        for pi in priors:
            minDCF = utils.kfolds(DTRpca, LTR, pi, model, (8, 'full'))[0]
            print(f'- with prior = {pi} -> minDCF = %.3f' % minDCF)
        print('GMM Diag (16 components)')
        for pi in priors:
            minDCF = utils.kfolds(DTRpca, LTR, pi, model, (16, 'diag'))[0]
            print(f'- with prior = {pi} -> minDCF = %.3f' % minDCF)
        print('GMM Tied (32 components)')
        for pi in priors:
            minDCF = utils.kfolds(DTRpca, LTR, pi, model, (32, 'tied'))[0]
            print(f'- with prior = {pi} -> minDCF = %.3f' % minDCF)
    print('\n\n')


# # Score calibration
def score_calibration_report(DTR, LTR):
    print('Score calibration report:')
    DTRpca = DTR
    print('Bayes Error Plot ...')
    p = numpy.linspace(-3, 3, 15)
    for model in [
            (GC.GaussianClassifier(), ([priors[0], 1 - priors[0]], 'MVG', True), 'tiedFullCov', 'Tied Full-Cov / PCA = 7'),
            (LR.LogisticRegression(), (1e-5, priors[0]), 'LogReg', 'Logistic Regression / λ = 1e-5 / PCA = 7'),
            (SVM.SVM(), ('linear', priors[0], False, 1, 1e-2), 'SVM','Linear SVM / C = 1e-2 / PCA = 7'),
            (GMM.GMM(), (8, 'full'), 'GMM','Tied GMM / 8 components / PCA = 7'),
        ]:
        minDCF = []
        actDCF = []
        for iP in p:
            iP = 1.0 / (1.0 + numpy.exp(-iP))
            minDCFtmp, actDCFtmp = utils.kfolds(DTRpca, LTR, iP, model[0], model[1])
            minDCF.append(minDCFtmp)
            actDCF.append(actDCFtmp)
        utils.bayes_error_plot(p, minDCF, actDCF, model[2], model[3])
    print('Done.')
    print('Bayes Error Plot Calibrated ...')
    p = numpy.linspace(-3, 3, 15)
    for model in [
            (GC.GaussianClassifier(), ([priors[0], 1 - priors[0]], 'MVG', True), 'calibrated_tiedFullCov', 'calibrated Tied Full-Cov / PCA = 7'),
            (LR.LogisticRegression(), (1e-5, priors[0]), 'calibrated_LogReg', 'calibrated  Logistic Regression / λ = 1e-5 / PCA = 7'),
            (SVM.SVM(), ('linear', priors[0], False, 1, 1e-2), 'calibrated_SVM','calibrated  Linear SVM / C = 1e-2 / PCA = 7'),
            (GMM.GMM(), (8, 'full'), 'calibrated_GMM','calibrated  Tied GMM / 8 components / PCA = 7'),
        ]:
        minDCF = []
        actDCF = []
        for iP in p:
            iP = 1.0 / (1.0 + numpy.exp(-iP))
            minDCFtmp, actDCFtmp = utils.kfolds(DTRpca, LTR, iP, model[0], model[1], calibrated=True)
            minDCF.append(minDCFtmp)
            actDCF.append(actDCFtmp)
        utils.bayes_error_plot(p, minDCF, actDCF, model[2], model[3])
    print('Done.')
    print('# # 5-folds')
    for i in range(2):  # raw, pca7
        print(f'# PCA m = {DTR.shape[0] - i}' if i > 0 else '# RAW')
        if(i > 0):
            PCA_ = utils.PCA(DTR, DTR.shape[0] - i)
            DTRpca = PCA_[0]
        print('Tied Full-Cov')
        for pi in priors:
            minDCF, actDCF = utils.kfolds(DTRpca, LTR, pi, GC.GaussianClassifier(), ([priors[0], 1 - priors[0]], 'MVG', True), calibrated=True)
            print(f'- with prior = {pi} -> minDCF = %.3f' % minDCF)
            print(f'- with prior = {pi} -> actDCF = %.3f' % actDCF)
        print('LogReg(λ = 1e-5, πT = 0.5)')
        for pi in priors:
            minDCF, actDCF = utils.kfolds(DTRpca, LTR, pi, LR.LogisticRegression(), (1e-5, priors[0]), calibrated=True)
            print(f'- with prior = {pi} -> minDCF = %.3f' % minDCF)
            print(f'- with prior = {pi} -> actDCF = %.3f' % actDCF)
        print('Linear SVM(C = 1e-2, πT = 0.5)')
        for pi in priors:
            minDCF, actDCF = utils.kfolds(DTRpca, LTR, pi, SVM.SVM(), ('linear', priors[0], False, 1, 1e-2), calibrated=True)
            print(f'- with prior = {pi} -> minDCF = %.3f' % minDCF)
            print(f'- with prior = {pi} -> actDCF = %.3f' % actDCF)
        print('GMM Full Cov(8 components)')
        for pi in priors:
            minDCF, actDCF = utils.kfolds(DTRpca, LTR, pi, GMM.GMM(), (8, 'full'), calibrated=True)
            print(f'- with prior = {pi} -> minDCF = %.3f' % minDCF)
            print(f'- with prior = {pi} -> actDCF = %.3f' % actDCF)
    print('\n\n')


# # Evaluation
def evaluation_report(DTR, LTR, DTE, LTE):
    print('Evaluation report:')
    PCA_ = utils.PCA(DTR, 7)
    DTRpca = PCA_[0]
    DTEpca = numpy.dot(PCA_[1].T, DTE)
    calibratedScores = []
    for model in [
            (GC.GaussianClassifier().trainClassifier(DTRpca, LTR, *([priors[0], 1 - priors[0]], 'MVG', True)), 'Tied Full-Cov'),
            (LR.LogisticRegression().trainClassifier(DTRpca, LTR, *(1e-5, priors[0])), 'LogReg(λ = 1e-5, πT = 0.5)'),
            (SVM.SVM().trainClassifier(DTRpca, LTR, *('linear', priors[0], False, 1, 1e-2)), 'Linear SVM(C = 1e-2)'),
            (GMM.GMM().trainClassifier(DTRpca, LTR, *(8, 'full')), 'GMM Full Cov (8 components)')
        ]:
        alpha, beta = utils.compute_calibrated_scores_param(model[0].computeLLR(DTRpca), LTR)
        scores = alpha * model[0].computeLLR(DTEpca) + beta - numpy.log(priors[0]/(1 - priors[0]))
        print(model[1])
        for pi in priors:
            minDCF = utils.minDCF(scores, LTE, pi, 1, 1)
            actDCF = utils.actDCF(scores, LTE, pi, 1, 1)
            print(f'- with prior = {pi} -> minDCF = %.3f' % minDCF)
            print(f'- with prior = {pi} -> actDCF = %.3f' % actDCF)
        calibratedScores.append(scores)
    utils.plot_ROC(zip(calibratedScores, [
        'Tied Full-Cov',
        'LogReg(λ = 1e-5, πT = 0.5)',
        'Linear SVM(C = 1e-2)',
        'GMM Full Cov (8 components)'
    ], [
        'r',
        'b',
        'g',
        'darkorange'

    ]), LTE, 'calibrated_classifiers', 'calibrated / PCA = 7')
    utils.plot_DET(zip(calibratedScores, [
        'Tied Full-Cov',
        'LogReg(λ = 1e-5, πT = 0.5)',
        'Linear SVM(C = 1e-2)',
        'GMM Full Cov (8 components)'
    ], [
        'r',
        'b',
        'g',
        'darkorange'

    ]), LTE, 'calibrated_classifiers', 'calibrated / PCA = 7')
    print('Done.')
    print('\n\n')


# # # # # # # FUNCTIONS # # # # # # #


if __name__ == '__main__':
    importlib.reload(utils)
    importlib.reload(GC)
    importlib.reload(LR)
    importlib.reload(SVM)
    importlib.reload(GMM)

    priors = [0.5, 0.1, 0.9]

    (DTR, LTR), (DTE, LTE) = load_data()
    plot_features(DTR, LTR)
    gaussian_classifier_report(DTR, LTR)
    logistic_regression_report(DTR, LTR)
    linear_svm_report(DTR, LTR)
    quadratic_svm_report(DTR, LTR)
    gmm_report(DTR, LTR)
    score_calibration_report(DTR, LTR)
    evaluation_report(DTR, LTR, DTE, LTE)

    print('\n\n ------ END ------')