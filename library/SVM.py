import numpy
import library.utils as utils
import scipy.optimize


def dual_wrapper(D, H, bounds):
    def LDual(alpha, H):
        Ha = numpy.dot(H, utils.vcol(alpha))
        aHa = numpy.dot(utils.vrow(alpha), Ha)
        a1 = alpha.sum()
        return 0.5 * aHa.ravel() - a1,  Ha.ravel() - numpy.ones(alpha.size)

    alphaStar, _x, _y = scipy.optimize.fmin_l_bfgs_b(
        LDual,
        numpy.zeros(D.shape[1]),
        args=(H,),
        bounds=bounds
    )

    return alphaStar


class SVM:

    def trainClassifier(self, D, L, type='linear', pi=0, balanced=False, K=1, C=0, c=0, d=2, gamma=0):
        self.Z = numpy.zeros(L.shape)
        self.Z[L == 1] = 1
        self.Z[L == 0] = -1
        self.DTR = D
        self.LTR = L
        self.type = type
        self.K = K
        self.C = C

        if(balanced):
            C1 = (C * pi) / (D[:, L == 1].shape[1] / D.shape[1])
            C0 = (C * (1 - pi)) / (D[:, L == 0].shape[1] / D.shape[1])
            self.bounds = [((0, C0) if x == 0 else (0, C1)) for x in L.tolist()]
        else:
            self.bounds = [(0, C)] * D.shape[1]

        if(type == 'linear'):
            DTRT = numpy.vstack([D, numpy.ones(D.shape[1]) * K])
            H = numpy.dot(DTRT.T, DTRT)
            H = numpy.dot(utils.vcol(self.Z), utils.vrow(self.Z)) * H
            alphaStar = dual_wrapper(D, H, self.bounds)
            self.w = numpy.dot(DTRT, utils.vcol(alphaStar) * utils.vcol(self.Z)).sum(axis = 1)
        if(type == 'RBF'):
            self.gamma = gamma
            Dist = utils.vcol((D ** 2).sum(0)) + utils.vrow((D ** 2).sum(0)) - 2 * numpy.dot(D.T, D)
            kernel = numpy.exp(-self.gamma * Dist) + (self.K ** 2)
            H = numpy.dot(utils.vcol(self.Z), utils.vrow(self.Z)) * kernel
            self.w = dual_wrapper(D, H, self.bounds)
        if(type == 'poly'):
            self.c = c
            self.d = d
            kernel = ((numpy.dot(D.T, D) + self.c) ** self.d) + (self.K ** 2)
            H = numpy.dot(utils.vcol(self.Z), utils.vrow(self.Z)) * kernel
            self.w = dual_wrapper(D, H, self.bounds)
        return self


    def computeLLR(self, D):
        if(self.type == 'linear'):
            DTET = numpy.vstack([D, numpy.ones(D.shape[1]) * self.K])
            return numpy.dot(self.w.T, DTET)
        if(self.type == 'RBF'):
            Dist = utils.vcol((self.DTR ** 2).sum(0)) + utils.vrow((D ** 2).sum(0)) - 2 * numpy.dot(self.DTR.T, D)
            kernel = numpy.exp(-self.gamma * Dist) + (self.K ** 2)
            return numpy.dot(self.w * self.Z, kernel)
        if(self.type == 'poly'):
            kernel = ((numpy.dot(self.DTR.T, D) + self.c) ** self.d) + (self.K ** 2)
            return numpy.dot(self.w * self.Z, kernel)