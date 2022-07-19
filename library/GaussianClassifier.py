import numpy
import scipy.special
import library.utils as utils


def ML_GAU(D):
    mu = utils.empirical_mean(D)
    sigma = utils.empirical_covariance(D)
    return mu, sigma


def logpdf_GAU_ND(D, mu, sigma):
    P = numpy.linalg.inv(sigma)
    c1 = 0.5 * D.shape[0] * numpy.log(2 * numpy.pi)
    c2 = 0.5 * numpy.linalg.slogdet(P)[1]
    c3 = 0.5 * (numpy.dot(P, (D - mu)) * (D - mu)).sum(0)
    return - c1 + c2 - c3


class GaussianClassifier:

    def trainClassifier(self, D, L, priors, type='MVG', tied=False):
        self.type = type
        self.tied = tied
        self.priors = priors

        self.mu0, sigma0 = ML_GAU(D[:, L == 0])
        self.mu1, sigma1 = ML_GAU(D[:, L == 1])
        if(not tied):
            self.sigma0 = sigma0
            self.sigma1 = sigma1
            if(type == 'NBG'):
                self.sigma0 *= numpy.eye(self.sigma0.shape[0], self.sigma0.shape[1])
                self.sigma1 *= numpy.eye(self.sigma1.shape[0], self.sigma1.shape[1])
        else:
            self.sigma = utils.empirical_withinclass_cov(D, L)
            if(type == 'NBG'):
                self.sigma *= numpy.eye(self.sigma.shape[0], self.sigma.shape[1])
        return self


    def computeLLR(self, D):
        logD0 = logpdf_GAU_ND(D, self.mu0, self.sigma0 if not self.tied else self.sigma)
        logD1 = logpdf_GAU_ND(D, self.mu1, self.sigma1 if not self.tied else self.sigma)
        return logD1 - logD0
