import math
import numpy as np
import numpy.linalg as LA


def ODL_updateD(D, E, F, iterations = 100, tol = 1e-8):
    """
    * The main algorithm in ODL.
    * Solving the optimization problem:
      `D = arg min_D -2trace(E'*D) + trace(D*F*D')` subject to: `||d_i||_2 <= 1`,
         where `F` is a positive semidefinite matrix.
    * Syntax `[D, iter] = ODL_updateD(D, E, F, opts)`
      - INPUT:
        + `D, E, F` as in the above problem.
        + `opts`. options:
          * `iterations`: maximum number of iterations.
          * `tol`: when the difference between `D` in two successive
                    iterations less than this value, the algorithm will stop.
      - OUTPUT:
        + `D`: solution.
    -----------------------------------------------
    Author: Tiep Vu, thv102@psu.edu, 04/07/2016
            (http://www.personal.psu.edu/thv102/)
    -----------------------------------------------
    """
    def calc_cost(D):
        return -2*np.trace(np.dot(E, D.T)) + np.trace(np.dot(np.dot(F, D.T), D))

    D_old = D.copy()
    it = 0
    for it in range(iterations):
        for i in range(D.shape[1]):
            if F[i,i] != 0:
                a = 1.0/F[i,i] * (E[:, i] - D.dot(F[:, i])) + D[:, i]
                D[:,i] = a/max(LA.norm(a, 2), 1)

        if LA.norm(D - D_old, 'fro')/D.size < tol:
            break
        D_old = D.copy()
    return D

def normF2(X):
    """
    * Return square of the Frobenius norm, which is sum of square of all
    elements in a matrix
    * Syntax: `res = normF2(X)`
    """
    # pass
    if X.shape[0]*X.shape[1] == 0:
        return 0
    return LA.norm(X, 'fro')**2


def norm1(X):
    """
    * Return norm 1 of a matrix, which is sum of absolute value of all element
    of that matrix.
    """
    # pass
    if X.shape[0]*X.shape[1] == 0:
        return 0
    return abs(X).sum()
    # return LA.norm(X, 1)


def shrinkage(U, alambda):
    """
    * Soft thresholding function.
    * Syntax: `X = shrinkage(U, lambda)`
    * Solve the following optimization problem:
    `X = arg min_X 0.5*||X - U||_F^2 + lambda||X||_1`
    where `U` and `X` are matrices with same sizes. `lambda` can be either
    positive a scalar or a positive matrix (all elements are positive) with
    same size as `X`. In the latter case, it is a weighted problem.
    """
    return np.maximum(0, U - alambda) + np.minimum(0, U + alambda)



def num_grad(func, X):
    """
    Calculating gradient of a function `func(X)` where `X` is a matrix or
    vector
    """
    grad = np.zeros_like(X)
    eps = 1e-4
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            # print X, '\n'
            Xp = X.copy()
            Xm = X.copy()
            Xp[i,j] += eps
            # print X
            fp = func(Xp)
            Xm[i,j] -= eps
            fm = func(Xm)
            grad[i,j] = (fp - fm)/(2*eps)
    return grad

class Fista(object):
    def __init__(self):
        """
        subclasses are required to have three following functions and lambd
        """
        self._grad = None
        self._calc_f = None
        self.lossF = None
        self.lambd = None

    def solve(self, Xinit = None, iterations = 100, tol = 1e-8, verbose = False):
        if Xinit is None:
            Xinit = np.zeros((self.D.shape[1], self.Y.shape[1]))
        Linv = 1/self.L
        lambdaLiv = self.lambd/self.L
        x_old = Xinit.copy()
        y_old = Xinit.copy()
        t_old = 1
        it = 0
        # cost_old = float("inf")
        for it in range(iterations):
            x_new = np.real(shrinkage(y_old - Linv*self._grad(y_old), lambdaLiv))
            t_new = .5*(1 + math.sqrt(1 + 4*t_old**2))
            y_new = x_new + (t_old - 1)/t_new * (x_new - x_old)
            e = norm1(x_new - x_old)/x_new.size
            if e < tol:
                break
            x_old = x_new.copy()
            t_old = t_new
            y_old = y_new.copy()
            if verbose:
                print('iter \t%d/%d, loss \t %4.4f'%(it + 1, iterations, self.lossF(x_new)))
        return x_new

    def check_grad(self, X):
        grad1 = self._grad(X)
        grad2 = num_grad(self._calc_f, X)
        dif = norm1(grad1 - grad2)/grad1.size
        print('grad difference = %.7f'%dif)



class Lasso(Fista):
    """
    Solving a Lasso problem using FISTA
    `X, = arg min_X 0.5*||Y - DX||_F^2 + lambd||X||_1
        = argmin_X f(X) + lambd||X||_1
        F(x) = f(X) + lamb||X||_1
    """
    def __init__(self, D, lambd = .1):
        self.D = D
        self.lambd = lambd
        self.DtD = np.dot(self.D.T, self.D)
        self.Y = None
        self.DtY = None
        self.L = np.max(LA.eig(self.DtD)[0])
        self.coef_ = None

    def fit(self, Y, Xinit = None, iterations = 100):
        self.Y = Y
        self.DtY = np.dot(self.D.T, self.Y)
        if Xinit is None:
            Xinit = np.zeros((self.D.shape[1], self.Y.shape[1]))
        self.coef_ = self.solve(Xinit = Xinit, iterations = iterations)

    def _grad(self, X):
        return np.dot(self.DtD, X) - self.DtY

    def _calc_f(self, X):
        return 0.5*normF2(self.Y - np.dot(self.D, X))

    def lossF(self, X):
        return self._calc_f(X) + self.lambd*norm1(X)