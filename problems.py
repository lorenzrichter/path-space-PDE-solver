#pylint: disable=invalid-name, no-member, too-many-arguments, unused-argument, missing-docstring, too-many-instance-attributes

import numpy as np
import torch as pt

from numpy import exp, log

from scipy.linalg import expm, inv, solve_banded


device = pt.device('cuda')


class LLGC():
    '''
        Ornstein-Uhlenbeck with linear terminal costs.
    '''
    def __init__(self, name='LLGC', d=1, off_diag=0, T=5, seed=42):

        pt.manual_seed(seed)
        self.name = name
        self.d = d
        self.T = T
        self.A = (-pt.eye(self.d) + off_diag * pt.randn(self.d, self.d)).to(device)
        self.B = (pt.eye(self.d) + off_diag * pt.randn(self.d, self.d)).to(device)
        self.alpha = pt.ones(self.d, 1).to(device)
        self.X_0 = pt.zeros(self.d).to(device)
        self.boundary = 'square'
        self.one_boundary = False
        self.X_l = -2.0
        self.X_r = 2.0

        if ~np.all(np.linalg.eigvals(self.A.cpu().numpy()) < 0):
            print('not all EV of A are negative')

    def b(self, x):
        return pt.mm(self.A, x.t()).t()

    def sigma(self, x):
        return self.B

    def f(self, x, t):
        return pt.zeros(x.shape[0]).to(device)

    def h(self, t, x, y, z):
        return -0.5 * pt.sum(z**2, dim=1)

    def g(self, x):
        return pt.mm(x, self.alpha)[:, 0]

    def u_true(self, x, t):
        return -self.sigma(x).cpu().numpy().T.dot(expm(self.A.cpu().numpy().T * (self.T - t)).dot(
            self.alpha.cpu().numpy()) * np.ones(x.shape).T)

    def v_true(self, x, t):
        delta_t = 0.001
        N = int(np.floor((self.T - t) / delta_t)) + 1
        Sigma_n = np.zeros([self.d, self.d])
        for t_n in np.linspace(t, self.T, N):
            Sigma_n += (expm(self.A.cpu().numpy() * (self.T - t_n))
                        .dot(self.sigma(np.zeros([self.d, self.d])).cpu())
                        .dot(self.sigma(np.zeros([self.d, self.d])).t().cpu())
                        .dot(expm(self.A.cpu().numpy().T * (self.T - t_n)))) * delta_t
        return ((expm(self.A.cpu().numpy() * (self.T - t)).dot(x.t()).T).dot(self.alpha.cpu().numpy())
                - 0.5 * self.alpha.cpu().numpy().T.dot(Sigma_n.dot(self.alpha.cpu())))


class LLGC_general_f():
    '''
        This problem demonstrates that most losses work also if there are no quadratic running costs on u.
    '''
    def __init__(self, name='LLGC', d=1, off_diag=0, T=5, seed=42):

        pt.manual_seed(seed)
        self.name = name
        self.d = d
        self.T = T
        self.A = 0 * (-pt.eye(self.d) + off_diag * pt.randn(self.d, self.d)).to(device)
        self.B = (pt.eye(self.d) + off_diag * pt.randn(self.d, self.d)).to(device)
        self.alpha = -pt.ones(self.d, 1).to(device)
        self.X_0 = pt.zeros(self.d).to(device)

        if ~np.all(np.linalg.eigvals(self.A.cpu().numpy()) < 0):
            print('not all EV of A are negative')

    def b(self, x):
        return pt.mm(self.A, x.t()).t()

    def sigma(self, x):
        return self.B

    def f(self, x, t):
        return pt.zeros(x.shape[0])

    def h(self, t, x, y, z):
        return -(0.8 * ((- z)**2)**(0.625) + x * pt.exp(self.T - t) - 0.8 * pt.exp(1.25 * (self.T - t)))[:, 0]

    def g(self, x):
        return pt.mm(x, self.alpha)[:, 0]

    def u_true(self, x, t):
        return -self.sigma(x).cpu().numpy().T.dot(expm(1 * self.B.cpu().numpy().T * (self.T - t)).dot(
            self.alpha.cpu().numpy()) * np.ones(x.shape).T)

    def v_true(self, x, t):
        delta_t = 0.001
        N = int(np.floor((self.T - t) / delta_t)) + 1
        Sigma_n = np.zeros([self.d, self.d])
        for t_n in np.linspace(t, self.T, N):
            Sigma_n += (expm(self.A.cpu().numpy() * t_n)
                        .dot(self.sigma(np.zeros([self.d, self.d])).cpu())
                        .dot(self.sigma(np.zeros([self.d, self.d])).t().cpu())
                        .dot(expm(self.A.cpu().numpy().T * t_n))) * delta_t
        return ((expm(self.A.cpu().numpy() * (self.T - t)).dot(x.t()).T).dot(self.alpha.cpu().numpy())
                - 0.5 * self.alpha.cpu().numpy().T.dot(Sigma_n.dot(self.alpha.cpu())))


class LQGC():
    '''
        Linear quadratic Gaussian control - Ornstein-Uhlenbeck with quadratic running and terminal costs.
    '''
    def __init__(self, name='LQGC', delta_t=0.05, d=1, off_diag=0, T=5, seed=42):

        pt.manual_seed(seed)
        self.name = name
        self.d = d
        self.T = T
        self.A = (-pt.eye(self.d) + off_diag * pt.randn(self.d, self.d)).to(device)
        self.B = (pt.eye(self.d) + off_diag * pt.randn(self.d, self.d)).to(device)
        self.delta_t = delta_t
        self.N = int(np.floor(self.T / self.delta_t))
        self.X_0 = pt.zeros(self.d)

        if ~np.all(np.linalg.eigvals(self.A.cpu().numpy()) < 0):
            print('not all EV of A are negative')

        self.P = 0.5 * pt.eye(self.d).to(device)
        self.Q = 0.5 * pt.eye(self.d).to(device)
        self.R = pt.eye(self.d).to(device)
        self.F = pt.zeros([self.N + 1, self.d, self.d]).to(device)
        self.F[self.N, :, :] = self.R
        for n in range(self.N, 0, -1):
            self.F[n - 1, :, :] = (self.F[n, :, :]
                                   + (pt.mm(self.A.t(), self.F[n, :, :])
                                      + pt.mm(self.F[n, :, :], self.A)
                                      - pt.mm(pt.mm(pt.mm(pt.mm(self.F[n, :, :], self.B),
                                                          self.Q.inverse()), self.B.t()),
                                              self.F[n, :, :]) + self.P) * self.delta_t)
        self.G = pt.zeros([self.N + 1])
        for n in range(self.N, 0, -1):
            self.G[n - 1] = (self.G[n] - pt.trace(pt.mm(pt.mm(self.B, self.F[n, :, :]), self.B))
                             * self.delta_t)

    def b(self, x):
        return pt.mm(self.A, x.t()).t()

    def sigma(self, x):
        return self.B

    def f(self, x, t):
        return pt.sum(x.t() * pt.mm(self.P, x.t()), 0)

    def g(self, x):
        return pt.sum(x.t() * pt.mm(self.R, x.t()), 0)

    def h(self, t, x, y, z):
        return -0.5 * pt.sum(z**2, dim=1) - self.f(x, t)

    def u_true(self, x, t):
        n = int(np.ceil(t / self.delta_t))
        return -pt.mm(pt.mm(pt.mm(self.Q.cpu().inverse(), self.B.cpu().t()), self.F[n, :, :].cpu()), x.t()).detach().numpy()

    def v_true(self, x, t):
        n = int(np.ceil(t / self.delta_t))
        return -pt.mm(x, pt.mm(self.F[n, :, :], x.t())).t() + self.G[n]


class DoubleWell():
    '''
        One-dimensional double well potential.
    '''
    def __init__(self, name='Double well', d=1, T=1, eta=1, kappa=1):
        self.name = name
        self.d = d
        self.T = T
        self.eta = eta
        self.kappa = kappa
        self.B = pt.eye(self.d).to(device)
        self.X_0 = -pt.ones(self.d).to(device)
        self.ref_sol_is_defined = False

        if self.d != 1:
            print('The double well example is only implemented for d = 1.')

    def V(self, x):
        return self.kappa * (x**2 - 1)**2

    def grad_V(self, x):
        return 4.0 * self.kappa * x * (x**2 - 1)

    def b(self, x):
        return -self.grad_V(x)

    def sigma(self, x):
        return self.B

    def f(self, x, t):
        return pt.zeros(x.shape[0]).to(device)

    def h(self, t, x, y, z):
        return -0.5 * pt.sum(z**2, dim=1)

    def g(self, x):
        return (self.eta * (x - 1)**2).squeeze()

    def compute_reference_solution(self, delta_t=0.005, xb=2.5, nx=1000):

        self.xb = xb # range of x, [-xb, xb]
        self.nx = nx # number of discrete interval
        self.dx = 2.0 * self.xb / self.nx
        self.delta_t = delta_t

        beta = 2

        self.xvec = np.linspace(-self.xb, self.xb, self.nx, endpoint=True)

        # A = D^{-1} L D
        # assumes Neumann boundary conditions

        A = np.zeros([self.nx, self.nx])
        for i in range(0, self.nx):

            x = -self.xb + (i + 0.5) * self.dx
            if i > 0:
                x0 = -self.xb + (i - 0.5) * self.dx
                x1 = -self.xb + i * self.dx
                A[i, i - 1] = -exp(beta * 0.5 * (self.V(x0) + self.V(x) - 2 * self.V(x1))) / self.dx**2
                A[i, i] = exp(beta * (self.V(x) - self.V(x1))) / self.dx**2
            if i < self.nx - 1:
                x0 = -self.xb + (i + 1.5) * self.dx
                x1 = -self.xb + (i + 1) * self.dx
                A[i, i + 1] = -exp(beta * 0.5 * (self.V(x0) + self.V(x) - 2 * self.V(x1))) / self.dx**2
                A[i, i] = A[i, i] + exp(beta * (self.V(x) - self.V(x1))) / self.dx**2

        A = -A / beta
        N = int(self.T / self.delta_t)

        D = np.diag(exp(beta * self.V(self.xvec) / 2))
        D_inv = np.diag(exp(-beta * self.V(self.xvec) / 2))

        np.linalg.cond(np.eye(self.nx) - self.delta_t * A)
        #w, vv = np.linalg.eigh(np.eye(self.nx) - self.delta_t * A)

        self.psi = np.zeros([N + 1, self.nx])
        self.psi[N, :] = exp(-self.g(self.xvec))

        for n in range(N - 1, -1, -1):
            band = - self.delta_t * np.vstack([np.append([0], np.diagonal(A, offset=1)),
                                               np.diagonal(A, offset=0) - N / self.T,
                                               np.append(np.diagonal(A, offset=1), [0])])

            self.psi[n, :] = D.dot(solve_banded([1, 1], band, D_inv.dot(self.psi[n + 1, :])))
            #psi[n, :] = np.dot(D, np.linalg.solve(np.eye(self.nx) - delta_t * A, D_inv.dot(psi[n + 1, :])));

        self.u = np.zeros([N + 1, self.nx - 1])
        for n in range(N + 1):
            for i in range(self.nx - 1):
                self.u[n, i] = -2 / beta * self.B * (- log(self.psi[n, i + 1]) + log(self.psi[n, i])) / self.dx
        #self.u = 2 / beta * np.gradient(np.log(self.psi), self.dx, 1)

    def v_true(self, x, t):
        i = np.floor((x.squeeze(0) + self.xb) / self.dx).long()
        i[-1] -= 2
        n = int(np.ceil(t / self.delta_t))
        return np.array(- log(self.psi[n, i])).reshape([1, len(i)])

    def u_true(self, x, t):
        i = np.floor((np.clip(x, -self.xb, self.xb - 2 * self.dx).squeeze(0) + self.xb) / self.dx).long()
        i[-1] -= 2
        n = int(np.ceil(t / self.delta_t))
        return np.array(self.u[n, i]).reshape([1, len(i)])
        #return interpolate.interp1d(self.xvec, self.u)(x)[:, n]


class DoubleWell_multidim():
    '''
        Multidimensional extension: one-dimensional double well potential in every dimension.
    '''
    def __init__(self, name='Double well', d=1, d_1=1, d_2=0, T=1, eta=1, kappa=1):
        self.name = name
        self.d = d
        self.d_1 = d_1
        self.d_2 = d_2
        self.T = T
        self.eta = eta
        self.eta_ = pt.tensor([eta] * d_1 + [1.0] * d_2).to(device)
        self.kappa = kappa
        self.kappa_ = pt.tensor([kappa] * d_1 + [1.0] * d_2).to(device)
        self.B = pt.eye(self.d).to(device)
        self.X_0 = -pt.ones(self.d).to(device)
        self.ref_sol_is_defined = False
        self.boundary = 'unbounded'
        self.boundary_distance = 2.0

    def V(self, x):
        return self.kappa * (x**2 - 1)**2

    def V_2(self, x):
        return (x**2 - 1)**2

    def grad_V(self, x):
        return 4.0 * self.kappa_ * (x * (x**2 - pt.ones(self.d).to(device)))

    def b(self, x):
        return -self.grad_V(x)

    def sigma(self, x):
        return self.B # self.B.repeat(x.shape[0], 1, 1)

    def h(self, t, x, y, z):
        return -0.5 * pt.sum(z**2, dim=1)

    def f(self, x, t):
        return pt.zeros(x.shape[0]).to(device)

    def g_1(self, x_1):
        return self.eta * (x_1 - 1)**2

    def g_2(self, x_1):
        return (x_1 - 1)**2

    def g(self, x):
        #return (self.eta * (pt.sum((x - pt.ones(self.d).to(device))**2, 1))).squeeze()
        return ((pt.sum(self.eta_ * (x - pt.ones(self.d).to(device))**2, 1))).squeeze()

    def compute_reference_solution(self, delta_t=0.005, xb=2.5, nx=1000):

        self.xb = xb # range of x, [-xb, xb]
        self.nx = nx # number of discrete interval
        self.dx = 2.0 * self.xb / self.nx
        self.delta_t = delta_t

        beta = 2

        self.xvec = np.linspace(-self.xb, self.xb, self.nx, endpoint=True)

        # A = D^{-1} L D
        # assumes Neumann boundary conditions

        A = np.zeros([self.nx, self.nx])
        for i in range(0, self.nx):

            x = -self.xb + (i + 0.5) * self.dx
            if i > 0:
                x0 = -self.xb + (i - 0.5) * self.dx
                x1 = -self.xb + i * self.dx
                A[i, i - 1] = -exp(beta * 0.5 * (self.V(x0) + self.V(x) - 2 * self.V(x1))) / self.dx**2
                A[i, i] = exp(beta * (self.V(x) - self.V(x1))) / self.dx**2
            if i < self.nx - 1:
                x0 = -self.xb + (i + 1.5) * self.dx
                x1 = -self.xb + (i + 1) * self.dx
                A[i, i + 1] = -exp(beta * 0.5 * (self.V(x0) + self.V(x) - 2 * self.V(x1))) / self.dx**2
                A[i, i] = A[i, i] + exp(beta * (self.V(x) - self.V(x1))) / self.dx**2

        A = -A / beta
        N = int(self.T / self.delta_t)

        D = np.diag(exp(beta * self.V(self.xvec) / 2))
        D_inv = np.diag(exp(-beta * self.V(self.xvec) / 2))

        np.linalg.cond(np.eye(self.nx) - self.delta_t * A)
        #w, vv = np.linalg.eigh(np.eye(self.nx) - self.delta_t * A)

        self.psi = np.zeros([N + 1, self.nx])
        self.psi[N, :] = exp(-self.g_1(self.xvec))

        for n in range(N - 1, -1, -1):
            band = - self.delta_t * np.vstack([np.append([0], np.diagonal(A, offset=1)),
                                               np.diagonal(A, offset=0) - N / self.T,
                                               np.append(np.diagonal(A, offset=1), [0])])

            self.psi[n, :] = D.dot(solve_banded([1, 1], band, D_inv.dot(self.psi[n + 1, :])))
            #psi[n, :] = np.dot(D, np.linalg.solve(np.eye(self.nx) - delta_t * A, D_inv.dot(psi[n + 1, :])));


        self.u = np.zeros([N + 1, self.nx - 1])
        for n in range(N + 1):
            for i in range(self.nx - 1):
                self.u[n, i] = -2 / beta * self.B[0, 0] * (- log(self.psi[n, i + 1]) + log(self.psi[n, i])) / self.dx
        #self.u = 2 / beta * np.gradient(np.log(self.psi), self.dx, 1)

    def v_true_1(self, x, t):
        i = np.floor((x.squeeze(0) + self.xb) / self.dx).long()
        i[-1] -= 2
        n = int(np.ceil(t / self.delta_t))
        return np.array(- log(self.psi[n, i])).reshape([1, len(i)])

    def u_true_1(self, x, t):
        x = x.unsqueeze(1)
        x = x.t()
        i = np.floor((np.clip(x, -self.xb, self.xb - 2 * self.dx).squeeze(0) + self.xb) / self.dx).long()
        i[-1] -= 2
        n = int(np.ceil(t / self.delta_t))
        return np.array(self.u[n, i]).reshape([1, len(i)])
        #return interpolate.interp1d(self.xvec, self.u)(x)[:, n]
        
    def compute_reference_solution_2(self, delta_t=0.005, xb=2.5, nx=1000):

        self.xb = xb # range of x, [-xb, xb]
        self.nx = nx # number of discrete interval
        self.dx = 2.0 * self.xb / self.nx
        self.delta_t = delta_t

        beta = 2

        self.xvec = np.linspace(-self.xb, self.xb, self.nx, endpoint=True)

        # A = D^{-1} L D
        # assumes Neumann boundary conditions

        A = np.zeros([self.nx, self.nx])
        for i in range(0, self.nx):

            x = -self.xb + (i + 0.5) * self.dx
            if i > 0:
                x0 = -self.xb + (i - 0.5) * self.dx
                x1 = -self.xb + i * self.dx
                A[i, i - 1] = -exp(beta * 0.5 * (self.V_2(x0) + self.V_2(x) - 2 * self.V_2(x1))) / self.dx**2
                A[i, i] = exp(beta * (self.V_2(x) - self.V_2(x1))) / self.dx**2
            if i < self.nx - 1:
                x0 = -self.xb + (i + 1.5) * self.dx
                x1 = -self.xb + (i + 1) * self.dx
                A[i, i + 1] = -exp(beta * 0.5 * (self.V_2(x0) + self.V_2(x) - 2 * self.V_2(x1))) / self.dx**2
                A[i, i] = A[i, i] + exp(beta * (self.V_2(x) - self.V_2(x1))) / self.dx**2

        A = -A / beta
        N = int(self.T / self.delta_t)

        D = np.diag(exp(beta * self.V_2(self.xvec) / 2))
        D_inv = np.diag(exp(-beta * self.V_2(self.xvec) / 2))

        np.linalg.cond(np.eye(self.nx) - self.delta_t * A)
        #w, vv = np.linalg.eigh(np.eye(self.nx) - self.delta_t * A)

        self.psi_2 = np.zeros([N + 1, self.nx])
        self.psi_2[N, :] = exp(-self.g_2(self.xvec))

        for n in range(N - 1, -1, -1):
            band = - self.delta_t * np.vstack([np.append([0], np.diagonal(A, offset=1)),
                                               np.diagonal(A, offset=0) - N / self.T,
                                               np.append(np.diagonal(A, offset=1), [0])])

            self.psi_2[n, :] = D.dot(solve_banded([1, 1], band, D_inv.dot(self.psi_2[n + 1, :])))
            #psi[n, :] = np.dot(D, np.linalg.solve(np.eye(self.nx) - delta_t * A, D_inv.dot(psi[n + 1, :])));


        self.u_2 = np.zeros([N + 1, self.nx - 1])
        for n in range(N + 1):
            for i in range(self.nx - 1):
                self.u_2[n, i] = -2 / beta * self.B[0, 0] * (- log(self.psi_2[n, i + 1]) + log(self.psi_2[n, i])) / self.dx
        #self.u = 2 / beta * np.gradient(np.log(self.psi), self.dx, 1)

    def u_true_2(self, x, t):
        x = x.unsqueeze(1)
        x = x.t()
        i = np.floor((np.clip(x, -self.xb, self.xb - 2 * self.dx).squeeze(0) + self.xb) / self.dx).long()
        i[-1] -= 2
        n = int(np.ceil(t / self.delta_t))
        return np.array(self.u_2[n, i]).reshape([1, len(i)])
        #return interpolate.interp1d(self.xvec, self.u)(x)[:, n]

    def v_true(self, x, t):
        return None

    def u_true(self, x, t):
        return np.concatenate([self.u_true_1(x[:, i], t).T for i in range(self.d_1)] + [self.u_true_2(x[:, i], t).T for i in range(self.d_1, self.d)], 1).T


class DoubleWell_multidim_for_general_solver():
    '''
        Multidimensional extension: one-dimensional double well potential in every dimension.
    '''
    def __init__(self, name='Double well', d=1, d_1=1, d_2=0, T=1, eta=1, kappa=1, modus='HJB'):
        self.name = name
        self.d = d
        self.d_1 = d_1
        self.d_2 = d_2
        self.T = T
        self.eta = eta
        self.eta_ = pt.tensor([eta] * d_1 + [1.0] * d_2).to(device)
        self.kappa = kappa
        self.kappa_ = pt.tensor([kappa] * d_1 + [1.0] * d_2).to(device)
        self.B = pt.eye(self.d).to(device)
        self.X_0 = -pt.ones(self.d).to(device)
        self.ref_sol_is_defined = False
        self.boundary = 'unbounded_square'
        self.X_l = -2.5
        self.X_r = 2.5
        self.modus = modus

    def V(self, x):
        return self.kappa * (x**2 - 1)**2

    def V_2(self, x):
        return (x**2 - 1)**2

    def grad_V(self, x):
        return 4.0 * self.kappa_ * (x * (x**2 - pt.ones(self.d).to(device)))

    def b(self, x):
        return -self.grad_V(x)

    def sigma(self, x):
        return self.B # self.B.repeat(x.shape[0], 1, 1)

    def h(self, t, x, y, z):
        if self.modus == 'linear':
            return pt.zeros(x.shape[0]).to(device)
        return -0.5 * pt.sum(z**2, dim=1)

    #def f(self, x, t):
    #    return pt.zeros(x.shape[0]).to(device)

    def g_1(self, x_1):
        return self.eta * (x_1 - 1)**2

    def g_2(self, x_1):
        return (x_1 - 1)**2

    def f(self, x):
        if self.modus == 'linear':
            return pt.exp(-((pt.sum(self.eta_ * (x - pt.ones(self.d).to(device))**2, 1))).squeeze())
        #return (self.eta * (pt.sum((x - pt.ones(self.d).to(device))**2, 1))).squeeze()
        return ((pt.sum(self.eta_ * (x - pt.ones(self.d).to(device))**2, 1))).squeeze()

    def compute_reference_solution(self, delta_t=0.005, xb=2.5, nx=1000):

        self.xb = xb # range of x, [-xb, xb]
        self.nx = nx # number of discrete interval
        self.dx = 2.0 * self.xb / self.nx
        self.delta_t = delta_t

        beta = 2

        self.xvec = np.linspace(-self.xb, self.xb, self.nx, endpoint=True)

        # A = D^{-1} L D
        # assumes Neumann boundary conditions

        A = np.zeros([self.nx, self.nx])
        for i in range(0, self.nx):

            x = -self.xb + (i + 0.5) * self.dx
            if i > 0:
                x0 = -self.xb + (i - 0.5) * self.dx
                x1 = -self.xb + i * self.dx
                A[i, i - 1] = -exp(beta * 0.5 * (self.V(x0) + self.V(x) - 2 * self.V(x1))) / self.dx**2
                A[i, i] = exp(beta * (self.V(x) - self.V(x1))) / self.dx**2
            if i < self.nx - 1:
                x0 = -self.xb + (i + 1.5) * self.dx
                x1 = -self.xb + (i + 1) * self.dx
                A[i, i + 1] = -exp(beta * 0.5 * (self.V(x0) + self.V(x) - 2 * self.V(x1))) / self.dx**2
                A[i, i] = A[i, i] + exp(beta * (self.V(x) - self.V(x1))) / self.dx**2

        A = -A / beta
        N = int(self.T / self.delta_t)

        D = np.diag(exp(beta * self.V(self.xvec) / 2))
        D_inv = np.diag(exp(-beta * self.V(self.xvec) / 2))

        np.linalg.cond(np.eye(self.nx) - self.delta_t * A)
        #w, vv = np.linalg.eigh(np.eye(self.nx) - self.delta_t * A)

        self.psi = np.zeros([N + 1, self.nx])
        self.psi[N, :] = exp(-self.g_1(self.xvec))

        for n in range(N - 1, -1, -1):
            band = - self.delta_t * np.vstack([np.append([0], np.diagonal(A, offset=1)),
                                               np.diagonal(A, offset=0) - N / self.T,
                                               np.append(np.diagonal(A, offset=1), [0])])

            self.psi[n, :] = D.dot(solve_banded([1, 1], band, D_inv.dot(self.psi[n + 1, :])))
            #psi[n, :] = np.dot(D, np.linalg.solve(np.eye(self.nx) - delta_t * A, D_inv.dot(psi[n + 1, :])));


        self.u = np.zeros([N + 1, self.nx - 1])
        for n in range(N + 1):
            for i in range(self.nx - 1):
                self.u[n, i] = -2 / beta * self.B[0, 0] * (- log(self.psi[n, i + 1]) + log(self.psi[n, i])) / self.dx
        #self.u = 2 / beta * np.gradient(np.log(self.psi), self.dx, 1)

    def v_true_1(self, x, t):
        i = np.floor((x.squeeze(0) + self.xb) / self.dx).long()
        i[-1] -= 2
        n = int(np.ceil(t / self.delta_t))
        if self.modus == 'linear':
            return np.array(self.psi[n, i]).reshape([1, len(i)])
        return np.array(- log(self.psi[n, i])).reshape([1, len(i)])

    def u_true_1(self, x, t):
        x = x.unsqueeze(1)
        x = x.t()
        i = np.floor((np.clip(x, -self.xb, self.xb - 2 * self.dx).squeeze(0) + self.xb) / self.dx).long()
        i[-1] -= 2
        n = int(np.ceil(t / self.delta_t))
        return np.array(self.u[n, i]).reshape([1, len(i)])
        #return interpolate.interp1d(self.xvec, self.u)(x)[:, n]

    def compute_reference_solution_2(self, delta_t=0.005, xb=2.5, nx=1000):

        self.xb = xb # range of x, [-xb, xb]
        self.nx = nx # number of discrete interval
        self.dx = 2.0 * self.xb / self.nx
        self.delta_t = delta_t

        beta = 2

        self.xvec = np.linspace(-self.xb, self.xb, self.nx, endpoint=True)

        # A = D^{-1} L D
        # assumes Neumann boundary conditions

        A = np.zeros([self.nx, self.nx])
        for i in range(0, self.nx):

            x = -self.xb + (i + 0.5) * self.dx
            if i > 0:
                x0 = -self.xb + (i - 0.5) * self.dx
                x1 = -self.xb + i * self.dx
                A[i, i - 1] = -exp(beta * 0.5 * (self.V_2(x0) + self.V_2(x) - 2 * self.V_2(x1))) / self.dx**2
                A[i, i] = exp(beta * (self.V_2(x) - self.V_2(x1))) / self.dx**2
            if i < self.nx - 1:
                x0 = -self.xb + (i + 1.5) * self.dx
                x1 = -self.xb + (i + 1) * self.dx
                A[i, i + 1] = -exp(beta * 0.5 * (self.V_2(x0) + self.V_2(x) - 2 * self.V_2(x1))) / self.dx**2
                A[i, i] = A[i, i] + exp(beta * (self.V_2(x) - self.V_2(x1))) / self.dx**2

        A = -A / beta
        N = int(self.T / self.delta_t)

        D = np.diag(exp(beta * self.V_2(self.xvec) / 2))
        D_inv = np.diag(exp(-beta * self.V_2(self.xvec) / 2))

        np.linalg.cond(np.eye(self.nx) - self.delta_t * A)
        #w, vv = np.linalg.eigh(np.eye(self.nx) - self.delta_t * A)

        self.psi_2 = np.zeros([N + 1, self.nx])
        self.psi_2[N, :] = exp(-self.g_2(self.xvec))

        for n in range(N - 1, -1, -1):
            band = - self.delta_t * np.vstack([np.append([0], np.diagonal(A, offset=1)),
                                               np.diagonal(A, offset=0) - N / self.T,
                                               np.append(np.diagonal(A, offset=1), [0])])

            self.psi_2[n, :] = D.dot(solve_banded([1, 1], band, D_inv.dot(self.psi_2[n + 1, :])))
            #psi[n, :] = np.dot(D, np.linalg.solve(np.eye(self.nx) - delta_t * A, D_inv.dot(psi[n + 1, :])));


        self.u_2 = np.zeros([N + 1, self.nx - 1])
        for n in range(N + 1):
            for i in range(self.nx - 1):
                self.u_2[n, i] = -2 / beta * self.B[0, 0] * (- log(self.psi_2[n, i + 1]) + log(self.psi_2[n, i])) / self.dx
        #self.u = 2 / beta * np.gradient(np.log(self.psi), self.dx, 1)

    def u_true_2(self, x, t):
        x = x.unsqueeze(1)
        x = x.t()
        i = np.floor((np.clip(x, -self.xb, self.xb - 2 * self.dx).squeeze(0) + self.xb) / self.dx).long()
        i[-1] -= 2
        n = int(np.ceil(t / self.delta_t))
        return np.array(self.u_2[n, i]).reshape([1, len(i)])
        #return interpolate.interp1d(self.xvec, self.u)(x)[:, n]

    def v_true_2(self, x, t):
        i = np.floor((x.squeeze(0) + self.xb) / self.dx).long()
        i[-1] -= 2
        n = int(np.ceil(t / self.delta_t))
        if self.modus == 'linear': 
            return np.array(self.psi_2[n, i]).reshape([1, len(i)])
        return np.array(- log(self.psi_2[n, i])).reshape([1, len(i)])

    def v_true(self, x, t):
        if self.modus == 'linear': 
            return np.prod(np.array([self.v_true_1(x[:, i], t).squeeze() for i in range(self.d_1)] + [self.v_true_2(x[:, i], t).squeeze() for i in range(self.d_1, self.d)]), 0)
        return np.sum(np.array([self.v_true_1(x[:, i], t).squeeze() for i in range(self.d_1)] + [self.v_true_2(x[:, i], t).squeeze() for i in range(self.d_1, self.d)]), 0)

    def u_true(self, x, t):
        return np.concatenate([self.u_true_1(x[:, i], t).T for i in range(self.d_1)] + [self.u_true_2(x[:, i], t).T for i in range(self.d_1, self.d)], 1).T


class DoubleWell_multidim_2():
    def __init__(self, name='Double well', d=1, T=1, alpha=1, kappa=1):
        self.name = name
        self.d = d
        self.T = T
        self.alpha = alpha
        self.kappa = kappa
        self.B = pt.eye(self.d).to(device)
        self.X_0 = (pt.ones(self.d) / pt.sqrt((pt.tensor(d).float()))).to(device)
        self.ref_sol_is_defined = False

    def V(self, x):
        return self.kappa * ((pt.sum(x**2, 1) - 3)**2 - 1)**2

    def grad_V(self, x):
        return 4.0 * self.kappa * (pt.sqrt(pt.sum(x**2, 1)) - 3).unsqueeze(1) * ((pt.sqrt(pt.sum(x**2, 1)) - 3).unsqueeze(1)**2 - 1) * x / pt.sqrt(pt.sum(x**2, 1)).unsqueeze(1)

    def b(self, x):
        return -self.grad_V(x)

    def sigma(self, x):
        return self.B # self.B.repeat(x.shape[0], 1, 1)

    def h(self, t, x, y, z):
        return -0.5 * pt.sum(z**2, dim=1)

    def f(self, x, t):
        return pt.zeros(x.shape[0]).to(device)

    def g(self, x):
        return (self.alpha * (pt.sqrt(pt.sum(x**2, 1)) - 2)**2).squeeze()

    def v_true(self, x, t):
        return pt.zeros(x.shape)

    def u_true(self, x, t):
        return pt.zeros(x.shape)


class DoubleWell_multidim_3():
    def __init__(self, name='Double well', d=1, T=1, eta=1, kappa=1):
        self.name = name
        self.d = d
        self.T = T
        self.eta = eta
        self.kappa = kappa
        self.B = pt.eye(self.d).to(device)
        self.X_0 = -pt.ones(self.d).to(device)
        self.ref_sol_is_defined = False

    def V(self, x):
        return self.kappa * (x**2 - 1)**2

    def grad_V(self, x):
        return 4.0 * self.kappa * (x * (x**2 - pt.ones(self.d).to(device)))

    def b(self, x):
        return -self.grad_V(x)

    def sigma(self, x):
        return self.B

    def f(self, x, t):
        return pt.zeros(x.shape[0]).to(device)

    def h(self, t, x, y, z):
        return -0.5 * pt.sum(z**2, dim=1)

    def g_1(self, x_1):
        return self.eta * (x_1 - 1)**2

    def g(self, x):
        return (self.eta * (pt.sum((x - pt.ones(self.d).to(device))**2, 1))).squeeze()

    def compute_reference_solution(self, delta_t=0.005, xb=2.5, nx=1000):

        self.xb = xb # range of x, [-xb, xb]
        self.nx = nx # number of discrete interval
        self.dx = 2.0 * self.xb / self.nx
        self.delta_t = delta_t

        beta = 2

        self.xvec = np.linspace(-self.xb, self.xb, self.nx, endpoint=True)

        # A = D^{-1} L D
        # assumes Neumann boundary conditions

        A = np.zeros([self.nx, self.nx])
        for i in range(0, self.nx):

            x = -self.xb + (i + 0.5) * self.dx
            if i > 0:
                x0 = -self.xb + (i - 0.5) * self.dx
                x1 = -self.xb + i * self.dx
                A[i, i - 1] = -exp(beta * 0.5 * (self.V(x0) + self.V(x) - 2 * self.V(x1))) / self.dx**2
                A[i, i] = exp(beta * (self.V(x) - self.V(x1))) / self.dx**2
            if i < self.nx - 1:
                x0 = -self.xb + (i + 1.5) * self.dx
                x1 = -self.xb + (i + 1) * self.dx
                A[i, i + 1] = -exp(beta * 0.5 * (self.V(x0) + self.V(x) - 2 * self.V(x1))) / self.dx**2
                A[i, i] = A[i, i] + exp(beta * (self.V(x) - self.V(x1))) / self.dx**2

        A = -A / beta
        N = int(self.T / self.delta_t)

        D = np.diag(exp(beta * self.V(self.xvec) / 2))
        D_inv = np.diag(exp(-beta * self.V(self.xvec) / 2))

        np.linalg.cond(np.eye(self.nx) - self.delta_t * A)
        #w, vv = np.linalg.eigh(np.eye(self.nx) - self.delta_t * A)

        self.psi = np.zeros([N + 1, self.nx])
        self.psi[N, :] = exp(-self.g_1(self.xvec))

        for n in range(N - 1, -1, -1):
            band = - self.delta_t * np.vstack([np.append([0], np.diagonal(A, offset=1)),
                                               np.diagonal(A, offset=0) - N / self.T,
                                               np.append(np.diagonal(A, offset=1), [0])])

            self.psi[n, :] = D.dot(solve_banded([1, 1], band, D_inv.dot(self.psi[n + 1, :])))
            #psi[n, :] = np.dot(D, np.linalg.solve(np.eye(self.nx) - delta_t * A, D_inv.dot(psi[n + 1, :])));


        self.u = np.zeros([N + 1, self.nx - 1])
        for n in range(N + 1):
            for i in range(self.nx - 1):
                self.u[n, i] = -2 / beta * self.B[0, 0] * (- log(self.psi[n, i + 1]) + log(self.psi[n, i])) / self.dx
        #self.u = 2 / beta * np.gradient(np.log(self.psi), self.dx, 1)

    def v_true_1(self, x, t):
        i = np.floor((x.squeeze(0) + self.xb) / self.dx).long()
        i[-1] -= 2
        n = int(np.ceil(t / self.delta_t))
        return np.array(- log(self.psi[n, i])).reshape([1, len(i)])

    def u_true_1(self, x, t):
        x = x.unsqueeze(1)
        x = x.t()
        i = np.floor((np.clip(x, -self.xb, self.xb - 2 * self.dx).squeeze(0) + self.xb) / self.dx).long()
        i[-1] -= 2
        n = int(np.ceil(t / self.delta_t))
        return np.array(self.u[n, i]).reshape([1, len(i)])
        #return interpolate.interp1d(self.xvec, self.u)(x)[:, n]

    def v_true(self, x, t):
        return None

    def u_true(self, x, t):
        return np.concatenate([self.u_true_1(x[:, i], t).T for i in range(self.d)], 1).T


class DoubleWell_OU():
    '''
        Combination of double well potential and Ornstein-Uhlenbeck.
    '''
    def __init__(self, name='Double well', d=1, T=1, alpha=1, kappa=1):
        self.name = name
        self.d = d
        self.T = T
        self.alpha = alpha
        self.kappa = kappa
        self.gamma = pt.ones(self.d - 1, 1).to(device) * 1
        self.a = 5
        self.B = pt.eye(self.d).to(device)
        self.X_0 = pt.tensor([-1.0] + [0.0] * (self.d - 1)).to(device)
        self.ref_sol_is_defined = False

    def V(self, x):
        return self.kappa * (x**2 - 1)**2

    def grad_V_1(self, x):
        return 4.0 * self.kappa * x * (x**2 - 1)

    def b(self, x):
        return -pt.cat([self.grad_V_1(x[:, 0]).unsqueeze(1), self.a * x[:, 1:]], 1)

    def sigma(self, x):
        return self.B # self.B.repeat(x.shape[0], 1, 1)

    def f(self, x, t):
        return pt.zeros(x.shape[0]).to(device)

    def h(self, t, x, y, z):
        return -0.5 * pt.sum(z**2, dim=1)

    def g_1(self, x_1):
        return self.alpha * (x_1 - 1)**2

    def g(self, x):
        return self.alpha * (x[:, 0] - 1)**2 + pt.mm(x[:, 1:], self.gamma)[:, 0]# pt.sum(x[:, 1:], 1)

    def compute_reference_solution_x_1(self, delta_t=0.005, xb=2.5, nx=1000):

        self.xb = xb # range of x, [-xb, xb]
        self.nx = nx # number of discrete interval
        self.dx = 2.0 * self.xb / self.nx
        self.delta_t = delta_t

        beta = 2

        self.xvec = np.linspace(-self.xb, self.xb, self.nx, endpoint=True)

        # A = D^{-1} L D
        # assumes Neumann boundary conditions

        A = np.zeros([self.nx, self.nx])
        for i in range(0, self.nx):

            x = -self.xb + (i + 0.5) * self.dx
            if i > 0:
                x0 = -self.xb + (i - 0.5) * self.dx
                x1 = -self.xb + i * self.dx
                A[i, i - 1] = -exp(beta * 0.5 * (self.V(x0) + self.V(x) - 2 * self.V(x1))) / self.dx**2
                A[i, i] = exp(beta * (self.V(x) - self.V(x1))) / self.dx**2
            if i < self.nx - 1:
                x0 = -self.xb + (i + 1.5) * self.dx
                x1 = -self.xb + (i + 1) * self.dx
                A[i, i + 1] = -exp(beta * 0.5 * (self.V(x0) + self.V(x) - 2 * self.V(x1))) / self.dx**2
                A[i, i] = A[i, i] + exp(beta * (self.V(x) - self.V(x1))) / self.dx**2

        A = -A / beta
        N = int(self.T / self.delta_t)

        D = np.diag(exp(beta * self.V(self.xvec) / 2))
        D_inv = np.diag(exp(-beta * self.V(self.xvec) / 2))

        np.linalg.cond(np.eye(self.nx) - self.delta_t * A)
        #w, vv = np.linalg.eigh(np.eye(self.nx) - self.delta_t * A)

        self.psi = np.zeros([N + 1, self.nx])
        self.psi[N, :] = exp(-self.g_1(self.xvec))

        for n in range(N - 1, -1, -1):
            band = - self.delta_t * np.vstack([np.append([0], np.diagonal(A, offset=1)),
                                               np.diagonal(A, offset=0) - N / self.T,
                                               np.append(np.diagonal(A, offset=1), [0])])

            self.psi[n, :] = D.dot(solve_banded([1, 1], band, D_inv.dot(self.psi[n + 1, :])))
            #psi[n, :] = np.dot(D, np.linalg.solve(np.eye(self.nx) - delta_t * A, D_inv.dot(psi[n + 1, :])));


        self.u = np.zeros([N + 1, self.nx - 1])
        for n in range(N + 1):
            for i in range(self.nx - 1):
                self.u[n, i] = -2 / beta * self.B[0, 0] * (- log(self.psi[n, i + 1]) + log(self.psi[n, i])) / self.dx
        #self.u = 2 / beta * np.gradient(np.log(self.psi), self.dx, 1)

    def v_true_1(self, x, t):
        i = np.floor((x.squeeze(0) + self.xb) / self.dx).long()
        i[-1] -= 2
        n = int(np.ceil(t / self.delta_t))
        return np.array(- log(self.psi[n, i])).reshape([1, len(i)])

    def u_true_1(self, x, t):
        x = x.unsqueeze(1)
        x = x.t()
        i = np.floor((np.clip(x, -self.xb, self.xb - 2 * self.dx).squeeze(0) + self.xb) / self.dx).long()
        i[-1] -= 2
        n = int(np.ceil(t / self.delta_t))
        return np.array(self.u[n, i]).reshape([1, len(i)])
        #return interpolate.interp1d(self.xvec, self.u)(x)[:, n]

    def v_true(self, x, t):
        return None

    def u_true(self, x, t):
        u_OU = -np.exp(self.a * (t - self.T)) * np.ones(x[:, 1:].shape) * self.gamma.cpu().numpy().T
        return np.concatenate([self.u_true_1(x[:, 0], t).T, u_OU], 1).T


class ExponentialOnSphere():
    def __init__(self, name='Exponential on sphere', d=2, alpha=1.0):
        self.name = name
        self.d = d
        self.alpha = alpha
        self.B = (pt.sqrt(pt.tensor(2.0)) * pt.eye(self.d)).to(device)
        self.X_0 = pt.zeros(self.d).to(device)
        self.Y_0 = pt.zeros(1).to(device)
        self.boundary = 'sphere'
        self.boundary_distance = 1.0

    def b(self, x):
        return pt.zeros(x.shape).to(device)

    def sigma(self, x):
        return self.B
   
    def f(self, x, t):
        return pt.zeros(x.shape[0]).to(device)

    def g(self, x):
        return pt.exp(self.alpha * pt.sum(x**2, 1))

    def h(self, x, y, z):
        return -self.alpha * y * (self.alpha * 4 * pt.sum(x**2, 1) + 2 * self.d)
   
    def u_true(self, x):
        return -2 * pt.sqrt(pt.tensor(2.0)) * self.alpha * x * pt.exp(self.alpha * pt.sum(x**2, 1).unsqueeze(1)) 
   
    def v_true(self, x):
        return pt.exp(self.alpha * pt.sum(x**2, 1)) 


class ExponentialOnBallNonlinear():
    def __init__(self, name='Exponential on ball nonlinear', d=2, alpha=1.0, boundary_type='Dirichlet'):
        self.name = name
        self.d = d
        self.alpha = alpha
        self.B = (pt.sqrt(pt.tensor(2.0)) * pt.eye(self.d)).to(device)
        self.X_0 = pt.zeros(self.d).to(device)
        self.Y_0 = pt.zeros(1).to(device)
        self.boundary = 'sphere'
        self.boundary_distance = 1.0
        self.boundary_type = boundary_type

    def b(self, x):
        return pt.zeros(x.shape).to(device)

    def sigma(self, x):
        return self.B

    def f(self, x, t):
        return pt.zeros(x.shape[0]).to(device)

    def g(self, x):
        if self.boundary_type == 'Neumann':
            return 2 * self.alpha * x * pt.exp(self.alpha * pt.sum(x**2, 1)).unsqueeze(1)
        return pt.exp(self.alpha * pt.sum(x**2, 1))

    def h(self, x, y, z):
        return -2 * self.alpha * y * (self.alpha * 2 * pt.sum(x**2, 1) + self.d) + pt.exp(2 * self.alpha * pt.sum(x**2, 1)) - y**2
   
    def u_true(self, x):
        return -2 * pt.sqrt(pt.tensor(2.0)) * self.alpha * x * pt.exp(self.alpha * pt.sum(x**2, 1).unsqueeze(1)) 
   
    def v_true(self, x):
        return pt.exp(self.alpha * pt.sum(x**2, 1)) 


class ExponentialOnBallNonlinearSin():
    def __init__(self, name='Exponential on ball nonlinear', d=2, alpha=1.0, boundary_type='Dirichlet'):
        self.name = name
        self.d = d
        self.alpha = alpha
        self.B = (pt.sqrt(pt.tensor(2.0)) * pt.eye(self.d)).to(device)
        self.X_0 = pt.zeros(self.d).to(device)
        self.Y_0 = pt.zeros(1).to(device)
        self.boundary = 'sphere'
        self.boundary_distance = 1.0
        self.boundary_type = boundary_type

    def b(self, x):
        return pt.zeros(x.shape).to(device)

    def sigma(self, x):
        return self.B

    def f(self, x, t):
        return pt.zeros(x.shape[0]).to(device)

    def g(self, x):
        if self.boundary_type == 'Neumann':
            return 2 * self.alpha * x * pt.exp(self.alpha * pt.sum(x**2, 1)).unsqueeze(1)
        return pt.exp(self.alpha * pt.sum(x**2, 1))

    def h(self, x, y, z):
        return -2 * self.alpha * y * (self.alpha * 2 * pt.sum(x**2, 1) + self.d) + pt.sin(pt.exp(2 * self.alpha * pt.sum(x**2, 1)) - y**2)
   
    def u_true(self, x):
        return -2 * pt.sqrt(pt.tensor(2.0)) * self.alpha * x * pt.exp(self.alpha * pt.sum(x**2, 1).unsqueeze(1)) 
   
    def v_true(self, x):
        return pt.exp(self.alpha * pt.sum(x**2, 1)) 


class ExponentialOnBallNonlinearSinHessian():
    def __init__(self, name='Exponential on ball nonlinear', d=2, alpha=1.0, boundary_type='Dirichlet'):
        self.name = name
        self.d = d
        self.alpha = alpha
        self.B = (pt.sqrt(pt.tensor(2.0) / self.d) * pt.ones(self.d, self.d)).to(device)
        self.X_0 = pt.zeros(self.d).to(device)
        self.Y_0 = pt.zeros(1).to(device)
        self.boundary = 'sphere'
        self.boundary_distance = 1.0
        self.boundary_type = boundary_type

    def b(self, x):
        return pt.zeros(x.shape).to(device)

    def sigma(self, x):
        return self.B

    def f(self, x, t):
        return pt.zeros(x.shape[0]).to(device)

    def g(self, x):
        if self.boundary_type == 'Neumann':
            return 2 * self.alpha * x * pt.exp(self.alpha * pt.sum(x**2, 1)).unsqueeze(1)
        return pt.exp(self.alpha * pt.sum(x**2, 1))

    def h(self, x, y, z):
        return -2 * self.alpha * y * (self.alpha * 2 * pt.sum(pt.bmm(x.unsqueeze(2), x.unsqueeze(1)), [1, 2]) + self.d) + pt.sin(pt.exp(2 * self.alpha * pt.sum(x**2, 1)) - y**2)
   
    def u_true(self, x):
        return -2 * pt.sqrt(pt.tensor(2.0)) * self.alpha * x * pt.exp(self.alpha * pt.sum(x**2, 1).unsqueeze(1)) 
   
    def v_true(self, x):
        return pt.exp(self.alpha * pt.sum(x**2, 1)) 


class ExponentialOnSphereParabolic():
    def __init__(self, name='Exponential on sphere', d=2, T=1.0, alpha=1.0):
        self.name = name
        self.d = d
        self.T = T
        self.alpha = alpha
        self.B = (pt.sqrt(pt.tensor(2.0)) * pt.eye(self.d)).to(device)
        self.X_0 = pt.zeros(self.d).to(device)
        self.Y_0 = pt.zeros(1).to(device)
        self.boundary = 'sphere'
        self.boundary_distance = 1.0

    def b(self, x):
        return pt.zeros(x.shape).to(device)

    def sigma(self, x):
        return self.B

    def f(self, x):
        return pt.exp(self.alpha * pt.sum(x**2, 1) + self.T)

    def g(self, x, t):
        return pt.exp(self.alpha * pt.sum(x**2, 1) + t)

    def h(self, t, x, y, z):
        return -y * (2 * self.alpha * (self.alpha * 2 * pt.sum(x**2, 1) + self.d) + 1)

    def u_true(self, x):
        return -2 * pt.sqrt(pt.tensor(2.0)) * self.alpha * x * pt.exp(self.alpha * pt.sum(x**2, 1).unsqueeze(1)) 

    def v_true(self, x, t):
        return pt.exp(self.alpha * pt.sum(x**2, 1) + t) 


class ExponentialOnSphereNonlinearParabolic():
    def __init__(self, name='Exponential on ball', d=2, T=1.0, alpha=1.0):
        self.name = name
        self.d = d
        self.T = T
        self.alpha = alpha
        self.B = (pt.sqrt(pt.tensor(2.0)) * pt.eye(self.d)).to(device)
        self.X_0 = pt.zeros(self.d).to(device)
        self.Y_0 = pt.zeros(1).to(device)
        self.boundary = 'sphere'
        self.boundary_distance = 1.0
        self.boundary_type = 'Dirichlet'

    def b(self, x):
        return pt.zeros(x.shape).to(device)

    def sigma(self, x):
        return self.B

    def f(self, x):
        return pt.exp(self.alpha * pt.sum(x**2, 1) + self.T)

    def g(self, x, t):
        if self.boundary_type == 'Neumann':
            return 2 * self.alpha * x * pt.exp(self.alpha * pt.sum(x**2, 1) + t).unsqueeze(1)
        return pt.exp(self.alpha * pt.sum(x**2, 1) + t)

    def h(self, t, x, y, z):
        #return -y * (2 * self.alpha * (self.alpha * 2 * pt.sum(x**2, 1) + self.d) + 1) + pt.exp(2 * self.alpha * pt.sum(x**2, 1)) - y**2
        return -2 * self.alpha * y * (self.alpha * 2 * pt.sum(x**2, 1) + self.d) - y + pt.sin(pt.exp(2 * self.alpha * pt.sum(x**2, 1) + 2 * t) - y**2)

    def u_true(self, x):
        return -2 * pt.sqrt(pt.tensor(2.0)) * self.alpha * x * pt.exp(self.alpha * pt.sum(x**2, 1).unsqueeze(1)) 

    def v_true(self, x, t):
        return pt.exp(self.alpha * pt.sum(x**2, 1) + t) 


class AllenCahn():
    def __init__(self, name='Allen-Cahn', d=1, T=0.3, seed=42, modus='np'):

        np.random.seed(seed)
        self.modus = modus
        self.name = name
        self.d = d
        self.T = T
        self.B = np.eye(self.d) * np.sqrt(2)
        self.B_pt = pt.tensor(self.B).float().to(device)
        self.alpha = np.ones([self.d, 1]) # not needed, can delete?
        self.alpha_pt = pt.tensor(self.alpha).float().to(device) # not needed, can delete?
        self.X_0 = np.zeros(self.d)
        self.sigma_modus = 'constant'
        self.boundary = 'unbounded'
        self.boundary_distance = 2.0

    def b(self, x):
        if self.modus == 'pt':
            return pt.zeros(x.shape).to(device)
        # return 0
        return np.zeros(x.shape)

    def sigma(self, x):
        if self.modus == 'pt':
            return self.B_pt
        return self.B

    def h(self, t, x, y, z):
        return y - y**3

    def f(self, x):
        if self.modus == 'pt':
            return 1 / (2 + 2 / 5 * pt.sum(x**2, 1))
        return 1 / (2 + 2 / 5 * np.linalg.norm(x, axis=1)**2)

    def u_true(self, x, t):
        print('no reference solution known')
        return 0

    def v_true(self, x, t):
        print('no reference solution known')
        return 0


class DoubleWell_stopping():
    def __init__(self, name='Double well', d=1, beta=1):
        self.name = name
        self.d = d
        self.beta = beta
        self.B = pt.eye(self.d).to(device)
        self.X_0 = -pt.ones(self.d).to(device)
        self.Y_0 = -pt.zeros(self.d).to(device)
        self.boundary = 'square'
        self.one_boundary = True
        self.X_l = -2.0
        self.X_r = 1.0

        if self.d != 1:
            print('The double well example is only implemented for d = 1.')

    def compute_reference_solution(self):
        # discretization of the state space

        f = 1
        sigma = self.B[0, 0].cpu().numpy()
        self.xr = [-2, 2]
        self.dx = 0.01
        Nx = int(np.ceil((self.xr[1] - self.xr[0]) / self.dx))
        x_val = np.linspace(self.xr[0], self.xr[1], Nx)

        # discretization of the generator

        L = np.zeros([Nx, Nx])

        L[0, 0] = - 2 * sigma**2 / 2 / self.dx**2 - self.grad_V(x_val[0]) / self.dx - f
        L[0, 1] = sigma**2 / self.dx
        L[Nx - 1, Nx - 2] = sigma**2 / 2 / self.dx**2 + self.grad_V(x_val[Nx - 1]) / self.dx
        L[Nx - 1, Nx - 1] = - sigma**2 / self.dx**2 - sigma * self.grad_V(x_val[Nx - 1]) / self.dx - f

        for i in range(1, Nx - 1):
            L[i, i - 1] = sigma**2 / 2 / self.dx**2 + self.grad_V(x_val[i]) / self.dx
            L[i, i] = - sigma**2 / self.dx**2 - self.grad_V(x_val[i]) / self.dx - f
            L[i, i + 1] = sigma**2 / 2 / self.dx**2

        d = np.zeros(Nx)

        # boundary condition, apply it for multiple values of x for numerical stability
        L[300:310, :] = 0
        for i in range(300, 310):
            L[i, i] = 1
        d[300:310] = 1 # e^{-g(x)}

        # additional stability: psi shall be flat on the boundary
        L[0, :] = 0
        L[0, 0] = 1
        L[0, 1] = -1
        d[0] = 0

        L[Nx - 1, :] = 0
        L[Nx - 1, Nx - 1] = 1
        L[Nx - 1, Nx - 2] = -1
        d[Nx - 1] = 0

        self.psi = np.linalg.solve(L, d)
        self.u = sigma * (- np.log(self.psi[:-1]) + np.log(self.psi[1:])) / self.dx

    def V(self, x):
        return self.beta * (x**2 - 1)**2

    def grad_V(self, x):
        return 4.0 * self.beta * x * (x**2 - 1)

    def b(self, x):
        return -self.grad_V(x)

    def sigma(self, x):
        return self.B

    def f(self, x):
        return pt.ones(x.shape[0]).to(device)

    def g(self, x):
        return pt.zeros(x.shape[0]).to(device)

    def h(self, x, y, z):
        return -0.5 * pt.sum(z**2, dim=1) + self.f(x)

    def u_true(self, x, t):
        i = pt.clamp(np.floor((x.squeeze(0) + self.xr[1]) / self.dx).long(), 0, 298).view(-1)
        return np.array(self.u[i]).reshape([1, len(i)])

    def v_true(self, x):
        i = pt.clamp(pt.floor((x.squeeze() + self.xr[1]) / self.dx).long(), 0, 298).view(-1)
        return np.array(-np.log(self.psi)[i]).reshape([1, len(i)])


class DoubleWell_stopping_linear():
    def __init__(self, name='Double well', d=1, beta=1):
        self.name = name
        self.d = d
        self.beta = beta
        self.B = pt.eye(self.d).to(device)
        self.X_0 = -pt.ones(self.d).to(device)
        self.Y_0 = -pt.zeros(self.d).to(device)
        self.boundary = 'square'
        self.one_boundary = True
        self.X_l = -2.0
        self.X_r = 1.0

        if self.d != 1:
            print('The double well example is only implemented for d = 1.')

    def compute_reference_solution(self):
        # discretization of the state space

        f = 1
        sigma = self.B[0, 0].cpu().numpy()
        self.xr = [-2, 2]
        self.dx = 0.01
        Nx = int(np.ceil((self.xr[1] - self.xr[0]) / self.dx))
        x_val = np.linspace(self.xr[0], self.xr[1], Nx)

        # discretization of the generator

        L = np.zeros([Nx, Nx])

        L[0, 0] = - 2 * sigma**2 / 2 / self.dx**2 - self.grad_V(x_val[0]) / self.dx - f
        L[0, 1] = sigma**2 / self.dx
        L[Nx - 1, Nx - 2] = sigma**2 / 2 / self.dx**2 + self.grad_V(x_val[Nx - 1]) / self.dx
        L[Nx - 1, Nx - 1] = - sigma**2 / self.dx**2 - sigma * self.grad_V(x_val[Nx - 1]) / self.dx - f

        for i in range(1, Nx - 1):
            L[i, i - 1] = sigma**2 / 2 / self.dx**2 + self.grad_V(x_val[i]) / self.dx
            L[i, i] = - sigma**2 / self.dx**2 - self.grad_V(x_val[i]) / self.dx - f
            L[i, i + 1] = sigma**2 / 2 / self.dx**2

        d = np.zeros(Nx)

        # boundary condition, apply it for multiple values of x for numerical stability
        L[300:310, :] = 0
        for i in range(300, 310):
            L[i, i] = 1
        d[300:310] = 1 # e^{-g(x)}

        # additional stability: psi shall be flat on the boundary
        L[0, :] = 0
        L[0, 0] = 1
        L[0, 1] = -1
        d[0] = 0

        L[Nx - 1, :] = 0
        L[Nx - 1, Nx - 1] = 1
        L[Nx - 1, Nx - 2] = -1
        d[Nx - 1] = 0

        self.psi = np.linalg.solve(L, d)
        self.u = sigma * (- np.log(self.psi[:-1]) + np.log(self.psi[1:])) / self.dx

    def V(self, x):
        return self.beta * (x**2 - 1)**2

    def grad_V(self, x):
        return 4.0 * self.beta * x * (x**2 - 1)

    def b(self, x):
        return -self.grad_V(x)

    def sigma(self, x):
        return self.B

    def f(self, x):
        return pt.ones(x.shape[0]).to(device)

    def g(self, x):
        return pt.ones(x.shape[0]).to(device)

    def h(self, x, y, z):
        return -self.f(x) * y

    def u_true(self, x, t):
        i = pt.clamp(np.floor((x.squeeze(0) + self.xr[1]) / self.dx).long(), 0, 298).view(-1)
        return np.array(self.u[i]).reshape([1, len(i)])

    def v_true(self, x):
        i = pt.clamp(np.floor((x.squeeze() + self.xr[1]) / self.dx).long(), 0, 298).view(-1)
        return np.array(self.psi[i]).reshape([1, len(i)])


class DoubleWell_expectation_hitting_time():
    def __init__(self, name='Double well', d=1, beta=1, dx=0.01, eta=2.0):
        self.name = name
        self.d = d
        self.beta = beta
        self.dx = dx
        self.B = eta * pt.eye(self.d).to(device)
        self.X_0 = -pt.ones(self.d).to(device)
        self.Y_0 = -pt.zeros(self.d).to(device)
        self.boundary = 'square'
        self.one_boundary = True
        self.X_l = -2.0
        self.X_r = 1.0

        if self.d != 1:
            print('The double well example is only implemented for d = 1.')

    def compute_reference_solution(self):
        # discretization of the state space

        f = 0
        sigma = self.B[0, 0].cpu().numpy()
        self.xr = [-2, 2]
        Nx = int(np.ceil((self.xr[1] - self.xr[0]) / self.dx))
        x_val = np.linspace(self.xr[0], self.xr[1], Nx)

        # discretization of the generator

        L = np.zeros([Nx, Nx])

        L[0, 0] = - 2 * sigma**2 / 2 / self.dx**2 - self.grad_V(x_val[0]) / self.dx - f
        L[0, 1] = sigma**2 / self.dx
        L[Nx - 1, Nx - 2] = sigma**2 / 2 / self.dx**2 + self.grad_V(x_val[Nx - 1]) / self.dx
        L[Nx - 1, Nx - 1] = - sigma**2 / self.dx**2 - sigma * self.grad_V(x_val[Nx - 1]) / self.dx - f

        for i in range(1, Nx - 1):
            L[i, i - 1] = sigma**2 / 2 / self.dx**2 + self.grad_V(x_val[i]) / self.dx
            L[i, i] = - sigma**2 / self.dx**2 - self.grad_V(x_val[i]) / self.dx - f
            L[i, i + 1] = sigma**2 / 2 / self.dx**2

        d = -np.ones(Nx)

        # boundary condition, apply it for multiple values of x for numerical stability

        index_r = int((self.X_r - self.X_l) / self.dx)

        L[index_r:int(index_r * 1.1), :] = 0
        for i in range(index_r, int(index_r * 1.1)):
            L[i, i] = 1
        d[index_r:int(index_r * 1.1)] = 0 #g(x)

        # additional stability: psi shall be flat on the boundary
        L[0, :] = 0
        L[0, 0] = 1
        L[0, 1] = -1
        d[0] = 0

        L[Nx - 1, :] = 0
        L[Nx - 1, Nx - 1] = 1
        L[Nx - 1, Nx - 2] = -1
        d[Nx - 1] = 0

        self.psi = np.linalg.solve(L, d)
        self.u = sigma * (- np.log(self.psi[:-1]) + np.log(self.psi[1:])) / self.dx

    def V(self, x):
        return self.beta * (x**2 - 1)**2

    def grad_V(self, x):
        return 4.0 * self.beta * x * (x**2 - 1)

    def b(self, x):
        return -self.grad_V(x)

    def sigma(self, x):
        return self.B

    def f(self, x):
        return pt.zeros(x.shape[0]).to(device)

    def g(self, x):
        return pt.zeros(x.shape[0]).to(device)

    def h(self, x, y, z):
        return pt.ones(y.shape[0]).to(device)

    def u_true(self, x):
        return x

    def v_true(self, x):
        index_r = int((self.X_r - self.X_l) / self.dx)
        i = pt.clamp(np.floor((x.squeeze() + self.xr[1]) / self.dx).long(), 0, index_r).view(-1)
        return np.array(self.psi[i]).reshape([1, len(i)])


class Committor_DoubleWell():
    def __init__(self, name='Double well', d=1, beta=1, dx=0.01, eta=2.0, T=1.0):
        self.name = name
        self.T = T
        self.d = d
        self.beta = beta
        self.dx = dx
        self.B = np.sqrt(eta) * pt.eye(self.d).to(device)
        self.X_0 = -pt.ones(self.d).to(device)
        self.Y_0 = -pt.zeros(self.d).to(device)
        self.boundary = 'square'
        self.one_boundary = True
        self.boundary_type = 'Dirichlet'
        self.X_l = -2.0
        self.X_r = 0.0

        if self.d != 1:
            print('The double well example is only implemented for d = 1.')

    def V(self, x):
        return self.beta * (x**2 - 1)**2

    def grad_V(self, x):
        return 4.0 * self.beta * x * (x**2 - 1)

    def b(self, x):
        return -self.grad_V(x)

    def sigma(self, x, t):
        return self.B

    def f(self, x):
        return pt.zeros(x.shape[0]).to(device)

    def g(self, x, t):
        return pt.ones(x.shape[0]).to(device)

    def h(self, x, t, y, z):
        return pt.zeros(y.shape[0]).to(device)

    def u_true(self, x, t):
        return pt.zeros(x.shape[0]).to(device)

    def v_true(self, x, t):
        return pt.zeros(x.shape[0]).to(device)


class Committor():
    def __init__(self, name='Committor', d=2, alpha=1.0):
        self.name = name
        self.a = 1.0
        self.c = 2.0
        self.d = d
        self.B = (pt.eye(self.d)).to(device)
        self.X_0 = pt.zeros(self.d).to(device)
        self.Y_0 = pt.zeros(1).to(device)
        self.boundary = 'two_spheres'
        self.boundary_distance_1 = self.a
        self.boundary_distance_2 = self.c

    def b(self, x):
        return pt.zeros(x.shape).to(device)

    def sigma(self, x):
        return self.B

    def f(self, x):
        return pt.zeros(x.shape[0]).to(device)

    def g(self, x):
        return (pt.sqrt(pt.sum(x**2, 1)) > self.a).float().to(device)

    def h(self, x, y, z):
        return pt.zeros(x.shape[0]).to(device)

    def u_true(self, x):
        return pt.zeros(x.shape)

    def v_true(self, x):
        return ((self.a**2 - pt.sqrt(pt.sum(x**2, 1))**(2 - self.d) * self.a**self.d) 
                 / (self.a**2 - self.c**(2 - self.d) * self.a**self.d))


class QuadraticGradient():
    def __init__(self, name='Quadratic Gradient', d=1, r=1.0):
        self.name = name
        self.d = d
        self.B = pt.sqrt(pt.tensor(2.0)).to(device) * pt.eye(self.d).to(device)
        self.X_0 = -pt.ones(self.d).to(device)
        self.Y_0 = -pt.zeros(self.d).to(device)
        self.boundary = 'sphere'
        self.boundary_distance = r

    def b(self, x):
        return pt.zeros(x.shape).to(device)

    def sigma(self, x):
        return self.B
   
    def f(self, x):
        return pt.zeros(x.shape[0]).to(device)

    def g(self, x):
        return pt.log((pt.sum(x**2, 1) + 1) / self.d)

    def h(self, x, y, z):
        return pt.sum(z**2, dim=1) / self.B[0, 0]**2 - 2 * pt.exp(-y)
   
    def u_true(self, x, t):
        return pt.zeros(x.shape[0], 1).to(device)
   
    def v_true(self, x):
        return pt.log((pt.sum(x**2, 1) + 1) / self.d)


class Helmholtz():
    def __init__(self, name='Helmholtz', d=2, r=1.0):
        self.name = name
        self.d = d
        self.B = pt.sqrt(pt.tensor(2.0)).to(device) * pt.eye(self.d).to(device)
        self.X_0 = -pt.ones(self.d).to(device)
        self.Y_0 = -pt.zeros(self.d).to(device)
        self.a_1 = 1.0
        self.a_2 = 4.0
        self.k = 1.0
        self.pi = pt.tensor(np.pi)
        self.boundary = 'square'
        self.one_boundary = False
        self.X_l = -1.0
        self.X_r = 1.0

        if d != 2:
            print('Only implemented for d = 2.')

    def b(self, x):
        return pt.zeros(x.shape).to(device)

    def sigma(self, x):
        return self.B
   
    def f(self, x):
        return pt.zeros(x.shape[0]).to(device)

    def g(self, x):
        return pt.sin(self.a_1 * self.pi * x[:, 0]) * pt.sin(self.a_2 * self.pi * x[:, 1])

    def h(self, x, y, z):
        return (self.k**2 * y + (self.a_1 * self.pi)**2 * pt.sin(self.a_1 * self.pi * x[:, 0]) * pt.sin(self.a_2 * self.pi * x[:, 1])
            + (self.a_2 * self.pi)**2 * pt.sin(self.a_1 * self.pi * x[:, 0]) * pt.sin(self.a_2 * self.pi * x[:, 1]) 
            - self.k**2 * pt.sin(self.a_1 * self.pi * x[:, 0]) * pt.sin(self.a_2 * self.pi * x[:, 1]))

    def u_true(self, x, t):
        return pt.zeros(x.shape[0], 1).to(device)

    def v_true(self, x):
        return pt.sin(self.a_1 * self.pi * x[:, 0]) * pt.sin(self.a_2 * self.pi * x[:, 1])


class Oscillations():
    def __init__(self, name='Oscillations', d=1, r=1.0):
        self.name = name
        self.d = d
        self.B = pt.sqrt(pt.tensor(2.0)).to(device) * pt.eye(self.d).to(device)
        self.X_0 = -pt.ones(self.d).to(device)
        self.Y_0 = -pt.zeros(self.d).to(device)
        self.pi = pt.tensor(np.pi)
        self.a = 5
        self.boundary = 'square'
        self.one_boundary = False
        self.X_l = 0.0
        self.X_r = 1.0

        if d != 1:
            print('Only implemented for d = 1.')

    def b(self, x):
        return pt.zeros(x.shape).to(device)

    def sigma(self, x):
        return self.B

    def f(self, x):
        return pt.zeros(x.shape[0]).to(device)

    def g(self, x):
        return pt.zeros(x.shape[0]).to(device)

    def h(self, x, y, z):
        return (2 * self.pi)**2 * pt.sin(2 * self.pi * x[:, 0]) + (self.a * self.pi)**2 * 0.1 * pt.sin(self.a * self.pi * x[:, 0])

    def u_true(self, x, t):
        return pt.zeros(x.shape[0], 1).to(device)

    def v_true(self, x):
        return pt.sin(2 * self.pi * x[:, 0]) + 0.1 * pt.sin(self.a * self.pi * x[:, 0])


class SinNorm2():
    def __init__(self, name='SinNorm2', d=1, r=1.0, linear=True, alpha=1.0):
        self.name = name
        self.d = d
        self.alpha = alpha
        self.B = self.alpha * pt.sqrt(pt.tensor(2.0) / self.d).to(device) * pt.ones(self.d, self.d).to(device)
        self.X_0 = -pt.ones(self.d).to(device)
        self.Y_0 = -pt.zeros(self.d).to(device)
        self.pi = pt.tensor(np.pi)
        self.linear = linear
        self.boundary = 'sphere'
        self.boundary_distance = 1.0

    def b(self, x):
        return pt.zeros(x.shape).to(device)

    def sigma(self, x):
        return self.B

    def f(self, x):
        return pt.zeros(x.shape[0]).to(device)

    def g(self, x):
        return pt.zeros(x.shape[0]).to(device)

    def h(self, x, y, z):
        if self.linear:
            return self.alpha**2 * (4 * self.pi**2 * pt.sin(self.pi * pt.sum(x**2, 1)) * pt.sum(x, 1)**2 - 2 * self.d * self.pi * pt.cos(self.pi * pt.sum(x**2, 1)))
        return self.alpha**2 * (4 * self.pi**2 * y * pt.sum(x, 1)**2 - 2 * self.d * self.pi * pt.cos(self.pi * pt.sum(x**2, 1)) + pt.sin(self.pi * pt.sum(x**2, 1))**2 - y**2)

    def u_true(self, x, t):
        return pt.zeros(x.shape[0], 1).to(device)

    def v_true(self, x):
        return pt.sin(self.pi * pt.sum(x**2, 1))


class HeatEquation():
    def __init__(self, name='Heat equation', d=1, T=1, seed=42):

        pt.manual_seed(seed)
        self.name = name
        self.d = d
        self.T = T
        self.B = pt.sqrt(pt.tensor(2.0)) * (pt.eye(self.d)).to(device)
        self.boundary = 'unbounded'
        self.boundary_type = 'Dirichlet'
        self.boundary_distance = 1.0

    def b(self, x):
        return pt.zeros(x.shape).to(device)

    def sigma(self, x):
        return self.B

    def g(self, x, t):
        return pt.zeros(x.shape[0]).to(device)

    def h(self, t, x, y, z):
        return pt.zeros(x.shape[0]).to(device)

    def f(self, x):
        return pt.sum(x**2, 1)

    def u_true(self, x, t):
        return None

    def v_true(self, x, t):
        return pt.sum(x**2, 1) + 2 * (self.T - t) * self.d
