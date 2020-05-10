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

        if ~np.all(np.linalg.eigvals(self.A.cpu().numpy()) < 0):
            print('not all EV of A are negative')

    def b(self, x):
        return pt.mm(self.A, x.t()).t()

    def sigma(self, x):
        return self.B

    def f(self, x, t):
        return pt.zeros(x.shape[0]).to(device)

    def h(self, t, x, y, z):
        return 0.5 * pt.sum(z**2, dim=1)

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
            Sigma_n += (expm(self.A.cpu().numpy() * t_n)
                        .dot(self.sigma(np.zeros([self.d, self.d])).cpu())
                        .dot(self.sigma(np.zeros([self.d, self.d])).t().cpu())
                        .dot(expm(self.A.cpu().numpy().T * t_n))) * delta_t
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
        return (0.8 * ((- z)**2)**(0.625) + x * pt.exp(self.T - t) - 0.8 * pt.exp(1.25 * (self.T - t)))[:, 0]

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
        return 0.5 * pt.sum(z**2, dim=1) - self.f(x, t)

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
        return 0.5 * pt.sum(z**2, dim=1)

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
        return 0.5 * pt.sum(z**2, dim=1)

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
        return 0.5 * pt.sum(z**2, dim=1)

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
        return 0.5 * pt.sum(z**2, dim=1)

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
        return 0.5 * pt.sum(z**2, dim=1)

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
