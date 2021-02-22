#pylint: disable=invalid-name, no-member, too-many-arguments, missing-docstring
#pylint: disable=too-many-instance-attributes, not-callable, no-else-return
#pylint: disable=inconsistent-return-statements, too-many-locals, too-many-return-statements
#pylint: disable=too-many-statements, too-many-public-methods

from copy import deepcopy
from datetime import date
import json
import numpy as np
import os
import time
import torch as pt

from function_space import DenseNet, Linear, NN, NN_Nik, SingleParam, MySequential
from utilities import compute_test_error, do_importance_sampling, do_importance_sampling_me


class Solver():

    def __init__(self, name, problem, lr=0.001, L=10000, K=50, delta_t=0.05,
                 approx_method='control', loss_method='log-variance', time_approx='outer',
                 learn_Y_0=False, adaptive_forward_process=True, detach_forward=False,
                 early_stopping_time=10000, random_X_0=False, compute_gradient_variance=0,
                 IS_variance_K=0, IS_variance_iter=1, metastability_logs=None, print_every=100, plot_trajectories=None,
                 seed=42, save_results=False, u_l2_error_flag=True, log_gradient=False, burgers_drift=False, verbose=True):
        self.problem = problem
        self.name = name
        self.date = date.today().strftime('%Y-%m-%d')
        self.d = problem.d
        self.T = problem.T
        self.X_0 = problem.X_0
        self.Y_0 = pt.tensor([0.0])
        self.X_u_opt = None

        # hyperparameters
        self.device = pt.device('cpu')
        self.seed = seed
        self.delta_t_np = delta_t
        self.delta_t = pt.tensor(self.delta_t_np).to(self.device) # step size
        self.sq_delta_t = pt.sqrt(self.delta_t).to(self.device)
        self.N = int(np.floor(self.T / self.delta_t_np)) # number of steps
        self.lr = lr # learning rate
        self.L = L # gradient steps
        self.K = K # batch size
        self.random_X_0 = random_X_0

        # learning properties
        self.loss_method = loss_method
        self.approx_method = approx_method
        self.learn_Y_0 = learn_Y_0
        self.adaptive_forward_process = adaptive_forward_process
        self.detach_forward = detach_forward
        self.early_stopping_time = early_stopping_time
        self.burgers_drift = burgers_drift

        self.has_ref_solution = hasattr(problem, 'u_true')
        self.u_l2_error_flag = u_l2_error_flag
        if self.has_ref_solution is False:
            self.u_l2_error_flag = False

        if self.loss_method == 'relative_entropy':
            self.adaptive_forward_process = True
        if self.loss_method == 'cross_entropy':
            self.learn_Y_0 = False

        # printing and logging
        self.print_every = print_every
        self.verbose = verbose
        self.verbose_NN = False
        self.save_results = save_results
        self.compute_gradient_variance = compute_gradient_variance
        self.IS_variance_K = IS_variance_K
        self.IS_variance_iter = IS_variance_iter
        self.metastability_logs = metastability_logs
        self.plot_trajectories = plot_trajectories
        self.metastability_logs = metastability_logs
        self.log_gradient = log_gradient
        self.print_gradient_norm = False

        # function approximation
        self.Phis = []
        self.time_approx = time_approx

        pt.manual_seed(seed)
        if self.approx_method == 'control':
            self.y_0 = SingleParam(lr=self.lr).to(self.device)
            if self.time_approx == 'outer':
                self.z_n = [DenseNet(d_in=self.d, d_out=self.d, lr=self.lr, seed=seed) for i in range(self.N)]
            elif self.time_approx == 'inner':
                #self.z_n = DenseNet(d_in=self.d + 1, d_out=self.d, lr=self.lr, seed=seed)
                self.z_n = MySequential(d_in=self.d + 1, d_out=self.d, lr=self.lr, seed=123)

        elif self.approx_method == 'value_function':
            if self.time_approx == 'outer':
                self.y_n = [DenseNet(d_in=self.d, d_out=1, lr=self.lr, seed=seed) for i in range(self.N)]
            elif self.time_approx == 'inner':
                self.y_n = [DenseNet(d_in=self.d + 1, d_out=1, lr=self.lr, seed=seed)]

        self.update_Phis()

        for phi in self.Phis:
            phi.train()

        if self.verbose_NN is True:
            if self.time_approx == 'outer':
                print('%d NNs, %d parameters in each network, total parameters = %d' 
                      % (self.N, self.p, self.p * self.N))
            else:
                print('%d NNs, %d parameters in each network, total parameters = %d'
                      % (1, self.p, self.p))

        # logging
        self.Y_0_log = []
        self.loss_log = []
        self.u_L2_loss = []
        self.IS_rel_log = []
        self.times = []
        self.grads_rel_error_log = []
        self.particles_close_to_target = []

    def b(self, x):
        return self.problem.b(x)

    def sigma(self, x):
        return self.problem.sigma(x)

    def h(self, t, x, y, z):
        return self.problem.h(t, x, y, z)

    def f(self, x, t):
        return self.problem.f(x, t)

    def g(self, x):
        return self.problem.g(x)

    def u_true(self, x, t):
        return self.problem.u_true(x, t)

    def v_true(self, x, t):
        return self.problem.v_true(x, t)

    def update_Phis(self):
        if self.approx_method == 'control':
            if self.learn_Y_0 is True:
                if self.time_approx == 'outer':
                    self.Phis = self.z_n + [self.y_0]
                elif self.time_approx == 'inner':
                    self.Phis = [self.z_n, self.y_0]
            else:
                if self.time_approx == 'outer':
                    self.Phis = self.z_n
                elif self.time_approx == 'inner':
                    self.Phis = [self.z_n]
        elif self.approx_method == 'value_function':
            self.Phis = self.y_n
        for phi in self.Phis:
            phi.to(self.device)
        self.p = sum([np.prod(params.size()) for params in
          filter(lambda params: params.requires_grad,
                 self.Phis[0].parameters())])
        if self.log_gradient is True:
            self.gradient_log = pt.zeros(self.L, self.p)

    def loss_function(self, X, Y, Z_sum, l):
        if self.loss_method == 'moment':
            return (Y - self.g(X)).pow(2).mean()
        elif self.loss_method == 'log-variance':
            return (Y - self.g(X)).pow(2).mean() - (Y - self.g(X)).mean().pow(2)
        elif self.loss_method == 'log-variance-repa':
            return (l % 2 * 2 - 1) * ((Y - self.g(X)).pow(2).mean() - (Y - self.g(X)).mean().pow(2))
        elif self.loss_method == 'variance':
            return pt.var(pt.exp(- self.g(X) + Y))
        elif self.loss_method == 'log-variance_red':
            return ((-u_int - self.g(X)).pow(2).mean() - 2 * ((-u_int - self.g(X)) * u_W_int).mean()
                    + 2 * u_int.mean() - (-u_int - self.g(X)).mean().pow(2))
        elif self.loss_method == 'log-variance_red_2':
            return ((-u_int - self.g(X)).pow(2).mean() + 2 * (self.g(X) * u_W_int).mean()
                    - double_int.mean() + 2 * u_int.mean() - (-u_int - self.g(X)).mean().pow(2))
        elif self.loss_method == 'relative_entropy':
            return (Z_sum + self.g(X)).mean()
        elif self.loss_method == 'relative_entropy_BSDE':
            return (Z_sum + self.g(X)).mean()
        elif self.loss_method == 'cross_entropy':
            if self.adaptive_forward_process is True:
                return (Y * pt.exp(-self.g(X) + Y.detach())).mean()
            return (Y * pt.exp(-self.g(X))).mean()
        elif self.loss_method == 'relative_entropy_log-variance':
            if l < 1000:
                return ((Z_sum + self.g(X))).mean()
            return (Y - self.g(X)).pow(2).mean() - (Y - self.g(X)).mean().pow(2)
        elif self.loss_method == 'reparametrization':
            return (Z_sum + self.g(X)).mean()

    def zero_grad(self):
        for phi in self.Phis:
            phi.optim.zero_grad()

    def optimization_step(self):
        for phi in self.Phis:
            phi.optim.step()

    def gradient_descent(self, X, Y, Z_sum, l, additional_loss):
        self.zero_grad()

        if self.loss_method == 'log-variance-y_0':

            loss_1 = pt.var(Y - self.g(X))
            loss_1.backward(retain_graph=True)
            self.z_n.optim.step()

            if self.learn_Y_0 is True:
                loss_2 = pt.mean(Y - self.g(X))**2
                loss_2.backward(retain_graph=True)
                self.y_0.optim.step()
            else:
                loss_2 = 0

            return loss_1 + loss_2

        loss = self.loss_function(X, Y, Z_sum, l) + additional_loss
        loss.backward()
        self.optimization_step()
        return loss

    def flatten_gradient(self, k, grads, grads_flat):
        i = 0
        for grad in grads:
            grad_flat = grad.reshape(-1)
            j = len(grad_flat)
            grads_flat[k, i:i + j] = grad_flat
            i += j
        return grads_flat

    def get_gradient_variances(self, X, Y):
        grads_mean = pt.zeros(self.N, self.p)
        grads_var = pt.zeros(self.N, self.p)

        for n in range(self.N):

            grads_Y_flat = pt.zeros(self.K, self.p)

            for k in range(self.K):
                self.zero_grad()
                Y[k].backward(retain_graph=True)

                grad_Y = [params.grad for params in list(filter(lambda params:
                                                                params.requires_grad,
                                                                self.z_n[n].parameters()))
                          if params.grad is not None]

                grads_Y_flat = self.flatten_gradient(k, grad_Y, grads_Y_flat)

            grads_g_X_flat = pt.zeros(self.K, self.p)

            if self.adaptive_forward_process is True:

                for k in range(self.K):
                    self.zero_grad()
                    self.g(X[0, :].unsqueeze(0)).backward(retain_graph=True)

                    grad_g_X = [params.grad for params in list(filter(lambda params:
                                                                      params.requires_grad,
                                                                      self.z_n[n].parameters()))
                                if params.grad is not None]

                    grads_g_X_flat = self.flatten_gradient(k, grad_g_X, grads_g_X_flat)

            if self.loss_method == 'moment':
                grads_flat = 2 * (Y - self.g(X)).unsqueeze(1) * (grads_Y_flat - grads_g_X_flat)
            elif self.loss_method == 'log-variance':
                grads_flat = 2 * (((Y - self.g(X)).unsqueeze(1)
                                   - pt.mean((Y - self.g(X)).unsqueeze(1), 0).unsqueeze(0))
                                  * (grads_Y_flat - grads_g_X_flat
                                     - pt.mean(grads_Y_flat - grads_g_X_flat, 0).unsqueeze(0)))

            grads_mean[n, :] = pt.mean(grads_flat, dim=0)
            grads_var[n, :] = pt.var(grads_flat, dim=0)

        grads_rel_error = pt.sqrt(grads_var) / grads_mean
        grads_rel_error[grads_rel_error != grads_rel_error] = 0
        return grads_rel_error

    def state_dict_to_list(self, sd):
        sd_list = {}
        for name in sd:
            if type(sd[name]) == pt.Tensor:
                sd_list[name] = sd[name].detach().cpu().numpy().tolist()
            else:
                sd_list[name] = sd[name]
        return sd_list

    def list_to_state_dict(self, l):
         return {param: pt.tensor(l[param]) if type(l[param]) == list else l[param] for param in l}

    def save_logs(self, model_name='model'):
        # currently does not work for all modi
        logs = {'name': self.name, 'date': self.date, 'd': self.d, 'T': self.T,
                'seed': self.seed, 'delta_t': self.delta_t_np, 'N': self.N, 'lr': self.lr,
                'K': self.K, 'loss_method': self.loss_method, 'learn_Y_0': self.learn_Y_0,
                'adaptive_forward_process': self.adaptive_forward_process,
                'Y_0_log': self.Y_0_log, 'loss_log': self.loss_log, 'u_L2_loss': self.u_L2_loss,
                'Phis_state_dict': [self.state_dict_to_list(z.cpu().state_dict()) for z in self.Phis]}

        path_name = 'logs/%s_%s_%s.json' % (model_name, self.name, self.date)
        i = 1
        while os.path.isfile(path_name):
            i += 1
            path_name = 'logs/%s_%s_%s_%d.json' % (model_name, self.name, self.date, i)

        with open(path_name, 'w') as f:
            json.dump(logs, f, indent=2)

    def save_networks(self):
        data_dict = {}
        idx = 0
        for z in self.Phis:
            key = 'nn%d' % idx
            data_dict[key] = z.state_dict()
            idx += 1
        path_name = 'output/%s_%s.pt' % (self.name, self.date)
        pt.save(data_dict, path_name)
        print('\nnetworks data has been stored to file: %s' % path_name)

    def load_networks(self, cp_name):
        print('\nload network data from file: %s' % cp_name)
        checkpoint = pt.load(cp_name)
        idx = 0
        for z in self.Phis:
            key = 'nn%d' % idx
            z.load_state_dict(checkpoint[key])
            z.eval()
            idx += 1

    def compute_grad_Y(self, X, n):
        Y_n_eval = self.Y_n(X, n).squeeze(1).sum() # compare to Jacobi-Vector trick
        Y_n_eval.backward(retain_graph=True) # do we need this?
        Z, = pt.autograd.grad(Y_n_eval, X, create_graph=True)
        Z = pt.mm(self.sigma(X), Z.t()).t()
        return Z

    def Y_n(self, X, t):
        n = int(np.ceil(t / self.delta_t))
        if self.time_approx == 'outer':
            return self.y_n[n](X)
        elif self.time_approx == 'inner':
            t_X = pt.cat([pt.ones([X.shape[0], 1]) * t, X], 1)
            return self.y_n[0](t_X)

    def Z_n_(self, X, n):
        if self.approx_method == 'control':
            if self.time_approx == 'outer':
                n = max(0, min(n, self.N - 1))
                return self.z_n[n](X)
            elif self.time_approx == 'inner':
                t_X = pt.cat([pt.ones([X.shape[0], 1]).to(self.device) * n * self.delta_t, X], 1)
                return self.z_n(t_X)
        if self.approx_method == 'value_function':
            return self.compute_grad_Y(X, n)

    def Z_n(self, X, t):
        n = int(pt.ceil(t / self.delta_t))
        return self.Z_n_(X, n)

    def initialize_training_data(self):
        X = self.X_0.repeat(self.K, 1).to(self.device)
        if self.random_X_0 is True:
            X = pt.randn(self.K, self.d).to(self.device)
        Y = self.Y_0.repeat(self.K).to(self.device)
        if self.approx_method == 'value_function':
            X = pt.autograd.Variable(X, requires_grad=True)
            Y = self.Y_n(X, 0)[:, 0]
        elif self.learn_Y_0 is True:
            Y = self.y_0(X)
            self.Y_0_log.append(Y[0].item())
        Z_sum = pt.zeros(self.K).to(self.device)
        u_L2 = pt.zeros(self.K).to(self.device)
        u_int = pt.zeros(self.K).to(self.device)
        u_W_int = pt.zeros(self.K).to(self.device)
        double_int = pt.zeros(self.K).to(self.device)

        xi = pt.randn(self.K, self.d, self.N + 1).to(self.device)
        return X, Y, Z_sum, u_L2, u_int, u_W_int, double_int, xi

    def train_LSE_with_reference(self):
        if self.approx_method != 'control':
            print('only learn control with reference solution!')
        if self.has_ref_solution == False:
            print('reference solution is needed!')

        print('\nd = %d, L = %d, K = %d, delta_t = %.2e, N = %d, lr = %.2e, %s, %s, %s, %s\n' % (self.d, self.L, self.K, self.delta_t_np, self.N, self.lr, self.approx_method, self.time_approx, self.loss_method, 'adaptive' if self.adaptive_forward_process else ''))

        xb = 2.0
        X = pt.linspace(-xb, xb, 200).unsqueeze(1)

        for l in range(self.L):
            t_0 = time.time()
            loss = 0.0
            for n in range(self.N):
                loss += pt.sum((- self.Z_n_(X, n) - pt.tensor(self.u_true(X, n * self.delta_t_np)).float())**2) * self.delta_t
            self.zero_grad()
            loss.backward()
            self.optimization_step()

            self.loss_log.append(loss.item())

            t_1 = time.time()
            self.times.append(t_1 - t_0)

            if l % self.print_every == 0:
                string = ('%d - loss: %.3e - time/iter: %.2fs' % (l, self.loss_log[-1],
                          np.mean(self.times[-self.print_every:])))
                print(string + ' - gradient l_inf: %.3e' %
                      (np.array([max([pt.norm(params.grad.data, float('inf')).item() for params
                                      in filter(lambda params: params.requires_grad,
                                                phi.parameters())])
                                 for phi in self.Phis]).max()))

        #   self.save_networks()

    def train(self):

        pt.manual_seed(self.seed)

        if self.verbose is True:
            print('d = %d, L = %d, K = %d, delta_t = %.2e, lr = %.2e, %s, %s, %s, %s'
                  % (self.d, self.L, self.K, self.delta_t_np, self.lr, self.approx_method,
                     self.time_approx, self.loss_method,
                     'adaptive' if self.adaptive_forward_process else ''))

        for l in range(self.L):
            t_0 = time.time()

            X, Y, Z_sum, u_L2, u_int, u_W_int, double_int, xi = self.initialize_training_data()
            additional_loss = pt.zeros(self.K)


            if self.loss_method == 'reparametrization':
                z_n_copy = deepcopy(self.z_n)

            for n in range(self.N):
                if self.approx_method == 'value_function':
                    if n > 0:
                        additional_loss += (self.Y_n(X, n)[:, 0] - Y).pow(2)
                if self.loss_method == 'log-variance-repa' and l % 2 == 0:
                    z_n_copy = deepcopy(self.z_n)
                    t_X = pt.cat([pt.ones([X.shape[0], 1]).to(self.device) * n * self.delta_t, X], 1)
                    Z = z_n_copy(t_X)
                else:
                    Z = self.Z_n_(X, n)

                c = pt.zeros(self.d, self.K).to(self.device)
                if self.adaptive_forward_process is True:
                    if self.burgers_drift is True:
                        c = pt.ones(self.d, self.K).to(self.device) * (Y.unsqueeze(0) - (2 + self.d) / (2 * self.d))
                    else:
                        c = -self.Z_n_(X, n).t()
                        #c = -Z.t()

                if self.loss_method == 'reparametrization':
                    if self.time_approx == 'outer':
                        n = max(0, min(n, self.N - 1))
                        z_n_copy = deepcopy(self.z_n[n])
                        v = -z_n_copy(X)
                    else:
                        t_X = pt.cat([pt.ones([X.shape[0], 1]).to(self.device) * n * self.delta_t, X], 1)
                        v = -z_n_copy(t_X)

                if self.detach_forward is True or (self.loss_method == 'log-variance-repa' and l % 2 == 1):
                    c = c.detach()

                X = (X + (self.b(X) + pt.mm(self.sigma(X), c).t()) * self.delta_t
                     + pt.mm(self.sigma(X), xi[:, :, n + 1].t()).t() * self.sq_delta_t)

                #X = (X + (self.b(X) + pt.bmm(self.sigma(X), c.t().unsqueeze(2)).squeeze(2)) * self.delta_t 
                #     + pt.bmm(self.sigma(X), xi[:, :, n + 1].unsqueeze(2)).squeeze(2) * self.sq_delta_t)

                Y = (Y + (-self.h(self.delta_t * n, X, Y, Z) + pt.sum(Z * c.t(), 1)) * self.delta_t
                     + pt.sum(Z * xi[:, :, n + 1], 1) * self.sq_delta_t)

                if self.loss_method == 'reparametrization':
                    Z_sum += (-0.5 * pt.sum(v**2, 1) * self.delta_t + pt.sum(v * c.t(), 1) * self.delta_t
                              + pt.sum(v * xi[:, :, n + 1], 1) * self.sq_delta_t)

                if 'relative_entropy' in self.loss_method:
                    #Z_sum += 0.5 * pt.sum((-Z)**2, 1) * self.delta_t
                    Z_sum += (0.5 * pt.sum(Z**2, dim=1) + self.f(X, n * self.delta_t)) * self.delta_t
                    #Z_sum += self.h(n * self.delta_t, X, Y, Z) * self.delta_t
                    if self.loss_method == 'relative_entropy_BSDE':
                        Z_sum += pt.sum(-Z * xi[:, :, n + 1], 1) * self.sq_delta_t

                if self.u_l2_error_flag is True:
                    u_L2 += pt.sum((-Z
                                    - pt.tensor(self.u_true(X.cpu().detach(), n * self.delta_t_np)).t().float().to(self.device))**2
                                   * self.delta_t, 1)

            if self.compute_gradient_variance > 0 and l % self.compute_gradient_variance == 0:
                self.grads_rel_error_log.append(pt.mean(self.get_gradient_variances(X, Y)).item())

            loss = self.gradient_descent(X, Y, Z_sum, l, additional_loss.mean())

            if self.log_gradient is True:
                grads = [params.grad for params in list(filter(lambda params: params.requires_grad, self.z_n.parameters()))
                        if params.grad is not None]
                grads_flat = pt.zeros(self.p)        
                i = 0
                for grad in grads:
                    grad_flat = grad.reshape(-1)
                    j = len(grad_flat)
                    grads_flat[i:i + j] = grad_flat
                    i += j
                
                self.gradient_log[l, :] = grads_flat.cpu().detach()

            self.loss_log.append(loss.item())
            self.u_L2_loss.append(pt.mean(u_L2).item())
            if self.metastability_logs is not None:
                target, epsilon = self.metastability_logs
                self.particles_close_to_target.append(pt.mean((pt.sqrt(pt.sum((X - target)**2, 1)) <
                                                               epsilon).float()))

            if self.IS_variance_K > 0 and l % self.IS_variance_iter == 0:
                _, _, rel_IS = do_importance_sampling_me(self.problem, self, self.IS_variance_K)
                #_, _, rel_naive, _, _, rel_IS = do_importance_sampling(self.problem, self,
                #                                                                 self.IS_variance_K,
                #                                                                 control='approx',
                #                                                                 verbose=False,
                #                                                                 plot_trajectories=self.plot_trajectories)
                self.IS_rel_log.append(rel_IS)

            t_1 = time.time()
            self.times.append(t_1 - t_0)

            if self.verbose is True:
                if l % self.print_every == 0:
                    string = ('%d - loss: %.4e - u L2: %.4e - time/iter: %.2fs'
                              % (l, self.loss_log[-1], self.u_L2_loss[-1],
                                 np.mean(self.times[-self.print_every:])))
                    if self.learn_Y_0 is True:
                            string += ' - Y_0: %.4e' % self.Y_0_log[-1]
                    if self.IS_variance_K > 0:
                        string += ' - rel IS: %.3e' % rel_IS
                    print(string)
                    if self.print_gradient_norm is True:
                        print('gradient l_inf: %.3e' %
                              (np.array([max([pt.norm(params.grad.data, float('inf')).item() for params in
                                              filter(lambda params: hasattr(params.grad, 'data'),
                                                     phi.parameters())])
                                         for phi in self.Phis]).max()))

            if self.early_stopping_time is not None:
                if ((l > self.early_stopping_time) and
                        (np.std(self.u_L2_loss[-self.early_stopping_time:])
                         / self.u_L2_loss[-1] < 0.02)):
                    break

        if self.save_results is True:
            self.save_logs()


class EllipticSolver():
    
    def __init__(self, problem, name, seed=42, delta_t=0.01, N=50, lr=0.001, L=100000, K=200, K_boundary=50,
                 alpha=[1.0, 1.0], adaptive_forward_process=False, detach_forward=True, print_every=100, verbose=True, 
                 approx_method='Y', sample_center=False, loss_method='ito', loss_with_stopped=False, K_test_log=None,
                 PINN_log_variance=False, log_loss_parts=False, boundary_loss=True, boundary_type='Dirichlet'):
        self.problem = problem
        self.name = name
        self.date = date.today().strftime('%Y-%m-%d')
        self.d = problem.d

        # hyperparameters
        self.device = pt.device('cpu')
        self.seed = seed
        self.delta_t_np = delta_t
        self.delta_t = pt.tensor(self.delta_t_np).to(self.device) # step size
        self.sq_delta_t = pt.sqrt(self.delta_t).to(self.device)
        self.N = N
        self.lr = lr # learning rate
        self.L = L # gradient steps
        self.K = K # batch size in domain
        self.K_boundary = K_boundary # batch size on boundary
        self.alpha = alpha
        self.boundary_type = boundary_type

        # learning properties
        self.adaptive_forward_process = adaptive_forward_process
        self.detach_forward = detach_forward
        self.approx_method = approx_method
        self.sample_center = sample_center
        self.loss_method = loss_method
        self.loss_with_stopped = loss_with_stopped
        self.boundary_loss = boundary_loss
        self.PINN_log_variance = PINN_log_variance

        # printing and logging
        self.print_every = print_every
        self.verbose = verbose

        # function approximation
        pt.manual_seed(seed)
        if self.approx_method == 'Y':
            self.V = DenseNet(d_in=self.d, d_out=1, lr=self.lr, seed=seed).to(self.device)
        elif self.approx_method == 'Z':
            self.y_0 = SingleParam(lr=self.lr).to(self.device)
            self.Z = DenseNet(d_in=self.d, d_out=self.d, lr=self.lr, seed=seed).to(self.device)
            
        # logging
        self.K_test_log = K_test_log
        self.Y_0_log = []
        self.loss_log = []
        self.loss_log_domain = []
        self.loss_log_boundary = []
        self.u_L2_log = []
        self.V_L2_log = []
        self.V_test_log = []
        self.times = []
        self.lambda_log = []
        self.log_loss_parts = log_loss_parts
        
    def train(self):
    
        pt.manual_seed(self.seed)

        if self.loss_method == 'PINN':
            self.train_PINN()
            return None
        
        for l in range(self.L):

            t_0 = time.time()

            loss = 0

            if self.sample_center:
                X_center = pt.zeros(1, 1)
                loss += pt.mean((self.V(X_center).squeeze() - self.problem.v_true(X_center).squeeze())**2)

            # sample uniformly on boundary
            if self.problem.boundary == 'sphere':
                X_boundary = pt.rand(self.K_boundary, self.d).to(self.device) * 2 - 1
                X_boundary = self.problem.boundary_distance * X_boundary / pt.sqrt(pt.sum(X_boundary**2, 1)).unsqueeze(1)
            elif self.problem.boundary == 'two_spheres':
                X_boundary = pt.rand(self.K_boundary, self.problem.d).to(self.device) * 2 - 1
                X_boundary = (pt.tensor([self.problem.boundary_distance_1] * int(self.K_boundary / 2) + 
                                        [self.problem.boundary_distance_2] * int(self.K_boundary / 2)).unsqueeze(1).to(self.device)
                              * X_boundary / pt.sqrt(pt.sum(X_boundary**2, 1)).unsqueeze(1))
            elif self.problem.boundary == 'square':
                s = np.concatenate([np.ones(int(self.K_boundary / 2))[:, np.newaxis], np.zeros([int(self.K_boundary / 2), self.d - 1])], 1)
                np.apply_along_axis(np.random.shuffle, 1, s)
                a = np.concatenate([s, np.zeros([int(self.K_boundary / 2), self.problem.d])]).astype(bool)
                b = np.concatenate([np.zeros([int(self.K_boundary / 2), self.problem.d]), s]).astype(bool)
                X_boundary = (self.problem.X_r - self.problem.X_l) * pt.rand(self.K_boundary, self.problem.d).to(self.device) + self.problem.X_l
                X_boundary[pt.tensor(a.astype(float)).byte()] = self.problem.X_l
                X_boundary[pt.tensor(b.astype(float)).byte()] = self.problem.X_r
                if self.problem.one_boundary:
                    X_boundary[pt.tensor(a.astype(float)).byte()] = self.problem.X_r
                    X_boundary[pt.tensor(b.astype(float)).byte()] = self.problem.X_r

            if self.loss_method != 'L4' and self.boundary_loss:
                if self.boundary_type == 'Dirichlet':
                    loss += self.alpha[1] * pt.mean((self.V(X_boundary).squeeze() - self.problem.g(X_boundary))**2)
                elif self.boundary_type == 'Neumann':
                    X_boundary = pt.autograd.Variable(X_boundary, requires_grad=True)
                    Y_ = self.V(X_boundary)
                    Y_eval = Y_.squeeze().sum() # compare to Jacobi-Vector trick
                    Y_eval.backward(retain_graph=True) # do we need this?
                    grad_V, = pt.autograd.grad(Y_eval, X_boundary, create_graph=True) # , allow_unused=True
                    loss += self.alpha[1] * pt.mean((pt.sum(grad_V * X_boundary, 1) - pt.sum(self.problem.g(X_boundary) * X_boundary, 1))**2)

            if self.problem.boundary == 'sphere':
                X = 2 * pt.rand(self.K, self.problem.d).to(self.device) - 1
                X = self.problem.boundary_distance * X / pt.sqrt(pt.sum(X**2, 1)).unsqueeze(1) * (pt.rand(self.K).unsqueeze(1)).to(self.device)
            elif self.problem.boundary == 'two_spheres':
                X = pt.rand(self.K, self.problem.d).to(self.device) * 2 - 1
                X = X / pt.sqrt(pt.sum(X**2, 1)).unsqueeze(1) * (pt.rand(self.K, self.problem.d).to(self.device) * (self.problem.boundary_distance_2 - self.problem.boundary_distance_1) + self.problem.boundary_distance_1)
            elif self.problem.boundary == 'square':
                X = (self.problem.X_r - self.problem.X_l) * pt.rand(self.K, self.problem.d).to(self.device) + self.problem.X_l

            X = pt.autograd.Variable(X, requires_grad=True)
            Y = pt.zeros(self.K).to(self.device)
            if self.loss_method in ['jentzen', 'L4']:
                Y = self.V(X).squeeze()

            #lambda_log.append(lambda_(X)[0].item())
            stopped = pt.zeros(self.K).byte().to(self.device)
            hitting_times = pt.zeros(self.K)
            V_L2 = pt.zeros(self.K)

            phi_0 = self.V(X).squeeze()

            for n in range(self.N):

                Y_ = self.V(X)
                Y_eval = Y_.squeeze().sum() # compare to Jacobi-Vector trick
                Y_eval.backward(retain_graph=True) # do we need this?
                Z, = pt.autograd.grad(Y_eval, X, create_graph=True) # , allow_unused=True
                Z = pt.mm(self.problem.sigma(X).t(), Z.t()).t()

                xi = pt.randn(self.K, self.d).to(self.device)

                selection = ~stopped
                K_selection = pt.sum(selection)
                if K_selection == 0:
                    break

                V_L2[selection] += ((self.V(X[selection]).squeeze() - pt.tensor(self.problem.v_true(X[selection])).float().squeeze())**2).detach().cpu() * self.delta_t_np

                c = pt.zeros(self.d, self.K).to(self.device)
                if self.adaptive_forward_process is True:
                    c = -Z.t()
                if self.detach_forward is True:
                    c = c.detach()

                X_proposal = (X + ((self.problem.b(X) + pt.mm(self.problem.sigma(X), c).t()) * self.delta_t
                     + pt.mm(self.problem.sigma(X), xi.t()).t() * self.sq_delta_t) * selection.float().unsqueeze(1).repeat(1, self.d))

                hitting_times[selection] += 1
                if self.problem.boundary == 'sphere':
                    new_selection = pt.all(pt.sqrt(pt.sum(X**2, 1)).unsqueeze(1) < self.problem.boundary_distance, 1).to(self.device)
                elif self.problem.boundary == 'two_spheres':
                    new_selection = ((pt.sqrt(pt.sum(X**2, 1)) > self.problem.boundary_distance_1) & (pt.sqrt(pt.sum(X**2, 1)) < self.problem.boundary_distance_2)).to(self.device)
                elif self.problem.boundary == 'square':
                    if self.problem.one_boundary:
                        new_selection = pt.all((X_proposal <= self.problem.X_r), 1).to(self.device)
                    else:
                        new_selection = pt.all((X_proposal >= self.problem.X_l) & (X_proposal <= self.problem.X_r), 1).to(self.device)

                if self.loss_method == 'jentzen':
                    loss += self.alpha[0] * pt.mean((Y_.squeeze() - Y)**2 * (new_selection & ~stopped).float())

                if self.loss_method in ['jentzen', 'L4']:
                    Y = (Y + ((- self.problem.h(X, Y, Z) #- lambda_(X) * Y_.squeeze() #  lambda_(X) 
                               + pt.sum(Z * c.t(), 1)) * self.delta_t + pt.sum(Z * xi, 1) * self.sq_delta_t) * (new_selection & ~stopped).float())
                else:
                    Y = (Y + ((- self.problem.h(X, Y_.squeeze(), Z) #- lambda_(X) * Y_.squeeze() #  lambda_(X) 
                               + pt.sum(Z * c.t(), 1)) * self.delta_t + pt.sum(Z * xi, 1) * self.sq_delta_t) * (new_selection & ~stopped).float())

                X_ = X
                X = (X * (~new_selection | stopped).float().unsqueeze(1).repeat(1, self.d) 
                     + X_proposal * (new_selection & ~stopped).float().unsqueeze(1).repeat(1, self.d))

                if pt.sum(~new_selection & ~stopped) > 0:
                    stopped[~new_selection & ~stopped] += True

                if self.loss_method == 'raissi':
                    loss += self.alpha[0] * pt.mean((self.V(X).squeeze() - Y_.squeeze() + (self.problem.h(X_, Y_.squeeze(), Z)
                                                                 - pt.sum(Z * c.t(), 1)) * self.delta_t 
                                     - pt.sum(Z * xi, 1) * self.sq_delta_t)**2 * (new_selection & ~stopped).float())

            if self.loss_method == 'ito':
                loss += self.alpha[0] * pt.mean((self.V(X).squeeze() - phi_0 - Y)**2)
            self.V.zero_grad()
            #lambda_.zero_grad()

            if self.loss_method == 'L4':
                loss += pt.mean((Y - self.problem.g(X))**2)

            if self.loss_with_stopped:
                loss += self.alpha[0] * pt.mean((Y[stopped] - self.problem.g(X[stopped, :]))**2)
            loss.backward(retain_graph=True)
            self.V.optim.step()

            #lambda_.optim.step()

            self.loss_log.append(loss.item())
            self.V_L2_log.append(pt.mean(V_L2).item())
            if self.K_test_log is not None:
                self.V_test_log.append(compute_test_error(self, self.problem, self.K_test_log, self.device))

            t_1 = time.time()
            self.times.append(t_1 - t_0)

            if self.verbose:
                if l % self.print_every == 0:
                    print('%d - loss = %.4e, v L2 error = %.4e, n = %d, active: %d/%d, %.2f' % 
                          (l, self.loss_log[-1], self.V_L2_log[-1], n, K_selection, self.K, np.mean(self.times[-self.print_every:])))

    def train_PINN(self):
        for l in range(self.L):

            t_0 = time.time()

            if self.problem.boundary == 'sphere':
                X_boundary = pt.rand(self.K_boundary, self.problem.d).to(self.device) * 2 - 1
                X_boundary = self.problem.boundary_distance * X_boundary / pt.sqrt(pt.sum(X_boundary**2, 1)).unsqueeze(1)
            elif self.problem.boundary == 'two_spheres':
                X_boundary = pt.rand(self.K_boundary, self.problem.d).to(self.device) * 2 - 1
                X_boundary = (pt.tensor([self.problem.boundary_distance_1] * int(self.K_boundary / 2) + 
                                        [self.problem.boundary_distance_2] * int(self.K_boundary / 2)).unsqueeze(1).to(self.device)
                              * X_boundary / pt.sqrt(pt.sum(X_boundary**2, 1)).unsqueeze(1))
            elif self.problem.boundary == 'square':
                s = np.concatenate([np.ones(int(self.K_boundary / 2))[:, np.newaxis], np.zeros([int(self.K_boundary / 2), self.d - 1])], 1)
                np.apply_along_axis(np.random.shuffle, 1, s)
                a = np.concatenate([s, np.zeros([int(self.K_boundary / 2), self.problem.d])]).astype(bool)
                b = np.concatenate([np.zeros([int(self.K_boundary / 2), self.problem.d]), s]).astype(bool)
                X_boundary = (self.problem.X_r - self.problem.X_l) * pt.rand(self.K_boundary, self.problem.d).to(self.device) + self.problem.X_l
                X_boundary[pt.tensor(a.astype(float)).byte()] = self.problem.X_l
                X_boundary[pt.tensor(b.astype(float)).byte()] = self.problem.X_r
                if self.problem.one_boundary:
                    X_boundary[pt.tensor(a.astype(float)).byte()] = self.problem.X_r
                    X_boundary[pt.tensor(b.astype(float)).byte()] = self.problem.X_r

            if self.problem.boundary == 'sphere':
                X = 2 * pt.rand(self.K, self.problem.d).to(self.device) - 1
                X = self.problem.boundary_distance * X / pt.sqrt(pt.sum(X**2, 1)).unsqueeze(1) * (pt.rand(self.K).unsqueeze(1)).to(self.device)
            elif self.problem.boundary == 'two_spheres':
                X = pt.rand(self.K, self.problem.d).to(self.device) * 2 - 1
                X = X / pt.sqrt(pt.sum(X**2, 1)).unsqueeze(1) * (pt.rand(self.K, self.problem.d).to(self.device) * (self.problem.boundary_distance_2 - self.problem.boundary_distance_1) + self.problem.boundary_distance_1)
            elif self.problem.boundary == 'square':
                X = (self.problem.X_r - self.problem.X_l) * pt.rand(self.K, self.problem.d).to(self.device) + self.problem.X_l

            laplacian = pt.zeros(X.shape[0]).to(self.device)

            X = pt.autograd.Variable(X, requires_grad=True)

            #for i, x in enumerate(X):
            #    #hess = hessian(self.V(x.unsqueeze(0)), x.unsqueeze(0)).squeeze()
            #    hess = hessian(self.V, x.unsqueeze(0), create_graph=True) # pt.autograd.functional.
            #    laplacian[i] = pt.diagonal(hess.view(d, d), offset=0).sum()

            V_eval = self.V(X).squeeze()
            grad = pt.autograd.grad(V_eval, X, grad_outputs=pt.ones(self.K).to(self.device), create_graph=True)[0]

            for k in range(grad.shape[1]):
                rad_grad_u_xx = pt.autograd.grad(grad[:, k], X, grad_outputs=pt.ones(self.K).to(self.device), create_graph=True)[0][:, k]
                laplacian += rad_grad_u_xx

            # works only for diagonal diffusion matrix with same entries
            if self.PINN_log_variance:
                loss = self.alpha[0] * pt.var(self.problem.B[0, 0]**2 * 0.5 * laplacian + pt.sum(self.problem.b(X) * grad, 1) 
                                              + self.problem.h(X, self.V(X).squeeze(), pt.mm(self.problem.B, grad.t()).t()))
            else:
                loss = self.alpha[0] * pt.mean((self.problem.B[0, 0]**2 * 0.5 * laplacian + pt.sum(self.problem.b(X) * grad, 1) 
                                                + self.problem.h(X, self.V(X).squeeze(), pt.mm(self.problem.B, grad.t()).t()))**2) 
            if self.log_loss_parts:
                self.loss_log_domain.append(loss.item() / self.alpha[0])
            if self.boundary_loss:
                loss += self.alpha[1] * pt.mean((self.V(X_boundary).squeeze() - self.problem.g(X_boundary))**2)
            if self.log_loss_parts:
                self.loss_log_boundary.append(pt.mean((self.V(X_boundary).squeeze() - self.problem.g(X_boundary))**2).item())

            self.V.zero_grad()

            loss.backward(retain_graph=True)
            self.V.optim.step()

            self.V_L2_log.append(pt.mean(((self.V(X).squeeze() - pt.tensor(self.problem.v_true(X.detach())).float().squeeze())**2).detach().cpu() * self.delta_t_np).item())
            self.loss_log.append(loss.item())
            if self.K_test_log is not None:
                self.V_test_log.append(compute_test_error(self, self.problem, self.K_test_log, self.device))

            t_1 = time.time()
            self.times.append(t_1 - t_0)

            if l % self.print_every == 0:
                print('%d - loss = %.4e - v L2 error = %.4e - %.2f' % (l, self.loss_log[-1], self.V_L2_log[-1], np.mean(self.times[-self.print_every:])))





class GeneralSolver():
    
    def __init__(self, problem, name, seed=42, delta_t=0.01, N=50, lr=0.001, L=100000, K=200, K_boundary=50,
                 alpha=[1.0, 1.0, 1.0], adaptive_forward_process=False, detach_forward=True, print_every=100, verbose=True, 
                 approx_method='Y', sample_center=False, loss_method='ito', loss_with_stopped=False, K_test_log=None,
                 PINN_log_variance=False, log_loss_parts=False, boundary_loss=True):
        self.problem = problem
        self.name = name
        self.date = date.today().strftime('%Y-%m-%d')
        self.d = problem.d

        # hyperparameters
        self.device = pt.device('cpu')
        self.seed = seed
        self.delta_t_np = delta_t
        self.delta_t = pt.tensor(self.delta_t_np).to(self.device) # step size
        self.sq_delta_t = pt.sqrt(self.delta_t).to(self.device)
        self.N = N
        self.lr = lr # learning rate
        self.L = L # gradient steps
        self.K = K # batch size in domain
        self.K_boundary = K_boundary # batch size on boundary
        self.alpha = alpha

        # learning properties
        self.adaptive_forward_process = adaptive_forward_process
        self.detach_forward = detach_forward
        self.approx_method = approx_method
        self.sample_center = sample_center
        self.loss_method = loss_method
        self.loss_with_stopped = loss_with_stopped
        self.boundary_loss = boundary_loss
        self.PINN_log_variance = PINN_log_variance

        # printing and logging
        self.print_every = print_every
        self.verbose = verbose

        # function approximation
        pt.manual_seed(seed)
        if self.approx_method == 'Y':
            self.V = DenseNet(d_in=self.d + 1, d_out=1, lr=self.lr, seed=seed).to(self.device)
        elif self.approx_method == 'Z':
            self.y_0 = SingleParam(lr=self.lr).to(self.device)
            self.Z = DenseNet(d_in=self.d + 1, d_out=self.d, lr=self.lr, seed=seed).to(self.device)
            
        # logging
        self.K_test_log = K_test_log
        self.Y_0_log = []
        self.loss_log = []
        self.loss_log_domain = []
        self.loss_log_boundary = []
        self.u_L2_log = []
        self.V_L2_log = []
        self.V_test_log = []
        self.times = []
        self.lambda_log = []
        self.log_loss_parts = log_loss_parts
        
    def train(self):
    
        pt.manual_seed(self.seed)

        if self.loss_method == 'PINN':
            self.train_PINN()
            return None
        
        for l in range(self.L):

            t_0 = time.time()

            loss = 0

            if self.sample_center:
                X_center = pt.zeros(1, 1)
                loss += pt.mean((self.V(X_center).squeeze() - self.problem.v_true(X_center).squeeze())**2)

            # sample uniformly on boundary
            if self.problem.boundary == 'sphere':
                X_boundary = pt.rand(self.K_boundary, self.d).to(self.device) * 2 - 1
                X_boundary = self.problem.boundary_distance * X_boundary / pt.sqrt(pt.sum(X_boundary**2, 1)).unsqueeze(1)
            elif self.problem.boundary == 'two_spheres':
                X_boundary = pt.rand(self.K_boundary, self.problem.d).to(self.device) * 2 - 1
                X_boundary = (pt.tensor([self.problem.boundary_distance_1] * int(self.K_boundary / 2) + 
                                        [self.problem.boundary_distance_2] * int(self.K_boundary / 2)).unsqueeze(1).to(self.device)
                              * X_boundary / pt.sqrt(pt.sum(X_boundary**2, 1)).unsqueeze(1))
            elif self.problem.boundary == 'square':
                s = np.concatenate([np.ones(int(self.K_boundary / 2))[:, np.newaxis], np.zeros([int(self.K_boundary / 2), self.d - 1])], 1)
                np.apply_along_axis(np.random.shuffle, 1, s)
                a = np.concatenate([s, np.zeros([int(self.K_boundary / 2), self.problem.d])]).astype(bool)
                b = np.concatenate([np.zeros([int(self.K_boundary / 2), self.problem.d]), s]).astype(bool)
                X_boundary = (self.problem.X_r - self.problem.X_l) * pt.rand(self.K_boundary, self.problem.d).to(self.device) + self.problem.X_l
                X_boundary[pt.tensor(a.astype(float)).byte()] = self.problem.X_l
                X_boundary[pt.tensor(b.astype(float)).byte()] = self.problem.X_r
                if self.problem.one_boundary:
                    X_boundary[pt.tensor(a.astype(float)).byte()] = self.problem.X_r
                    X_boundary[pt.tensor(b.astype(float)).byte()] = self.problem.X_r

            if self.problem.boundary in ['sphere', 'unbounded']:
                X = 2 * pt.rand(self.K, self.problem.d).to(self.device) - 1
                X = self.problem.boundary_distance * X / pt.sqrt(pt.sum(X**2, 1)).unsqueeze(1) * (pt.rand(self.K).unsqueeze(1)).to(self.device)
            elif self.problem.boundary == 'two_spheres':
                X = pt.rand(self.K, self.problem.d).to(self.device) * 2 - 1
                X = X / pt.sqrt(pt.sum(X**2, 1)).unsqueeze(1) * (pt.rand(self.K, self.problem.d).to(self.device) * (self.problem.boundary_distance_2 - self.problem.boundary_distance_1) + self.problem.boundary_distance_1)
            elif self.problem.boundary == 'square':
                X = (self.problem.X_r - self.problem.X_l) * pt.rand(self.K, self.problem.d).to(self.device) + self.problem.X_l

            if self.problem.boundary != 'unbounded':
                t_n_boundary = pt.rand(self.K_boundary, 1) * self.problem.T
                X_t_n_boundary = pt.cat([X_boundary, t_n_boundary], 1)

            if self.loss_method != 'L4' and self.boundary_loss:
                #loss += self.alpha[1] * pt.mean((self.V(X_boundary).squeeze() - self.problem.g(X_boundary))**2)
                X_T = pt.cat([X[:self.K_boundary, :], self.problem.T * pt.ones(self.K_boundary).unsqueeze(1)], 1)
                loss += self.alpha[1] * pt.mean((self.V(X_T).squeeze() - self.problem.f(X[:self.K_boundary, :]))**2)
                if self.problem.boundary != 'unbounded':
                    loss += self.alpha[2] * pt.mean((self.V(X_t_n_boundary).squeeze() - self.problem.g(X_boundary, t_n_boundary.squeeze()))**2)


            X = pt.autograd.Variable(X, requires_grad=True)
            Y = pt.zeros(self.K).to(self.device)
            t_n = pt.rand(self.K, 1) * self.problem.T
            X_t_n = pt.cat([X, t_n], 1)
            if self.loss_method in ['jentzen', 'L4']:
                Y = self.V(X_t_n).squeeze()

            #lambda_log.append(lambda_(X)[0].item())
            stopped = pt.zeros(self.K).byte().to(self.device)
            hitting_times = pt.zeros(self.K)
            V_L2 = pt.zeros(self.K)

            phi_0 = self.V(X_t_n).squeeze()

            for n in range(self.N):

                Y_ = self.V(X_t_n)
                Y_eval = Y_.squeeze().sum() # compare to Jacobi-Vector trick
                Y_eval.backward(retain_graph=True) # do we need this?
                Z, = pt.autograd.grad(Y_eval, X, create_graph=True) # , allow_unused=True
                Z = pt.mm(self.problem.sigma(X).t(), Z.t()).t()

                xi = pt.randn(self.K, self.d).to(self.device)

                selection = ~stopped
                K_selection = pt.sum(selection)
                if K_selection == 0:
                    break

                #V_L2[selection] += ((self.V(X_t_n[selection]).squeeze() - pt.tensor(self.problem.v_true(X[selection], t_n[selection])).float().squeeze())**2).detach().cpu() * self.delta_t_np

                c = pt.zeros(self.d, self.K).to(self.device)
                if self.adaptive_forward_process is True:
                    c = -Z.t()
                if self.detach_forward is True:
                    c = c.detach()

                X_proposal = (X + ((self.problem.b(X) + pt.mm(self.problem.sigma(X), c).t()) * self.delta_t
                     + pt.mm(self.problem.sigma(X), xi.t()).t() * self.sq_delta_t) * selection.float().unsqueeze(1).repeat(1, self.d))

                new_selection = pt.ones(self.K).byte()
                hitting_times[selection] += 1
                if self.problem.boundary == 'sphere':
                    new_selection = pt.all(pt.sqrt(pt.sum(X**2, 1)).unsqueeze(1) < self.problem.boundary_distance, 1).to(self.device)
                elif self.problem.boundary == 'two_spheres':
                    new_selection = ((pt.sqrt(pt.sum(X**2, 1)) > self.problem.boundary_distance_1) & (pt.sqrt(pt.sum(X**2, 1)) < self.problem.boundary_distance_2)).to(self.device)
                elif self.problem.boundary == 'square':
                    if self.problem.one_boundary:
                        new_selection = pt.all((X_proposal <= self.problem.X_r), 1).to(self.device)
                    else:
                        new_selection = pt.all((X_proposal >= self.problem.X_l) & (X_proposal <= self.problem.X_r), 1).to(self.device)

                new_selection = new_selection & ((t_n.squeeze() + self.delta_t) <= self.problem.T)

                if self.loss_method == 'jentzen':
                    loss += self.alpha[0] * pt.mean((Y_.squeeze() - Y)**2 * (new_selection & ~stopped).float())

                if self.loss_method in ['jentzen', 'L4']:
                    Y = (Y + ((- self.problem.h(n * self.delta_t, X, Y, Z) #- lambda_(X) * Y_.squeeze() #  lambda_(X) 
                               + pt.sum(Z * c.t(), 1)) * self.delta_t + pt.sum(Z * xi, 1) * self.sq_delta_t) * (new_selection & ~stopped).float())
                else:
                    Y = (Y + ((- self.problem.h(n * self.delta_t, X, Y_.squeeze(), Z) #- lambda_(X) * Y_.squeeze() #  lambda_(X) 
                               + pt.sum(Z * c.t(), 1)) * self.delta_t + pt.sum(Z * xi, 1) * self.sq_delta_t) * (new_selection & ~stopped).float())

                X_ = X
                X = (X * (~new_selection | stopped).float().unsqueeze(1).repeat(1, self.d) 
                     + X_proposal * (new_selection & ~stopped).float().unsqueeze(1).repeat(1, self.d))

                t_n += self.delta_t * (new_selection & ~stopped).float().unsqueeze(1)
                X_t_n = pt.cat([X, t_n], 1)

                if pt.sum(~new_selection & ~stopped) > 0:
                    stopped[~new_selection & ~stopped] += True

                if self.loss_method == 'raissi':
                    loss += self.alpha[0] * pt.mean((self.V(X).squeeze() - Y_.squeeze() + (self.problem.h(X_, Y_.squeeze(), Z)
                                                                 - pt.sum(Z * c.t(), 1)) * self.delta_t 
                                     - pt.sum(Z * xi, 1) * self.sq_delta_t)**2 * (new_selection & ~stopped).float())

            if self.loss_method == 'ito':
                loss += self.alpha[0] * pt.mean((self.V(X_t_n).squeeze() - phi_0 - Y)**2)
            self.V.zero_grad()
            #lambda_.zero_grad()

            if self.loss_method == 'L4':
                loss += pt.mean((Y - self.problem.g(X))**2)

            if self.loss_with_stopped:
                loss += self.alpha[0] * pt.mean((Y[stopped] - self.problem.g(X[stopped, :]))**2)
            loss.backward(retain_graph=True)
            self.V.optim.step()

            #lambda_.optim.step()

            self.loss_log.append(loss.item())
            self.V_L2_log.append(pt.mean(V_L2).item())
            if self.K_test_log is not None:
                self.V_test_log.append(compute_test_error(self, self.problem, self.K_test_log, self.device, 'parabolic'))

            t_1 = time.time()
            self.times.append(t_1 - t_0)

            if self.verbose:
                if l % self.print_every == 0:
                    print('%d - loss = %.4e, v L2 error = %.4e, n = %d, active: %d/%d, %.2f' % 
                          (l, self.loss_log[-1], self.V_L2_log[-1], n, K_selection, self.K, np.mean(self.times[-self.print_every:])))

