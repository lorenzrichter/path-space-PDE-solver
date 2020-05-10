#pylint: disable=invalid-name, no-member, too-many-arguments, missing-docstring
#pylint: disable=too-many-instance-attributes, not-callable, no-else-return
#pylint: disable=inconsistent-return-statements, too-many-locals, too-many-return-statements
#pylint: disable=too-many-statements, too-many-public-methods

from datetime import date
import json
import numpy as np
import os
import time
import torch as pt

from function_space import DenseNet, Linear, NN, NN_Nik, SingleParam, MySequential
from utilities import do_importance_sampling, do_importance_sampling_me


class Solver():

    def __init__(self, name, problem, lr=0.001, L=10000, K=50, delta_t=0.05,
                 approx_method='control', loss_method='log-variance', time_approx='outer',
                 learn_Y_0=False, adaptive_forward_process=True, detach_forward=False,
                 early_stopping_time=10000, random_X_0=False, compute_gradient_variance=0,
                 IS_variance_K=0, IS_variance_iter=1, metastability_logs=None, print_every=100, plot_trajectories=None,
                 seed=42, save_results=False, u_l2_error_flag=True, log_gradient=False, verbose=True):
        self.problem = problem
        self.name = name
        self.date = date.today().strftime('%Y-%m-%d')
        self.d = problem.d
        self.T = problem.T
        self.X_0 = problem.X_0
        self.Y_0 = pt.tensor([0.0])
        self.X_u_opt = None

        # hyperparameters
        self.device = pt.device('cuda')
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

    def zero_grad(self):
        for phi in self.Phis:
            phi.optim.zero_grad()

    def optimization_step(self):
        for phi in self.Phis:
            phi.optim.step()

    def gradient_descent(self, X, Y, Z_sum, l, additional_loss):
        self.zero_grad()
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
            self.gradient_descent()

            self.loss_log.append(loss.item())

            t_1 = time.time()
            self.times.append(t_1 - t_0)

            if l % self.print_every == 0:
                string = ('%d - loss: %.3e - time/iter: %.2fs' % (l, self.loss_log[-1],
                          np.mean(self.times[-self.print_every:])))
                print(string)
                print('gradient l_inf: %.3e\n' %
                      (np.array([max([pt.norm(params.grad.data, float('inf')).item() for params
                                      in filter(lambda params: params.requires_grad,
                                                phi.parameters())])
                                 for phi in self.Phis]).max()))

        self.save_networks()

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

            for n in range(self.N):
                if self.approx_method == 'value_function':
                    if n > 0:
                        additional_loss += (self.Y_n(X, n)[:, 0] - Y).pow(2)
                Z = self.Z_n_(X, n)

                c = pt.zeros(self.d, self.K).to(self.device)
                if self.adaptive_forward_process is True:
                    c = -Z.t()
                if self.detach_forward is True:
                    c = c.detach()
                X = (X + (self.b(X) + pt.mm(self.sigma(X), c).t()) * self.delta_t
                     + pt.mm(self.sigma(X), xi[:, :, n + 1].t()).t() * self.sq_delta_t)
                Y = (Y + (self.h(self.delta_t * n, X, Y, Z) + pt.sum(Z * c.t(), 1)) * self.delta_t
                     + pt.sum(Z * xi[:, :, n + 1], 1) * self.sq_delta_t)

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
