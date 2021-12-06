#pylint: disable=invalid-name, no-member, too-many-arguments, missing-docstring
#pylint: disable=too-many-branches


from datetime import date
import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch as pt


COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown',
          'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']


def plot_loss_logs(experiment_name, models):
    variance_log = False
    if sum([sum(model.IS_rel_log) == 0 for model in models]) == 0:
        variance_log = True
        fig, ax = plt.subplots(1, 3, figsize=(15, 3))
    else:
        fig, ax = plt.subplots(1, 3, figsize=(15, 3))
    fig.suptitle('%s, d = %d' % (experiment_name, models[0].d))

    for model in models:
        if 'entropy' in model.loss_method:
            ax[0].plot(np.array(model.loss_log) - np.min(np.array(model.loss_log)),
                       label=model.name)
            ax[0].set_yscale('log')
        else:
            ax[0].plot(model.loss_log, label=model.name)
            ax[0].set_yscale('log')
        ax[1].plot(model.u_L2_loss, label=model.name)
        if variance_log is True:
            ax[2].plot(model.IS_rel_log)
            ax[2].set_yscale('log')
            ax[2].set_title('IS relative error')
    ax[1].set_yscale('log')
    ax[0].legend()
    ax[0].set_title('loss')
    ax[1].set_title(r'$\mathbb{E}\left[\|u - u^* \|^2_{L_2}\right]$')

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    return fig


def plot_loss_logs_extended(experiment_name, models):
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.5))

    fig.suptitle(r'%s, $d = %d$' % (experiment_name, models[0].problem.d))

    for model in models:
        ax[0].plot(model.loss_log, label=model.name)
        ax[0].set_yscale('log');
        ax[0].legend()
        ax[1].plot(model.V_L2_log)
        ax[1].set_xlabel('iterations')
        ax[1].set_yscale('log');
        #if model.loss_method == 'PINN':
        #    ax[2].plot(np.linspace(1, len(model.V_L2_log) * model.K, len(model.V_L2_log)), model.V_L2_log)
        #else:
        #    ax[2].plot(np.linspace(1, len(model.V_L2_log) * model.N * model.K, len(model.V_L2_log)), model.V_L2_log)
        #ax[2].set_yscale('log');
        #ax[2].set_xlabel('samples')
        ax[2].plot(model.V_test_rel_abs)
        ax[2].set_xlabel('iterations')
        ax[2].set_yscale('log');

        ax[3].plot(model.V_test_L2)
        ax[3].set_xlabel('iterations')
        ax[3].set_yscale('log');
    ax[0].set_title('loss')
    ax[1].set_title(r'$L^2$ error $V$');
    ax[2].set_title(r'relative absolute test error')
    ax[3].set_title(r'$L^2$ test error')

    fig.tight_layout(rect=[0, 0.03, 1, 0.93])

    return fig


def plot_moving_average(experiment_name, models, moving_span=400):
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.5))

    ax[0].set_title('test error')
    for model in models:
        ax[0].plot(model.V_test_L2, label=model.name)
    ax[0].set_yscale('log')
    ax[0].legend()

    ax[1].set_title('moving average relative absolute test error')
    for model in models:
        ax[1].plot([np.mean(model.V_test_rel_abs[i:i + moving_span]) for i in range(len(model.V_test_rel_abs) - moving_span)], label=model.name)
    ax[1].set_yscale('log')
    ax[1].legend()

    ax[2].set_title(r'moving average $L^2$ test error')
    for model in models:
        ax[2].plot([np.mean(model.V_test_L2[i:i + moving_span]) for i in range(len(model.V_test_L2) - moving_span)], label=model.name)
    ax[2].set_yscale('log')
    ax[2].legend()

    return fig


def plot_solution(model, x, t, components, ylims=None):
    if len(components) > 10:
        print('You can display at most 10 components.')
        return None
    n = int(np.ceil(t / model.delta_t_np))
    t_range = np.linspace(0, model.problem.T, model.N)
    x_val = pt.linspace(-3, 3, 100)

    if model.approx_method == 'control':
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    elif model.approx_method == 'value_function':
        fig, ax = plt.subplots(1, 4, figsize=(15, 4))

    fig.suptitle(model.name)

    X = pt.autograd.Variable(x_val.unsqueeze(1).repeat(1, model.d), requires_grad=True).to(model.device)

    ax[0].set_title('control, t = %.2f' % t)
    for j in components:
        if model.u_true(x_val.unsqueeze(1).repeat(1, model.d), t) is not None:
            ax[0].plot(x_val.numpy(), model.u_true(x_val.unsqueeze(1).repeat(1, model.d), t)[j, :],
                       label=r'true $x_%d$' % (j + 1), color=COLORS[j])
        ax[0].plot(x_val.numpy(), -model.Z_n(X, t).cpu().detach().numpy()[:, j], '--',

                   label=r'approx $x_%d$' % (j + 1), color=COLORS[j])
    if ylims is not None:
        ax[0].set_ylim(ylims[0][0], ylims[0][1])
    ax[0].legend()

    X = pt.autograd.Variable(pt.tensor([[x] * model.d]), requires_grad=True).to(model.device)

    ax[1].set_title('control, x = %.2f' % x)
    for j in components:
        if model.u_true(X.cpu().detach(), n * model.delta_t_np) is not None:
            ax[1].plot(t_range, [model.u_true(X.cpu().detach(), n * model.delta_t_np)[j].item() for n in
                                 range(model.N)], label=r'true $x_%d$' % (j + 1), color=COLORS[j])
        ax[1].plot(t_range, [-model.Z_n(X, t)[0, j].item() for t in t_range], '--',
                   label=r'approx $x_%d$' % (j + 1), color=COLORS[j])
    if ylims is not None:
        ax[1].set_ylim(ylims[1][0], ylims[1][1])

    if model.approx_method == 'value_function':

        X = pt.autograd.Variable(x_val.unsqueeze(1).repeat(1, model.d), requires_grad=True).to(model.device)

        ax[2].set_title('value function, t = %.2f' % t)
        if model.v_true(x_val.unsqueeze(1).repeat(1, model.d), t) is not None:
            ax[2].plot(x_val.numpy(), model.v_true(x_val.unsqueeze(1).repeat(1, model.d), t))
        ax[2].plot(x_val.numpy(), model.Y_n(X, n)[:, 0].detach().numpy(), '--')
        if ylims is not None:
            ax[2].set_ylim(ylims[2][0], ylims[2][1])

        X = pt.autograd.Variable(pt.tensor([[x] * model.d]), requires_grad=True).to(model.device)

        ax[3].set_title('value function, x = %.2f' % x)
        if model.v_true(X.detach(), n * model.delta_t_np) is not None:
            ax[3].plot(t_range, [model.v_true(X.detach(), n * model.delta_t_np).item()
                                 for n in range(model.N)])
        ax[3].plot(t_range, [model.Y_n(X, t)[0, 0].detach().numpy() for t in t_range], '--')
        if ylims is not None:
            ax[3].set_ylim(ylims[3][0], ylims[3][1])

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    return fig


def plot_control_for_DoubleWell1d(model, fig_file_name_prefix, plot_ref_control_flag):
    xb = 2.0
    X = pt.linspace(-xb, xb, 200).unsqueeze(1).to(model.device)

    if plot_ref_control_flag is True:
        fig_1, ax_1 = plt.subplots(1, 2, figsize=(15, 6))
    else:
        fig_1, ax_1 = plt.subplots(1, 1, figsize=(15, 6))

    Z = np.array([-model.Z_n(X, n*model.delta_t).cpu().detach().numpy().squeeze() for n in range(model.N)])
    u_min = -1.0
    u_max = 4.0
    im = ax_1[0].imshow(Z, cmap=cm.jet, extent=[-xb, xb, 0, model.T], vmin=u_min, vmax=u_max,
                        origin='lower', interpolation='none')

    if plot_ref_control_flag is True:
        Z = np.array([model.u_true(X.cpu(), n * model.delta_t_np).squeeze() for n in range(model.N)])
        im = ax_1[1].imshow(Z, cmap=cm.jet, extent=[-xb, xb, 0, model.T], vmin=u_min, vmax=u_max,
                            origin='lower', interpolation='none' )

    #cax = fig_1.add_axes([0.08, 0.04, .84, 0.04])
    #fig_1.colorbar(im, cax=cax, orientation='horizontal', cmap=cm.jet)
    #cax.tick_params(labelsize=10)

    fig_file_name = '%s-2d.eps' % fig_file_name_prefix
    print('\n2d control u has been stored to file: %s' % fig_file_name)
    plt.savefig(fig_file_name)

    lc = ['r', 'k', 'b', 'g', 'c', 'y']

    fig_2, ax_2 = plt.subplots(1, 1, figsize=(10, 6))

    t_vec = [0.0, 0.2, 0.5, 0.7, 0.97]
    nt = len(t_vec)

    for idx in range(nt):
        Zt = -model.Z_n(X, t_vec[idx]).cpu().detach().numpy().squeeze()
        Zt_ref = model.u_true(X.cpu(), t_vec[idx]).squeeze()
        ax_2.plot(X.cpu(), Zt, '-', color=lc[idx], label=r'$t=%.2f' % t_vec[idx])
        ax_2.plot(X.cpu(), Zt_ref, '--', color=lc[idx], label=r'$t=%.2f' % t_vec[idx])

    ax_2.set_ylim(u_min, u_max)
    ax_2.legend()
    ax_2.set_title(r'Control $u$')

    fig_file_name = '%s-1d.eps' % fig_file_name_prefix
    print('\n1d control u has been stored to file: %s' % fig_file_name)
    plt.savefig(fig_file_name)

    return fig_1, fig_2


def do_importance_sampling(problem, model, K, control='approx', verbose=True, delta_t=0.01, plot_trajectories=None, plot_dim=0):

    if plot_trajectories is not None and model.X_u_opt is None:
        do_importance_sampling(problem, model, K, control='true', verbose=False, delta_t=0.01, plot_trajectories=None)

    sq_delta_t = np.sqrt(delta_t)
    N = int(np.ceil(problem.T / delta_t))

    X = pt.zeros([K, N + 1, problem.d]).to(model.device)
    X_u = pt.zeros([K, N + 1, problem.d]).to(model.device)
    X[:, 0, :] = problem.X_0.repeat(K, 1).to(model.device)
    X_u[:, 0, :] = problem.X_0.repeat(K, 1).to(model.device)
    ito_int = pt.zeros(K).to(model.device)
    riemann_int = pt.zeros(K).to(model.device)
    f_int = pt.zeros(K).to(model.device)
    f_int_u = pt.zeros(K).to(model.device)

    for n in range(N):
        xi = pt.randn(K, problem.d).to(model.device)

        X[:, n + 1, :] = (X[:, n, :] + problem.b(X[:, n, :]) * delta_t
                          + pt.mm(problem.sigma(X[:, n, :]), xi.t()).t() * sq_delta_t)
        if control == 'approx' or model.u_l2_error_flag is False:
            ut = -model.Z_n(X_u[:, n, :], n * delta_t)
        elif control == 'true':
            ut = pt.tensor(problem.u_true(X_u[:, n, :].cpu(), n * delta_t)).t().float().to(model.device)
        X_u[:, n + 1, :] = (X_u[:, n, :] + (problem.b(X_u[:, n, :]) + pt.mm(problem.sigma(X_u[:, n, :]), ut.t()).t()) * delta_t
                            + pt.mm(problem.sigma(X_u[:, n, :]), xi.t()).t() * sq_delta_t)
        ito_int += pt.sum(ut * xi, 1) * sq_delta_t
        riemann_int += pt.sum(ut**2, 1) * delta_t
        f_int += model.f(X[:, n, :], n * delta_t) * delta_t
        f_int_u += model.f(X_u[:, n, :], n * delta_t) * delta_t

    if control == 'true' and  model.X_u_opt is None:
        model.X_u_opt = X_u

    girsanov = pt.exp(- ito_int - 0.5 * riemann_int)

    mean_naive = pt.mean(pt.exp(- f_int - problem.g(X[:, N, :]))).item()
    variance_naive = pt.var(pt.exp(- f_int - problem.g(X[:, N, :]))).item()
    rel_error_naive = np.sqrt(variance_naive) / mean_naive
    mean_IS = pt.mean(pt.exp(- f_int_u - problem.g(X_u[:, N, :])) * girsanov).item()
    variance_IS = pt.var(pt.exp(- f_int_u - problem.g(X_u[:, N, :])) * girsanov).item()
    rel_error_IS = np.sqrt(variance_IS) / mean_IS

    if verbose is True:
        print('naive mean: %.4e, naive variance: %.4e, naive RE %.4e' % (mean_naive, variance_naive, rel_error_naive))
        print('IS mean: %.4e, IS variance: %.4e, IS RE %.4e' % (mean_IS, variance_IS, rel_error_IS))

    if plot_trajectories is not None:
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].plot(np.linspace(0, problem.T, N + 1), X_u[:plot_trajectories, :, plot_dim].detach().cpu().numpy().T)
        ax[1].hist(model.X_u_opt[:, N, plot_dim].detach().cpu().numpy(), bins=50, density=True)
        ax[1].hist(X_u[:, N, plot_dim].detach().cpu().numpy(), bins=50, alpha=0.4, density=True)
        plt.show()

    return mean_naive, variance_naive, rel_error_naive, mean_IS, variance_IS, rel_error_IS


def do_importance_sampling_me(problem, model, K, control='approx', simulate_naive=False, verbose=False, delta_t=0.01,
                              on_cpu=False, cross_statistics=None):
    '''
    Memory efficient importance sampling, do not keep whole trajectories.
    '''
    if on_cpu:
        device = pt.device('cuda')
        model.z_n = model.z_n.to(device)
    else:
        device = model.device

    sq_delta_t = np.sqrt(delta_t)
    N = int(np.ceil(problem.T / delta_t))

    if simulate_naive:
        X = problem.X_0.repeat(K, 1).to(device)
    X_u = problem.X_0.repeat(K, 1).to(device)
    ito_int = pt.zeros(K).to(device)
    riemann_int = pt.zeros(K).to(device)
    f_int = pt.zeros(K).to(model.device)
    f_int_u = pt.zeros(K).to(model.device)

    for n in range(N):
        xi = pt.randn(K, problem.d).to(device)

        if simulate_naive:
            X = (X + problem.b(X) * delta_t + pt.mm(problem.sigma(X), xi.t()).t() * sq_delta_t)
            f_int += model.f(X, n * delta_t) * delta_t
        if control == 'approx' or model.u_l2_error_flag is False:
            if on_cpu:
                t_X_u = pt.cat([pt.ones([X_u.shape[0], 1]).to(device) * n * delta_t, X_u], 1)
                ut = -model.z_n(t_X_u)
            else:
                ut = -model.Z_n(X_u, n * delta_t)
        elif control == 'true':
            ut = pt.tensor(problem.u_true(X_u.cpu(), n * delta_t)).t().float().to(device)
        X_u = (X_u + (problem.b(X_u) + pt.mm(problem.sigma(X_u), ut.t()).t()) * delta_t
                            + pt.mm(problem.sigma(X_u), xi.t()).t() * sq_delta_t)
        ito_int += pt.sum(ut * xi, 1) * sq_delta_t
        riemann_int += pt.sum(ut**2, 1) * delta_t
        f_int_u += model.f(X_u, n * delta_t) * delta_t

    girsanov = pt.exp(- ito_int - 0.5 * riemann_int)

    if simulate_naive:
        mean_naive = pt.mean(pt.exp(- f_int - problem.g(X))).item()
        variance_naive = pt.var(pt.exp(- f_int - problem.g(X))).item()
        rel_error_naive = np.sqrt(variance_naive) / mean_naive
    mean_IS = pt.mean(pt.exp(- f_int_u - problem.g(X_u)) * girsanov).item()
    variance_IS = pt.var(pt.exp(- f_int_u - problem.g(X_u)) * girsanov).item()
    rel_error_IS = np.sqrt(variance_IS) / mean_IS

    if verbose is True:
        string = ''
        if simulate_naive:
            string += 'naive mean: %.4e, naive variance: %.4e, naive RE %.4e' % (mean_naive, variance_naive, rel_error_naive)
            if cross_statistics is not None:
                crossed = pt.sum(X > cross_statistics)
                string += ', crossed: %d/%d' % (crossed, X.shape[0])
            string += '\n'
        string += 'IS mean: %.4e, IS variance: %.4e, IS RE %.4e' % (mean_IS, variance_IS, rel_error_IS)
        if cross_statistics is not None:
            crossed_u = pt.sum(X_u > cross_statistics)
            string += ', crossed: %d/%d' % (crossed_u, X_u.shape[0])

        print(string)

    if on_cpu:
        model.z_n.to(model.device)

    if simulate_naive:
        return mean_naive, variance_naive, rel_error_naive, mean_IS, variance_IS, rel_error_IS
    return mean_IS, variance_IS, rel_error_IS


def do_importance_sampling_Wei(problem, model, K, control='approx', verbose=True, delta_t=0.01):

    with pt.no_grad():

        X = problem.X_0.repeat(K, 1).to(model.device)
        X_u = problem.X_0.repeat(K, 1).to(model.device)
        ito_int = pt.zeros(K).to(model.device)
        riemann_int = pt.zeros(K).to(vdevice)

        sq_delta_t = pt.sqrt(pt.tensor(delta_t)).to(model.device)
        N = int(np.ceil(problem.T / delta_t))

        for n in range(N):
            xi = pt.randn(K, problem.d).to(model.device)
            X = X + problem.b(X) * delta_t + pt.bmm(problem.sigma(X), xi.unsqueeze(2)).squeeze(2) * sq_delta_t
            if control == 'approx':
                ut = -model.Z_n(X_u, n * delta_t)
            if control == 'true':
                ut = pt.tensor(problem.u_true(X_u.cpu(), n * delta_t)).float().to(model.device)
            X_u = X_u + (problem.b(X_u) + pt.bmm(problem.sigma(X_u), ut.unsqueeze(2)).squeeze(2)) * delta_t + pt.bmm(problem.sigma(X_u), xi.unsqueeze(2)).squeeze(2) * sq_delta_t
            ito_int += pt.sum(ut * xi, 1) * sq_delta_t
            riemann_int += pt.sum(ut**2, 1) * delta_t

        girsanov = pt.exp(- ito_int - 0.5 * riemann_int)

        mean_naive = pt.mean(pt.exp(-problem.g(X))).item()
        variance_naive = pt.var(pt.exp(-problem.g(X))).item()
        mean_IS = pt.mean(pt.exp(-problem.g(X_u)) * girsanov).item()
        variance_IS = pt.var(pt.exp(-problem.g(X_u)) * girsanov).item()

        if verbose is True:
            print('\n(mean, variance) of naive estimator: (%.4e, %.4e)' % (mean_naive, variance_naive))
            print('(mean, variance) of importance sampling estimator: (%.4e, %.4e)' % (mean_IS, variance_IS))

        return variance_naive, variance_IS


# Currently, this function only works for 1D double well potential
def plot_path_ensemble(problem, model, K, fig_file_name_prefix, control='zero', delta_t=0.01, how_often=10):
    X_u = problem.X_0.repeat(K, 1)
    sq_delta_t = np.sqrt(delta_t)
    N = int(np.ceil(problem.T / delta_t))
    ut = pt.zeros(K, 1)
    N_output = N // how_often
    path_vec = pt.zeros(K, N_output)
    tvec = pt.zeros(N_output)

    for n in range(N):
        if n % how_often == 0:
            idx = n // how_often
            path_vec[:, idx] = X_u.squeeze(1)
            tvec[idx] = n * delta_t
        xi = pt.randn(K, problem.d)
        if control == 'approx':
            ut = -model.Z_n(X_u, n * delta_t)
        if control == 'true':
            ut = pt.tensor(problem.u_true(X_u, n * delta_t).float())
        X_u = X_u + (problem.b(X_u) + pt.bmm(problem.sigma(X_u), ut.unsqueeze(2)).squeeze(2)) * delta_t + pt.bmm(problem.sigma(X_u), xi.unsqueeze(2)).squeeze(2) * sq_delta_t

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    cross_barrier_num = 0
    for idx in range(K):
        if max(path_vec[idx, :]) > 0.0:
            cross_barrier_num += 1
        ax.plot(tvec, path_vec[idx, :].detach().numpy().squeeze(), '-', color='k')

    print ("\nAmong %d paths, %d paths have crossed the barrier (ratio=%.2f)." % (K, cross_barrier_num, cross_barrier_num * 1.0 / K))

    ax.set_ylim(-2, 2)
    ax.set_title('path ensemble')

    fig_file_name = '%s_%s.eps' % (fig_file_name_prefix, control)
    print('\n1d control u has been stored to file: %s' % fig_file_name)

    plt.savefig(fig_file_name)


def compute_test_error(model, problem, K, device=pt.device('cuda'), modus='elliptic'):
    if problem.boundary in ['sphere', 'unbounded']:
        #X = pt.rand(K, problem.d).to(device) * 2 - 1
        #X = problem.boundary_distance * X / pt.sqrt(pt.sum(X**2, 1)).unsqueeze(1) * (pt.rand(K).unsqueeze(1)).to(device)
        X = pt.randn(K, problem.d).to(device)
        X = problem.boundary_distance * X / pt.sqrt(pt.sum(X**2, 1)).unsqueeze(1) * (pt.rand(K).unsqueeze(1)**(1 / problem.d)).to(device)
    elif problem.boundary == 'two_spheres':
        X = pt.randn(K, problem.d).to(device)
        X = problem.boundary_distance_2 * X / pt.sqrt(pt.sum(X**2, 1)).unsqueeze(1) * (pt.rand(K).unsqueeze(1)**(1 / problem.d)).to(device)
        selection = pt.sqrt(pt.sum(X**2, 1)) > problem.boundary_distance_1
        X = X[selection, :]
        #X = pt.rand(self.K, self.problem.d).to(self.device) * 2 - 1
        #X = X / pt.sqrt(pt.sum(X**2, 1)).unsqueeze(1) * (pt.rand(self.K, self.problem.d).to(self.device) * (self.problem.boundary_distance_2 - self.problem.boundary_distance_1) + self.problem.boundary_distance_1)
    elif problem.boundary in ['square', 'unbounded_square']:
        X = (problem.X_r - problem.X_l) * pt.rand(K, problem.d).to(device) + problem.X_l

    if modus == 'parabolic':
        t_n = pt.rand(K, 1).to(device) * problem.T
        X_t_n = pt.cat([X, t_n], 1)        
        v_true = np.array(problem.v_true(X.detach().cpu(), t_n.cpu().squeeze()).squeeze())
        v_est = model.V(X_t_n).squeeze().detach().cpu().numpy()
        L2_error = np.mean((v_true - v_est)**2)
        relative_L2_error = np.mean(((v_true - v_est) / (1 + np.abs(v_true)))**2)
        mean_abolute_error = np.mean(np.abs(v_true - v_est))
        mean_relative_error = np.mean(np.abs(v_true - v_est) / v_true)
    else:
        v_true = np.array(problem.v_true(X.detach().cpu()).squeeze())
        v_est = model.V(X).squeeze().detach().cpu().numpy()
        L2_error = np.mean((v_true - v_est)**2)
        relative_L2_error = np.mean(((v_true - v_est) / (1 + np.abs(v_true)))**2)
        mean_abolute_error = np.mean(np.abs(v_true - v_est))
        mean_relative_error = np.mean(np.abs(v_true - v_est) / v_true)
    return L2_error, mean_abolute_error, mean_relative_error # relative_L2_error


def save_exp_logs(models, name):
    exp_log = {}
    for model in models:
        exp_log[model.name] = {}
        exp_log[model.name]['loss'] = model.loss_log
        exp_log[model.name]['u_L2_loss'] = model.u_L2_loss
        exp_log[model.name]['IS_rel_log'] = model.IS_rel_log
    filename = '%s_%s.json' % (name, date.today().strftime('%Y-%m-%d'))
    with open('logs/%s' % filename, 'w') as f:
        json.dump(exp_log, f)


def load_exp_logs(filename):
    with open('logs/%s' % filename, 'r') as f:
        exp_log = json.load(f)
    return exp_log
