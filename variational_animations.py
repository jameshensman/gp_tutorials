import sys
import numpy as np
import matplotlib.pyplot as plt
# plt.style.use('ops.mplstyle')
import gpflow
import tensorflow as tf
import matplotlib.animation as animation

import matplotlib
matplotlib.rcParams['figure.figsize'] = (10, 6)
matplotlib.rcParams['lines.linewidth'] = 3

M = 20
x = np.random.rand(30,1) * 10
y = np.sin(x) + np.cos(3 * x) + np.random.randn(x.size, 1) * 0.1
Z = np.linspace(2,8,M)[:, None]
m = gpflow.models.SVGP(inducing_variable=Z, kernel=gpflow.kernels.Matern52(1, lengthscale=3.), likelihood=gpflow.likelihoods.Gaussian())
m.likelihood.variance.assign(0.01)
m.q_sqrt.assign((np.eye(M) * .2).reshape(1, M, M))
m.kernel.lengthscale.trainable = False
m.kernel.variance.trainable = False
m.likelihood.variance.trainable = False
num_samples = 30
SAVE=True
# WRITER = animation.ImageMagickWriter(fps=25)
WRITER = animation.FFMpegWriter(fps=25)

#colors
new_black = '#1C1B42'
line_color= '#19B8FF'
data_color = '#5E5095'


def interpolate(x_start, x_end, i, num_iter, loop=False):
    fac = 2 * np.pi if loop else np.pi
    alpha = np.cos(fac * float(i) / float(num_iter-1)) / 2 + .5
    return x_start * alpha + (1-alpha) * x_end


def animate_mean():
    np.random.seed(12)
    q_mu_start = np.random.randn(M,1) * 3 
    q_mu_end = np.random.randn(M,1)* 0.1
    fig, ax = plt.subplots(1, 1)
    x_test = np.linspace(0, 10, 200)[:, None]
    mean, var = m.predict_f(x_test, full_cov=True)
    mean = mean.numpy()
    var = var.numpy().squeeze()
    diag_var = np.diag(var)
    L = np.linalg.cholesky(var)
    samples = np.dot(L, np.random.randn(200, num_samples))
    lower, upper = -2 * np.sqrt(diag_var), 2 * np.sqrt(diag_var)

    samples_lines = ax.plot(x_test, samples, "C0", lw=1)
    ip_line, = ax.plot(m.inducing_variable.Z.numpy(), m.predict_f(m.inducing_variable.Z.numpy())[0], "C1o", ms=3)

    ax.set_xlim(0, 10)
    ax.set_ylim(-2.6, 2.6)

    def init():
        return samples_lines + [ip_line]

    def animate(i):
        q_mu = interpolate(q_mu_start, q_mu_end, i, 100)
        m.q_mu.assign(q_mu)
        mean, _ = m.predict_f(x_test)
        mean = mean.numpy().squeeze()
        [l.set_data(x_test.flatten(), s + mean) for l, s in zip(samples_lines, samples.T)]
        mu_Z = m.predict_f(m.inducing_variable.Z.numpy())[0].numpy().squeeze()
        ip_line.set_data(Z.flatten(), mu_Z)
        return samples_lines + [ip_line]

    ani = animation.FuncAnimation(
        fig, animate, init_func=init, interval=40, blit=True, save_count=200)
    if SAVE:
        ani.save('mean.mp4', writer=WRITER)
    return ani

def animate_variance():
    np.random.seed(12)
    m.q_mu.assign(np.random.randn(M, 1))
    q_sqrt_start = np.random.randn(1, M*(M+1)//2) * 0.01
    q_sqrt_end = np.random.randn(1, M*(M+1)//2) * 0.4
    fig, ax = plt.subplots(1, 1)
    x_test = np.linspace(0, 10, 200)[:, None]
    mean, _ = m.predict_f(x_test)
    mean = mean.numpy()
    white_samples = np.random.randn(200, num_samples)
    samples_lines = ax.plot(x_test, white_samples, "C0", lw=1)

    ax.set_xlim(0, 10)
    ax.set_ylim(-2.6, 2.6)

    def init():
        return samples_lines

    def animate(i):
        q_sqrt = interpolate(q_sqrt_start, q_sqrt_end, i, 100)
        m.q_sqrt.assign(q_sqrt)
        _, var = m.predict_f(x_test, full_cov=True)
        var = var.numpy().squeeze()
        L = np.linalg.cholesky(var)
        samples = np.dot(L, white_samples) + mean
        [l.set_data(x_test.flatten(), s) for l, s in zip(samples_lines, samples.T)]
        return samples_lines

    ani = animation.FuncAnimation(
        fig, animate, init_func=init, interval=40, blit=True, save_count=200)
    if SAVE:
        ani.save('L.mp4', writer=WRITER)

def animate_Z():
    np.random.seed(3)
    m.q_mu = np.random.randn(M, 1)
    m.q_sqrt = np.random.randn(1, M, M) * 0.1
    Z_start = np.sort(np.random.rand(M, 1) * 2 + 4, axis=0)
    Z_end = np.sort(np.random.rand(M, 1) * 10, axis=0)
    fig, ax = plt.subplots(1, 1)
    x_test = np.linspace(0, 10, 200)[:, None]
    white_samples = np.random.randn(200, num_samples)

    samples_lines = ax.plot(x_test, white_samples, "C0", lw=1)
    Z_line, = ax.plot(Z_start, Z_start * 0 - 2.55, 'C1|', mew=4, ms=8)

    ax.set_xlim(0, 10)
    ax.set_ylim(-2.6, 2.6)

    def init():
        return samples_lines + [Z_line]

    def animate(i):
        Z = interpolate(Z_start, Z_end, i, 100)
        mu, var = m.predict_f_full_cov(x_test, feed_dict={m.feature.Z.parameter_tensor:Z})
        L = np.linalg.cholesky(var.squeeze())
        samples = np.dot(L, white_samples) + mu
        [l.set_data(x_test.flatten(), s) for l, s in zip(samples_lines, samples.T)]
        Z_line.set_xdata(Z)
        return samples_lines + [Z_line]

    ani = animation.FuncAnimation(
        fig, animate, init_func=init, interval=40, blit=True, save_count=200)
    if SAVE:
        ani.save('Z.mp4', writer=WRITER)


def animate_fit(plot_Z=False):
    m.kernel.lengthscale.assign(1.0)
    np.random.seed(0)
    m.q_mu.assign(np.random.randn(M, 1))
    m.q_sqrt.assign(np.random.randn(1, M, M) * 0.1)
    x_test = np.linspace(0, 10, 200)[:, None]
    white_samples = np.random.randn(200, num_samples)

    fig, ax = plt.subplots(1, 1)
    ax.spines['bottom'].set_color(new_black)
    ax.spines['left'].set_color(new_black)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    samples_lines = ax.plot(x_test, white_samples, line_color, lw=1)
    data_line, = ax.plot(x, y, 'x', color=data_color, mew=3, ms=6)
    data_line.set_zorder(1e6)
    text = ax.text(0.99, 0.99, '', horizontalalignment='right', verticalalignment='top', transform=ax.transAxes, fontsize=18, color=new_black)
    Z_line, = ax.plot(np.zeros(M), np.zeros(M)-2.6, marker='|', color=new_black, mew=4, alpha=1.0*plot_Z)

    ax.set_xlim(0, 10)
    ax.set_ylim(-2.6, 2.6)

    opt = tf.optimizers.Adam(0.1)

    def init():
        return samples_lines + [data_line, text, Z_line]

    def animate(i):
        opt.minimize(lambda : - m.elbo(x, y), m.trainable_variables)
        mean, var = m.predict_f(x_test, full_cov=True)
        L = np.linalg.cholesky(var.numpy().squeeze())
        samples = np.dot(L, white_samples) + mean.numpy()
        [l.set_data(x_test.flatten(), s) for l, s in zip(samples_lines, samples.T)]
        text.set_text('ELBO:{0:.2f}'.format(m.elbo(x, y).numpy().squeeze()))
        Z_line.set_xdata(m.inducing_variable.Z.numpy().flatten())
        print(m.inducing_variable.Z.numpy().flatten())
        return samples_lines + [data_line, text, Z_line]

    ani = animation.FuncAnimation(
        fig, animate, init_func=init, interval=40, blit=True, save_count=500)

    if SAVE:
        ani.save('fit.mp4', writer=WRITER)
    return ani


def animate_inference():
    np.random.seed(0)
    x_test = np.linspace(0, 10, 200)[:, None]
    means, covs = [], []
    for num_data in range(x.size):
        k = gpflow.kernels.RBF(1)
        m = gpflow.models.GPR(x[:num_data], y[:num_data], kern=k)
        m.likelihood.variance = 0.01
        mu, var = m.predict_f_full_cov(x_test)
        means.append(mu.squeeze())
        covs.append(var.squeeze())
        gpflow.reset_default_graph_and_session()
        print(num_data)

    sqrts = [np.linalg.cholesky(c + np.eye(200) * 1e-6) for c in covs]

    fig, ax = plt.subplots(1, 1)
    white_samples = np.random.randn(200, num_samples)
    samples_lines = ax.plot(x_test, white_samples, "C0", lw=1)

    data_line, = ax.plot(x, y, 'C1x', mew=2, ms=6)
    ax.set_xlim(0, 10)
    ax.set_ylim(-2.6, 2.6)

    def init():
        return samples_lines + [data_line]

    def animate(i):
        i = i % (10 * x.size) # loop.
        j = i//10
        mean = interpolate(means[j], means[j+1], i %10, 10)
        L = interpolate(sqrts[j], sqrts[j+1], i%10, 10)
        samples = np.dot(L, white_samples)
        [l.set_data(x_test.flatten(), s + mean) for l, s in zip(samples_lines, samples.T)]
        data_line.set_xdata(x[:j+1])
        data_line.set_ydata(y[:j+1])
        return samples_lines + [data_line]

    ani = animation.FuncAnimation(
        fig, animate, init_func=init, interval=40, blit=True, save_count=500)

    if SAVE:
        ani.save('inference.mp4', writer=WRITER)
