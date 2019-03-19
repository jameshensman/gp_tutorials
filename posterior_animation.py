import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
matplotlib.rcParams['figure.figsize'] = (10, 6)
np.random.seed(3)


M = 20
noise_var = 0.01
x = np.random.rand(20,1)
y = np.sin(10 * x) + np.cos(30 * x) + np.random.randn(x.size, 1) * np.sqrt(noise_var)

def k(X1, X2, log_sigma2, log_ls):
    X1 = X1 / np.exp(log_ls)
    X2 = X2 / np.exp(log_ls)
    X1_sq = np.sum(np.square(X1), 1)[:, None]
    X2_sq = np.sum(np.square(X2), 1)[:, None]
    sq_dist = X1_sq + X2_sq.T -2*np.matmul(X1, X2.T)
    return np.exp(log_sigma2 - 0.5 * sq_dist)

def posterior(xtest, x, y, log_sigma2, log_ls):
    Kxx = k(x, x, log_sigma2, log_ls)
    Kxs = k(x, xtest, log_sigma2, log_ls)
    Kss = k(xtest, xtest, log_sigma2, log_ls)

    L = np.linalg.cholesky(Kxx + np.eye(x.size) * noise_var)
    LiKxs = np.linalg.solve(L, Kxs)
    alpha = np.linalg.solve(L, y)
    mu = np.dot(LiKxs.T, alpha)
    cov = Kss - np.dot(LiKxs.T, LiKxs)
    log_marginal = -0.5 * x.size * np.log(2*np.pi) - 0.5 * np.sum(np.log(np.square(np.diag(L)))) - 0.5 * np.sum(np.square(alpha))
    return mu, cov, log_marginal


xx = np.linspace(0, 1, 200)[:, None]
white_samples = np.random.randn(200, 50)
fig = plt.figure()
main_ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])
main_ax.set_xlim(0, 1)
little_ax = fig.add_axes([0.12, 0.13, 0.2, 0.2])
main_ax.set_xticks([])
main_ax.set_yticks([])
little_ax.set_xlabel('log(sigma2)')
little_ax.set_ylabel('log(lengthscale)')
little_ax.set_xlim(-2, 0)
little_ax.set_ylim(-3, -1)
lines = main_ax.plot(xx, white_samples, 'C0', lw=0.6)
point, = little_ax.plot(0, 0, 'C0o')
data_line, = main_ax.plot(x, y, 'C1x', mew=2, ms=6)
data_line.set_zorder(1e6)
text = main_ax.text(0.99, 0.99, '123,45', horizontalalignment='right', verticalalignment='top', transform=main_ax.transAxes, fontsize=18)

def init():
    return lines + [point, data_line, text]

def animate(i):
    # move on a circle:
    log_sigma2 = np.cos(i * 2 * np.pi / 300) - 1
    log_ls = np.sin(i * 2 * np.pi / 300) - 2
    mu, cov, ll = posterior(xx, x, y, log_sigma2, log_ls)

    text.set_text('{0:.2f}'.format(ll))

    point.set_xdata(log_sigma2)
    point.set_ydata(log_ls)

    L = np.linalg.cholesky(cov + np.eye(200) * 1e-6)
    samples = np.dot(L, white_samples) + mu
    [l.set_ydata(s) for l, s in zip(lines, samples.T)]
    return lines + [point, data_line, text]

ani = animation.FuncAnimation(
    fig, animate, init_func=init, interval=10, blit=True, save_count=300)

w = animation.ImageMagickWriter(fps=25)
ani.save('posterior_samples.gif', writer=w)

