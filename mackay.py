import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Ellipse

import matplotlib
matplotlib.rcParams['figure.figsize'] = (16, 6)


def samples_2D():
    np.random.seed(12)
    num_samples = 100
    Ks = [np.array([[1., 0.8], [0.8, 1.]]),
          np.array([[1., 0.32], [0.32, 1.]]),
          np.array([[1., 0.98], [0.98, 1.]])]

    for i, K in enumerate(Ks):
        fig, axes = plt.subplots(1, 2)
        L = np.linalg.cholesky(K)
        samples = np.dot(L, np.random.randn(2, num_samples))
        line, = axes[0].plot([], [], 'C0x', mew=2, ms=8)

        evals, evecs = np.linalg.eigh(K)
        ell = Ellipse(xy=(0, 0),
                  width=np.sqrt(evals[0])*4, height=np.sqrt(evals[1])*4,
                  angle=np.rad2deg(np.arccos(evecs[0, 0])))
        ell.set_facecolor('none')
        ell.set_edgecolor('k')

        axes[0].add_artist(ell)
        axes[0].set_xlabel('f(x_1)')
        axes[0].set_ylabel('f(x_2)')
        axes[0].set_xlim(-2.5, 2.5)
        axes[0].set_ylim(-2.5, 2.5)

        axes[1].imshow(K, cmap=plt.cm.gray, extent=[0, 2, 0, 2], origin='lower', interpolation='nearest', vmin=0.2, vmax=1.0)
        axes[1].set_xticks([])
        axes[1].set_yticks([])
        xx, yy = np.mgrid[0:1:2j, 0:1:2j]
        xx, yy = xx + 0.5, yy + 0.5
        [axes[1].text(x, y, k, color='C1',
                      horizontalalignment='center',
                      verticalalignment='center',
                      fontsize=20)
         for x, y, k in zip(xx.flat, yy.flat, K.flat)]

        def init():
            return [line]

        def animate(i):
            line.set_xdata(samples[0, :i])
            line.set_ydata(samples[1, :i])
            return [line]

        ani = animation.FuncAnimation(
            fig, animate, init_func=init, interval=100, blit=True, save_count=num_samples)
        w = animation.ImageMagickWriter(fps=25)
        ani.save('twoD_samples{}.gif'.format(i+1), writer=w)


def samples_2D_gp_plot():
    np.random.seed(12)
    num_samples = 100
    Ks = [np.array([[1., 0.8], [0.8, 1.]]),
          np.array([[1., 0.32], [0.32, 1.]]),
          np.array([[1., 0.98], [0.98, 1.]])]

    for i, K in enumerate(Ks):
        fig, axes = plt.subplots(1, 2)
        L = np.linalg.cholesky(K)
        samples = np.dot(L, np.random.randn(2, num_samples))
        line1, = axes[0].plot([], [], 'C0x', mew=2, ms=8)

        evals, evecs = np.linalg.eigh(K)
        ell = Ellipse(xy=(0, 0),
                  width=np.sqrt(evals[0])*4, height=np.sqrt(evals[1])*4,
                  angle=np.rad2deg(np.arccos(evecs[0, 0])))
        ell.set_facecolor('none')
        ell.set_edgecolor('k')

        axes[0].add_artist(ell)
        axes[0].set_xlabel('f(x_1)')
        axes[0].set_ylabel('f(x_2)')
        axes[0].set_xlim(-2.5, 2.5)
        axes[0].set_ylim(-2.5, 2.5)

        axes[1].set_xticks([-1, 1])
        axes[1].set_xticklabels(['x_1', 'x_2'])
        axes[1].set_yticks([])
        axes[1].set_ylim(-2.5, 2.5)
        axes[1].set_xlim(-1.5, 1.5)
        line2, = axes[1].plot([], [], 'C0--', mew=2, ms=8, marker='x')

        def init():
            return [line1, line2]

        def animate(i):
            line1.set_xdata(samples[0, :i])
            line1.set_ydata(samples[1, :i])
            line2.set_xdata(np.tile([[-1, 1, np.nan]], [i, 1]).flatten())
            line2.set_ydata(np.vstack([samples[:, :i], (np.empty(i) * np.nan)]).T.flatten())
            return [line1, line2]

        ani = animation.FuncAnimation(
            fig, animate, init_func=init, interval=100, blit=True, save_count=num_samples)
        w = animation.ImageMagickWriter(fps=25)
        ani.save('twoD_samples_gpplot{}.gif'.format(i+1), writer=w)


def samples_2D_gp_plot_cond():
    np.random.seed(12)
    num_samples = 100
    cond_val = 1.1
    Ks = [np.array([[1., 0.8], [0.8, 1.]]),
          np.array([[1., 0.32], [0.32, 1.]]),
          np.array([[1., 0.98], [0.98, 1.]])]

    for i, K in enumerate(Ks):
        fig, axes = plt.subplots(1, 2)
        L = np.linalg.cholesky(K)

        mean = K[0, 1] / K[0, 0] * cond_val
        var = K[1, 1] - K[0 , 1]**2 / K[0, 0]
        samples = np.random.randn(num_samples) * np.sqrt(var) + mean

        x_line = np.linspace(-2.5, 2.5, 200)
        density = np.exp(-0.5 * np.log(2 * np.pi) - 0.5 * np.log(var) - 0.5 * (x_line - mean)**2 / var)
        axes[0].plot(cond_val + density * 0.3, x_line, 'C1', lw=1.2)

        line1, = axes[0].plot([], [], 'C0x', mew=2, ms=8)
        line2, = axes[1].plot([], [], 'C0--', mew=2, ms=8, marker='x')

        evals, evecs = np.linalg.eigh(K)
        ell = Ellipse(xy=(0, 0),
                  width=np.sqrt(evals[0])*4, height=np.sqrt(evals[1])*4,
                  angle=np.rad2deg(np.arccos(evecs[0, 0])))
        ell.set_facecolor('none')
        ell.set_edgecolor('k')

        axes[0].add_artist(ell)
        axes[0].set_xlabel('f(x_1)')
        axes[0].set_ylabel('f(x_2)')
        axes[0].set_xlim(-2.5, 2.5)
        axes[0].set_ylim(-2.5, 2.5)

        axes[1].set_xticks([-1, 1])
        axes[1].set_xticklabels(['x_1', 'x_2'])
        axes[1].set_yticks([])
        axes[1].set_ylim(-2.5, 2.5)
        axes[1].set_xlim(-1.5, 1.5)

        def init():
            return [line1, line2]

        def animate(i):
            line1.set_xdata(np.zeros(i) + cond_val)
            line1.set_ydata(samples[:i])
            line2.set_xdata(np.tile([[-1, 1, np.nan]], [i, 1]).flatten())
            line2.set_ydata(np.vstack([np.zeros(i) + cond_val, samples[:i], (np.empty(i) * np.nan)]).T.flatten())
            return [line1, line2]

        ani = animation.FuncAnimation(
            fig, animate, init_func=init, interval=100, blit=True, save_count=num_samples)
        w = animation.ImageMagickWriter(fps=25)
        ani.save('twoD_samples_gpplot_cond{}.gif'.format(i+1), writer=w)



def samples_2D_conditioned():
    np.random.seed(12)
    num_samples = 100
    cond_val = 1.1
    Ks = [np.array([[1., 0.8], [0.8, 1.]]),
          np.array([[1., 0.32], [0.32, 1.]]),
          np.array([[1., 0.98], [0.98, 1.]])]

    for i, K in enumerate(Ks):
        fig, axes = plt.subplots(1, 2)
        mean = K[0, 1] / K[0, 0] * cond_val
        var = K[1, 1] - K[0 , 1]**2 / K[0, 0]
        samples = np.random.randn(num_samples) * np.sqrt(var) + mean
        line, = axes[0].plot([], [], 'C0x', mew=2, ms=8)

        x_line = np.linspace(-2.5, 2.5, 200)
        density = np.exp(-0.5 * np.log(2 * np.pi) - 0.5 * np.log(var) - 0.5 * (x_line - mean)**2 / var)
        axes[0].plot(cond_val + density * 0.3, x_line, 'C1', lw=1.2)

        evals, evecs = np.linalg.eigh(K)
        ell = Ellipse(xy=(0, 0),
                  width=np.sqrt(evals[0])*4, height=np.sqrt(evals[1])*4,
                  angle=np.rad2deg(np.arccos(evecs[0, 0])))
        ell.set_facecolor('none')
        ell.set_edgecolor('k')

        axes[0].add_artist(ell)
        axes[0].set_xlabel('f(x_1)')
        axes[0].set_ylabel('f(x_2)')
        axes[0].set_xlim(-2.5, 2.5)
        axes[0].set_ylim(-2.5, 2.5)

        axes[1].imshow(K, cmap=plt.cm.gray, extent=[0, 2, 0, 2], origin='lower', interpolation='nearest', vmin=0.2, vmax=1.0)
        axes[1].set_xticks([])
        axes[1].set_yticks([])
        xx, yy = np.mgrid[0:1:2j, 0:1:2j]
        xx, yy = xx + 0.5, yy + 0.5
        [axes[1].text(x, y, k, color='C1',
                      horizontalalignment='center',
                      verticalalignment='center',
                      fontsize=20)
         for x, y, k in zip(xx.flat, yy.flat, K.flat)]

        def init():
            return [line]

        def animate(i):
            line.set_xdata(np.zeros(i) + cond_val)
            line.set_ydata(samples[:i])
            return [line]

        ani = animation.FuncAnimation(
            fig, animate, init_func=init, interval=100, blit=True, save_count=num_samples)
        w = animation.ImageMagickWriter(fps=25)
        ani.save('twoD_samples_cond{}.gif'.format(i+1), writer=w)


def samples_6D_gp_plot():
    np.random.seed(12)
    num_samples = 100
    x = np.linspace(-1, 1, 6)
    K = np.exp(-0.5 * np.square(x - x[:, None]))

    fig, axes = plt.subplots(1, 2)

    line, = axes[1].plot([], [], 'C0--', mew=2, ms=8, marker='x')

    L = np.linalg.cholesky(K)
    samples = np.dot(L, np.random.randn(6, num_samples))
    axes[0].imshow(K, cmap=plt.cm.gray, extent=[0, 6, 0, 6], origin='lower', interpolation='nearest', vmin=0.2, vmax=1.0)
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    xx, yy = np.mgrid[0:5:6j, 0:5:6j]
    xx, yy = xx + 0.5, yy + 0.5
    [axes[0].text(x, y, '{0:.2f}'.format(k), color='C1',
                  horizontalalignment='center',
                  verticalalignment='center',
                  fontsize=12)
     for x, y, k in zip(xx.flat, yy.flat, K.flat)]



    axes[1].set_xticks(x)
    axes[1].set_xticklabels(['x_{}'.format(i+1) for i in range(len(x))])
    axes[1].set_yticks([])
    axes[1].set_ylim(-2.5, 2.5)
    axes[1].set_xlim(-1.2, 1.2)

    def init():
        return [line]

    def animate(i):
        line.set_xdata(np.tile(np.hstack([x, [np.nan]]), [i, 1]).flatten())
        line.set_ydata(np.vstack([samples[:, :i], (np.empty(i) * np.nan)]).T.flatten())
        print(i)
        return [line]

    ani = animation.FuncAnimation(
        fig, animate, init_func=init, interval=200, blit=True, save_count=num_samples)
    w = animation.ImageMagickWriter(fps=10)
    ani.save('sixD_samples_gpplot.gif', writer=w)


def samples_6D_gp_plot_cond():
    np.random.seed(12)
    num_samples = 100
    x = np.linspace(-1, 1, 6)
    K = np.exp(-0.5 * np.square(x - x[:, None]))

    i_cond = [1, 3]
    f_cond = np.array([[1.], [-0.2]])
    Kxx = K[i_cond][:, i_cond]
    kxX = K[i_cond]
    mu = np.dot(kxX.T, np.linalg.solve(Kxx, f_cond))
    var = K - np.dot(kxX.T, np.linalg.solve(Kxx, kxX))

    fig, axes = plt.subplots(1, 2)

    line, = axes[1].plot([], [], 'C0--', mew=2, ms=8, marker='x')

    L = np.linalg.cholesky(var + np.eye(6) * 1e-6)
    samples = np.dot(L, np.random.randn(6, num_samples)) + mu
    axes[0].imshow(K, cmap=plt.cm.gray, extent=[0, 6, 0, 6], origin='lower', interpolation='nearest', vmin=0.2, vmax=1.0)
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    xx, yy = np.mgrid[0:5:6j, 0:5:6j]
    xx, yy = xx + 0.5, yy + 0.5
    [axes[0].text(x, y, '{0:.2f}'.format(k), color='C1',
                  horizontalalignment='center',
                  verticalalignment='center',
                  fontsize=12)
     for x, y, k in zip(xx.flat, yy.flat, K.flat)]



    axes[1].set_xticks(x)
    axes[1].set_xticklabels(['x_{}'.format(i+1) for i in range(len(x))])
    axes[1].set_yticks([])
    axes[1].set_ylim(-2.5, 2.5)
    axes[1].set_xlim(-1.2, 1.2)

    def init():
        return [line]

    def animate(i):
        line.set_xdata(np.tile(np.hstack([x, [np.nan]]), [i, 1]).flatten())
        line.set_ydata(np.vstack([samples[:, :i], (np.empty(i) * np.nan)]).T.flatten())
        return [line]

    ani = animation.FuncAnimation(
        fig, animate, init_func=init, interval=200, blit=True, save_count=num_samples)
    w = animation.ImageMagickWriter(fps=10)
    ani.save('sixD_samples_gpplot_cond.gif', writer=w)


