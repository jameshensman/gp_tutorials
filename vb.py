import numpy as np
import matplotlib.pyplot as plt
import gpflow
import matplotlib.animation as animation
import matplotlib
matplotlib.rcParams['figure.figsize'] = (10, 6)
matplotlib.rcParams['lines.linewidth'] = 3


# plot a prior, likelihood and posterior
xx = np.linspace(-10, 10, 200)
prior_var = 5
prior = np.exp(-0.5 * np.log(2 * np.pi) -0.5 * np.log(prior_var) - 0.5 * np.square(xx) / prior_var)

likelihood = np.exp(3 * xx) / (1 + np.exp(3 * xx))

plt.plot(xx, prior)
plt.plot(xx, likelihood * 0.2) # rescale for prettiness

post = prior * likelihood
post = post / np.sum(post * (xx[1] - xx[0]) )
plt.plot(xx, post)
plt.xlim(-10, 10)
plt.yticks([])


# make a gpflow model (!)
m = gpflow.models.VGP(X=np.zeros((1,1)), Y=np.ones((1, 1)),
                      kern=gpflow.kernels.White(1, variance=prior_var),
                      likelihood=gpflow.likelihoods.Bernoulli())
m.kern.variance.set_trainable(False)


mu_grid = np.linspace(-5, 10, 20)
logvar_grid = np.linspace(np.log(0.01), np.log(30), 20)
mm, vv = np.meshgrid(mu_grid, logvar_grid)
ELBOS = []
for mu_i, lv_i in zip(mm.flat, vv.flat):
    m.q_mu = np.array(mu_i).reshape(1, 1)
    m.q_sqrt = np.sqrt(np.exp(lv_i)).reshape(1,1,1)
    ELBOS.append(m.compute_log_likelihood())




o = gpflow.train.GradientDescentOptimizer(0.2)
# set initial conditions
m.q_mu = np.array(9.5).reshape(1,1)
m.q_sqrt = np.sqrt(np.exp(3)).reshape(1,1,1)


def go():
        
    q_mu = m.q_mu.read_value().squeeze()
    q_var = np.square(m.q_sqrt.read_value()).squeeze()
    q_post = np.exp(-0.5 * np.log(2 * np.pi) -0.5 * np.log(q_var) - 0.5 * np.square(xx - q_mu) / q_var)
    
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.contour(mm, vv, -np.array(ELBOS).reshape(*mm.shape), [-50, -40, -30, -20, -10, -5, -3, -2, -1, -0.8], colors='k', alpha=0.5, lw=1.6)
    ax1.set_xlabel('mean')
    ax1.set_ylabel('log(variance)')
    ball_line, = ax1.plot(q_mu, np.log(q_var), 'C0o')
    
    ax2.plot(xx, post, 'C1')
    q_line, = ax2.plot(xx, q_post, 'C0')
    ax2.set_yticks([])
    ax2.set_xlim(-10, 10)
    ax2.set_ylim(-.01, 0.5)

    opt = gpflow.train.GradientDescentOptimizer(0.1)
    optimizer_tensor = opt.make_optimize_tensor(m)
    session = gpflow.get_default_session()

    def init():
        return ball_line, q_line

    def animate(i):
        print(i)
        print(m.compute_log_likelihood())
        session.run(optimizer_tensor)
        m.anchor(session)

        q_mu = m.q_mu.value.squeeze()
        q_var = np.square(m.q_sqrt.value).squeeze()
        q_post = np.exp(-0.5 * np.log(2 * np.pi) -0.5 * np.log(q_var) - 0.5 * np.square(xx - q_mu) / q_var)
        q_line.set_ydata(q_post)
        ball_line.set_xdata(q_mu)
        ball_line.set_ydata(np.log(q_var))

        return ball_line, q_line


    ani = animation.FuncAnimation(
        fig, animate, init_func=init, interval=40, blit=True, save_count=40)

    w = animation.ImageMagickWriter(fps=25)
    ani.save('vb.gif', writer=w)
