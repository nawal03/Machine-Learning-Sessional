import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def multivariate_gaussian_pdf(x, mean, covariance):
    num_features = mean.shape[0]
    coeff = (2 * np.pi) ** (-num_features / 2) * np.linalg.det(covariance) ** (-0.5)
    expon = np.exp(-0.5 * np.sum(np.dot((x - mean), np.linalg.inv(covariance)) * (x - mean), axis=1))
    return coeff*expon


def initialization(data, num_components):
    num_samples, num_features = data.shape

    means = []
    covariances = []
    weights = []

    for _ in range(num_components):
        mean = np.zeros(num_features)
        for i in range(num_features):
            mean[i] = np.random.uniform(low=np.min(data[:, i]), high=np.max(data[:, i]))
        means.append(mean)
        covariances.append(np.identity(num_features))
        weights.append(1/num_components)

    means = np.array(means)
    covariances = np.array(covariances)
    weights = np.array(weights)

    return means, covariances, weights


def expectation(data, means, covariances, weights):
    num_components = means.shape[0]
    num_samples = data.shape[0]

    probabilities = np.zeros((num_samples, num_components))

    for i in range(num_components):
        probabilities[:, i] = multivariate_gaussian_pdf(data, means[i], covariances[i]) * weights[i]

    probabilities /= np.sum(probabilities, axis=1, keepdims=True)

    return probabilities


def maximization(data, probabilities):
    num_components = probabilities.shape[1]
    num_samples, num_features = data.shape

    n = np.sum(probabilities, axis=0)
    means = []
    covariances = []
    weights = []
    epsilon = 1e-5
    for i in range(num_components):
        means.append(np.dot(probabilities[:, i], data)/n[i])
        covariance = np.dot(probabilities[:, i] * (data - means[i]).T, (data - means[i])) / n[i]
        covariance +=  epsilon* np.identity(num_features)  # Regularization
        covariances.append(covariance)
        weights.append(n[i]/num_samples)
    
    means = np.array(means)
    covariances = np.array(covariances)
    weights = np.array(weights)

    return means, covariances, weights


def log_likelihood(data, means, covariances, weights):
    num_components = means.shape[0]
    num_samples = data.shape[0]
    
    likelihoods = np.zeros((num_components, num_samples))

    for i in range(num_components):
        likelihoods[i, :] = multivariate_gaussian_pdf(data, means[i], covariances[i]) * weights[i]

    log_likelihoods = np.log(np.sum(likelihoods, axis=0))

    return np.sum(log_likelihoods)


def EM(data, num_components, num_trials, num_iterations):
    best_log_likelihood = -1e18
    best_param = None
    best_init_param = None
    for _ in range(num_trials):
        means, covariances, weights = initialization(data, num_components)
        init_param = (means, covariances, weights)
        for _ in range(num_iterations):
            probabilities = expectation(data, means, covariances, weights)
            means, covariances, weights = maximization(data, probabilities)
            
        curr_log_likelihood = log_likelihood(data, means, covariances, weights)
    
        if best_log_likelihood < curr_log_likelihood:
            best_log_likelihood = curr_log_likelihood
            best_param = (means, covariances, weights)
            best_init_param = init_param
    
    return best_log_likelihood, best_param, best_init_param


def PCA(data, num_features):
    centered_data = data - np.mean(data, axis=0)
    _, _, Vt = np.linalg.svd(centered_data)
    principal_axes = Vt[:num_features, :]
    projected_data = np.dot(centered_data, principal_axes.T)
    return projected_data


def plot_before_EM(data, file_name):
    plt.scatter(data[:, 0], data[:, 1], alpha=0.5)
    plt.savefig(file_name)
    plt.close()


def plot_log_likelihoods_vs_num_components(best_log_likelihoods, num_components, file_name):
    plt.plot(num_components, best_log_likelihoods, marker='o')
    plt.title('Log-Likelihood vs. Number of Components (K)')
    plt.xlabel('Number of Components (K)')
    plt.ylabel('Log-Likelihood')
    plt.savefig(file_name)
    plt.close()


def plot_after_EM(data, means, covariances, weights, file_name, num_components):
    probabilities = expectation(data, means, covariances, weights)
    assigned_components = np.argmax(probabilities, axis=1)

    plt.scatter(data[:, 0], data[:, 1], c=assigned_components, alpha=0.5)
    plt.title(f'k = {num_components}')
    plt.savefig(file_name)
    plt.close()


def plot_after_EM_range(data, best_params, min_k, max_k, file_name):
    num_subplots = max_k - min_k + 1
    num_cols = 4
    num_rows = (num_subplots + num_cols - 1) // num_cols

    _, axs = plt.subplots(num_rows, num_cols, figsize=(15, 4 * num_rows))

    for i, ax in enumerate(axs.flatten()):
        k = i + min_k
        if k <= max_k:
            means, covariances, weights = best_params[k - min_k]
            probabilities = expectation(data, means, covariances, weights)
            assigned_components = np.argmax(probabilities, axis=1)

            ax.scatter(data[:, 0], data[:, 1], c=assigned_components, alpha=0.5)
            ax.set_title(f'k = {k}')

    plt.savefig(file_name)
    plt.close()


def plot_after_EM_animation(data, initial_means, initial_covariances, initial_weights, num_iterations, file_name):
    fig, ax = plt.subplots()
    params = {'means': initial_means.copy(), 'covariances': initial_covariances.copy(), 'weights': initial_weights.copy()}

    def update(frame, ax, data, params):
        ax.clear()
        means = params['means']
        covariances = params['covariances']
        weights = params['weights']
        probabilities = expectation(data, means, covariances, weights)

        ax.scatter(data[:, 0], data[:, 1], alpha=0.5)

        for i in range(means.shape[0]):
            plot_contour(ax, means[i], covariances[i], color='red', alpha=0.5)

        means, covariances, weights = maximization(data, probabilities)
        params['means'] = means.copy()
        params['covariances'] = covariances.copy()
        params['weights'] = weights.copy()

        ax.set_title(f'Iteration: {frame + 1}')

    anim = FuncAnimation(fig, update, fargs=(ax, data, params),
                         frames=num_iterations, repeat=False)
    writer = PillowWriter(fps=3)  # Use PillowWriter for saving
    anim.save(file_name, writer=writer)


def plot_contour(ax, mean, covariance, color, alpha):
    # create grid
    x_min, x_max = data[:, 0].min(), data[:, 0].max()
    y_min, y_max = data[:, 1].min(), data[:, 1].max()
    x, y = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

    # flatten
    points = np.c_[x.ravel(), y.ravel()]

    # calculate pdf
    pdf_values = multivariate_gaussian_pdf(points, mean, covariance)
    pdf_values = pdf_values.reshape(x.shape)

    ax.contour(x, y, pdf_values, levels=8, colors=color, alpha=alpha)


if __name__ == "__main__":
    np.random.seed(42)

    file_name, min_k, max_k, best_k, num_trials, num_iterations = sys.argv[1:]
    min_k = int(min_k)
    max_k = int(max_k)
    best_k = int(best_k)
    num_trials = int(num_trials)
    num_iterations = int(num_iterations)

    df = pd.read_csv(file_name, header=None)
    data = df.to_numpy()

    if data.shape[1]>2:
        data = PCA(data, 2)

    plot_before_EM(data, 'before EM')

    best_log_likelihoods = []
    best_params = []
    best_init_params = []
    num_components = []
    for k in range(min_k, max_k+1):
        best_log_likelihood, best_param, best_init_param = EM(data, k, num_trials, num_iterations)
        best_log_likelihoods.append(best_log_likelihood)
        best_params.append(best_param)
        best_init_params.append(best_init_param)
        num_components.append(k)

    plot_log_likelihoods_vs_num_components(best_log_likelihoods, num_components, 'log likelihoods vs num components')

    plot_after_EM_range(data, best_params, min_k, max_k, 'after EM range')

    # graphs for best k
    best_idx = best_k - min_k
    
    means, covariances, weights = best_params[best_idx]
    plot_after_EM(data, means, covariances, weights, 'after EM', best_k)

    initial_means, initial_covariances, initial_weights = best_init_params[best_idx]
    plot_after_EM_animation(data, initial_means, initial_covariances, initial_weights, num_iterations, 'EM animation.gif')
        
    
    
    


    
    