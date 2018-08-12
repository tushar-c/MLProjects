"""Logistic Regression on the  UCI breast cancer dataset. Model trained using Newton-Raphson Method"""

import numpy as np 
import sklearn.datasets


def train(X, y, w_init, n_iterations):
	y_ = prepare_predictions(X, w_init)
	R = np.zeros((X.shape[0], X.shape[0]))
	for k in range(R.shape[0]):
		R[k][k] = grad_logistic_sigmoid(y_[k])
		
	H = np.matmul(X.T, np.matmul(R, X))
	w = w_init
	grad_E = np.matmul(X.T, (y_ - y))
	for n in range(n_iterations):
		error = cross_entropy_error(y_, labels)
		print('error at iteration {} = {}'.format(n + 1, error))
		w = newton_raphson(w, H, grad_E)
		y_ = prepare_predictions(X, w)
		updated_params = update_matrices(X, y_, y)
		grad_E, H, R = updated_params[0], updated_params[1], updated_params[2]		
	return w
		

def update_matrices(X, y_, y):
	R = np.zeros((X.shape[0], X.shape[0]))
	for k in range(R.shape[0]):
		R[k][k] = grad_logistic_sigmoid(y_[k])
		H = np.matmul(X.T, np.matmul(R, X))
		grad_E = np.matmul(X.T, (y_ - y))
	return grad_E, H, R
		
		
def cross_entropy_error(y_, labels):
	error = 0
	for n in range(len(y_)):
		error += (labels[n] * stable_log(y_[n])) + (1 - labels[n]) * stable_log(1 - y_[n])
	return -error
	
		
def prepare_predictions(X, w, classify=False):
	y_ = [0 for j in range(X.shape[0])]
	for k in range(X.shape[0]):
		sample = X[k, :]
		sample = sample.reshape(sample.shape[0], 1)
		prediction = predictions(sample, w, classify)
		y_[k] = prediction
	y_ = np.array(y_)
	return y_.reshape(y_.shape[0], 1)


def logistic_sigmoid(x):
	return stable_sigmoid(x)
	
	
def grad_logistic_sigmoid(x):
	return logistic_sigmoid(x) * (1 - logistic_sigmoid(x))


def predictions(x, w, classify=False):
	linear_comb = np.matmul(w.T, x)
	pred = logistic_sigmoid(linear_comb)
	if classify:
		if pred >= 0.5:
			pred = 1
		else:
			pred = 0
	return pred


def newton_raphson(w_old, H, grad_E):
	w_new = w_old - np.matmul(np.linalg.inv(H), grad_E)
	return w_new


def stable_sigmoid(x):
    squashed_x = np.zeros(x.shape[0]).reshape(x.shape[0], 1)
    for j in range(len(x)):
        if x[j] >= 0:
            z = np.power(np.e, -x[j])
            squashed_x[j] = 1 / (1 + z)
        else:
            z = np.power(np.e, x[j])
            squashed_x[j] = z / (1 + z)
    return squashed_x
	
	
def stable_log(x, delta=1e-10, epsilon=1e10):
	if x >= epsilon:
		return np.log(epsilon)
	if x <= delta:
		return np.log(delta)
	return np.log(x)
	

dataset = sklearn.datasets.load_breast_cancer()
features = dataset['data']
labels = dataset['target']
labels = labels.reshape(labels.shape[0], 1)

n_iterations = 10
random_vars = np.array([np.random.normal() for j in range(features.shape[1])])
random_vars = random_vars.reshape(random_vars.shape[0], 1)
w_init = random_vars
f_train = features[:500]
l_train = labels[:500]
f_test = features[500:]
l_test = labels[500:]
w_final = train(f_train, l_train, w_init, n_iterations)	

for j in range(f_test.shape[0]):
	sample = f_test[j, :]
	pred = predictions(sample, w_final, classify=True)
	print('predicted class {} : actual class {}'.format(pred, l_test[j]))
