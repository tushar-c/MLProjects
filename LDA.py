import numpy as np
import process_data


def get_sigma(X, Y, class_label, mu_vector):
    v = np.zeros((4, 4))
    N = len(X)
    cols = 4
    for i in range(len(Y)):
        if Y[i] == class_label:
            diff = (X[i] - mu_vector).reshape(cols, 1)
            res = np.matmul(diff,diff.T)
            v += res / (N - 2)
    return v


def lda_predictions(x, classes, pi_k, mu_k, sigma):
    sigma_inv = np.linalg.inv(sigma)
    predictions = []
    for j in range(len(classes)):
        mu_ = mu_k[j]
        pi_ = pi_k[j]
        t1 = np.matmul(x.T, np.matmul(sigma_inv, mu_))
        t2 = np.matmul(mu_.T, np.matmul(sigma_inv, mu_))
        t3 = np.log(pi_)
        predictions.append(t1 - t2 + t3)
    final_predict = predictions.index(max(predictions))
    return final_predict



def LDA(X, Y):
    N = len(Y)
    class1 = [(i, Y[i]) for i in range(len(Y)) if Y[i] == 0]
    class2 = [(i, Y[i]) for i in range(len(Y)) if Y[i] == 1]
    pi1, pi2 = len(class1)/N, len(class2)/N
    mu1, mu2 = 0, 0
    for i in range(len(Y)):
        if Y[i] == 0:
            mu1 += X[i]
        else:
            mu2 += X[i]
    mu1, mu2 = mu1 / N, mu2 / N
    s1 = get_sigma(X, Y, 0, mu1)
    s2 = get_sigma(X, Y, 1, mu2)
    sigma = s1 + s2
    pi_vector = [pi1, pi2]
    mu_vector = [mu1, mu2]
    return (pi_vector, mu_vector, sigma)


pre_data = process_data.get_data()
data = process_data.process_data(pre_data)
features = data[0]
labels = data[1]
train_features = np.array(features[:80])
train_labels = np.array(labels[:80]).reshape(80, 1)
test_features = np.array(features[80:])
test_labels = np.array(labels[80:]).reshape(20, 1)
X = train_features
y = train_labels
X_ = test_features
y_ = test_labels
l_fit = LDA(X, y)
pi_k = l_fit[0]
mu_k = l_fit[1]
sigma = l_fit[2]
lda_correct = 0

for i in range(len(X_)):
    lda_predict = lda_predictions(X_[i], [0, 1], pi_k, mu_k, sigma)
    if lda_predict == y_[i]:
        lda_correct += 1
lda_accuracy = (lda_correct / len(X_)) * 100
print('(lda)correct predictions = {}, accuracy = {}%'.format(lda_correct, lda_accuracy))
