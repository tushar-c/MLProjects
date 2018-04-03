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


def get_sigma_QDA(X, Y, classes, mu_vectors):
    qda_sigmas = []
    for i in range(len(Y)):
        if Y[i] == class_label:
            s = get_sigma(X, Y, class_label, mu_vectors[i])
            qda_sigmas.append(s)
    return qda_sigmas


def QDA_predictions(X, pi_k, mu_k, sigmas):
    predictions = []
    for i in range(len(sigmas)):
        eigs = np.linalg.eig(sigmas[i])
        eigvals = eigs[0]
        eigvec = eigs[1]
        D = np.diag(eigvals)
        t1 = np.sum(np.log(eigvals))/-2
        diff = (X - mu_k[i]).reshape(4, 1)
        t2 = np.matmul(diff.T, np.matmul(np.linalg.inv(sigmas[i]), diff)) / -2
        t3 = np.log(pi_k[i])
        predictions.append(t1 + t2 + t3)
    final_predict = predictions.index(max(predictions))
    return final_predict


def QDA(X, Y):
    N = len(Y)
    class1 = [(X[i], Y[i]) for i in range(len(Y)) if Y[i] == 0]
    class2 = [(X[i], Y[i]) for i in range(len(Y)) if Y[i] == 1]
    pi1, pi2 = len(class1)/N, len(class2)/N
    mu1, mu2 = 0, 0
    for i in range(len(Y)):
        if Y[i] == 0:
            mu1 += X[i]
        else:
            mu2 += X[i]
    mu1, mu2 = mu1 / N, mu2 / N
    dummy_X = []
    for i in range(len(X)):
        raw_elems = [j for j in X[i]]
        dummy_X.append(raw_elems)
    class1_vectors = [(i[0], i[1]) for i in class1]
    class2_vectors = [(j[0], j[1]) for j in class2]
    class1_X = [i[0] for i in class1_vectors]
    class1_Y = [i[1] for i in class1_vectors]
    class2_X = [j[0] for j in class2_vectors]
    class2_Y = [j[1] for j in class2_vectors]
    sigmas = [get_sigma(class1_X, class1_Y, 0, mu1), \
              get_sigma(class2_X, class2_Y, 1, mu2)]
    s1 = sigmas[0]
    s2 = sigmas[1]
    pi_vector = [pi1, pi2]
    mu_vector = [mu1, mu2]
    return (pi_vector, mu_vector, sigmas)


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
q_fit = QDA(X, y)
sigmas = q_fit[2]
qda_correct = 0

for i in range(len(X_)):
    qda_predict = QDA_predictions(X_[i], pi_k, mu_k, sigmas)
    if qda_predict == y_[i]:
        qda_correct += 1
qda_accuracy = (qda_correct / len(X_)) * 100
print('(qda)correct predictions = {}, accuracy = {}%'.format(qda_correct, qda_accuracy))
