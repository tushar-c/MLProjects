import process_data
import numpy as np
import ffnn


pre_data = process_data.get_data()
data = process_data.process_data_2(pre_data)
features = data[0]
labels = data[1]
train_features = np.array(features[:80]).reshape(80, 4)
test_features = np.array(features[80:]).reshape(20, 4)
train_labels = np.array(labels[:80]).reshape(80, 2)
test_labels = labels[80:]


folds = 8
fold_size = int(80 / folds)
divided_data = []
start = 0
end = fold_size
for f in range(folds):
    extract_features = train_features[start:end]
    extract_labels = train_labels[start:end]
    divided_data.append((extract_features, extract_labels))
    start += fold_size
    end += fold_size

test_fold = divided_data[0]
cv_mses = []
for d in range(len(divided_data[1:])):
    X = divided_data[d][0]
    y = divided_data[d][1]
    B1 = np.linalg.inv(np.matmul(X.T, X))
    B = np.matmul(B1, np.matmul(X.T, y))
    p = np.matmul(test_fold[0], B)
    error = ffnn.mse(p, y)
    cv_mses.append(error)
    print('fold {} error = {}'.format(d + 1, error))

cv_error = np.mean(cv_mses)
print('cross validation error = {}'.format(cv_error))
