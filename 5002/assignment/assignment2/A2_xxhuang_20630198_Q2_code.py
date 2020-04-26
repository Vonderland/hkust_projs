import numpy as np
from numpy.random import choice


class AdaBoostClassifier:
    def __init__(self,
                 n_estimators=3,
                 step_size=0.5):
        self.n_estimators = n_estimators
        self.step_size = step_size
        self.classifiers = []
        # prevent being divided by zero when calculating the error rate
        self.epsilon = 1e-16

    def _get_classification(self, X, threshold, cond):
        n = np.shape(X)[0]
        labels = np.ones(n)
        # y does not satisfy the condition and get -1
        if cond == "lt":
            labels[X >= threshold] = -1
        else:
            labels[X <= threshold] = -1
        return labels

    def _get_error_rate(self, X, y, threshold, cond, D, update_weights=False):
        n = np.shape(X)[0]
        pred_y = self._get_classification(X, threshold, cond)
        error = np.ones_like(y)
        error[pred_y == y] = 0
        error_rate = np.dot(error, D) / n
        # use epsilon to avoid being divided by zero
        alpha = 0.5 * np.log((1 - error_rate) / max(error_rate, self.epsilon))
        new_D = D
        if update_weights:
            flag = np.ones_like(y)
            flag[pred_y == y] = -1
            new_D = D * np.exp(flag * error_rate)
            # normalization
            new_D = new_D / new_D.sum()
        return error_rate, new_D, alpha

    def _get_weak_classifier(self, X, y, D):
        # the range for current threshold
        range_min = X.min()
        range_max = X.max()
        i = 0
        min_error = float("inf")
        classifier = {}
        while range_min + i * self.step_size <= range_max:
            threshold = range_min + i * self.step_size
            # two conditions: less than or greater than the threshold
            for cond in ["lt", "gt"]:
                error_rate, _, _ = self._get_error_rate(X, y, threshold, cond, D)
                # find better classifier and update it
                if error_rate < min_error:
                    min_error = error_rate
                    classifier["cond"] = cond
                    classifier["threshold"] = threshold
            i += 1
        return classifier

    def fit(self, X, y):
        self.classifiers = []
        n = np.shape(X)[0]
        D = np.ones(n)/n
        for i in range(self.n_estimators):
            # sampling with weights D
            samples_idx = choice(n, n, replace=True, p=D)
            sample_x = X[samples_idx]
            sample_y = y[samples_idx]
            sample_D = D[samples_idx]
            # get classifier using sampled data
            classifier = self._get_weak_classifier(sample_x, sample_y, sample_D)
            # calculate alpha and update weights
            _, D, alpha = self._get_error_rate(X, y, classifier["threshold"], classifier["cond"], D, True)
            classifier["alpha"] = alpha
            self.classifiers.append(classifier)

    def predict(self, X):
        values = 1.0 * np.ones_like(X)
        for c in self.classifiers:
            y = self._get_classification(X, c["threshold"], c["cond"])
            values += c["alpha"] * y
        # return the sign of weighted y
        return np.sign(values)

    def report(self):
        # print the expression for the final classifier
        final_str = "C*(x) = sign["
        for index, c in enumerate(self.classifiers):
            final_str += ("{:.5}C{}(x) + ".format(c["alpha"], index + 1))
        final_str = final_str[:-3]
        final_str += "]"
        print("final classifier:")
        print(final_str)
        # print the details of basic classifiers
        print("-------- basic classifiers details --------")
        for index, c in enumerate(self.classifiers):
            print("classifier {}:".format(index + 1))
            if c["cond"] == "lt":
                print("x < {},\t1".format(c["threshold"]))
                print("x >= {},\t-1".format(c["threshold"]))
            else:
                print("x > {},\t1".format(c["threshold"]))
                print("x <= {},\t-1".format(c["threshold"]))
            print("-------------------------------------------")


# change x and y into numpy array for convenience
X = [i for i in range(10)]
X = np.array(X)
y = [1, 1, 1, -1, -1, -1, 1, 1, 1, -1]
y = np.array(y)
clf = AdaBoostClassifier(n_estimators=5)
clf.fit(X, y)
clf.report()
pred = clf.predict(X)
print("predict:")
print(pred)
print("accuracy =", sum(pred == y) / np.shape(X)[0])
