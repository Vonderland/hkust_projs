import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import time
import warnings
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# limit the rows for concise
pd.set_option('display.max_rows', 28)

# preprocessing the training dataset and change it into dataframe
data = []
train_file_path = "Q1_dataset/letter_train"
with open(train_file_path) as f:
    raw = f.readlines()
    cur = []
    for line in raw:
        cur = line.split()
        for i in range(1, 17):
            cur[i] = cur[i].split(":")[1]
        data.append(cur)
train_df = pd.DataFrame(data)
print("train_df:")
print(train_df)

# preprocessing the testing dataset and change it into dataframe
test_data = []
test_file_path = "Q1_dataset/letter_test"
with open(test_file_path) as f:
    raw = f.readlines()
    cur = []
    for line in raw:
        cur = line.split()
        for i in range(1, 17):
            cur[i] = cur[i].split(":")[1]
        test_data.append(cur)
test_df = pd.DataFrame(test_data)
print("test_df:")
print(test_df)

# split x and y from training
train_y = train_df[0]
train_x = train_df.drop(columns=[0])
# print(train_y)

# split x and y from testing
test_y = test_df[0]
test_x = test_df.drop(columns=[0])
# print(test_y)

# the distribution of the training data
print(train_y.value_counts())
# the distribution of the test data
print(test_y.value_counts())

# define a function to print matrics for convenience
def print_matrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    # all use macro as average because this is a multi-class problem
    prec = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    print("acc: {:.6}".format(acc))
    print("precision: {:.6}".format(prec))
    print("recall: {:.6}".format(recall))
    print("f1: {:.6}".format(f1))

# for the concise output...
warnings.filterwarnings("ignore")

# params for DecisionTreeClassifier: depths and criterion
depths = [5,10,15,20,25]
criterion = ["gini", "entropy"]
print("-------------------------------------------")
print("DecisionTreeClassifier")
for c in criterion:
    for d in depths:
        print("------- criterion = {}, depth = {} -------".format(c, d))
        start = time.perf_counter()
        clf = DecisionTreeClassifier(random_state=0, criterion=c, max_depth=d)
        clf.fit(train_x, train_y)
        end = time.perf_counter()
        print("training time: {:.6}".format(end-start))
        print("tree depth =", clf.get_depth())
        print("matrics for training:")
        pred_y = clf.predict(train_x)
        print_matrics(train_y, pred_y)
        print("matrics for testing:")
        pred_y = clf.predict(test_x)
        print_matrics(test_y, pred_y)

# KNeighborsClassifier
n_neighbors = [1, 3, 5]
print("-------------------------------------------")
print("KNeighborsClassifier")
for n in n_neighbors:
    print("------- n_neighbors = {} -------".format(n))
    start = time.perf_counter()
    knn = KNeighborsClassifier(n_neighbors=n)
    knn.fit(train_x, train_y)
    end = time.perf_counter()
    print("training time: {:.6}".format(end-start))
    print("matrics for training:")
    pred_y = knn.predict(train_x)
    print_matrics(train_y, pred_y)
    print("matrics for testing:")
    pred_y = knn.predict(test_x)
    print_matrics(test_y, pred_y)

# RandomForestClassifier
print("-------------------------------------------")
print("RandomForestClassifier")
max_depth = [10, 15, 20]
for d in max_depth:
    print("------- max_depth = {} -------".format(d))
    start = time.perf_counter()
    rf = RandomForestClassifier(max_depth=d)
    rf.fit(train_x, train_y)
    end = time.perf_counter()
    print("training time: {:.6}".format(end-start))
    print("matrics for training:")
    pred_y = rf.predict(train_x)
    print_matrics(train_y, pred_y)
    print("matrics for testing:")
    pred_y = rf.predict(test_x)
    print_matrics(test_y, pred_y)