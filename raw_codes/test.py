import pandas
import numpy
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectPercentile

# Loading default iris data set
iris = load_iris()
print(iris['target_names'])
print((iris['feature_names']))

x, y = iris.data, iris.target
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=0)
print(x_train.shape)
print(x_test.shape)
print(y_test.shape)

# Feature Selection
select = SelectPercentile(percentile=75)
select.fit(x_train, y_train)
x_train_compressed = select.transform(x_train)
print(x_train_compressed.shape)
x_test_compressed = select.transform(x_test)

# iris_data_frame = pandas.DataFrame(x_train, columns=iris.feature_names)
# axis = iris_data_frame.plot.scatter(x=iris.feature_names[0], y=iris.feature_names[1], c='Darkblue')

# Applying KNN
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train, y_train)
y_train_predict = knn.predict(x_train)
y_test_predict = knn.predict(x_test)

# Testing Scores
score = accuracy_score(y_test, y_test_predict)
train_score = accuracy_score(y_train, y_train_predict)

print("Before")
print(f'Test Accuracy: {score}')
print(f'Train Accuracy: {train_score}')

# Applying KNN on reduced features
knn.fit(x_train_compressed, y_train)
y_train_predict = knn.predict(x_train_compressed)
y_test_predict = knn.predict(x_test_compressed)

score = accuracy_score(y_test, y_test_predict)
train_score = accuracy_score(y_train, y_train_predict)

print("Compressed")
print(f'Test Accuracy: {score}')
print(f'Train Accuracy: {train_score}')

print('--Completed--')

"""
# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_test_predict.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# First aggregate all false positive rates
all_fpr = numpy.unique(numpy.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = numpy.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plot.figure()
lw = 5
plot.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plot.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plot.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plot.plot([0, 1], [0, 1], 'k--', lw=lw)
plot.xlim([0.0, 1.0])
plot.ylim([0.0, 1.05])
plot.xlabel('False Positive Rate')
plot.ylabel('True Positive Rate')
plot.title('Some extension of Receiver operating characteristic to multi-class')
plot.legend(loc="lower right")
plot.show()
"""
