import pandas
import numpy
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, \
     roc_auc_score, roc_curve, auc, plot_precision_recall_curve
from sklearn.feature_selection import SelectPercentile
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer, SimpleImputer
from matplotlib import pyplot as plot

pandas.set_option('display.max_columns', 50)
pandas.set_option('display.width', 1000)

data_frame = pandas.read_csv('ehresp_2014.csv')
data_frame = data_frame.dropna()
print(f'Raw shape: {data_frame.shape}')

# Converting BMI to categorical data
# bmi_category = pandas.cut(data_frame.erbmi, bins=[-5, 0, 18.5, 24.9, 29.9, 200],
#                           labels=['invalid_BMI', 'underweight', 'normal', 'overweight', 'obese'])

data_frame = data_frame[data_frame.erbmi.values >= 0]

data_frame.mask(data_frame < 0, inplace=True)
data_frame = data_frame.dropna(thresh=0.8*len(data_frame), axis=1)

bmi_category = pandas.cut(data_frame.erbmi.values, bins=[0, 29.9, 200],
                          labels=[0, 1])

data_frame['ert_seat'] = pandas.cut(data_frame.ertseat.values, bins=[0, 30, 60, 120, 240, 480, 800, 1200, 1440],
                                    labels=[1, 2, 3, 4, 5, 6, 7, 8])

data_frame['ert_preat'] = pandas.cut(data_frame.ertpreat.values, bins=[0, 30, 60, 120, 240, 480, 800, 1200, 1440],
                                     labels=[1, 2, 3, 4, 5, 6, 7, 8])

# data_frame['weight'] = pandas.cut(data_frame.euwgt.values, bins=[0, 30, 60, 120, 240, 480, 800, 1200, 1440],
#                                     labels=[1, 2, 3, 4, 5, 6, 7, 8])

data_frame.drop('ertseat', axis=1, inplace=True)
data_frame.drop('ertpreat', axis=1, inplace=True)

# data_frame.insert(0, 'ertseat', ert_seat)

# data_frame = discrete.fit_transform(data_frame)

# print(f'Max : {max(data_frame.ertpreat)} Min : {min(data_frame.ertpreat)}')

data_frame.drop('erbmi', axis=1, inplace=True)
data_frame.insert(3, 'BMI_category', bmi_category)

target = data_frame.BMI_category.values
data_frame.drop(['BMI_category', 'tulineno', 'tucaseid', 'euinclvl', 'eufinlwgt', 'eugenhth', 'ethgt',
                 'etwgt', 'exincome1', 'erhhch'],
                axis=1, inplace=True)

print(data_frame.describe(percentiles=[0, 1/3, 2/3, 1]))


print(f'New shape: {data_frame.shape}')

# Removing constant features
# data_frame = data_frame.loc[:, (data_frame != data_frame.iloc[0]).any()]
data_column_names = list(data_frame.columns)
print(f'\n\nColumns: {data_column_names} Length: {len(data_column_names)}')


# knn_impute = KNNImputer(n_neighbors=3, weights='uniform')
# data_frame = pandas.DataFrame(knn_impute.fit_transform(data_frame))
imp = SimpleImputer(strategy="most_frequent")
data_frame = pandas.DataFrame(imp.fit_transform(data_frame))

# standard_scale = StandardScaler()
# data_frame = pandas.DataFrame(standard_scale.fit_transform(data_frame.to_numpy()))

# print(data_frame)

# Train/Test Split
x_train, x_test, y_train, y_test = train_test_split(data_frame, target, random_state=0)
print(f'\nTrain data shape: {x_train.shape}')
print(f'Test data shape{x_test.shape}')
print(f'Target shape {y_test.shape}')

# Feature Selection
selection = SelectPercentile(percentile=25)
selection.fit(x_train, y_train)
x_train_compressed = selection.transform(x_train)
print(f'\nTrain shape after selection: {x_train_compressed.shape}')
selection_status = list(selection.get_support())
print(f'Selection Status: {selection_status} Length: {len(selection_status)}')
x_test_compressed = selection.transform(x_test)

# Printing Selected Column Names
i = 0
selected_columns = []
for status in selection_status:
    if status:
        selected_columns.append(data_column_names[i])
    i += 1
print(f'Columns After Feature Selection: {selected_columns} Length: {len(selected_columns)}')

# Applying KNN
dummy = DummyClassifier(strategy='most_frequent')
dummy.fit(x_train, y_train)
y_train_predict = dummy.predict(x_train)
y_test_predict = dummy.predict(x_test)

# Testing Scores

print("\nBefore Feature Selection")
print(f'Test Accuracy: {accuracy_score(y_test, y_test_predict)}')
print(f'Test Precision: {precision_score(y_test, y_test_predict)}')
print(f'Test Recall: {recall_score(y_test, y_test_predict)}')
print(f'Test F1: {f1_score(y_test, y_test_predict)}')
print(f'Test ROC Accuracy: {roc_auc_score(y_test, y_test_predict)}')
print()
print(f'Train Accuracy: {accuracy_score(y_train, y_train_predict)}')
print(f'Train Precision: {precision_score(y_train, y_train_predict)}')
print(f'Train Recall: {recall_score(y_train, y_train_predict)}')
print(f'Train F1: {f1_score(y_train, y_train_predict)}')
print(f'Train ROC Accuracy: {roc_auc_score(y_train, y_train_predict)}')

# Applying KNN on reduced features
dummy.fit(x_train_compressed, y_train)
y_train_predict = dummy.predict(x_train_compressed)
y_test_predict = dummy.predict(x_test_compressed)


print("\nAfter Feature Selection")
print(f'Test Accuracy: {accuracy_score(y_test, y_test_predict)}')
print(f'Test Precision: {precision_score(y_test, y_test_predict)}')
print(f'Test Recall: {recall_score(y_test, y_test_predict)}')
print(f'Test F1: {f1_score(y_test, y_test_predict)}')
print(f'Test ROC Accuracy: {roc_auc_score(y_test, y_test_predict)}')
print()
print(f'Train Accuracy: {accuracy_score(y_train, y_train_predict)}')
print(f'Train Precision: {precision_score(y_train, y_train_predict)}')
print(f'Train Recall: {recall_score(y_train, y_train_predict)}')
print(f'Train F1: {f1_score(y_train, y_train_predict)}')
print(f'Train ROC Accuracy: {roc_auc_score(y_train, y_train_predict)}')


# Compute ROC curve and ROC area
def show_roc_curve(y, y_predict, label):
    fpr, tpr, _ = roc_curve(y, y_predict)
    roc_auc = auc(fpr, tpr)

    plot.figure()
    lw = 2
    plot.plot(fpr, tpr, color='darkorange',
              lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plot.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plot.xlim([0.0, 1.0])
    plot.ylim([0.0, 1.05])
    plot.xlabel('False Positive Rate')
    plot.ylabel('True Positive Rate')
    plot.title('ROC for: ' + label)
    plot.legend(loc="lower right")
    plot.show()


def show_pr_curve(classifier, x, y):
    display = plot_precision_recall_curve(classifier, x, y)
    display.ax_.set_title('Precision-Recall curve: ')


show_pr_curve(dummy, x_test_compressed, y_test)
show_roc_curve(y_train, y_train_predict, label='Zero R (Training Set)')
show_roc_curve(y_test, y_test_predict, label='Zero R (Test Set)')
print('\n\n--Completed--')
