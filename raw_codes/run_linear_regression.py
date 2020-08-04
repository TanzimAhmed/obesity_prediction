import pandas
import numpy
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectPercentile
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer

data_frame = pandas.read_csv('ehresp_2014.csv')
data_frame = data_frame.dropna()
print(f'Raw shape: {data_frame.shape}')

# BMI
data_frame = data_frame[data_frame.erbmi > 0]

target = pandas.DataFrame(data_frame.erbmi)
target = target.assign(y=1)
data_frame.drop(['erbmi', 'tulineno', 'tucaseid', 'euinclvl', 'eufinlwgt', 'eugenhth', 'euhgt', 'euwgt'],
                axis=1, inplace=True)

data_frame.mask(data_frame < 0, inplace=True)
data_frame = data_frame.dropna(thresh=0.7*len(data_frame), axis=1)

print(f'New shape: {data_frame.shape}')

# Removing constant features
# data_frame = data_frame.loc[:, (data_frame != data_frame.iloc[0]).any()]
data_column_names = list(data_frame.columns)
print(f'Columns: {data_column_names} Length: {len(data_column_names)}')


knn_impute = KNNImputer(n_neighbors=3, weights='uniform')
data_frame = pandas.DataFrame(knn_impute.fit_transform(data_frame))

min_max_scale = MinMaxScaler()
data_frame = pandas.DataFrame(min_max_scale.fit_transform(data_frame.to_numpy()))

print(data_frame)
numpy.seterr(divide='ignore', invalid='ignore')

# Train/Test Split
x_train, x_test, y_train, y_test = train_test_split(data_frame, target, random_state=0)
print(f'\nTrain data shape: {x_train.shape}')
print(f'Test data shape{x_test.shape}')
print(f'Target shape {y_test.shape}')

# Feature Selection
selection = SelectPercentile(percentile=50)
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
print(f'Selected Columns: {selected_columns} Length: {len(selected_columns)}')

# Applying Linear Regression
regression = LinearRegression()
regression.fit(x_train, y_train)
y_train_predict = regression.predict(x_train)
y_test_predict = regression.predict(x_test)

# Testing Scores
score = regression.score(y_test, y_test_predict)
train_score = regression.score(y_train, y_train_predict)

print("\nBefore Feature Selection")
print(f'Test Accuracy: {score}')
print(f'Train Accuracy: {train_score}')

# Applying KNN on reduced features
regression.fit(x_train_compressed, y_train)
y_train_predict = regression.predict(x_train_compressed)
y_test_predict = regression.predict(x_test_compressed)

score = regression.score(y_test, y_test_predict)
train_score = regression.score(y_train, y_train_predict)

print("\nAfter Feature Selection")
print(f'Test Accuracy: {score}')
print(f'Train Accuracy: {train_score}')

print('\n\n--Completed--')
