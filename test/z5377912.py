import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


# Data Exploration:
train = pd.read_csv('training.csv')
test = pd.read_csv('test.csv')


# In[3]:
train.info()

# In[4]:
train.select_dtypes(include=['object'])

# We find that there are 3 types of data in the train data set. There are some object types data with several categories. For example, in NAME_INCOME_TYPE, there are State servant, Working, Pensioner. There data are very hard to train, we want to transform them as numerical data (1, 2, 3...)
# So firstly we need to find out these types of object columns.

# In[5]:
length2_columns = train.columns[train.apply(lambda x:len(x.unique())) == 2]
boolen_data_list = [0, 1, "Y", "N"]
boolen_YN_list = ['Y','N']
if_bool = train[length2_columns].isin(boolen_data_list)
boolen_YN = train[length2_columns].isin(boolen_YN_list)
boolen_YN_data = train[boolen_YN == True]
boolen_data = train[if_bool == True]
boolen_data = boolen_data.dropna(axis=1)
boolen_features = boolen_data.columns.values
boolen_features

# We find the boolean types features with the data 1 or 0 or Y or N. We know that boolean types data only have 1/0 or Y/N, so it only have 2 different data in its column.

# In[6]:
numerical_data = train.select_dtypes(include=['int64', 'float64'])
all_numerical_features = numerical_data.columns.values
numerical_features = np.setdiff1d(all_numerical_features, boolen_features)
numerical_features

# In[7]:
train_features = train.columns.values
object_features = np.setdiff1d(train_features, boolen_features)
object_features = np.setdiff1d(object_features, numerical_features)
object_features

# We have 2 target: TARGET, AMT_INCOME_TOTAL

# In[8]:
train.select_dtypes("object")

# In[9]:
plt.figure(figsize= (10,20))
plt.style.use('seaborn-talk')

fig, axes = plt.subplots(nrows=2, ncols=1)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, hspace=0.5)
fig.suptitle('Target Pictures')

color = ['r', 'g']
ax0 = train.TARGET.value_counts().plot(kind = 'barh', ax = axes[0], color = color)
ax0.set_title('TARGET')

ax1 = sns.kdeplot(np.log(train.AMT_INCOME_TOTAL), color="y", ax = axes[1])
ax1.set_title('AMT_INCOME_TOTAL')

# In[10]:
# sns.heatmap(train[['']])

# From the Target Pictures, we can find that the data of TARGET, which is 1 and 0, their amount are unbalanced. Also, from the AMT_INCOME_TOTAL picture we find that the data focus on the middle part.

# Data Preprocessing

# In[11]:
def prep1_train(data):
    #     label encoder
    le = LabelEncoder()
    for col in data.select_dtypes("object").columns:
        data[col] = le.fit_transform(data[col])
        
    #     replace missing data
    total = data.isnull().sum()
    percent = data.isnull().sum()/ data.isnull().count()
    missing = pd.concat([total,percent], axis=1, keys=["Total", "Percent"])
    data = data[missing[missing.Percent < 0.05].index]
    data = data.fillna(0)
    #         outlier filter
    iso = IsolationForest(contamination=0.1)
    ycap = iso.fit_predict(data.values)
    mask = ycap != -1
    data = data[mask]
    return data

def prep1_test(data):
    #     label encoder
    le = LabelEncoder()
    for col in data.select_dtypes("object").columns:
        data[col] = le.fit_transform(data[col])
        
    #     replace missing data
    total = data.isnull().sum()
    percent = data.isnull().sum()/ data.isnull().count()
    missing = pd.concat([total,percent], axis=1, keys=["Total", "Percent"])
    data = data[missing[missing.Percent < 0.05].index]
    data = data.fillna(0)
    #         outlier filter
#     iso = IsolationForest(contamination=0.1)
#     ycap = iso.fit_predict(data.values)
#     mask = ycap != -1
#     data = data[mask]
    return data

# In[12]:
# Feature Selection
# model
from sklearn.feature_selection import SelectKBest, SelectPercentile, SelectFromModel, SequentialFeatureSelector

# estimator
## regression
from sklearn.feature_selection import f_regression, mutual_info_regression

## classification
from sklearn.feature_selection import f_classif, mutual_info_classif, chi2

# basic model
from sklearn.ensemble import RandomForestRegressor

# In[13]:
new_train = prep1_train(train)
new_test = prep1_test(test)

# In[14]:
# bai
# from sklearn.feature_selection import VarianceThreshold
# filter = VarianceThreshold(threshold=0.01)
# filter.fit_transform(data)

# new_data = data[data.columns[filter.get_support()]]
# new_data = data

# In[15]:
def separateX_y(data, target):
    y = data[target]
    X = data.drop(target,axis=1)
    return X, y
X_train,y_train = separateX_y(new_train, 'AMT_INCOME_TOTAL')
X_test,y_test = separateX_y(new_test, 'AMT_INCOME_TOTAL')

# In[16]:
# bai
# selector = SequentialFeatureSelector(RandomForestRegressor(),
#                                     n_features_to_select=20,
#                                     n_jobs=-1,
#                                     direction="forward",
#                                     scoring="r2",
#                                     cv=3)

# In[17]:
# bai
# from sklearn.linear_model import Lasso

# selector = SelectFromModel(Lasso(alpha=100), max_features=10)
# selector.fit(X_train,y_train)
# X_train = X_train[X_train.columns[selector.get_support()]]

# In[18]:
# from sklearn.tree import DecisionTreeClassifier
# from sklearn import tree
# selector = tree.DecisionTreeClassifier(criterion='entropy')
# selector.fit(X_train,y_train)

# In[19]:
features = X_train.columns.values
features

# In[20]:
# test_data = prep(test)
# test_data
# train_data = prep(train)
# train_data

# We find that in the train_data and test_data the data in different columns have very big difference. In some columns the number is very big and in some columns the number is very small. So we scale them as the same order of magnitude.

# In[21]:
# scaler
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler

def scaler(X_data, y_data, scaler_way):
    scaler = scaler_way
    X = scaler.fit_transform(X_data.values)
    y = np.log(y_data)
    return X,y

X_train, y_train = scaler(X_train, y_train, MaxAbsScaler())
X_test, y_test = scaler(X_test, y_test, MaxAbsScaler())

# In[22]:
# regression
from sklearn.linear_model import LinearRegression, SGDRegressor , Ridge, ElasticNet

from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor

from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from scipy.stats import randint, uniform

model = LinearRegression()
model.fit(X_train, y_train)
prediction = model.predict(X_test)
prediction = np.exp(prediction)
prediction

# In[24]:
# LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score
from scipy.stats import pearsonr
mean_squared_error(y_test, prediction), pearsonr(y_test, prediction)[0]

# Ridge
model = Ridge()
model.fit(X_train, y_train)
prediction = model.predict(X_test)
mean_squared_error(y_test, prediction), pearsonr(y_test, prediction)[0]

# SGD
model = SGDRegressor(n_iter_no_change=250, penalty=None, max_iter=100000, tol=0.001)
model.fit(X_train, y_train)
prediction = model.predict(X_test)
mean_squared_error(y_test, prediction), pearsonr(y_test, prediction)[0]

# AdaBoost
model = AdaBoostRegressor(LinearRegression(), n_estimators= 1000)
model.fit(X_train, y_train)
prediction = model.predict(X_test)
mean_squared_error(y_test, prediction), pearsonr(y_test, prediction)[0]

# GBoost
model = GradientBoostingRegressor()
model.fit(X_train, y_train)
prediction = model.predict(X_test)
mean_squared_error(y_test, prediction), pearsonr(y_test, prediction)[0]

#output performance
MSE = round(mean_squared_error(y_test, prediction), 2)
correlation = round(pearsonr(y_test, prediction)[0], 2)
# print(MSE, correlation)
df_part1 = pd.DataFrame(columns=['zid', 'MSE', 'correlation'])
df_part1['zid'] = ['z5377912']
df_part1['MSE'] = MSE
df_part1['correlation'] = correlation
df_part1.to_csv('z5377912.PART1.summary.csv', index=False)

# part1 output
prediction = np.exp(prediction)
df_part1_prediction = pd.DataFrame(columns=['SK_ID_CURR', 'predicted_income'])
df_part1_prediction['SK_ID_CURR'] = range(1,12001)
df_part1_prediction['predicted_income'] = prediction
df_part1_prediction.to_csv('z5377912.PART1.output.csv', index=False)

# Classification
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import StackingClassifier

from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import VarianceThreshold

def prep2_train(data):
    data = data.copy()
    
    #label encoder
    le = LabelEncoder()
    for col in data.select_dtypes("object").columns:
        data[col] = le.fit_transform(data[col])
    
    #replace missing data
    total = data.isnull().sum()
    percent = data.isnull().sum()/ data.isnull().count()
    missing = pd.concat([total,percent], axis=1, keys=["Total", "Percent"])
    data = data[missing[missing.Percent < 0.05].index]
    data = data.fillna(0)
    
    #remove single
    single_column = data.columns[data.apply(lambda x: len(x.unique())) == 1].values
    data = data.drop(columns=single_column)
    
    return data

def prep2_test(data):
    data = data.copy()
    
    #label encoder
    le = LabelEncoder()
    for col in data.select_dtypes("object").columns:
        data[col] = le.fit_transform(data[col])
    
    #replace missing data
    total = data.isnull().sum()
    percent = data.isnull().sum()/ data.isnull().count()
    missing = pd.concat([total,percent], axis=1, keys=["Total", "Percent"])
    data = data[missing[missing.Percent < 0.05].index]
    data = data.fillna(0)
    
    #remove single
    single_column = data.columns[data.apply(lambda x: len(x.unique())) == 1].values
    data = data.drop(columns=single_column)
    
    return data


# In[35]:
part2_train = prep2_train(train)
part2_test = prep2_test(test)

# In[36]:
X_train,y_train = separateX_y(part2_train, 'TARGET')
X_test,y_test = separateX_y(part2_test, 'TARGET')

# In[37]:
selector = SelectKBest(f_classif, k=10)
selector.fit(X_train, y_train)

features = X_train.columns[selector.get_support()]
X_train, X_test = X_train[features], X_test[features]
X_test = X_test.dropna()
y_test = y_test[X_test.index.values]

# In[38]:
model = KNeighborsClassifier(15, weights="distance")
model.fit(X_train, y_train)
prediction = model.predict(X_test)

# In[39]:
# output
df_part2_prediction = pd.DataFrame(columns=['SK_ID_CURR', 'predicted_target'])
df_part2_prediction['SK_ID_CURR'] = range(1,12001)
df_part2_prediction['predicted_target'] = prediction
# df_part2_prediction
df_part2_prediction.to_csv('z5377912.PART2.output.csv', index=False)

# In[40]:
from sklearn.metrics import classification_report

# print(classification_report(y_test, prediction))
report = classification_report(y_test, prediction, output_dict=True)
# report["accuracy"], report["macro avg"]["precision"],  report["macro avg"]["recall"]
average_precision = round(report["macro avg"]["precision"], 2)
average_recall = round(report["macro avg"]["recall"], 2)
accuracy = round(report["accuracy"], 2)
df_part2 = pd.DataFrame(columns=['zid', 'average_precision', 'average_recall', 'accuracy'])
df_part2['zid'] = ['z5377912']
df_part2['average_precision'] = average_precision
df_part2['average_recall'] = average_recall
df_part2['accuracy'] = accuracy
df_part2.to_csv('z5377912.PART2.summary.csv', index=False)

# if __name__ == '__main__':
#     if len(sys.argv) != 3:
#         sys.exit("Please input paths of the training.csv and test.csv!")
#     path_train = sys.argv[1]
#     path_test = sys.argv[2]
#     # path_train = 'training.csv'
#     # path_validation = 'test.csv'
#     try:
#         train = pd.read_csv(path_train)
#         test = pd.read_csv(path_validation)
#     except:
#         sys.exit("Invalid csv file path!")




