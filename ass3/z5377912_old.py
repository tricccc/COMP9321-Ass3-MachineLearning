import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


# Data Exploration:
train = pd.read_csv('training.csv')
test = pd.read_csv('test.csv')

# train.info()

# train.select_dtypes(include=['object'])


# We find that there are 3 types of data in the train data set. There are some object types data with several categories. For example, in NAME_INCOME_TYPE, there are State servant, Working, Pensioner. There data are very hard to train, we want to transform them as numerical data (1, 2, 3...)
# So firstly we need to find out these types of object columns.

# In[30]:

length2_columns = train.columns[train.apply(lambda x:len(x.unique())) == 2]
boolen_data_list = [0, 1, "Y", "N"]
if_bool = train[length2_columns].isin(boolen_data_list)
boolen_data = train[if_bool == True]
boolen_data = boolen_data.dropna(axis=1)
boolen_features = boolen_data.columns.values
# boolen_features


# We find the boolean types features with the data 1 or 0 or Y or N. We know that boolean types data only have 1/0 or Y/N, so it only have 2 different data in its column.

# In[31]:


numerical_data = train.select_dtypes(include=['int64', 'float64'])
all_numerical_features = numerical_data.columns.values
numerical_features = np.setdiff1d(all_numerical_features, boolen_features)
# numerical_features


# In[32]:


train_features = train.columns.values
object_features = np.setdiff1d(train_features, boolen_features)
object_features = np.setdiff1d(object_features, numerical_features)
# object_features


# We have 2 target: TARGET, AMT_INCOME_TOTAL

# In[33]:


# train.select_dtypes("object")


# In[34]:


# plt.figure(figsize= (10,20))
# plt.style.use('seaborn-talk')
#
# fig, axes = plt.subplots(nrows=2, ncols=1)
# plt.subplots_adjust(left=None, bottom=None, right=None, top=None, hspace=0.5)
# fig.suptitle('Target Pictures')
#
# color = ['r', 'g']
# ax0 = train.TARGET.value_counts().plot(kind = 'barh', ax = axes[0], color = color)
# ax0.set_title('TARGET')
#
# ax1 = sns.kdeplot(np.log(train.AMT_INCOME_TOTAL), color="y", ax = axes[1])
# ax1.set_title('AMT_INCOME_TOTAL')


# From the Target Pictures, we can find that the data of TARGET, which is 1 and 0, their amount are unbalanced. Also, from the AMT_INCOME_TOTAL picture we find that the data focus on the middle part.

# In[35]:


# missing data
# total = train.isnull().sum()
# percent = train.isnull().sum()/ train.isnull().count()
# missing = pd.concat([total,percent], axis=1, keys=["Total", "Percent"])
# data = train[missing[missing.Percent < 0.001].index]
# data = data.dropna()
# data


# In[36]:


# label encoding
# le = LabelEncoder()
# for col in data.select_dtypes("object").columns:
#     data[col] = le.fit_transform(data[col])
# data


# In[37]:


# outlier filter
# iso = IsolationForest(contamination=0.1)
# ycap = iso.fit_predict(data.values)
# mask = ycap != -1
# data = data[mask]
# data


# In[38]:


def prep(data):
    total = train.isnull().sum()
    percent = train.isnull().sum()/ train.isnull().count()
    missing = pd.concat([total,percent], axis=1, keys=["Total", "Percent"])
    data = train[missing[missing.Percent < 0.001].index]
    data = data.dropna()
    le = LabelEncoder()
    for col in data.select_dtypes("object").columns:
        data[col] = le.fit_transform(data[col])
    iso = IsolationForest(contamination=0.1)
    ycap = iso.fit_predict(data.values)
    mask = ycap != -1
    data = data[mask]
    return data


# In[63]:


from sklearn.feature_selection import VarianceThreshold

def prep2(data, target, miss_ratio=None, encoder=None):
    data = data.copy()
    
    #remove missing data
    if miss_ratio!=None:
        total = data.isnull().sum()
        percent = data.isnull().sum()/ data.isnull().count()
        missing = pd.concat([total,percent], axis=1, keys=["Total", "Percent"])
        data = data[missing[missing.Percent < miss_ratio].index.values]
        data = data.dropna()
    
    #remove single
    single_column = data.columns[data.apply(lambda x: len(x.unique())) == 1].values
    data = data.drop(columns=single_column)
    
    #label encoder
    if encoder:
        for col in data.select_dtypes(include="object").columns:
            data[col] = encoder.fit_transform(data[col])
            
    #split features and target
    y = data[target]
    x = data.drop(columns=[target])

#     #outlier filter
#     iso = IsolationForest(contamination=0.1)
#     ycap = iso.fit_predict(x.values)
#     mask = ycap != -1
#     x = x[mask]
    
#     filter = VarianceThreshold(threshold=0.01)
#     filter.fit_transform(x)
#     x = x[x.columns[filter.get_support()]]
    
    return x, y


# In[64]:


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


# In[65]:
new_data = prep(train)

# In[66]:
from sklearn.feature_selection import VarianceThreshold
filter = VarianceThreshold(threshold=0.01)
filter.fit_transform(new_data)

new_data = new_data[new_data.columns[filter.get_support()]]

# In[67]:
selector = SelectKBest(f_regression, k=20)

# use def prep
# X_train,y_train = prep2(train, target="AMT_INCOME_TOTAL", encoder=le, miss_ratio=0.05)
# X_test,y_test = prep2(test, target="AMT_INCOME_TOTAL", encoder=le, miss_ratio=0.05)

y_train = new_data["AMT_INCOME_TOTAL"]
X_train = new_data.drop("AMT_INCOME_TOTAL",axis=1)

selector.fit(X_train, y_train)


# In[68]:
selector = SequentialFeatureSelector(RandomForestRegressor(),
                                    n_features_to_select=20,
                                    n_jobs=-1,
                                    direction="forward",
                                    scoring="r2",
                                    cv=3)
# In[69]:
from sklearn.linear_model import Lasso

selector = SelectFromModel(Lasso(alpha=100), max_features=20)
selector.fit(X_train,y_train)

# In[70]:
X_train = X_train[X_train.columns[selector.get_support()]]

# In[71]:
features = X_train.columns.values
# features

# In[72]:
# scaler
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.values)
y_train = np.log(y_train)

# X_test = scaler.fit_transform(X_test.values)
# y_test = np.log(y_test)
test_data = prep(test)
y_test = np.log(test_data["AMT_INCOME_TOTAL"])
X_test = scaler.transform(test_data[features].values)

# In[73]:
# regression
from sklearn.linear_model import LinearRegression, SGDRegressor , Ridge, ElasticNet
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from scipy.stats import randint, uniform

# In[74]:
model = LinearRegression()
model.fit(X_train, y_train)
prediction = model.predict(X_test)

# # In[75]:
# # LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score
from scipy.stats import pearsonr
# mean_squared_error(y_test, prediction), pearsonr(y_test, prediction)[0]
#
# # In[76]:
# # Ridge
# model = Ridge()
# model.fit(X_train, y_train)
# prediction = model.predict(X_test)
# mean_squared_error(y_test, prediction), pearsonr(y_test, prediction)[0]
#
# # In[77]:
# # SGD
# model = SGDRegressor(n_iter_no_change=250, penalty=None, max_iter=100000, tol=0.001)
# model.fit(X_train, y_train)
# prediction = model.predict(X_test)
# mean_squared_error(y_test, prediction), pearsonr(y_test, prediction)[0]
#
# # In[78]:
# # AdaBoost
# model = AdaBoostRegressor(LinearRegression(), n_estimators= 1000)
# model.fit(X_train, y_train)
# prediction = model.predict(X_test)
# mean_squared_error(y_test, prediction), pearsonr(y_test, prediction)[0]

# In[79]:
# GBoost
model = GradientBoostingRegressor()
model.fit(X_train, y_train)
prediction = model.predict(X_test)
mean_squared_error(y_test, prediction), pearsonr(y_test, prediction)[0]
MSE = round(mean_squared_error(y_test, prediction), 2)
correlation = round(pearsonr(y_test, prediction)[0], 2)
# print(MSE, correlation)
df_part1 = pd.DataFrame(columns=['zid', 'MSE', 'correlation'])
df_part1['zid'] = ['z5377912']
df_part1['MSE'] = MSE
df_part1['correlation'] = correlation
df_part1.to_csv('z5377912.PART1.summary.csv', index=False)


# In[80]:
# model = GradientBoostingRegressor(n_estimators=742,learning_rate=0.01468301725168444)
# model.fit(X_train, y_train)
# prediction = model.predict(X_test)
# mean_squared_error(y_test, prediction), pearsonr(y_test, prediction)[0]

# In[82]:


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

# In[84]:


# def prep2(data, target, miss_ratio=None, encoder=None):
#     data = data.copy()
    
#     #remove missing
#     if miss_ratio!=None:
#         total = data.isnull().sum()
#         percent = data.isnull().sum()/ data.isnull().count()
#         missing = pd.concat([total,percent], axis=1, keys=["Total", "Percent"])
#         data = data[missing[missing.Percent < miss_ratio].index.values]
#         data = data.dropna()
    
#     #remove single
#     single_column = data.columns[data.apply(lambda x: len(x.unique())) == 1].values
#     data = data.drop(columns=single_column)
    
#     #label encoder
#     if encoder:
#         for col in data.select_dtypes(include="object").columns:
#             data[col] = encoder.fit_transform(data[col])
            
#     #split features and target
#     y = data[target]
#     data = data.drop(columns=[target])

#     #outlier filter
# #     iso = IsolationForest(contamination=0.1)
# #     ycap = iso.fit_predict(data.values)
# #     mask = ycap != -1
# #     data = data[mask]
# #     data
#     return data, y

# In[85]:
selector = SelectKBest(f_classif, k=10)
le = LabelEncoder()
X_train,y_train = prep2(train, target="TARGET", encoder=le, miss_ratio=0.05)
X_test,y_test = prep2(test, target="TARGET", encoder=le, miss_ratio=0.05)
selector.fit(X_train, y_train)
features = X_train.columns[selector.get_support()]
X_train, X_test = X_train[features], X_test[features]
X_test = X_test.dropna()
y_test = y_test[X_test.index.values]

# In[86]:
model = KNeighborsClassifier(15, weights="distance")
model.fit(X_train, y_train)
prediction = model.predict(X_test)

# In[88]:
from sklearn.metrics import classification_report
report = classification_report(y_test, prediction, output_dict=True)

# In[89]:
average_precision = round(report["macro avg"]["precision"], 2)
average_recall = round(report["macro avg"]["recall"], 2)
accuracy = round(report["accuracy"], 2)
# print(average_precision, average_recall, accuracy)
# report
df_part2 = pd.DataFrame(columns=['zid', 'average_precision', 'average_recall', 'accuracy'])
df_part2['zid'] = ['z5377912']
df_part2['average_precision'] = average_precision
df_part2['average_recall'] = average_recall
df_part2['accuracy'] = accuracy
df_part2.to_csv('z5377912.PART2.summary.csv', index=False)

# output
df_part2_prediction = pd.DataFrame(columns=['SK_ID_CURR', 'predicted_target'])
df_part2_prediction['SK_ID_CURR'] = range(1,11881)
df_part2_prediction['predicted_target'] = prediction
# df_part2_prediction
df_part2_prediction.to_csv('z5377912.PART2.output.csv', index=False)

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




