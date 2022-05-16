import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import sys

if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit("Please input paths of the training.csv and test.csv!")
    path_train = sys.argv[1]
    path_test = sys.argv[2]
    # path_train = 'training.csv'
    # path_validation = 'test.csv'
    try:
        train = pd.read_csv('training.csv')
        test = pd.read_csv('test.csv')
    except:
        sys.exit("Invalid csv file path!")

    # Data Exploration:
    # train = pd.read_csv('training.csv')
    # test = pd.read_csv('test.csv')

    # train.info()
    # train.select_dtypes(include=['object'])

    # We find that there are 3 types of data in the train data set. There are some object types data with several categories. For example, in NAME_INCOME_TYPE, there are State servant, Working, Pensioner. There data are very hard to train, we want to transform them as numerical data (1, 2, 3...)
    # So firstly we need to find out these types of object columns.

    # In[5]:
    # length2_columns = train.columns[train.apply(lambda x:len(x.unique())) == 2]
    # boolen_data_list = [0, 1, "Y", "N"]
    # boolen_YN_list = ['Y','N']
    # if_bool = train[length2_columns].isin(boolen_data_list)
    # boolen_YN = train[length2_columns].isin(boolen_YN_list)
    # boolen_YN_data = train[boolen_YN == True]
    # boolen_data = train[if_bool == True]
    # boolen_data = boolen_data.dropna(axis=1)
    # boolen_features = boolen_data.columns.values
    # boolen_features

    # We find the boolean types features with the data 1 or 0 or Y or N. We know that boolean types data only have 1/0 or Y/N, so it only have 2 different data in its column.

    # boolen_YN_features = boolen_YN_data.dropna(axis=1)
    # boolen_YN_features


    # We find that there are 2 boolean types columns filled with Y/N. So we let them stay in the object types to do the encoder to tranform Y as 1, N as 0 later.
    # numerical_data = train.select_dtypes(include=['int64', 'float64'])
    # all_numerical_features = numerical_data.columns.values
    # numerical_features = np.setdiff1d(all_numerical_features, boolen_features)
    # numerical_features

    # Numerical data is very good to use, so I don't do much on it.
    # train_features = train.columns.values
    # object_features = np.setdiff1d(train_features, boolen_features)
    # object_features = np.setdiff1d(object_features, numerical_features)
    # object_features

    # This is the pure object features, but as I mentioned above, we need to encoder them as well as 2 special Y/N boolean features.

    # train.select_dtypes("object")


    # So all the object features including 14 pure object features and 2 special boolean features are shown above. We need to encoder them to numbers.

    # We have 2 target: TARGET, AMT_INCOME_TOTAL

    # draw
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

    # Data Preprocessing
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

        return data

    # model
    from sklearn.feature_selection import SelectKBest, SelectPercentile, SelectFromModel, SequentialFeatureSelector

    # estimator regression
    from sklearn.feature_selection import f_regression, mutual_info_regression

    ## classification
    from sklearn.feature_selection import f_classif, mutual_info_classif, chi2

    from sklearn.ensemble import RandomForestRegressor

    new_train = prep1_train(train)
    new_test = prep1_test(test)

    from sklearn.feature_selection import VarianceThreshold
    filter = VarianceThreshold(threshold=0.01)
    filter.fit_transform(new_train)

    new_train = new_train[new_train.columns[filter.get_support()]]

    def separateX_y(data, target):
        y = data[target]
        X = data.drop(target,axis=1)
        return X, y
    X_train,y_train = separateX_y(new_train, 'AMT_INCOME_TOTAL')
    X_test,y_test = separateX_y(new_test, 'AMT_INCOME_TOTAL')


    # Feature Selection
    selector = SelectKBest(f_regression, k=35)
    selector.fit(X_train, y_train)
    features = X_train.columns[selector.get_support()]

    # features
    X_train, X_test = X_train[features], X_test[features]


    # We find that in the train_data and test_data the data in different columns have very big difference.
    # In some columns the number is very big and in some columns the number is very small.
    # The very big data may have bigger influence to the prediction results.
    # So we scale them as the same order of magnitude.
    #
    # Like here we know AMT_INCOME_TOTAL data has very big numbers, so we log them to get the smaller numbers,
    # then this column's data won't have too much difference between other smaller columns' data.
    # it is more suitable for our regression model below.
    # scaler
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler

    def scaler(X_data, y_data, scaler_way):
        scaler = scaler_way
        X = scaler.fit_transform(X_data.values)
        y = np.log(y_data)
        return X,y

    X_train, y_train = scaler(X_train, y_train, MaxAbsScaler())
    X_test, y_test = scaler(X_test, y_test, MaxAbsScaler())


    # Regression
    from sklearn.linear_model import LinearRegression, SGDRegressor , Ridge, ElasticNet

    from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor

    from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
    from scipy.stats import randint, uniform

    from sklearn.metrics import mean_squared_error, accuracy_score
    from scipy.stats import pearsonr
    def regression(regression_way, X_train, y_train, X_test, y_test):
        model = regression_way
        model.fit(X_train, y_train)
        prediction_log = model.predict(X_test)
        #Log the data in y_data above, now exp it back to original data.
        prediction = np.exp(prediction_log)
        MSE = mean_squared_error(np.exp(y_test), prediction)
        correlation = pearsonr(y_test, prediction_log)[0]
        return prediction, MSE, correlation

    # Use different regression model to get the prediction, mean_squared_error, and correlation.
    # Compare all the model output correlation to choose the model with the best correlation,
    # use this model to get the predicted income.

    # LinearRegression
    # prediction, MSE, correlation = regression(LinearRegression(), X_train, y_train, X_test, y_test)
    # # prediction
    # MSE, correlation

    # In[23]:

    # # Ridge
    # prediction, MSE, correlation = regression(Ridge(), X_train, y_train, X_test, y_test)
    # # prediction
    # MSE, correlation


    # In[24]:

    # SGD
    # prediction, MSE, correlation = regression(SGDRegressor(), X_train, y_train, X_test, y_test)
    # # prediction
    # MSE, correlation

    # In[25]:

    # # AdaBoost
    # prediction, MSE, correlation = regression(AdaBoostRegressor(), X_train, y_train, X_test, y_test)
    # # prediction
    # MSE, correlation

    # In[26]:

    # GBoost
    prediction, MSE, correlation = regression(GradientBoostingRegressor(), X_train, y_train, X_test, y_test)
    # prediction
    MSE, correlation

    # GradientBoostingRegressor has the highest performance,
    # so I choose to use GradientBoostingRegressor model to get the predicted income.

    # get the SK_ID_CURR
    id_list = []
    for id in test.SK_ID_CURR:
        id_list.append(id)

    # part1 output
    # csv performance
    df_part1 = pd.DataFrame(columns=['zid', 'MSE', 'correlation'])
    df_part1['zid'] = ['z5377912']
    df_part1['MSE'] = round(MSE, 2)
    df_part1['correlation'] = round(correlation, 2)
    df_part1.to_csv('z5377912.PART1.summary.csv', index=False)

    # csv output
    prediction = prediction.astype(int)
    df_part1_prediction = pd.DataFrame(columns=['SK_ID_CURR', 'predicted_income'])
    df_part1_prediction['SK_ID_CURR'] = id_list
    df_part1_prediction['predicted_income'] = prediction
    df_part1_prediction.to_csv('z5377912.PART1.output.csv', index=False)

    # Classification
    from sklearn.svm import SVC, LinearSVC
    from sklearn.neural_network import MLPClassifier

    # Data Preprocessing

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

    part2_train = prep2_train(train)
    part2_test = prep2_test(test)

    X_train,y_train = separateX_y(part2_train, 'TARGET')
    X_test,y_test = separateX_y(part2_test, 'TARGET')


    # Feature Selection
    selector = SelectKBest(f_classif, k=25)
    selector.fit(X_train, y_train)

    features = X_train.columns[selector.get_support()]
    X_train, X_test = X_train[features], X_test[features]
    X_test = X_test.dropna()
    y_test = y_test[X_test.index.values]


    # Classification
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

    from sklearn.metrics import classification_report

    def classification(classification_way, X_train, y_train, X_test, y_test):
        model = classification_way
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        report = classification_report(y_test, prediction, output_dict=True)
        average_precision = report["macro avg"]["precision"]
        average_recall = report["macro avg"]["recall"]
        accuracy = report["accuracy"]
        return prediction, average_precision, average_recall, accuracy


    # Use different models to get the accuracy, choose the model with the best accuracy to get the predication.

    # In[35]:


    # KNeighborsClassifier
    # prediction, average_precision, average_recall, accuracy = classification(KNeighborsClassifier(), X_train, y_train, X_test, y_test)
    # average_precision, average_recall, accuracy


    # In[36]:


    # GaussianNB
    # prediction, average_precision, average_recall, accuracy = classification(GaussianNB(), X_train, y_train, X_test, y_test)
    # average_precision, average_recall, accuracy


    # In[37]:


    # RandomForestClassifier
    # prediction, average_precision, average_recall, accuracy = classification(RandomForestClassifier(), X_train, y_train, X_test, y_test)
    # average_precision, average_recall, accuracy


    # In[38]:


    # AdaBoostClassifier
    # prediction, average_precision, average_recall, accuracy = classification(AdaBoostClassifier(), X_train, y_train, X_test, y_test)
    # average_precision, average_recall, accuracy


    # In[39]:


    # BaggingClassifier
    # prediction, average_precision, average_recall, accuracy = classification(BaggingClassifier(), X_train, y_train, X_test, y_test)
    # average_precision, average_recall, accuracy


    # In[40]:


    # ExtraTreesClassifier
    # prediction, average_precision, average_recall, accuracy = classification(ExtraTreesClassifier(), X_train, y_train, X_test, y_test)
    # average_precision, average_recall, accuracy


    # In[41]:


    # GradientBoostingClassifier
    prediction, average_precision, average_recall, accuracy = classification(GradientBoostingClassifier(), X_train, y_train, X_test, y_test)
    average_precision, average_recall, accuracy


    # GradientBoostingClassifier has the highest performance, so I choose to use GradientBoostingRegressor model to get the predicted target.

    # part2 output
    # csv performance
    average_precision = round(average_precision, 2)
    average_recall = round(average_recall, 2)
    accuracy = round(accuracy, 2)
    df_part2 = pd.DataFrame(columns=['zid', 'average_precision', 'average_recall', 'accuracy'])
    df_part2['zid'] = ['z5377912']
    df_part2['average_precision'] = average_precision
    df_part2['average_recall'] = average_recall
    df_part2['accuracy'] = accuracy
    df_part2.to_csv('z5377912.PART2.summary.csv', index=False)

    # csv output
    df_part2_prediction = pd.DataFrame(columns=['SK_ID_CURR', 'predicted_target'])
    df_part2_prediction['SK_ID_CURR'] = id_list
    df_part2_prediction['predicted_target'] = prediction
    df_part2_prediction.to_csv('z5377912.PART2.output.csv', index=False)

