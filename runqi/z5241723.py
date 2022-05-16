import sys
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.utils import shuffle
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn import tree


def json_to_sum(df, col):
    df_new = df.copy()
    for i in range(len(df[col])):
        tmp_num = len(json.loads(df[col][i]))
        df_new.loc[i, col] = tmp_num
    mean_value = df_new[col].mean()
    for i in range(len(df_new[col])):
        if df_new[col][i] == 0:
            df_new.loc[i, col] = mean_value
    return df_new


def json_to_label(df, col):
    df_new = df.copy()
    for i in range(len(df[col])):
        try:
            tmp_name = json.loads(df[col][i])[0]['name']
            df_new.loc[i, col] = tmp_name
        except:
            df_new.loc[i, col] = 'null'
    lc = LabelEncoder()
    df_new[col] = lc.fit_transform(df_new[col].values)
    return df_new


def process_homepage(df, col):
    df_new = df.copy()
    for i in range(len(df[col])):
        if df[col][i] == 'null':
            df_new.loc[i, col] = 0
        else:
            df_new.loc[i, col] = 1
    return df_new


def process_original_language(df, col):
    df_new = df.copy()
    # all_languages = set(list(df[col]))
    lc = LabelEncoder()
    df_new[col] = lc.fit_transform(df[col].values)
    return df_new


def process_release_date(df, col):
    df_new = df.copy()
    for i in range(len(df[col])):
        try:
            df_new.loc[i, col] = int(df[col][i][0:4])
        except:
            df_new.loc[i, col] = 0
    df_new[col] = df_new[col].apply(pd.to_numeric)
    return df_new


def process_numeric(df, col):
    df_new = df.copy()
    for i in range(len(df[col])):
        if df[col][i] == 'null':
            df_new.loc[i, col] = df[col].mean()
    return df_new


def data_process_part_1(df):
    # Specify the columns we want keep
    # Drop description columns like original_tile, overview and tagline
    # Drop the column has same values: status
    columns = ['cast', 'crew', 'budget', 'genres', 'homepage', 'keywords', 'original_language', 'production_companies',
               'production_countries', 'release_date', 'runtime', 'revenue']
    df = df[columns]
    df = df.fillna('null')
    # print(df.loc[159])
    # exit(99)
    # process list columns cast, crew, production_companies, production_countries, concerting it to the number of items
    for i in ['cast', 'crew', 'production_companies', 'production_countries']:
        df = json_to_sum(df, i)
    # process list columns genres, keywords
    # process genres and keywords, get the first item and label it
    for i in ['genres', 'keywords']:
        df = json_to_label(df, i)
    # process homepage, setting to 0 or 1 depends on exist
    df = process_homepage(df, 'homepage')
    # process original_language, label it
    df = process_original_language(df, 'original_language')
    # process release_date, converting it to year
    df = process_release_date(df, 'release_date')
    # process numeric columns budget and runtime
    for i in ['budget', 'runtime']:
        df = process_numeric(df, i)
    # According to drawing scatter picture, dropping these columns
    # draw_data(df, 'revenue')

    # According to SelectKBest to choose features
    # print(df.loc[0, :])
    # print("Select: ")
    # x_new = SelectKBest(score_func=f_regression, k=7)\
    #     .fit_transform(df.drop(columns='revenue'), df['revenue'])
    # print(x_new[0])

    # According to Decision Tree to choose features
    # clf = tree.DecisionTreeClassifier(criterion='entropy')
    # print(clf)
    # clf.fit(np.array(df.drop(columns='revenue')), np.array(df['revenue']))
    # index = 0
    # for i in df.columns[:-1]:
    #     print(i, ': ', clf.feature_importances_[index])
    #     index += 1
    # print(clf.feature_importances_)

    drop_columns = ['cast', 'crew', 'keywords', 'original_language']
    df.drop(drop_columns, inplace=True, axis=1)
    return df


def data_process_part_2(df):
    # Specify the columns we want to keep
    # Drop description columns like original_tile, overview and tagline
    # Drop the column has same values: status
    columns = ['cast', 'crew', 'budget', 'genres', 'homepage', 'keywords', 'original_language', 'production_companies',
               'production_countries', 'release_date', 'runtime', 'rating']
    df = df[columns]
    df = df.fillna('null')
    # process list columns cast, crew, production_companies, production_countries, concerting it to the number of items
    for i in ['cast', 'crew', 'production_companies', 'production_countries']:
        df = json_to_sum(df, i)
    # process list columns genres, keywords
    # process genres and keywords, get the first item and label it
    for i in ['genres', 'keywords']:
        df = json_to_label(df, i)
    # process homepage, setting to 0 or 1 depends on exist
    df = process_homepage(df, 'homepage')
    # process original_language, label it
    df = process_original_language(df, 'original_language')
    # process release_date, converting it to year
    df = process_release_date(df, 'release_date')
    # process numeric columns budget and runtime
    for i in ['budget', 'runtime']:
        df = process_numeric(df, i)
    # According to drawing scatter picture, dropping these columns
    # draw_data(df, 'rating')
    # According to SelectKBest to choose features
    # print(df.loc[0, :])
    # print("Select: ")
    # x_new = SelectKBest(score_func=f_regression, k=7)\
    #     .fit_transform(df.drop(columns='rating'), df['rating'])
    # print(x_new[0])
    # exit(99)

    # According to Decision Tree to choose features
    # clf = tree.DecisionTreeClassifier(criterion='entropy')
    # print(clf)
    # clf.fit(np.array(df.drop(columns='rating')), np.array(df['rating']))
    # index = 0
    # for i in df.columns[:-1]:
    #     print(i, ': ', clf.feature_importances_[index])
    #     index += 1
    # print(clf.feature_importances_)
    # exit(99)
    drop_columns = ['cast', 'crew', 'keywords', 'original_language']
    df.drop(drop_columns, inplace=True, axis=1)
    return df


def training_and_predict_part_1(t_x, t_y, v_x):
    # Gboost = GradientBoostingRegressor(n_estimators=2000, learning_rate=0.01, max_depth=15, max_features='sqrt',
    #                                    min_samples_leaf=10, min_samples_split=10, loss='ls', random_state=42)
    # Gboost.fit(t_x, t_y)
    # prediction = Gboost.predict(v_x)

    # model = RandomForestRegressor()
    # param_grid = {'n_estimators': [i for i in range(10, 100, 10)],
    #               'max_depth': [10],
    #               'random_state': [0]
    #               }
    # grid_search = GridSearchCV(model, param_grid, n_jobs=8, verbose=1, scoring='neg_mean_squared_error')
    # grid_search.fit(t_x, t_y)
    # print(grid_search.best_params_)
    # print(grid_search.best_score_)
    # print(grid_search.best_estimator_)

    model = RandomForestRegressor(n_estimators=90, max_depth=10, random_state=0)
    model.fit(t_x, t_y)
    prediction = model.predict(v_x)

    # model = LinearRegression()
    # model.fit(t_x, t_y)
    # prediction = model.predict(v_x)

    return prediction


def output_part_1(pre, v_df, v_y):
    # output.csv
    df = pd.DataFrame(columns=['movie_id', 'predicted_revenue'])
    df['movie_id'] = v_df['movie_id']
    df['predicted_revenue'] = np.around(pre)
    df.to_csv('z5241723.PART1.output.csv', index=False)
    # summary.csv
    msr = round(mean_squared_error(v_y, pre), 2)
    coef, p_value = stats.pearsonr(v_y, pre)
    df = pd.DataFrame(columns=['zid', 'MSR', 'correlation'])
    df['zid'] = ['z5241723']
    df['MSR'] = msr
    df['correlation'] = round(coef, 2)
    df.to_csv('z5241723.PART1.summary.csv', index=False)


def training_and_predict_part_2(t_x, t_y, v_x):
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(t_x, t_y)
    prediction = knn.predict(v_x)
    return prediction


def output_part_2(pre, v_df, v_y):
    # output.csv
    df = pd.DataFrame(columns=['movie_id', 'predicted_rating'])
    df['movie_id'] = v_df['movie_id']
    df['predicted_rating'] = np.around(pre)
    df.to_csv('z5241723.PART2.output.csv', index=False)
    # summary.csv
    avg_pre = round(precision_score(v_y, pre, average=None).mean(), 2)
    avg_re = round(recall_score(v_y, pre, average=None).mean(), 2)
    accuracy = round(accuracy_score(v_y, pre), 2)
    df = pd.DataFrame(columns=['zid', 'average_precision', 'average_recall', 'accuracy'])
    df['zid'] = ['z5241723']
    df['average_precision'] = avg_pre
    df['average_recall'] = avg_re
    df['accuracy'] = accuracy
    df.to_csv('z5241723.PART2.summary.csv', index=False)


def draw_data(df, column):
    df_tmp = df.copy()
    if column == 'revenue':
        df_tmp[column] = (df_tmp[column] - df_tmp[column].min()) / (
                    df_tmp[column].max() - df_tmp[column].min())
    n = len(df_tmp.columns)
    col = list(df_tmp.columns)
    plt.figure(figsize=(40, 40), dpi=100)
    for i in range(1, n + 1):
        plt.subplot(4, 3, i)
        plt.scatter(df_tmp[col[i - 1]], df_tmp[column])
        plt.xlabel(col[i - 1])
    plt.savefig("part2.png")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit("Please input paths of the training data and validation data!")
    path_train = sys.argv[1]
    path_validation = sys.argv[2]
    # path_train = 'training.csv'
    # path_validation = 'validation.csv'
    try:
        train_df = pd.read_csv(path_train)
        validation_df = pd.read_csv(path_validation)
    except:
        sys.exit("Invalid csv file path!")

    ##################################### PART-1 ######################################
    train_df_r = data_process_part_1(train_df)
    validation_df_r = data_process_part_1(validation_df)
    train_x_r = train_df_r.drop(['revenue'], axis=1).values
    train_y_r = train_df_r['revenue'].values
    validation_x_r = validation_df_r.drop(['revenue'], axis=1).values
    validation_y_r = validation_df_r['revenue'].values
    prediction = training_and_predict_part_1(train_x_r, train_y_r, validation_x_r)
    output_part_1(prediction, validation_df, validation_y_r)

    ###################################### PART-2 ######################################
    train_df_c = data_process_part_2(train_df)
    validation_df_c = data_process_part_2(validation_df)
    train_x_c = train_df_c.drop(['rating'], axis=1).values
    train_y_c = train_df_c['rating'].values
    validation_x_c = validation_df_c.drop(['rating'], axis=1).values
    validation_y_c = validation_df_c['rating'].values
    pred = training_and_predict_part_2(train_x_c, train_y_c, validation_x_c)
    output_part_2(pred, validation_df, validation_y_c)

