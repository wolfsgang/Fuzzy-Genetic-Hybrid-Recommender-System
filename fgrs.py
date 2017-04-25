import load_data
import numpy as np
from fuzzy_sets import Age, GIM
import gim
import pandas as pd


# Load data form MovieLens data


def eucledian(A, B):
    return np.linalg.norm(A - B)


def fuzzy_dist(a, b, A, B):
    return abs(a - b) * eucledian(A, B)


# print fuzzy_dist(35, 40, np.array([3, 4 ,5]), np.array([4, 5, 6]))
# Data Details
# users_cols ='user_id', 'age', 'sex', 'occupation', 'zip_code'
# ratings_cols = 'user_id', 'movie_id', 'rating', 'unix_timestamp'
# i_cols = ['movie_id', 'movie_title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
# 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
# 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
# All in one DataFrame
# mr_ur = pd.merge(users, ratings, on='user_id')
# df = pd.merge(df1, items, on='movie_id')

# Users who has rated movies atleast 60 movies
top_users = load_data.df.groupby('user_id').size().sort_values(ascending=False)[:497]
m_cols = ['unknown', 'Action', 'Adventure',
          'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
          'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western', 'user_id',
          'age']
model_data_au = pd.DataFrame(m_cols)
# Users who has rated movies atleast 60 movies

# Making fuzzy sets for age 13 to 69
age_columns = ['age', 'age_young', 'age_middle', 'age_old']
tu_fuzzy_ages = pd.DataFrame(columns=age_columns)
a = Age()
j = 0
# Our user have age in between 13 and 69
for i in range(13, 70):
    x = [i, a.young(i), a.middle(i), a.old(i)]
    tu_fuzzy_ages.loc[j] = x
    j = j + 1

for i in range(0, 5):
    # active_users and passive_users - pd.Series()
    active_users = top_users.sample(frac=0.10)

    # 34 - 66 split for a user
    training_active_users = active_users.sample(frac=0.34)
    testing_active_users = active_users.drop(training_active_users.index)

    passive_users = top_users.drop(active_users.index)

    tau_data = load_data.df.loc[load_data.df['user_id'].isin(training_active_users)][:10]
    for key, value in tau_data.iterrows():
        user_ui_movies = load_data.df.loc[load_data.df['user_id'] == value['user_id']]
        # print user_ui_movies.shape
        feature_array = []
        feature_array = gim.gim_final(user_ui_movies, value['user_id'])
        #print 'GIM array', feature_array

        feature_array[19], feature_array[20] = value['user_id'], value['age']
        #print feature_array.shape

        feature_row = pd.DataFrame(feature_array,m_cols)
        model_data_au.append(feature_row)