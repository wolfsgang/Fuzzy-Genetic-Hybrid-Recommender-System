import load_data
import pandas as pd
import numpy as np
from fuzzy_sets import Age, GIM
import gim
import operator

a = Age()
g = GIM()


def eucledian(A, B):
    return np.linalg.norm(np.array(A) - np.array(B))


def dist_func(a, b, A, B):
    return abs(a - b) * eucledian(A, B)


def fuzzy_distance(ui, uj):
    fuzzy_v = [0] * 21
    for i in range(0, 19):
        ui_gim = [g.very_bad(ui[i]), g.bad(ui[i]), g.average(ui[i]), g.good(ui[i]), g.very_good(ui[i]),
                  g.excellent(ui[i])]
        uj_gim = [g.very_bad(uj[i]), g.bad(uj[i]), g.average(uj[i]), g.good(uj[i]), g.very_good(uj[i]),
                  g.excellent(uj[i])]
        fuzzy_v[i] = dist_func(ui[i], uj[i], ui_gim, uj_gim)
    i = 19
    ui_gim = [a.young(ui[i]), a.middle(ui[i]), a.old(ui[i])]
    uj_gim = [a.young(uj[i]), a.middle(uj[i]), a.old(uj[i])]
    fuzzy_v[i] = dist_func(ui[i], uj[i], ui_gim, uj_gim)
    # adding user_id of second user
    fuzzy_v[20] = uj['user_id']
    # print fuzzy_v
    return fuzzy_v


def recommend(neighbours, testing):
    for key, item in testing.iterrows():
        ar,m_id = item['rating'],item['movie_id']
        n_ratings = []
        for i in neighbours:
            temp = df.loc[df['user_id'] == i].loc[df['movie_id'] == m_id]
            for k, it in temp.iterrows():
                n_ratings.append(it['rating'])
        pr = sum(n_ratings) / len(n_ratings) if len(n_ratings) else 0
    return abs(pr - ar)


# print fuzzy_dist(35, 40, np.array([3, 4 ,5]), np.array([4, 5, 6]))
# Data Details
# users_cols ='user_id', 'age', 'sex', 'occupation', 'zip_code'
# ratings_cols = 'user_id', 'movie_id', 'rating', 'unix_timestamp'
# i_cols = ['movie_id', 'movie_title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
# 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
# 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
# All in one DataFrame
mr_ur = pd.merge(load_data.users, load_data.ratings, on='user_id')
df = pd.merge(mr_ur, load_data.items, on='movie_id')
m_cols = ['unknown', 'Action', 'Adventure',
          'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
          'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western', 'age',
          'user_id']
model_data_au = pd.DataFrame(columns=m_cols)
feature_row = pd.DataFrame(columns=m_cols)
model_data_pu = pd.DataFrame(columns=m_cols)
# Users who has rated movies atleast 60 movies
top_users = load_data.df.groupby('user_id').size().sort_values(ascending=False)[:497]

for i in range(0, 5):
    # active_users and passive_users - pd.Series()
    active_users = top_users.sample(frac=0.10)

    # training_active_users = active_users.sample(frac=0.34)
    # testing_active_users = active_users.drop(training_active_users.index)

    passive_users = top_users.drop(active_users.index)
    # To change this # active users
    tau_data = df.loc[df['user_id'].isin(active_users)][:10]
    index = np.arange(0, tau_data.shape[0])
    i = 0
    for key, value in tau_data.iterrows():
        user_ui_movies = df.loc[df['user_id'] == value['user_id']]

        training_user_movies = user_ui_movies.sample(frac=0.34)
        feature_array = gim.gim_final(training_user_movies, value['user_id'])

        # print 'GIM array', feature_array
        feature_array[19], feature_array[20] = value['age'], value['user_id']
        # print feature_array.shape
        model_data_au.loc[i] = feature_array
        i = i + 1
    # print model_data_au
    # print model_data_au
    # Working with passive users
    # change # of passive users

    i = 0
    pu_data = df.loc[df['user_id'].isin(passive_users)][:10]

    for key, value in pu_data.iterrows():
        user_ui_movies = df.loc[df['user_id'] == value['user_id']]
        feature_array_p = gim.gim_final(user_ui_movies, value['user_id'])
        # print 'GIM array', feature_array
        feature_array_p[19], feature_array_p[20] = value['age'], value['user_id']
        # print feature_array.shape
        model_data_pu.loc[i] = feature_array_p
        i = i + 1
    # print model_data_au
    # print model_data_pu

    # fuzzy_df = pd.DataFrame(columns=m_cols)
    fuzzy_vec = []
    error = []
    for key, value in model_data_au.iterrows():
        i = 0
        for key1, value1 in model_data_pu.iterrows():
            fuzzy_vec.append(fuzzy_distance(value, value1))
            # print value[i], value1[i]
            fuzzy_vec[i] = [sum(x * 0.5 for x in fuzzy_vec[i][:-1]), fuzzy_vec[i][-1]]
            i = i + 1
        # print fuzzy_vec
        neighbours = [n[1] for n in sorted(fuzzy_vec, key=operator.itemgetter(0), reverse=True)][:30]  # taking top 30
        # print neighbours
        testing_user = df.loc[df['user_id'] == value['user_id']].sample(frac=0.66)
        e = recommend(neighbours, testing_user)
        print e
        error.append(e)
    MAE = sum(error) / len(error)
    print "MAE is" + MAE
