import pandas as pd
import numpy as np

# Для работы с матрицами
from scipy.sparse import csr_matrix

# Матричная факторизация
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import ItemItemRecommender  # нужен для одного трюка
from implicit.nearest_neighbours import bm25_weight, tfidf_weight


class MainRecommender:
    """Рекоммендации, которые можно получить из ALS

    Input
    -----
    user_item_matrix: pd.DataFrame
        Матрица взаимодействий user-item
    """

    def __init__(self, data, N=5, weighting=True):
        # your_code. Это не обязательная часть. Но если вам удобно что-либо посчитать тут - можно это сделать

        self.data = data
        self.user_item_matrix = self.prepare_matrix(data)  # pd.DataFrame
        self.id_to_itemid, self.id_to_userid, \
            self.itemid_to_id, self.userid_to_id = self.prepare_dicts()
        self.popularity = self.get_n_popularity_item(N)

        if weighting:
            self.user_item_matrix = bm25_weight(self.user_item_matrix.T).T

        self.model = self.fit(self.user_item_matrix)
        self.own_recommender = self.fit_own_recommender(self.user_item_matrix)

    @staticmethod
    def prepare_matrix(data):
        """Создаем user-item матрицу"""

        user_item_matrix = pd.pivot_table(data,
                                          index='user_id', columns='item_id',
                                          values='quantity',  # Можно пробовать другие варианты
                                          aggfunc='count',
                                          fill_value=0
                                          )

        user_item_matrix = user_item_matrix.astype(float)  # необходимый тип матрицы для implicit

        return user_item_matrix

    def prepare_dicts(self):
        """Подготавливает вспомогательные словари"""

        self.userids = self.user_item_matrix.index.values
        self.itemids = self.user_item_matrix.columns.values

        self.matrix_userids = np.arange(len(self.userids))
        self.matrix_itemids = np.arange(len(self.itemids))

        self.id_to_itemid = dict(zip(self.matrix_itemids, self.itemids))
        self.id_to_userid = dict(zip(self.matrix_userids, self.userids))

        self.itemid_to_id = dict(zip(self.itemids, self.matrix_itemids))
        self.userid_to_id = dict(zip(self.userids, self.matrix_userids))

        return self.id_to_itemid, self.id_to_userid, self.itemid_to_id, self.userid_to_id

    @staticmethod
    def fit_own_recommender(user_item_matrix):
        """Обучает модель, которая рекомендует товары, среди товаров, купленных юзером"""

        own_recommender = ItemItemRecommender(K=1, num_threads=4)
        own_recommender.fit(csr_matrix(user_item_matrix).T.tocsr())

        return own_recommender

    def fit(self, n_factors=20, regularization=0.001, iterations=15, num_threads=4):
        """Обучает ALS"""

        model = AlternatingLeastSquares(factors=n_factors,
                                        regularization=regularization,
                                        iterations=iterations,
                                        num_threads=num_threads)
        model.fit(csr_matrix(self.user_item_matrix).T.tocsr())

        return model

    def get_n_popularity_item(self, N=5):
        """Берем топ-N товаров юзеров"""

        popularity = self.data.groupby(['user_id', 'item_id'])['quantity'].count().reset_index()
        popularity.sort_values('quantity', ascending=False, inplace=True)

        popularity = popularity[popularity['item_id'] != 999999]

        popularity = popularity.groupby('user_id').head(N)

        popularity.sort_values(by=['user_id', 'quantity'], ascending=False, inplace=True)

        return popularity

    def get_rec_similar_items(self, x):
        """Берем товар наиболее похожий на целевой"""

        recs = self.model.similar_items(self.itemid_to_id[x], N=2)
        top_rec = recs[1][0]

        return self.id_to_itemid[top_rec]

    def get_similar_items_recommendation(self, user, N=5):
        """Рекомендуем товары, похожие на топ-N купленных юзером товаров"""

        popularity = self.popularity.loc[self.popularity['user_id'] == user]
        popularity['similar_recommendation_bpr'] = popularity['item_id'].apply(lambda x: self.get_rec_similar_items(x))

        recommendation_similar_items = popularity.groupby('user_id')['similar_recommendation_bpr']. \
        unique().reset_index()
        recommendation_similar_items.columns = ['user_id', 'similar_recommendation_bpr']
        res = recommendation_similar_items['similar_recommendation_bpr'].loc[
            recommendation_similar_items['user_id'] == 1].tolist()[0]
        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res


def get_rec_similar_users(self, x, N=16, treshold=0.45):
    """Получаем список юзеров, похожих на целевого юзера"""

    simil_usr = self.model.similar_users(self.userid_to_id[x], N)[1:]
    simil_usr_list = np.array([list(x) for x in simil_usr if x[1] > treshold])[:, 0].astype('int').tolist()

    return [self.id_to_userid[x] for x in simil_usr_list]


def get_list_items_simil_usrs(self, x, N=16, treshold=0.45):
    """Получаем и сортируем по попуярности товары, купленные похожими юзерами"""

    simil_usr_list = self.get_rec_similar_users(x, N, treshold)
    list_items = self.data[self.data['user_id'].isin(simil_usr_list)]
    list_items = list_items.groupby(['item_id'])['quantity'].count().reset_index()
    list_items.sort_values('quantity', ascending=False, inplace=True)

    return list_items


def get_similar_users_recommendation(self, user, N=5, n_users=16, treshold=0.45):
    """Рекомендуем топ-N товаров, среди купленных похожими юзерами"""

    popularity_items_users = self.get_list_items_simil_usrs(user, n_users, treshold)
    res = popularity_items_users['item_id'].head(N).values

    assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
    return res
