import numpy as np
import pandas as pd
from implicit.als import AlternatingLeastSquares

def add_fiction_id(data, take_n_popular):
    popularity_n = data.groupby('item_id')['quantity'].sum().reset_index()
    popularity_n.rename(columns={'quantity': 'n_sold'}, inplace=True)

    top_n = popularity_n.sort_values('n_sold', ascending=False).head(take_n_popular).item_id.tolist()
    # Заведем фиктивный item_id

    data.loc[~data['item_id'].isin(top_n), 'item_id'] = 999999

    return data


def prefilter_items(data, item_features, take_n_popular, category: list):

    data['price'] = data['sales_value'] / (np.maximum(data['quantity'], 1))

    # Уберем самые популярные товары (их и так купят)
    #     может я чего не догоняю, но мне не нравится вариант, что item_id тоже делится, поэтому предлагаю коррекция кода: вместо
    #     popularity = data_train.groupby('item_id')['user_id'].nunique().reset_index() / data_train['user_id'].nunique()
    #     вот это:
    popularity = data.groupby('item_id')['user_id'].nunique().reset_index()
    popularity['user_id'] = popularity['user_id'] / data['user_id'].nunique()
    popularity.rename(columns={'user_id': 'share_unique_users'}, inplace=True)

    top_popular = popularity[popularity['share_unique_users'] > 0.5].item_id.tolist()
    data = data[~data['item_id'].isin(top_popular)]

    # Уберем самые НЕ популярные товары (их и так НЕ купят)
    top_notpopular = popularity[popularity['share_unique_users'] < 0.01].item_id.tolist()
    data = data[~data['item_id'].isin(top_notpopular)]

    # Уберем товары, которые не продавались за последние 12 месяцев
    last_week = data.groupby('item_id')['week_no'].max().reset_index()
    last_week = last_week.loc[last_week['week_no'] < data['week_no'].max() - 52].item_id.tolist()
    data = data[~data['item_id'].isin(last_week)]

    # Уберем не интересные для рекоммендаций категории (department)
    # всвяязи с тем, что не очень понятно какие категории нам не интересны просто добавил передаваемый в функцию
    # аргумент со списком категорий, товары из которых нужно откинуть
    non_interesting_cat_item = item_features['item_id'].loc[item_features['department'].isin(category)].tolist()
    data = data[~data['item_id'].isin(non_interesting_cat_item)]

    # Уберем слишком дешевые товары (на них не заработаем). 1 покупка из рассылок стоит 60 руб.
    # по курсу на 2 янв 60 руб = 0.8 usd
    price_ = data.groupby('item_id')['price'].max().reset_index()
    low_price = price_.loc[price_['price'] < 0.8].item_id.tolist()
    data = data[~data['item_id'].isin(low_price)]

    # Уберем слишком дорогие товары
    # пожалуй стоит убрать все, что дороже 80 usd (аккурат 6000 руб)
    high_price = price_.loc[price_['price'] > 80].item_id.tolist()
    data = data[~data['item_id'].isin(high_price)]

    data = add_fiction_id(data, take_n_popular)

    return data


def postfilter_items(user_id, recommednations):
    pass
