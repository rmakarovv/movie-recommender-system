import os
import wget

if not os.path.exists('movie-recommender-system/data/raw/ml-100k.zip'):
    url = 'https://files.grouplens.org/datasets/movielens/ml-100k.zip'
    wget.download(url, out='movie-recommender-system/data/raw/ml-100k.zip')
