import numpy as np
import pandas as pd
movies =  pd.read_csv(r"F:\project\movie recommdation\dataset\tmdb_5000_movies.csv")
credits = pd.read_csv(r"F:\project\movie recommdation\dataset\tmdb_5000_credits.csv")
print(movies.head())
print(credits.head())
print(movies.shape)
print(credits.shape)
#merge both dataset base on title feature  name and creat new dataet 
movies = movies.merge(credits, on='title')
print(movies.head())
print(movies.shape)
print(credits.head())
