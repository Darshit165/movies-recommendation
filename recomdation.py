import numpy as np
import pandas as pd
moveis =  pd.read_csv(r"F:\project\movie recommdation\dataset\tmdb_5000_movies.csv")
credits = pd.read_csv(r"F:\project\movie recommdation\dataset\tmdb_5000_credits.csv")
print(moveis.head())
print(credits.head())
print(moveis.shape)
print(credits.shape)