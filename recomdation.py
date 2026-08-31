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
#remove unwanted columns from the dataset and create tags in our analyse we dnt need numeric colums because we recommend based on the content of the movie so we will remove numeric columns and keep only text based columns
#remove columns [ budget,homepage, original_language,original_title,popularity,pruduction_companies,production_countries,release_date,revenue,runtime,spoken_languages,status,tagline,vote_average,vote_count,movie_id]
#keep columns [genres,id , keywords,title,overview,cast,crew]
movies = movies[['genres','movie_id','keywords','title','overview','cast','crew']]
print(movies.head())
print(movies.shape)
#create a new column called tags and combine [genres, keywords, overview, cast, crew]
#before combining we need to convert the data in these columns into string format because some of the data is in list format so we will convert them into string format and then combine them
#remove missing values and duplicates from the dataset
print(movies.isnull().sum())
movies = movies.dropna()
print(movies.isnull().sum())
print(movies.duplicated().sum())
#clean genre columns with string conversion
print(movies.iloc[0].genres)
#convert to ["action","adventure","fantasy","science fiction"] but problem is full column is in string format so we will use ast.literal_eval to convert it into list format and then we will extract the name of the genre from the list of dictionaries and then we will convert it into string format and then we will combine it with other columns
import ast
def convert(obj):
    l=[]
    for i in ast.literal_eval(obj):
        l.append(i['name'])
    return l
movies['genres'] = movies['genres'].apply(convert)
print(movies["genres"].head())
#convert keywords column in the same way as genres
movies['keywords'] = movies['keywords'].apply(convert)
print(movies["keywords"].head())
#convert cast column in the same way as genres but we only want to keep the first 3 cast members because they are the main actors of the movie and we will convert them into string format and then we will combine them with other columns

print (movies.iloc[0].cast)
def convert_cast(obj):
    l=[]

    counter=0
    for i in ast.literal_eval(obj):
        if counter<3:
            l.append(i['name'])
            counter+=1
        else:
            break
    return l
movies['cast'] = movies['cast'].apply(convert_cast)
print(movies["cast"].head(1))
#convert crew column in the same way as genres but we only want to keep the director of the movie because he is the main person behind the movie and we will convert them into string format and then we will combine them with other columns
def convert_crew(obj):
    l=[]
    for i in ast.literal_eval(obj):
        if i['job']=='Director':
            l.append(i['name'])
            break
    return l
movies['crew'] = movies['crew'].apply(convert_crew)
print(movies["crew"].head(1))
#convert overview column is in string so convert into list of words and then we will combine it with other columns
movies["overview"] = movies["overview"].apply(lambda x: x.split())
print(movies["overview"].head())
#apply trasnformation for removing spaces in the words of genres, keywords, cast and crew columns because we will combine them with other columns and we will use CountVectorizer to convert them into vector format and CountVectorizer will consider space as a separator so we will remove spaces from the words in these columns
movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ","") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ","") for i in x])
movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ","") for i in x])
movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ","") for i in x])
print(movies.head())
#combine all the columns into a new column called tags and convert it into string format
movies['tags'] = movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew'] + movies['overview']
print(movies['tags'].head())
print(movies.head())
#create a new dataframe with only title, movie_id and tags columns
new_movies = movies[[ 'movie_id','title', 'tags']]
print(new_movies.head())
#convert tags column into string format
new_movies['tags'] = new_movies['tags'].apply(lambda x: " ".join(x))
print(new_movies.head())
#convert tags column into lower case
new_movies['tags'] = new_movies['tags'].apply(lambda x: x.lower())
print(new_movies.head())
print(new_movies["tags"][0])
#we have movies id , title and tags columns for build the recommendation system we have to convert the tags column into vector format and then we will use cosine similarity to find the similarity between the movies and then we will recommend the movies based on the similarity score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
#we differnt words with similar meaning like "fight" and "fighting" so we will use stemming to convert them into the same word and then we will convert them into vector format
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

new_movies['tags'] = new_movies['tags'].apply(stem)
print(new_movies['tags'][0]) 

#convert tags column into vector format
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_movies['tags']).toarray()
print(cv.get_feature_names_out())
