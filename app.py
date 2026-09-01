import streamlit as st
import pandas as pd
import pickle

from recomdation import similarity

movies_dict = pickle.load(open('movies_dict.pkl', 'rb'))
movies = pd.DataFrame(movies_dict)
similarity=pickle.load(open('similarity.pkl', 'rb'))
def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    recommended_movie =[]
    for i in movies_list:
        recommended_movie.append(movies.iloc[i[0]].title)
    return recommended_movie


st.title('movies recommendation system')
selected_movie_name=st.selectbox(
    "Enter movies name",
    movies['title'].values
)
if st.button('RECOMMEND'):
    recommdatations = recommend(selected_movie_name)
    for i in recommdatations:
        st.write(i)
