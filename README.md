# My_project
step1
import numpy as np # linear algebra
import pandas as pd


step2
movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')


step3
movies.head(2)


step4
movies.shape

step5

credits.head()


step6
movies = movies.merge(credits,on='title')

step7

movies.head()

step8

movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]


step9
movies.head()

step10
import ast

step11

def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name']) 
    return L 
	
step12
movies.dropna(inplace=True)


step13
movies['genres'] = movies['genres'].apply(convert)
movies.head()


step14

movies['keywords'] = movies['keywords'].apply(convert)
movies.head()

step15

import ast
ast.literal_eval('[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]')

step16

def convert3(text):
    L = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter < 3:
            L.append(i['name'])
        counter+=1
    return L 
	
step17

movies['cast'] = movies['cast'].apply(convert)
movies.head()


step18

movies['cast'] = movies['cast'].apply(lambda x:x[0:3])


step19
def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
    return L 
	
step20
movies['crew'] = movies['crew'].apply(fetch_director)


step21
#movies['overview'] = movies['overview'].apply(lambda x:x.split())
movies.sample(5)

step22

def collapse(L):
    L1 = []
    for i in L:
        L1.append(i.replace(" ",""))
    return L1
	
	
step23
movies['cast'] = movies['cast'].apply(collapse)
movies['crew'] = movies['crew'].apply(collapse)
movies['genres'] = movies['genres'].apply(collapse)
movies['keywords'] = movies['keywords'].apply(collapse)


step24
movies.head()

step25

movies['overview'] = movies['overview'].apply(lambda x:x.split())

step26
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

step27
new = movies.drop(columns=['overview','genres','keywords','cast','crew'])
#new.head()

step28

new['tags'] = new['tags'].apply(lambda x: " ".join(x))
new.head()


step29

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')

step30
vector = cv.fit_transform(new['tags']).toarray()

step31
vector.shape

step31
from sklearn.metrics.pairwise import cosine_similarity

step32
similarity = cosine_similarity(vector)

step33
similarity

step34

new[new['title'] == 'The Lego Movie'].index[0]


step35

def recommend(movie):
    index = new[new['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])),reverse=True,key = lambda x: x[1])
    for i in distances[1:6]:
        print(new.iloc[i[0]].title)
step36

recommend('Gandhi')

step37

import pickle

step38
pickle.dump(new,open('movie_list.pkl','wb'))
pickle.dump(similarity,open('similarity.pkl','wb'))



****************************************frontend********************
import pickle
import streamlit as st
import requests

def fetch_poster(movie_id):
    url = "https://api.themoviedb.org/3/movie/{}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US".format(movie_id)
    data = requests.get(url)
    data = data.json()
    poster_path = data['poster_path']
    full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
    return full_path

def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_movie_names = []
    recommended_movie_posters = []
    for i in distances[1:6]:
        # fetch the movie poster
        movie_id = movies.iloc[i[0]].movie_id
        recommended_movie_posters.append(fetch_poster(movie_id))
        recommended_movie_names.append(movies.iloc[i[0]].title)

    return recommended_movie_names,recommended_movie_posters


st.header('watch your favorite movie')
movies = pickle.load(open('movie_list.pkl','rb'))
similarity = pickle.load(open('similarity.pkl','rb'))

movie_list = movies['title'].values
selected_movie = st.selectbox(
    "Type or select a movie",
    movie_list
)

if st.button('Show Recommendation'):
    recommended_movie_names,recommended_movie_posters = recommend(selected_movie)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.text(recommended_movie_names[0])
        st.image(recommended_movie_posters[0])
    with col2:
        st.text(recommended_movie_names[1])
        st.image(recommended_movie_posters[1])

    with col3:
        st.text(recommended_movie_names[2])
        st.image(recommended_movie_posters[2])
    with col4:
        st.text(recommended_movie_names[3])
        st.image(recommended_movie_posters[3])
    with col5:
        st.text(recommended_movie_names[4])
        st.image(recommended_movie_posters[4])







		
		
https://streamlit.io/cloud
 PS E:\machine project> pip install streamlit
PS E:\machine project> streamlit run ram.py





