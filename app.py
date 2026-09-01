import pickle
import streamlit as st
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Movie Recommender System",
    page_icon="🎬",
    layout="wide"
)


# ============================================================
# TMDB API KEY
# ============================================================

TMDB_API_KEY = "779ea3031b6109df9b58f47621a6cb2d"


# ============================================================
# LOAD PKL FILES
# ============================================================

@st.cache_resource
def load_data():

    movies = pickle.load(
        open("movies.pkl", "rb")
    )

    similarity = pickle.load(
        open("similarity.pkl", "rb")
    )

    return movies, similarity


movies, similarity = load_data()


# ============================================================
# CREATE REQUEST SESSION
# ============================================================

@st.cache_resource
def create_session():

    session = requests.Session()

    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )

    adapter = HTTPAdapter(
        max_retries=retry_strategy
    )

    session.mount(
        "https://",
        adapter
    )

    session.mount(
        "http://",
        adapter
    )

    session.headers.update({
        "User-Agent": "Mozilla/5.0"
    })

    return session


session = create_session()


# ============================================================
# FETCH POSTER FROM TMDB
# ============================================================

@st.cache_data(ttl=86400)
def fetch_poster(movie_id):

    url = f"https://api.themoviedb.org/3/movie/{movie_id}"

    params = {
        "api_key": TMDB_API_KEY,
        "language": "en-US"
    }

    try:

        response = session.get(
            url,
            params=params,
            timeout=10
        )

        response.raise_for_status()

        data = response.json()

        poster_path = data.get("poster_path")

        if poster_path:

            return (
                "https://image.tmdb.org/t/p/w500"
                + poster_path
            )

        return None

    except requests.exceptions.RequestException:

        return None

    except Exception:

        return None


# ============================================================
# RECOMMENDATION FUNCTION
# ============================================================

def recommend(movie):

    # Find movie index
    movie_indices = movies[
        movies["title"] == movie
    ].index

    if len(movie_indices) == 0:

        return [], []

    index = movie_indices[0]

    # Get similarity scores
    distances = sorted(
        list(
            enumerate(similarity[index])
        ),
        key=lambda x: x[1],
        reverse=True
    )

    recommended_movie_names = []
    recommended_movie_posters = []
    recommended_movie_scores = []

    # Skip selected movie itself
    for i, score in distances[1:6]:

        movie_name = movies.iloc[i]["title"]

        movie_id = movies.iloc[i]["movie_id"]

        poster = fetch_poster(movie_id)

        recommended_movie_names.append(
            movie_name
        )

        recommended_movie_posters.append(
            poster
        )

        recommended_movie_scores.append(
            score
        )

    return (
        recommended_movie_names,
        recommended_movie_posters,
        recommended_movie_scores
    )


# ============================================================
# USER INTERFACE
# ============================================================

st.title("🎬 Movie Recommender System")

st.write(
    "Select a movie and get five similar movie recommendations."
)


# ============================================================
# MOVIE DROPDOWN
# ============================================================

movie_list = movies["title"].tolist()


selected_movie = st.selectbox(
    "Type or select a movie",
    movie_list
)


# ============================================================
# RECOMMEND BUTTON
# ============================================================

if st.button(
    "🎥 Show Recommendation",
    type="primary"
):

    with st.spinner(
        "Finding similar movies..."
    ):

        result = recommend(
            selected_movie
        )

    recommended_movie_names = result[0]
    recommended_movie_posters = result[1]
    recommended_movie_scores = result[2]


    # ========================================================
    # CHECK RESULT
    # ========================================================

    if not recommended_movie_names:

        st.error(
            "Could not find recommendations for this movie."
        )

    else:

        st.subheader(
            f"Movies similar to: {selected_movie}"
        )


        # ====================================================
        # CREATE 5 COLUMNS
        # ====================================================

        columns = st.columns(5)


        for column, name, poster, score in zip(
            columns,
            recommended_movie_names,
            recommended_movie_posters,
            recommended_movie_scores
        ):

            with column:

                # Movie poster
                if poster:

                    st.image(
                        poster,
                        use_container_width=True
                    )

                else:

                    st.info(
                        "Poster unavailable"
                    )


                # Movie name
                st.markdown(
                    f"**{name}**"
                )


                # Similarity score
                st.caption(
                    f"Similarity: {score:.3f}"
                )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("About")

    st.write(
        """
        This movie recommender uses:

        • Content-Based Filtering  
        • CountVectorizer  
        • Cosine Similarity  
        • TMDB API  

        Select a movie to find the
        five most similar movies.
        """
    )