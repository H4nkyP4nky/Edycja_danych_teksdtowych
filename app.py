import html
import os
import re

import altair as alt
import pandas as pd
import requests
import streamlit as st
from transformers import pipeline


TMDB_API_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_URL = "https://image.tmdb.org/t/p/w342"
TMDB_HEADERS = {"Accept": "application/json"}
REVIEWS_PER_PAGE = 20
SENTIMENT_MODEL = "nlptown/bert-base-multilingual-uncased-sentiment"

st.set_page_config(
    page_title="Analiza recenzji filmowych",
    page_icon="🎬",
    layout="wide",
)
st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"],
    [data-testid="stHeader"] {
        background-color: #e5e7eb;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("Analiza sentymentu recenzji filmowych")
st.write(
    "Wyszukaj film, pobierz anglojęzyczne recenzje z TMDB "
    "i przeanalizuj ich sentyment gotowym modelem przeznaczonym do recenzji."
)


def get_configured_token():
    token = os.getenv("TMDB_API_READ_ACCESS_TOKEN", "").strip()
    if token:
        return token
    try:
        return st.secrets.get("TMDB_API_READ_ACCESS_TOKEN", "").strip()
    except (FileNotFoundError, KeyError):
        return ""


configured_token = get_configured_token()
if configured_token:
    access_token = configured_token
    st.success("Token TMDB jest skonfigurowany dla aplikacji.")
else:
    st.info(
        "Do korzystania z API potrzebny jest bezpłatny TMDB API Read Access Token. "
        "Możesz go wpisać poniżej albo skonfigurować raz jako sekret aplikacji."
    )
    access_token = st.text_input(
        "TMDB API Read Access Token",
        type="password",
        help=(
            "Token jest używany tylko podczas działania aplikacji. "
            "Nie wklejaj go do kodu ani nie udostępniaj publicznie."
        ),
    )


@st.cache_resource
def get_analyzer():
    return pipeline(
        "sentiment-analysis",
        model=SENTIMENT_MODEL,
        tokenizer=SENTIMENT_MODEL,
        device=-1,
    )


def clean_review(text):
    text = re.sub(r"<br\s*/?>|</p\s*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def tmdb_get(path, params, token):
    response = requests.get(
        f"{TMDB_API_URL}{path}",
        params=params,
        headers={
            **TMDB_HEADERS,
            "Authorization": f"Bearer {token}",
        },
        timeout=20,
    )
    if response.status_code == 401:
        raise ValueError(
            "TMDB odrzuciło token. Sprawdź, czy używasz API Read Access Token "
            "z ustawień API konta TMDB."
        )
    if response.status_code == 403:
        raise ValueError(
            "TMDB odmówiło dostępu. Sprawdź, czy token API jest aktywny."
        )
    if response.status_code == 429:
        raise ValueError(
            "TMDB ograniczyło liczbę zapytań. Odczekaj chwilę i spróbuj ponownie."
        )
    response.raise_for_status()
    try:
        data = response.json()
    except requests.exceptions.JSONDecodeError as error:
        raise ValueError("TMDB zwróciło nieprawidłową odpowiedź zamiast JSON.") from error
    if not isinstance(data, dict):
        raise ValueError("TMDB zwróciło dane w nieoczekiwanym formacie.")
    return data


def search_movies(query, token):
    data = tmdb_get(
        "/search/movie",
        {"query": query, "language": "en-US", "include_adult": "false"},
        token,
    )
    results = data.get("results")
    if not isinstance(results, list):
        raise ValueError("TMDB zwróciło odpowiedź bez listy filmów.")
    return [
        movie
        for movie in results
        if isinstance(movie, dict)
        and isinstance(movie.get("id"), int)
        and isinstance(movie.get("title"), str)
    ]


def fetch_reviews(movie_id, limit, token):
    reviews = []
    max_pages = (limit + REVIEWS_PER_PAGE - 1) // REVIEWS_PER_PAGE

    for page in range(1, max_pages + 1):
        data = tmdb_get(
            f"/movie/{movie_id}/reviews",
            {"language": "en-US", "page": page},
            token,
        )
        page_reviews = data.get("results")
        if not isinstance(page_reviews, list):
            raise ValueError("TMDB zwróciło odpowiedź bez listy recenzji.")
        reviews.extend(page_reviews)

        total_pages = data.get("total_pages", page)
        if not isinstance(total_pages, int) or page >= total_pages:
            break
        if len(reviews) >= limit:
            break

    return reviews[:limit]


def fetch_director(movie_id, token):
    data = tmdb_get(
        f"/movie/{movie_id}/credits",
        {"language": "en-US"},
        token,
    )
    crew = data.get("crew")
    if not isinstance(crew, list):
        raise ValueError("TMDB zwróciło dane ekipy filmowej w nieoczekiwanym formacie.")
    return next(
        (
            person["name"]
            for person in crew
            if isinstance(person, dict)
            and person.get("job") == "Director"
            and isinstance(person.get("name"), str)
        ),
        None,
    )


def analyze_reviews(reviews):
    texts = []
    valid_reviews = []
    for review in reviews:
        if not isinstance(review, dict):
            continue
        raw_text = review.get("content")
        if not isinstance(raw_text, str):
            continue
        text = clean_review(raw_text)
        if not text:
            continue
        texts.append(text)
        valid_reviews.append(review)

    if not texts:
        return []

    predictions = get_analyzer()(
        texts,
        truncation=True,
        max_length=512,
        batch_size=8,
    )
    analyzed = []
    for text, review, prediction in zip(texts, valid_reviews, predictions):
        match = re.search(r"\b([1-5])\s*stars?\b", prediction["label"], re.I)
        if not match:
            raise ValueError(
                f"Model zwrócił nieoczekiwaną etykietę: {prediction['label']}"
            )
        predicted_stars = int(match.group(1))
        sentiment = (
            "pozytywny"
            if predicted_stars >= 4
            else "negatywny"
            if predicted_stars <= 2
            else "neutralny"
        )
        author_details = review.get("author_details")
        if not isinstance(author_details, dict):
            author_details = {}
        analyzed.append({
            "text": text,
            "sentiment": sentiment,
            "predicted_stars": predicted_stars,
            "confidence": prediction["score"],
            "author": review.get("author") or "nieznany",
            "rating": author_details.get("rating"),
            "created_at": review.get("created_at"),
            "url": review.get("url"),
        })
    return analyzed


def average_author_rating(reviews):
    ratings = []
    for review in reviews:
        if not isinstance(review, dict):
            continue
        author_details = review.get("author_details")
        if not isinstance(author_details, dict):
            continue
        rating = author_details.get("rating")
        if isinstance(rating, (int, float)) and not isinstance(rating, bool):
            ratings.append(float(rating))
    if not ratings:
        return None
    return sum(ratings) / len(ratings), len(ratings)


with st.form("movie_search"):
    movie_query = st.text_input(
        "Tytuł filmu",
        placeholder="Np. Inception",
        help="Wyszukiwanie filmów w katalogu TMDB.",
    )
    search_submitted = st.form_submit_button("Szukaj filmu")

if search_submitted:
    if not access_token.strip():
        st.warning("Wpisz TMDB API Read Access Token przed wyszukaniem filmu.")
    elif len(movie_query.strip()) < 2:
        st.warning("Wpisz co najmniej dwa znaki.")
    else:
        st.session_state.pop("movies", None)
        st.session_state.pop("analysis", None)
        try:
            st.session_state["movies"] = search_movies(
                movie_query.strip(), access_token.strip()
            )
            if not st.session_state["movies"]:
                st.info("Nie znaleziono filmu. Spróbuj innego tytułu.")
        except (requests.RequestException, ValueError) as error:
            st.error(f"Nie udało się wyszukać filmu w TMDB: {error}")

movies = st.session_state.get("movies", [])
if movies:
    movie_options = {}
    for movie in movies:
        year = movie.get("release_date", "")[:4]
        label = f"{movie['title']} ({year})" if year else movie["title"]
        movie_options[f"{label} · TMDB {movie['id']}"] = movie

    selected_label = st.selectbox("Wybierz film", list(movie_options))
    selected_movie = movie_options[selected_label]
    review_limit = st.select_slider(
        "Maksymalna liczba recenzji do pobrania",
        options=[20, 40, 60, 100],
        value=40,
    )

    if st.button("Pobierz i analizuj recenzje", type="primary"):
        if not access_token.strip():
            st.warning("Wpisz TMDB API Read Access Token, aby pobrać recenzje.")
        else:
            st.session_state.pop("analysis", None)
            try:
                with st.spinner(
                    "Pobieranie recenzji i analiza modelem "
                    "od recenzji (pierwsze uruchomienie może potrwać)..."
                ):
                    reviews = fetch_reviews(
                        selected_movie["id"],
                        review_limit,
                        access_token.strip(),
                    )
                    st.session_state["analysis"] = analyze_reviews(reviews)
                st.session_state["fetched_count"] = len(reviews)
                st.session_state["selected_movie"] = selected_movie
                st.session_state["average_author_rating"] = (
                    average_author_rating(reviews)
                )
                try:
                    st.session_state["director"] = fetch_director(
                        selected_movie["id"],
                        access_token.strip(),
                    )
                    st.session_state.pop("director_error", None)
                except (requests.RequestException, ValueError) as error:
                    st.session_state["director"] = None
                    st.session_state["director_error"] = str(error)
            except (
                requests.RequestException,
                ValueError,
                OSError,
                RuntimeError,
            ) as error:
                st.error(
                    "Nie udało się pobrać recenzji lub przeprowadzić analizy: "
                    f"{error}"
                )

analysis = st.session_state.get("analysis")
if analysis is not None:
    selected_movie = st.session_state["selected_movie"]
    average_rating = st.session_state.get("average_author_rating")
    movie_heading = f"Recenzje filmu: {selected_movie['title']}"
    if average_rating is not None:
        average_value, rated_review_count = average_rating
        movie_heading += (
            f" · średnia ocen autorów: {average_value:.1f}/10"
            f" ({rated_review_count} ocen)"
        )
    poster_path = selected_movie.get("poster_path")
    poster_column, details_column = st.columns([1, 4], gap="large")
    with poster_column:
        if isinstance(poster_path, str) and poster_path.startswith("/"):
            st.image(
                f"{TMDB_IMAGE_URL}{poster_path}",
                width=220,
                caption="Plakat filmu · TMDB",
            )
        else:
            st.markdown(
                """
                <div style="
                    width: 220px;
                    height: 320px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    border: 1px solid #cbd5e1;
                    border-radius: 12px;
                    background: rgba(255, 255, 255, 0.55);
                    color: #64748b;
                    text-align: center;
                ">Brak plakatu dla tego filmu</div>
                """,
                unsafe_allow_html=True,
            )
    with details_column:
        st.subheader(movie_heading)
        st.markdown(
            "[Źródło danych: TMDB API]"
            "(https://developer.themoviedb.org/docs)"
        )
        st.caption(
            f"Pobrano {st.session_state['fetched_count']} recenzji; "
            f"przeanalizowano {len(analysis)} niepustych tekstów. "
            "Wyniki dotyczą recenzji dostępnych w języku angielskim."
        )
        director = st.session_state.get("director")
        if director:
            st.markdown(f"**Reżyser:** {director}")
        director_error = st.session_state.get("director_error")
        if director_error:
            st.warning(f"Nie udało się pobrać nazwiska reżysera: {director_error}")
        overview = selected_movie.get("overview")
        if isinstance(overview, str) and overview.strip():
            st.markdown(f"**Opis filmu:** {overview.strip()}")

    if not analysis:
        st.info(
            "TMDB nie zwróciło recenzji w języku angielskim dla tego filmu."
        )
    else:
        counts = {
            label: sum(item["sentiment"] == label for item in analysis)
            for label in ("negatywny", "neutralny", "pozytywny")
        }
        sentiment_colors = {
            "negatywny": "#dc2626",
            "neutralny": "#ca8a04",
            "pozytywny": "#16a34a",
        }
        columns = st.columns(3)
        for column, (label, count) in zip(columns, counts.items()):
            color = sentiment_colors[label]
            percentage = count / len(analysis)
            column.markdown(
                f"""
                <div style="
                    box-sizing: border-box;
                    min-height: 128px;
                    padding: 16px 18px;
                    border: 1px solid {color}55;
                    border-left: 6px solid {color};
                    border-radius: 10px;
                    background: rgba(255, 255, 255, 0.7);
                    color: {color};
                ">
                    <div style="font-size: 0.95rem; font-weight: 600;">{label}</div>
                    <div style="font-size: 2.2rem; font-weight: 700; line-height: 1.3;">{count}</div>
                    <div style="font-size: 0.9rem;">{percentage:.0%} recenzji</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        chart_data = pd.DataFrame(
            {
                "Sentyment": list(counts),
                "Liczba recenzji": list(counts.values()),
            }
        )
        sentiment_chart = (
            alt.Chart(chart_data)
            .mark_bar(size=52, cornerRadiusEnd=5)
            .encode(
                x=alt.X(
                    "Sentyment:N",
                    sort=list(counts),
                    title=None,
                    axis=alt.Axis(labelAngle=0, labelFontSize=14),
                    scale=alt.Scale(paddingInner=0.45, paddingOuter=0.3),
                ),
                y=alt.Y(
                    "Liczba recenzji:Q",
                    title="Liczba recenzji",
                    scale=alt.Scale(domainMin=0, nice=True),
                ),
                color=alt.Color(
                    "Sentyment:N",
                    scale=alt.Scale(
                        domain=["negatywny", "neutralny", "pozytywny"],
                        range=["#dc2626", "#eab308", "#16a34a"],
                    ),
                    legend=None,
                ),
                tooltip=["Sentyment:N", "Liczba recenzji:Q"],
            )
            .properties(height=280)
        )
        st.write("")
        with st.container(border=True):
            st.markdown(
                "<h4 style='text-align: center; margin: 0 0 12px;'>"
                "Podsumowanie sentymentu</h4>",
                unsafe_allow_html=True,
            )
            chart_columns = st.columns([1, 8, 1])
            chart_columns[1].altair_chart(
                sentiment_chart,
                width="stretch",
            )
        st.subheader("Przeanalizowane recenzje")
        for index, item in enumerate(analysis, start=1):
            rating = (
                f" · ocena autora: {item['rating']}/10"
                if isinstance(item["rating"], (int, float))
                else ""
            )
            with st.expander(
                f"{index}. {item['sentiment']} · "
                f"model: {item['predicted_stars']}/5 gwiazdek · "
                f"{item['author']}{rating}"
            ):
                st.write(item["text"])
                st.caption(
                    f"Pewność modelu dla przewidywanej liczby gwiazdek: "
                    f"{item['confidence']:.0%}. Ocena autora jest osobną "
                    "wartością i nie wpływa na przewidywanie modelu."
                )
                if item["created_at"]:
                    st.caption(f"Data recenzji: {item['created_at']}")
                if item["url"]:
                    st.link_button("Otwórz recenzję w TMDB", item["url"])

st.divider()
st.caption(
    "This product uses the TMDB API but is not endorsed or certified by TMDB. "
    f"Model {SENTIMENT_MODEL} przewiduje ocenę recenzji w skali 1–5 na "
    "podstawie jej tekstu: 1–2 oznacza sentyment negatywny, 3 neutralny, "
    "4–5 pozytywny. Wynik może różnić się od oceny autora."
)
