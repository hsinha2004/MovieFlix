from flask import Flask, render_template, request, redirect, url_for, jsonify, make_response, session, flash
import json
import logging
import numpy as np
from collections import defaultdict
import random
import sqlite3
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps

http_session = requests.Session()
retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
http_session.mount('https://', HTTPAdapter(max_retries=retries))
http_session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})

from functools import lru_cache
from datetime import datetime
import gzip
from io import BytesIO
import time

import os
from dotenv import load_dotenv

basedir = os.path.abspath(os.path.dirname(__file__))
env_path = os.path.join(basedir, '.env')
load_dotenv(env_path)

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "fallback_secret_key")
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 86400
app.config['TEMPLATES_AUTO_RELOAD'] = False

def gzip_response(response):
    accept_encoding = request.headers.get('Accept-Encoding', '')
    if 'gzip' not in accept_encoding.lower():
        return response
    if (response.status_code < 200 or response.status_code >= 300 or 'Content-Encoding' in response.headers):
        return response
    response.direct_passthrough = False
    if response.data:
        gzip_buffer = BytesIO()
        with gzip.GzipFile(mode='wb', fileobj=gzip_buffer) as gzip_file:
            gzip_file.write(response.data)
        response.data = gzip_buffer.getvalue()
        response.headers['Content-Encoding'] = 'gzip'
        response.headers['Content-Length'] = len(response.data)
        response.headers['Vary'] = 'Accept-Encoding'
    return response

@app.after_request
def add_cache_headers(response):
    if response.status_code >= 400:
        return response
    response = gzip_response(response)
    if request.path.startswith('/static/'):
        response.headers['Cache-Control'] = 'public, max-age=86400'
    elif request.path.startswith('/movie/') or request.path == '/':
        response.headers['Cache-Control'] = 'public, max-age=3600'
    response.headers['X-Response-Time'] = f"{time.time() - request.start_time:.4f}s"
    return response

@app.before_request
def before_request():
    request.start_time = time.time()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger('urllib3').setLevel(logging.WARNING)

GENRES = ["Action", "Adventure", "Animation", "Comedy", "Crime",
          "Drama", "Fantasy", "Horror", "Mystery", "Romance",
          "Sci-Fi", "Thriller"]
DB_PATH = os.path.join(basedir, "movieflix.db")
CURRENT_YEAR = datetime.now().year
MIN_RATING = 6.0

TMDB_API_KEY = os.environ.get("TMDB_API_KEY")
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_URL = "https://image.tmdb.org/t/p/w342"

q_table = defaultdict(lambda: defaultdict(lambda: np.zeros(len(GENRES) + 3)))
movie_vectors = {}
movie_cache = {}

@lru_cache(maxsize=1000)
def get_movie_vector_from_genres(genres_tuple):
    genres = list(genres_tuple)
    return np.array([1 if g in genres else 0 for g in GENRES])

def get_movie_vector(movie):
    imdb_id = movie.get('imdb_id') or movie.get('imdbID')
    if imdb_id and imdb_id in movie_vectors:
        return movie_vectors[imdb_id]

    # Fetch full movie details if we only have the ID
    if imdb_id and ('genres' not in movie and 'Genre' not in movie):
        full_movie = get_movie_details(imdb_id)
        if full_movie:
            movie.update(full_movie)

    if not movie:
        return np.zeros(len(GENRES) + 3)

    genres = []
    if 'genres' in movie:
        genres_data = movie.get('genres', [])
        if isinstance(genres_data, str):
            genres = [g.strip() for g in genres_data.split(',') if g.strip()]
        elif isinstance(genres_data, list):
            genres = [genre.get('name', '') if isinstance(genre, dict) else str(genre) for genre in genres_data]
    elif 'Genre' in movie:
        genre_str = movie.get('Genre', '')
        if genre_str and isinstance(genre_str, str):
            genres = [g.strip() for g in genre_str.split(',') if g.strip()]

    # Optimization: Do not call TMDB API during vector precomputation to prevent 60s load time
    if not genres:
        genres = []

    genre_vector = get_movie_vector_from_genres(tuple(genres)) if genres else np.zeros(len(GENRES))

    rating = 0
    if 'imdbRating' in movie and movie['imdbRating'] != 'N/A':
        try:
            rating = float(movie['imdbRating']) / 10.0
        except (ValueError, TypeError):
            pass
    elif 'vote_average' in movie:
        try:
            rating = float(movie['vote_average']) / 10.0
        except (ValueError, TypeError):
            pass

    year = 0
    year_str = movie.get('Year') or (movie.get('release_date', '').split('-')[0] if movie.get('release_date') else 'N/A')
    if year_str != 'N/A':
        try:
            year = (int(year_str) - 1900) / (CURRENT_YEAR - 1900)
        except (ValueError, TypeError):
            pass

    director = 0
    # Optimization: Director affinity is removed from precomputation to save 2.5 mins load time


    vector = np.concatenate([genre_vector, [rating, year, director]])
    if imdb_id:
        movie_vectors[imdb_id] = vector
    return vector

def get_movie_details(imdb_id):
    if imdb_id in movie_cache and movie_cache[imdb_id]:
        return movie_cache[imdb_id]

    # Try DB cache first
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.execute("SELECT details, timestamp FROM movie_cache WHERE imdb_id = ?", (imdb_id,))
            row = cursor.fetchone()
            if row:
                cached_details = json.loads(row[0])
                movie_cache[imdb_id] = cached_details
                return cached_details
    except Exception as e:
        logger.error(f"Error retrieving movie details from DB cache: {e}")

    try:
        movie_id = None
        if imdb_id.startswith('tmdb_'):
            movie_id = imdb_id.replace('tmdb_', '')
        else:
            params = {
                'api_key': TMDB_API_KEY,
                'external_source': 'imdb_id'
            }
            response = http_session.get(f"{TMDB_BASE_URL}/find/{imdb_id}", params=params, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data.get('movie_results'):
                    movie_id = data['movie_results'][0]['id']

        if movie_id:
            details_response = http_session.get(f"{TMDB_BASE_URL}/movie/{movie_id}", params={'api_key': TMDB_API_KEY}, timeout=5)
            if details_response.status_code == 200:
                details = details_response.json()
                # Safe extraction of potentially null fields
                release_date = details.get('release_date') or ''
                year = release_date.split('-')[0] if release_date else 'N/A'
                
                genres_data = details.get('genres') or []
                genre_str = ', '.join(g.get('name', '') for g in genres_data) if genres_data else 'N/A'

                formatted_movie = {
                    'Title': details.get('title') or 'Unknown',
                    'Poster': details.get('poster_path') or 'N/A',
                    'imdbRating': details.get('vote_average') or 'N/A',
                    'Year': year,
                    'Runtime': details.get('runtime') or 'N/A',
                    'Rated': 'R' if details.get('adult') else 'PG',
                    'Plot': details.get('overview') or 'N/A',
                    'Genre': genre_str,
                    'Director': 'N/A',
                    'Actors': 'N/A',
                    'Writer': 'N/A',
                    'Language': details.get('original_language') or 'N/A',
                    'Awards': 'N/A'
                }
                movie_cache[imdb_id] = formatted_movie
                try:
                    with sqlite3.connect(DB_PATH) as conn:
                        current_time = int(datetime.now().timestamp())
                        conn.execute("INSERT OR REPLACE INTO movie_cache (imdb_id, details, timestamp) VALUES (?, ?, ?)",
                                    (imdb_id, json.dumps(formatted_movie), current_time))
                except Exception as e:
                    logger.error(f"Error caching TMDB data in DB: {e}")
                return formatted_movie
    except Exception as e:
        logger.error(f"Failed to fetch movie {imdb_id} from TMDB: {e}")
    return {}

def get_streaming_providers(title, imdb_id=None):
    """Get real streaming providers from TMDB watch/providers API."""
    if not TMDB_API_KEY:
        return []
    try:
        tmdb_id = None
        if imdb_id and imdb_id.startswith('tmdb_'):
            tmdb_id = imdb_id.replace('tmdb_', '')
        
        if not tmdb_id and imdb_id:
            resp = http_session.get(f"{TMDB_BASE_URL}/find/{imdb_id}",
                params={'api_key': TMDB_API_KEY, 'external_source': 'imdb_id'}, timeout=3)
            if resp.status_code == 200:
                results = resp.json().get('movie_results', [])
                if results:
                    tmdb_id = results[0]['id']
        
        if not tmdb_id:
            # Search by title as fallback
            resp = http_session.get(f"{TMDB_BASE_URL}/search/movie",
                params={'api_key': TMDB_API_KEY, 'query': title}, timeout=3)
            if resp.status_code == 200:
                results = resp.json().get('results', [])
                if results:
                    tmdb_id = results[0]['id']
        
        if tmdb_id:
            resp = http_session.get(f"{TMDB_BASE_URL}/movie/{tmdb_id}/watch/providers",
                params={'api_key': TMDB_API_KEY}, timeout=3)
            if resp.status_code == 200:
                data = resp.json().get('results', {})
                # Check IN (India) and US regions
                providers = []
                for region in ['IN', 'US']:
                    region_data = data.get(region, {})
                    for provider_type in ['flatrate', 'free', 'ads']:
                        for p in region_data.get(provider_type, []):
                            name = p.get('provider_name', '')
                            if name and name not in providers:
                                providers.append(name)
                    if providers:
                        break
                return providers[:5]
    except Exception as e:
        logger.debug(f"Could not fetch streaming providers for {title}: {e}")
    return []

formatted_movie_cache = {}

def format_movie(movie):
    if not movie or not isinstance(movie, dict):
        return None

    imdb_id = movie.get('imdbID') or movie.get('imdb_id')
    if imdb_id and imdb_id in formatted_movie_cache:
        return formatted_movie_cache[imdb_id]

    title = movie.get('Title') or movie.get('title')
    if not imdb_id or not title:
        logger.error(f"Missing critical movie data: {movie}")
        return None

    try:
        poster_path = movie.get('Poster') or movie.get('poster_path', 'N/A')
        genres = []
        if 'genres' in movie:
            genres_data = movie.get('genres', [])
            if isinstance(genres_data, str):
                genres = [g.strip() for g in genres_data.split(',') if g.strip()]
            elif isinstance(genres_data, list):
                genres = [genre.get('name', '') if isinstance(genre, dict) else str(genre) for genre in genres_data]
        elif 'Genre' in movie:
            genre_str = movie.get('Genre', '')
            if genre_str and isinstance(genre_str, str):
                genres = [g.strip() for g in genre_str.split(',') if g.strip()]

        if poster_path != 'N/A' and not poster_path.startswith(('http://', 'https://')):
            if poster_path.startswith('/'):
                poster_path = "https://image.tmdb.org/t/p/w342" + poster_path
            else:
                poster_path = "https://image.tmdb.org/t/p/w342/" + poster_path

        # Extract rating directly from movie data - avoid API call
        rating_val = 'N/A'
        if 'imdbRating' in movie and movie['imdbRating'] != 'N/A':
            try:
                rating_val = str(float(movie['imdbRating']))
            except (ValueError, TypeError):
                pass
        elif 'vote_average' in movie and movie['vote_average']:
            try:
                rating_val = str(float(movie['vote_average']))
            except (ValueError, TypeError):
                pass

        formatted = {
            'title': title,
            'imdb_id': imdb_id,
            'poster': poster_path,
            'rating': rating_val,
            'year': movie.get('Year') or (movie.get('release_date', '').split('-')[0] if movie.get('release_date') else 'N/A'),
            'streaming': [],
            'genres': genres
        }

        # Only call API if poster is truly missing - rating already extracted above
        if formatted['poster'] == 'N/A':
            details = get_movie_details(imdb_id)
            if details and 'Poster' in details and details['Poster'] != 'N/A':
                poster = details['Poster']
                if poster and not poster.startswith(('http://', 'https://')):
                    if poster.startswith('/'):
                        poster = "https://image.tmdb.org/t/p/w342" + poster
                    else:
                        poster = "https://image.tmdb.org/t/p/w342/" + poster
                formatted['poster'] = poster
            if details and formatted['rating'] == 'N/A':
                if 'imdbRating' in details and details['imdbRating'] != 'N/A':
                    try:
                        formatted['rating'] = str(float(details['imdbRating']))
                    except (ValueError, TypeError):
                        pass

        # Streaming providers are fetched on-demand on the movie details page only
        # to avoid slow API calls during home page rendering

        if imdb_id:
            formatted_movie_cache[imdb_id] = formatted
        return formatted
    except Exception as e:
        logger.error(f"Error in format_movie for {imdb_id}: {e}")
        return None

def load_q_table():
    expected_shape = (len(GENRES) + 3,)
    q_values = defaultdict(lambda: defaultdict(lambda: np.zeros(expected_shape)))
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.execute("SELECT user_id, imdb_id, q_values FROM q_table_users")
            invalid_entries = []
            for user_id, imdb_id, values in cursor.fetchall():
                try:
                    q_array = np.array(json.loads(values))
                    if q_array.shape != expected_shape:
                        logger.warning(f"Invalid shape for {imdb_id}: expected {expected_shape}, got {q_array.shape}")
                        invalid_entries.append((user_id, imdb_id))
                    else:
                        q_values[user_id][imdb_id] = q_array
                except Exception as e:
                    logger.error(f"Error parsing Q-values for {imdb_id}: {e}")
                    invalid_entries.append((user_id, imdb_id))

            if invalid_entries:
                logger.warning(f"Removing {len(invalid_entries)} invalid Q-table entries")
                for uid, mid in invalid_entries:
                    conn.execute("DELETE FROM q_table_users WHERE user_id = ? AND imdb_id = ?", (uid, mid))
                conn.commit()
    except sqlite3.OperationalError:
        pass  # Table might not exist yet on fresh DB
    except Exception as e:
        logger.error(f"Error loading Q-table: {e}")
    return q_values

def save_q_table(q_table, user_id, imdb_id):
    try:
        if user_id not in q_table or imdb_id not in q_table[user_id]:
            return
            
        with sqlite3.connect(DB_PATH) as conn:
            q_values = q_table[user_id][imdb_id]
            if isinstance(q_values, np.ndarray):
                q_values_list = q_values.tolist()
            else:
                q_values_list = list(q_values)
                
            conn.execute("INSERT OR REPLACE INTO q_table_users (user_id, imdb_id, q_values) VALUES (?, ?, ?)",
                         (user_id, imdb_id, json.dumps(q_values_list)))
            conn.commit()
    except Exception as e:
        logger.error(f"Error saving Q-table for {imdb_id}: {e}")

def fetch_movies(keyword, pages=2):
    all_movies = []
    
    try:
        for page in range(1, pages + 1):
            params = {
                'api_key': TMDB_API_KEY,
                'query': keyword,
                'page': page
            }
            response = http_session.get(f"{TMDB_BASE_URL}/search/movie", params=params, timeout=5)
            if response.status_code == 200:
                data = response.json()
                tmdb_movies = data.get('results', [])
                for movie in tmdb_movies:
                    movie['imdb_id'] = movie.get('imdb_id', f"tmdb_{movie['id']}")
                    movie['imdbID'] = movie['imdb_id']
                    all_movies.append(movie)
    except Exception as e:
        logger.error(f"Error fetching movies from TMDB for keyword '{keyword}': {e}")

    try:
        with sqlite3.connect(DB_PATH) as conn:
            current_time = int(datetime.now().timestamp())
            cache_key = f"{keyword}_pages{pages}"
            conn.execute(
                "INSERT OR REPLACE INTO search_cache (query, results, timestamp) VALUES (?, ?, ?)",
                (cache_key, json.dumps(all_movies), current_time)
            )
            conn.commit()
    except Exception as e:
        logger.error(f"Error caching search results for '{keyword}' with {pages} pages: {e}")

    return all_movies

popular_movies_cache = {}
POPULAR_MOVIES_CACHE_DURATION = 3600

def fetch_popular_movies(limit=10, exclude_imdb_ids=None):
    if exclude_imdb_ids is None:
        exclude_imdb_ids = set()
    elif not isinstance(exclude_imdb_ids, set):
        exclude_imdb_ids = set(exclude_imdb_ids)

    cache_key = f"popular_{limit}_{hash(frozenset(exclude_imdb_ids))}"
    current_time = int(datetime.now().timestamp())
    if cache_key in popular_movies_cache and 'timestamp' in popular_movies_cache[cache_key]:
        if current_time - popular_movies_cache[cache_key]['timestamp'] < POPULAR_MOVIES_CACHE_DURATION:
            cached_movies = popular_movies_cache[cache_key]['movies']
            filtered_movies = [m for m in cached_movies if m.get('imdb_id') not in exclude_imdb_ids]
            if len(filtered_movies) >= min(limit, len(cached_movies) - len(exclude_imdb_ids)):
                return filtered_movies[:limit]

    all_movies = []
    try:
        params = {
            'api_key': TMDB_API_KEY,
            'sort_by': 'popularity.desc',
            'vote_count.gte': 500,
            'page': random.randint(1, 10)  # Randomize pages to get different popular movies
        }
        response = http_session.get(f"{TMDB_BASE_URL}/discover/movie", params=params, timeout=5)
        if response.status_code == 200:
            data = response.json()
            tmdb_movies = data.get('results', [])
            for movie in tmdb_movies:
                imdb_id = movie.get('imdb_id', f"tmdb_{movie['id']}")
                if imdb_id in exclude_imdb_ids:
                    continue
                rating = movie.get('vote_average', 'N/A')
                try:
                    if rating != 'N/A' and float(rating) < MIN_RATING:
                        continue
                except (ValueError, TypeError):
                    continue
                movie['imdbID'] = imdb_id
                movie['imdb_id'] = imdb_id
                all_movies.append(movie)
    except Exception as e:
        logger.error(f"Error fetching popular movies from TMDB: {e}")

    random.shuffle(all_movies)
    formatted_movies = []
    for movie in all_movies:
        if len(formatted_movies) >= limit:
            break
        try:
            formatted = format_movie(movie)
            if formatted and formatted.get('poster') != 'N/A':
                formatted_movies.append(formatted)
        except Exception as e:
            logger.error(f"Error formatting movie {movie.get('imdb_id')}: {str(e)}")

    if formatted_movies:
        popular_movies_cache[cache_key] = {
            'movies': formatted_movies,
            'timestamp': current_time
        }
    return formatted_movies[:limit]

recommendations_cache = {}
RECOMMENDATIONS_CACHE_DURATION = 600  # 10 min cache

def _get_seen_and_disliked():
    seen_ids = set()
    disliked_titles = []
    user_id = session.get('user_id', 1)
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.execute("SELECT imdb_id, action FROM preferences WHERE user_id = ?", (user_id,))
            for row in cursor.fetchall():
                seen_ids.add(row[0])
    except Exception as e:
        logger.error(f"Error loading preferences: {e}")
    return seen_ids, disliked_titles

def _score_candidates(candidates, genre_preference, disliked_titles):
    stop_words = {'the', 'and', 'of', 'a', 'an', 'in', 'to', 'for', 'is', 'it'}
    scored = []
    for movie in candidates:
        try:
            imdb_id = movie.get('imdb_id') or movie.get('imdbID')
            movie_vec = get_movie_vector(movie)
            score = float(np.dot(movie_vec, genre_preference))

            # Boost recent movies (Netflix-style: favor recent content)
            year_str = movie.get('year', '') or movie.get('release_date', '')[:4] if movie.get('release_date') else ''
            try:
                year = int(str(year_str)[:4])
                if year >= 2024:
                    score *= 1.5  # Strong boost for very recent
                elif year >= 2020:
                    score *= 1.25  # Moderate boost
                elif year >= 2015:
                    score *= 1.1
            except (ValueError, TypeError):
                pass

            # Boost highly rated movies
            rating = movie.get('rating', '') or movie.get('vote_average', 0)
            try:
                r = float(rating)
                if r >= 8.0:
                    score *= 1.2
                elif r >= 7.0:
                    score *= 1.1
            except (ValueError, TypeError):
                pass

            # Penalize titles similar to disliked movies
            current_title = movie.get('title', '').lower()
            current_words = set(current_title.split())
            for dt in disliked_titles:
                common = set(dt.split()) & current_words
                meaningful = common - stop_words
                if len(common) >= 2 or len(meaningful) >= 1:
                    score -= 0.5

            scored.append((score, movie))
        except Exception as e:
            logger.error(f"Error scoring movie {movie.get('imdb_id')}: {e}")
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored

def _get_genre_preference():
    """Compute genre preference vector from Q-table. Returns None if no valid data."""
    user_id = session.get('user_id', 1)
    if user_id not in q_table or not q_table[user_id]:
        return None
    expected_shape = (len(GENRES) + 3,)
    valid = [q_table[user_id][k] for k in q_table[user_id] if isinstance(q_table[user_id][k], np.ndarray) and q_table[user_id][k].shape == expected_shape]
    if not valid:
        return None
    return np.sum(valid, axis=0)

def _get_top_genres(genre_preference, n=3):
    """Return top N genre names from preference vector."""
    genre_scores = list(zip(GENRES, genre_preference[:len(GENRES)]))
    genre_scores.sort(key=lambda x: x[1], reverse=True)
    return [g for g, s in genre_scores[:n] if s > 0]

def _fetch_genre_section(genre_name, limit=12, exclude_ids=None):
    if exclude_ids is None: exclude_ids = set()
    genre_mapping = {"Action": 28, "Adventure": 12, "Animation": 16, "Comedy": 35, "Crime": 80, "Drama": 18, "Fantasy": 14, "Horror": 27, "Mystery": 9648, "Romance": 10749, "Sci-Fi": 878, "Thriller": 53}
    genre_id = genre_mapping.get(genre_name, 28)
    results = []
    try:
        resp = http_session.get(f"{TMDB_BASE_URL}/discover/movie",
            params={'api_key': TMDB_API_KEY, 'with_genres': genre_id, 'sort_by': 'popularity.desc', 'vote_count.gte': 100, 'page': random.randint(1, 10)}, timeout=5)
        if resp.status_code == 200:
            for m in resp.json().get('results', []):
                imdb_id = m.get('imdb_id', f"tmdb_{m['id']}")
                if imdb_id in exclude_ids: continue
                m['imdbID'] = imdb_id
                m['imdb_id'] = imdb_id
                fmt = format_movie(m)
                if fmt and fmt.get('poster') != 'N/A':
                    results.append(fmt)
                    exclude_ids.add(imdb_id)
                if len(results) >= limit: break
    except Exception as e: pass
    return results

def _fetch_top_rated(limit=12, exclude_ids=None):
    if exclude_ids is None: exclude_ids = set()
    results = []
    try:
        resp = http_session.get(f"{TMDB_BASE_URL}/movie/top_rated",
            params={'api_key': TMDB_API_KEY, 'page': 1}, timeout=5)
        if resp.status_code == 200:
            for m in resp.json().get('results', []):
                imdb_id = m.get('imdb_id', f"tmdb_{m['id']}")
                if imdb_id in exclude_ids: continue
                m['imdbID'] = imdb_id
                m['imdb_id'] = imdb_id
                fmt = format_movie(m)
                if fmt and fmt.get('poster') != 'N/A':
                    results.append(fmt)
                    exclude_ids.add(imdb_id)
                if len(results) >= limit: break
    except Exception as e: pass
    return results

def _fetch_recent_releases(limit=12, exclude_ids=None):
    if exclude_ids is None: exclude_ids = set()
    results = []
    try:
        resp = http_session.get(f"{TMDB_BASE_URL}/trending/movie/week",
            params={'api_key': TMDB_API_KEY, 'page': 1}, timeout=5)
        if resp.status_code == 200:
            for m in resp.json().get('results', []):
                imdb_id = m.get('imdb_id', f"tmdb_{m['id']}")
                if imdb_id in exclude_ids: continue
                m['imdbID'] = imdb_id
                m['imdb_id'] = imdb_id
                fmt = format_movie(m)
                if fmt and fmt.get('poster') != 'N/A':
                    results.append(fmt)
                    exclude_ids.add(imdb_id)
                if len(results) >= limit: break
    except Exception as e: pass
    return results

def build_home_sections():
    try:
        current_time = int(datetime.now().timestamp())
        if 'sections' in recommendations_cache and 'timestamp' in recommendations_cache:
            if current_time - recommendations_cache['timestamp'] < RECOMMENDATIONS_CACHE_DURATION:
                return recommendations_cache['sections']

        seen_ids, disliked_titles = _get_seen_and_disliked()
        genre_pref = _get_genre_preference()
        all_shown_ids = set(seen_ids)
        sections = []

        trending = _fetch_recent_releases(limit=12, exclude_ids=all_shown_ids)
        if trending:
            sections.append({'title': 'Trending Now', 'movies': trending})

        if genre_pref is not None:
            pool = []
            try:
                resp = http_session.get(f"{TMDB_BASE_URL}/movie/popular", params={'api_key': TMDB_API_KEY, 'page': 1}, timeout=5)
                if resp.status_code == 200:
                    for m in resp.json().get('results', []):
                        m['imdbID'] = m.get('imdb_id', f"tmdb_{m['id']}")
                        m['imdb_id'] = m['imdbID']
                        if m['imdbID'] not in all_shown_ids: pool.append(m)
            except: pass

            if pool:
                scored = _score_candidates(pool, genre_pref, disliked_titles)
                for_you = []
                for _, movie in scored:
                    if len(for_you) >= 12: break
                    fmt = format_movie(movie)
                    if fmt and fmt.get('poster') != 'N/A':
                        for_you.append(fmt)
                        all_shown_ids.add(fmt['imdb_id'])
                if for_you:
                    sections.append({'title': 'Recommended For You', 'movies': for_you})

            top_genres = _get_top_genres(genre_pref, n=3)
            for genre in top_genres:
                genre_movies = _fetch_genre_section(genre, limit=12, exclude_ids=all_shown_ids)
                if len(genre_movies) >= 4:
                    sections.append({'title': f'Best in {genre}', 'movies': genre_movies})

        top_rated = _fetch_top_rated(limit=12, exclude_ids=all_shown_ids)
        if top_rated:
            sections.append({'title': 'All-Time Favorites', 'movies': top_rated})

        if not sections:
            popular = fetch_popular_movies(20)
            if popular: sections.append({'title': 'Popular Movies', 'movies': popular})

        recommendations_cache['sections'] = sections
        recommendations_cache['timestamp'] = current_time
        return sections
    except Exception as e:
        logger.error(f"Error building home sections: {e}")
        popular = fetch_popular_movies(20)
        return [{'title': 'Popular Movies', 'movies': popular}] if popular else []

def check_and_add_timestamp_column():
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.execute("PRAGMA table_info(movie_cache)")
            columns = [row[1] for row in cursor.fetchall()]
            if 'timestamp' not in columns:
                logger.info("Adding timestamp column to movie_cache table")
                conn.execute("ALTER TABLE movie_cache ADD COLUMN timestamp INTEGER DEFAULT 0")
                conn.commit()
                current_time = int(datetime.now().timestamp())
                conn.execute("UPDATE movie_cache SET timestamp = ?", (current_time,))
                conn.commit()
    except Exception as e:
        logger.error(f"Error checking or adding timestamp column: {e}")

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS users
                        (id INTEGER PRIMARY KEY,
                         username TEXT UNIQUE,
                         password TEXT)''')
        conn.execute('''CREATE TABLE IF NOT EXISTS preferences
                        (id INTEGER PRIMARY KEY,
                         user_id INTEGER,
                         imdb_id TEXT,
                         action TEXT)''')
        try:
            conn.execute("ALTER TABLE preferences ADD COLUMN user_id INTEGER DEFAULT 1")
        except: pass
        conn.execute('''CREATE TABLE IF NOT EXISTS q_table_users (user_id INTEGER, imdb_id TEXT, q_values TEXT, PRIMARY KEY(user_id, imdb_id))''')
        conn.execute('''CREATE TABLE IF NOT EXISTS movie_cache
                        (imdb_id TEXT PRIMARY KEY,
                         details TEXT,
                         timestamp INTEGER)''')
        conn.execute('''CREATE TABLE IF NOT EXISTS search_cache
                        (query TEXT PRIMARY KEY,
                         results TEXT,
                         timestamp INTEGER)''')
        conn.execute('''CREATE TABLE IF NOT EXISTS ratings
                        (id INTEGER PRIMARY KEY,
                         user_id TEXT,
                         imdb_id TEXT,
                         rating INTEGER)''')
        conn.commit()

    check_and_add_timestamp_column()

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute('CREATE INDEX IF NOT EXISTS idx_preferences_imdb_id ON preferences(imdb_id)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_movie_cache_timestamp ON movie_cache(timestamp)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_search_cache_timestamp ON search_cache(timestamp)')
        conn.commit()

    global q_table
    q_table = load_q_table()


    try:
        current_time = int(datetime.now().timestamp())
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("DELETE FROM search_cache WHERE timestamp < ?", (current_time - 604800,))
            conn.execute("DELETE FROM movie_cache WHERE timestamp < ?", (current_time - 2592000,))
            conn.commit()
    except Exception as e:
        logger.error(f"Error cleaning up cache: {e}")

    logger.info(f"Database initialized. {len(movie_vectors)} movie vectors precomputed.")
    logger.info(f"Loaded {len(q_table)} Q-table entries.")


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('landing'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/landing')
def landing():
    if 'user_id' in session:
        return redirect(url_for('home'))
    return render_template('landing.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session:
        return redirect(url_for('home'))
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.execute("SELECT id, password FROM users WHERE username = ?", (username,))
            user = cursor.fetchone()
            if user and check_password_hash(user[1], password):
                session['user_id'] = user[0]
                session['username'] = username
                return redirect(url_for('home'))
            flash('Invalid username or password', 'error')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if 'user_id' in session:
        return redirect(url_for('home'))
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        hashed = generate_password_hash(password)
        try:
            with sqlite3.connect(DB_PATH) as conn:
                conn.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed))
                conn.commit()
                # auto login
                cursor = conn.execute("SELECT id FROM users WHERE username = ?", (username,))
                session['user_id'] = cursor.fetchone()[0]
                session['username'] = username
            return redirect(url_for('home'))
        except sqlite3.IntegrityError:
            flash('Username already exists! Please sign in.', 'error')
            return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('username', None)
    return redirect(url_for('landing'))

@app.route('/')
@login_required
def home():
    start_time = datetime.now()
    sections = build_home_sections()
    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info(f"Home page built in {elapsed:.2f}s with {len(sections)} sections")
    response = make_response(render_template('recommendations.html', sections=sections))
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

def get_swipe_movie(exclusion_list):
    genre_pref = _get_genre_preference()
    candidates = []
    
    # 75% chance for personalized recommendation, 25% for random popular movie
    if genre_pref is not None and random.random() < 0.75:
        top_genres = _get_top_genres(genre_pref, n=2)
        if top_genres:
            genre = random.choice(top_genres)
            candidates = _fetch_genre_section(genre, limit=20, exclude_ids=exclusion_list)
            
    # Fallback or 25% random branch
    if not candidates:
        candidates = fetch_popular_movies(limit=20, exclude_imdb_ids=exclusion_list)
        
    if candidates:
        return random.choice(candidates)
    return None

@app.route('/swipe', methods=['GET', 'POST'])
@login_required
def swipe():
    recommendations_cache.clear()

    if request.method == 'POST':
        if 'imdb_id' not in request.form or 'action' not in request.form:
            logger.error("Missing required form data in swipe POST")
            return "Error: Missing form data", 400

        imdb_id = request.form['imdb_id']
        action = request.form['action']
        logger.debug(f"Received swipe: {imdb_id} - {action}")

        try:
            with sqlite3.connect(DB_PATH) as conn:
                user_id = session.get('user_id', 1)
                conn.execute("INSERT INTO preferences (user_id, imdb_id, action) VALUES (?, ?, ?)",
                             (user_id, imdb_id, action))
                conn.commit()
                logger.debug(f"Successfully saved swipe: {action} on {imdb_id}")

            movie_obj = {'imdbID': imdb_id}
            movie_vec = get_movie_vector(movie_obj)
            reward = 1 if action == 'like' else -1
            user_id = session.get('user_id', 1)
            existing_q = q_table[user_id][imdb_id] if imdb_id in q_table[user_id] else np.zeros(len(GENRES) + 3)
            q_table[user_id][imdb_id] = existing_q + (movie_vec * reward)
            save_q_table(q_table, user_id, imdb_id)

            recently_shown = session.get('recently_shown', [])
            if imdb_id in recently_shown:
                recently_shown.remove(imdb_id)
            recently_shown.append(imdb_id)
            if len(recently_shown) > 20:
                recently_shown = recently_shown[-20:]
            session['recently_shown'] = recently_shown

        except sqlite3.Error as e:
            logger.error(f"Database error: {e}")
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify({'success': False, 'error': 'Database error'}), 500
            return "Database error", 500

        # If it's an AJAX request, fetch the next movie and return it as JSON
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            try:
                with sqlite3.connect(DB_PATH) as conn:
                    user_id = session.get('user_id', 1)
                    seen_imdb_ids = [row[0] for row in conn.execute("SELECT imdb_id FROM preferences WHERE user_id = ?", (user_id,))]

                recently_shown = session.get('recently_shown', [])
                exclusion_list = set(seen_imdb_ids + recently_shown)

                next_movie = get_swipe_movie(exclusion_list)
                if next_movie:
                    return jsonify({'success': True, 'movie': next_movie})
                return jsonify({'success': False, 'error': 'No more movies found'}), 404
            except Exception as e:
                logger.error(f"Error finding next movie: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        return redirect(url_for('swipe'))

    try:
        with sqlite3.connect(DB_PATH) as conn:
            user_id = session.get('user_id', 1)
            seen_imdb_ids = [row[0] for row in conn.execute("SELECT imdb_id FROM preferences WHERE user_id = ?", (user_id,))]

        recently_shown = session.get('recently_shown', [])
        exclusion_list = set(seen_imdb_ids + recently_shown)
        logger.debug(f"Finding a movie the user hasn't seen yet (excluded {len(exclusion_list)} movies)")

        movie = get_swipe_movie(exclusion_list)
        if movie:
            logger.debug(f"Presenting movie for swiping: {movie.get('title')}")
            return render_template('swipe.html', movie=movie)

        logger.warning("No movies found after retries")
        return render_template('swipe.html', movie=None)
    except Exception as e:
        logger.error(f"Error in swipe route: {str(e)}")
        return render_template('error.html', error=str(e))
@app.route('/feedback', methods=['POST'])
def feedback():
    recommendations_cache.clear()
    imdb_id = request.form.get('imdb_id')
    action = request.form.get('action')

    if not imdb_id or not action:
        return jsonify({'success': False, 'error': 'Missing parameters'}), 400

    normalized_action = action
    if action == 'up':
        normalized_action = 'like'
    elif action == 'down':
        normalized_action = 'dislike'

    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("INSERT INTO preferences (imdb_id, action) VALUES (?, ?)",
                         (imdb_id, normalized_action))
            conn.commit()
            logger.debug(f"Successfully saved feedback: {normalized_action} on {imdb_id}")

        movie_obj = {'imdbID': imdb_id}
        movie_vec = get_movie_vector(movie_obj)
        reward = 1 if normalized_action == 'like' else -1
        user_id = session.get('user_id', 1)
        existing_q = q_table[user_id][imdb_id] if imdb_id in q_table[user_id] else np.zeros(len(GENRES) + 3)
        q_table[user_id][imdb_id] = existing_q + (movie_vec * reward)
        save_q_table(q_table, user_id, imdb_id)

        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({'success': True, 'refresh': True})
        return redirect(url_for('home'))

    except sqlite3.Error as e:
        logger.error(f"Database error in feedback: {e}")
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({'success': False, 'error': str(e)}), 500
        return "Database error", 500

@app.route('/rate', methods=['POST'])
def rate_movie():
    user_id = session.get('user_id', 'default_user')
    imdb_id = request.form.get('imdb_id')
    rating = request.form.get('rating')

    if not imdb_id or not rating:
        flash("Error: Missing rating or movie ID.", "error")
        return redirect(request.referrer or url_for('home'))

    try:
        rating = int(rating)
        if rating < 1 or rating > 5:
            flash("Error: Rating must be between 1 and 5.", "error")
            return redirect(request.referrer or url_for('home'))

        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("INSERT INTO ratings (user_id, imdb_id, rating) VALUES (?, ?, ?)",
                         (user_id, imdb_id, rating))
            conn.commit()

        movie_vec = get_movie_vector({'imdbID': imdb_id})
        reward = (rating - 3) / 2
        user_id = session.get('user_id', 1)
        existing_q = q_table[user_id][imdb_id] if imdb_id in q_table[user_id] else np.zeros(len(GENRES) + 3)
        q_table[user_id][imdb_id] = existing_q + (movie_vec * reward)
        save_q_table(q_table, user_id, imdb_id)
        recommendations_cache.clear()

        flash("Thank you for rating the movie!", "success")
        return redirect(request.referrer or url_for('home'))
    except (ValueError, sqlite3.Error) as e:
        logger.error(f"Error saving rating: {e}")
        flash("Error: Unable to save your rating. Please try again.", "error")
        return redirect(request.referrer or url_for('home'))

def get_recommendation_explanation(imdb_id):
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.execute("SELECT imdb_id, action FROM preferences")
            preferences = cursor.fetchall()

        if not preferences:
            return "This movie is popular among users right now."

        movie = get_movie_details(imdb_id)
        movie_vec = get_movie_vector({'imdbID': imdb_id})
        genre_scores = movie_vec[:len(GENRES)]
        top_genres = [GENRES[i] for i in np.argsort(genre_scores)[-2:] if genre_scores[i] > 0]

        liked_genres = []
        for pref_id, action in preferences:
            if action == 'like':
                liked_movie = get_movie_details(pref_id)
                liked_vec = get_movie_vector({'imdbID': pref_id})
                liked_genres.extend([GENRES[i] for i in range(len(GENRES)) if liked_vec[i] > 0])

        common_genres = set(top_genres) & set(liked_genres)
        if common_genres:
            return f"We recommend this movie because you enjoyed other {', '.join(common_genres)} movies."

        if movie_vec[-1] > 0:
            return "We recommend this movie because you liked other movies by the same director."
        if movie_vec[-2] > 0.8:
            return "We recommend this movie because you seem to enjoy recent releases."
        if movie_vec[-3] > 0.8:
            return "We recommend this movie because you tend to like highly rated films."

        return "This movie was recommended based on your viewing history and preferences."
    except Exception as e:
        logger.error(f"Error generating recommendation explanation: {e}")
        return "Recommended based on popularity."

@app.route('/search')
def search():
    keyword = request.args.get('q', '')
    if not keyword:
        return redirect(url_for('home'))

    try:
        # Always search TMDB first for latest results
        formatted_movies = []
        seen_ids = set()
        
        if TMDB_API_KEY:
            try:
                resp = http_session.get(f"{TMDB_BASE_URL}/search/movie",
                    params={'api_key': TMDB_API_KEY, 'query': keyword, 'page': 1}, timeout=5)
                if resp.status_code == 200:
                    for movie in resp.json().get('results', []):
                        movie['imdb_id'] = movie.get('imdb_id', f"tmdb_{movie['id']}")
                        movie['imdbID'] = movie['imdb_id']
                        movie['title'] = movie.get('title', 'Unknown')
                        fmt = format_movie(movie)
                        if fmt and fmt.get('poster') != 'N/A':
                            formatted_movies.append(fmt)
                            seen_ids.add(movie['imdb_id'])
            except Exception as e:
                logger.error(f"TMDB search error: {e}")
        
        return render_template('search_results.html', keyword=keyword, movies=formatted_movies)
    except Exception as e:
        logger.error(f"Error in search route: {e}")
        return render_template('error.html', message="An error occurred while searching. Please try again.")

@app.route('/movie/<imdb_id>')
def movie_details(imdb_id):
    try:
        details = get_movie_details(imdb_id)
        if not details:
            return render_template('error.html', message="Movie not found.")

        explanation = get_recommendation_explanation(imdb_id)
        poster_base_url = "https://image.tmdb.org/t/p/w342"

        if details.get('Poster') and not details['Poster'].startswith('http'):
            if details['Poster'].startswith('/'):
                details['Poster'] = poster_base_url + details['Poster']
            else:
                details['Poster'] = poster_base_url + '/' + details['Poster']

        if 'additional_backdrops' in details and details['additional_backdrops']:
            for i, backdrop in enumerate(details['additional_backdrops']):
                if backdrop and not backdrop.startswith('http'):
                    if backdrop.startswith('/'):
                        details['additional_backdrops'][i] = poster_base_url + backdrop
                    else:
                        details['additional_backdrops'][i] = poster_base_url + '/' + backdrop

        streaming_data = get_streaming_providers(details.get('Title', ''), imdb_id=imdb_id)
        return render_template('movie_details.html',
                               movie=details,
                               explanation=explanation,
                               poster_base_url=poster_base_url,
                               streaming=streaming_data)
    except Exception as e:
        logger.error(f"Error in movie_details route: {e}")
        return render_template('error.html', message="An error occurred while loading movie details. Please try again.")

init_db()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)