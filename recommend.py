import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')


# Stopwörter entfernen
def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word.lower() not in stop_words]
    return ' '.join(filtered_text)


# Empfehlungen bezüglich Ähnlichkeiten ermitteln
def get_recommendations(title, df, cosine_sim, top_n):
    idx = df[df['Title'].str.contains(title, case=False, na=False)].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n + 1]
    book_indices = [i[0] for i in sim_scores]
    return df.iloc[book_indices], [i[1] for i in sim_scores]


# Daten aus der CSV Datei laden/lesen
df = pd.read_csv('OTL.csv', usecols=['Title', 'Description'])
df = df.fillna('')
df['combined'] = df['Title'] + ' ' + df['Description']
df['combined'] = df['combined'].apply(remove_stopwords)


# Vektorisieren des kombinierten Textes
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df['combined'])


# Ähnlichkeitsmatrix erstellen
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)


# Ausgabe: Empfehlungen bezüglich einer Suche erhalten
title = 'Financial Accounting'
top_n = 10
recommend_df, scores = get_recommendations(title, df, cosine_sim, top_n)
print(f'Empfehlungen für: {title}')
for recommendation, score in zip(recommend_df['Title'], scores):
    print(f"Empfohlen: {recommendation} mit einem Ähnlichkeitsscore von {score:.2f}")


