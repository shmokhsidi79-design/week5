# WEEK 5 – NLP Projects

This repository contains **three NLP projects** implemented from scratch:

1. **Word2Vec – Book Recommendation System**
2. **TF-IDF from Scratch**
3. **Arabic Word2Vec – Supermarket Recommendation (Bonus)**

---

# 1️⃣ Word2Vec – Book Recommendation System

This project builds a simple **book recommendation system** using **Word2Vec embeddings** from the `gensim` library.

### Idea
- Each **reading list** is treated as a *sentence*
- Each **book title** is treated as a *token*
- Books that appear together in reading lists learn similar vector representations

---

## Project Goal
- Train a Word2Vec model on custom reading lists
- Generate book recommendations using vector similarity (`most_similar`)

---

## Why Word2Vec?
Word2Vec learns embeddings based on **co-occurrence**:
- Books that appear together frequently become closer in the embedding space
- Similarity is calculated using **cosine similarity**

---

## Dataset
A small custom dataset was created manually.

- **11 reading lists**
- Categories:
  - Fantasy
  - Self-development
  - Classics
  - Science

Example reading list:
```python
['harry_potter', 'percy_jackson', 'hunger_games']
Model Configuration
python
Copy code
from gensim.models import Word2Vec

model = Word2Vec(
    sentences=reading_lists,
    vector_size=30,
    window=3,
    min_count=1,
    workers=2,
    sg=1
)
Parameter Notes
sentences: reading lists (training data)

vector_size=30: embedding dimension

window=3: context window size

min_count=1: keep all books

sg=1: Skip-gram (better for recommendation with small datasets)

Recommendation Function
python
Copy code
def recommend_books(book, top_n=4):
    if book not in model.wv:
        print("Book not found.")
        return

    for b, score in model.wv.most_similar(book, topn=top_n):
        print(f"{b} (similarity: {score:.2f})")
Notes
Because the dataset is small, some recommendations may look inconsistent

Increasing the number of reading lists improves results

Possible Improvements
Expand dataset to 50+ reading lists

Add more books per category

Reduce cross-category noise

Visualize embeddings using PCA or TSNE

# 2️⃣ TF-IDF from Scratch (NLP)
This project implements TF, IDF, and TF-IDF from scratch using Python and compares the results with scikit-learn.

Dataset / Corpus
python
Copy code
corpus = [
    "i love web development",
    "i love cats",
    "i hate dogs"
]
What I Implemented
Term Frequency (TF)

Inverse Document Frequency (IDF)

TF-IDF

Comparison with scikit-learn
python
Copy code
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(norm=None, smooth_idf=False)
sklearn_tfidf = vectorizer.fit_transform(corpus)
scikit-learn Output (No normalization, no smoothing)
Doc	cats	development	dogs	hate	love	web
0	0.0	2.098612	0.0	0.0	1.405465	2.098612
1	2.098612	0.0	0.0	0.0	1.405465	0.0
2	0.0	0.0	2.098612	2.098612	0.0	0.0

Why Numbers Differ (Expected)
scikit-learn uses different internal scaling

Tokenization differs from a simple .split()

Results remain correlated

Observations
Rare words (web, cats, dogs) get higher TF-IDF

Frequent words (i) get lower TF-IDF (IDF ≈ 0)

# 3️⃣ Arabic Word2Vec – Supermarket Recommendation (Bonus)
This mini-project applies Word2Vec to an Arabic dataset of supermarket shopping baskets.

Each basket = sentence

Each product = token

Dataset (Shopping Baskets)
python
Copy code
shopping_baskets = [
    ["تمر", "زبادي_يوناني", "فطور_الصباح"],
    ["رز", "دجاج"],
    ["خبز", "جبنة", "طماطم"],
    ["تفاح", "عنب", "رمان"],
    ["بقدونس", "قهوة", "شاي"],
    ["بصل", "ثوم", "دجاج"]
]
Model Training
python
Copy code
from gensim.models import Word2Vec

model = Word2Vec(
    sentences=shopping_baskets,
    vector_size=100,
    window=3,
    min_count=1,
    workers=2,
    sg=1
)
Recommendation Function
python
Copy code
def recommend(item, top_n=4):
    if item not in model.wv:
        print("العنصر غير موجود. جرّب واحد من:")
        print(list(model.wv.index_to_key))
        return

    print(f"اقتراحات مشابهة '{item}':")
    for x, score in model.wv.most_similar(item, topn=top_n):
        print(f"- {x} (تشابه: {score:.2f})")
Sample Output
Recommendations for تمر:

ثوم (0.12)

عنب (0.08)

دجاج (0.05)

قهوة (0.02)

Recommendations for دجاج:

طماطم (0.22)

رز (0.09)

عنب (0.09)

خبز (0.08)

Handling unknown item (بطيخ):

The system correctly detects unseen items and prints available options.

Notes
Similarity scores are low due to the small dataset (6 baskets)

Increasing baskets (30–100+) and reducing vector_size improves results

