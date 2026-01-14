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

---

# 2️⃣ TF-IDF from Scratch (NLP)

This project implements **TF (Term Frequency)**, **IDF (Inverse Document Frequency)**, and **TF-IDF** from scratch using Python, then compares the results with **scikit-learn**.

---

## Project Goal
- Implement TF, IDF, and TF-IDF without using libraries
- Understand how text is converted into numerical vectors
- Compare results with scikit-learn

---

## Dataset / Corpus
A small sample corpus was used:

[
"i love web development",
"i love cats",
"i hate dogs"
]

yaml
Copy code

---

## What Was Implemented
- Term Frequency (TF)
- Inverse Document Frequency (IDF)
- TF-IDF

---

## Comparison with scikit-learn
The implementation was compared with scikit-learn using:

TfidfVectorizer(norm=None, smooth_idf=False)

yaml
Copy code

---

## Results (scikit-learn)
Doc 0:
- development = 2.098612
- love = 1.405465
- web = 2.098612

Doc 1:
- cats = 2.098612
- love = 1.405465

Doc 2:
- dogs = 2.098612
- hate = 2.098612

---

## Observations
- Rare words (web, cats, dogs) receive higher TF-IDF values
- Frequent words (i) receive lower TF-IDF values (IDF ≈ 0)
- Results are correlated despite implementation differences

---

## Notes
Differences are expected because:
- scikit-learn uses slightly different internal scaling
- Tokenization differs from a simple split approach

---

# 3️⃣ Arabic Word2Vec – Supermarket Recommendation (Bonus)

This mini-project applies **Word2Vec** to an Arabic dataset of supermarket shopping baskets.

---

## Idea
- Each basket is treated as a sentence
- Each product is treated as a token
- Products appearing together learn similar vector representations

---

## Dataset (Shopping Baskets)
[
["تمر", "زبادي_يوناني", "فطور_الصباح"],
["رز", "دجاج"],
["خبز", "جبنة", "طماطم"],
["تفاح", "عنب", "رمان"],
["بقدونس", "قهوة", "شاي"],
["بصل", "ثوم", "دجاج"]
]

yaml
Copy code

---

## Model Training
Word2Vec(
sentences=shopping_baskets,
vector_size=100,
window=3,
min_count=1,
workers=2,
sg=1
)

yaml
Copy code

---

## Recommendation Function
recommend(item, top_n=4)

yaml
Copy code

---

## Sample Results

Recommendations for "تمر":
- ثوم
- عنب
- دجاج
- قهوة

Recommendations for "دجاج":
- طماطم
- رز
- عنب
- خبز

---

## Handling Unknown Items
When an item such as "بطيخ" is queried, the system correctly reports that the item is not in the vocabulary and shows available options.

---

## Notes
- Similarity scores are low due to the small dataset (6 baskets)
- Increasing the number of baskets (30–100+) improves recommendations
- Reducing vector_size (20–50) is recommended for small datasets


Visualize embeddings using PCA or TSNE

