# WEEK5
# Word2Vec Book Recommendation System (from Scratch Dataset)

This project builds a simple **book recommendation system** using **Word2Vec embeddings** from the `gensim` library.  
The idea is to treat each **reading list** as a “sentence” and each **book title** as a “token”. Books that appear together in the same reading lists learn similar vector representations, allowing us to recommend related books.

---

## Project Overview

### Goal
- Train a Word2Vec model on custom reading lists.
- Generate book recommendations using vector similarity (`most_similar`).

### Why Word2Vec?
Word2Vec learns embeddings based on **co-occurrence**:
- Books that appear together in many lists become closer in the embedding space.
- Similarity is calculated using cosine similarity.

---

## Dataset

A small custom dataset was created manually:
- **11 reading lists**
- Categories included:
  - Fantasy
  - Self-development
  - Classics
  - Science
Example reading list:
```python
['harry_potter', 'percy_jackson', 'hunger_games']
Parameter Notes:
sentences: the reading lists (training data)
vector_size=30: embedding dimension
window=3: context window size
min_count=1: keep all books
sg=1: use Skip-gram (often better for recommendation and smaller datasets)
Recommendation Function
Note: Because the dataset is small (only 11 lists), some recommendations may look less consistent. Increasing the number of reading lists will improve results.
How to Improve the Project
Expand the dataset to 50+ reading lists.
Add more books per category.
Remove cross-category noise or balance categories.
Visualize embeddings using PCA/TSNE.



# TF-IDF from Scratch (NLP)

This project implements **TF (Term Frequency)**, **IDF (Inverse Document Frequency)**, and **TF-IDF** from scratch using Python, then compares the results with **scikit-learn**.

## Project Files
- `TF_TFIDF_From_Scratch.ipynb` — the full notebook implementation and outputs
- `README.md` — project overview
## Dataset / Corpus
We use a small sample corpus:

```python
corpus = [
    "i love web development",
    "i love cats",
    "i hate dogs"
]
What I Implemented
1) Term Frequency (TF)
2) Inverse Document Frequency (IDF)
3) TF-IDF

Comparison with scikit-learn
I compared my results with:

python
Copy code
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(norm=None, smooth_idf=False)
sklearn_tfidf = vectorizer.fit_transform(corpus)
scikit-learn Output (No normalization, no smoothing)
Doc	cats	development	dogs	hate	love	web
0	0.00000	2.098612	0.00000	0.00000	1.405465	2.098612
1	2.098612	0.00000	0.00000	0.00000	1.405465	0.00000
2	0.00000	0.00000	2.098612	2.098612	0.00000	0.00000

Why Numbers Differ (Expected):
Even when we set norm=None and smooth_idf=False, scikit-learn still differs because:

It uses a different TF-IDF scaling/implementation details (including internal weighting choices).

Tokenization rules can also differ from a simple .split() approach.

However, the results should still be correlated:

Rare words like web, development, cats, hate, dogs get higher TF-IDF.

Frequent words like i get lower (IDF becomes 0).

