# WEEK5

## TF-IDF from Scratch (NLP)

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

