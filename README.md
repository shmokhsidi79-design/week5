# WEEK 5 – NLP PROJECTS

This repository contains three NLP projects implemented from scratch.

1. Word2Vec – Book Recommendation System  
2. TF-IDF from Scratch  
3. Arabic Word2Vec – Supermarket Recommendation (Bonus)

---

# 1) Word2Vec – Book Recommendation System

This project builds a simple book recommendation system using Word2Vec embeddings from the gensim library.

## Idea
Each reading list is treated as a sentence.  
Each book title is treated as a token.  
Books that appear together in reading lists learn similar vector representations.

## Project Goal
- Train a Word2Vec model on custom reading lists  
- Generate book recommendations using vector similarity  

## Dataset
A small custom dataset was created manually.

- 11 reading lists  
- Categories:
  - Fantasy
  - Self-development
  - Classics
  - Science

Example reading list:
['harry_potter', 'percy_jackson', 'hunger_games']

shell
Copy code

## Model Configuration
Word2Vec(
sentences=reading_lists,
vector_size=30,
window=3,
min_count=1,
workers=2,
sg=1
)

sql
Copy code

## Parameter Notes
- sentences: reading lists (training data)  
- vector_size: embedding dimension  
- window: context window size  
- min_count: keep all books  
- sg: Skip-gram (better for recommendation with small datasets)

## Recommendation Function
recommend_books(book, top_n=4)

yaml
Copy code

## Notes
Because the dataset is small, some recommendations may look inconsistent.  
Increasing the number of reading lists will improve results.

## Possible Improvements
- Expand dataset to 50+ reading lists  
- Add more books per category  
- Reduce cross-category noise  
- Visualize embeddings using PCA or TSNE  

---

# 2) TF-IDF from Scratch

This project implements TF, IDF, and TF-IDF from scratch using Python and compares the results with scikit-learn.

## Dataset / Corpus
[
"i love web development",
"i love cats",
"i hate dogs"
]

markdown
Copy code

## What Was Implemented
- Term Frequency (TF)  
- Inverse Document Frequency (IDF)  
- TF-IDF  

## Comparison with scikit-learn
TfidfVectorizer(norm=None, smooth_idf=False)

yaml
Copy code

## Observations
- Rare words such as web, cats, and dogs receive higher TF-IDF values  
- Frequent words such as "i" receive lower TF-IDF values  
- Results remain correlated despite implementation differences  

---

# 3) Arabic Word2Vec – Supermarket Recommendation (Bonus)

This mini-project applies Word2Vec to an Arabic dataset of supermarket shopping baskets.

## Idea
Each basket is treated as a sentence.  
Each product is treated as a token.

## Dataset
[
["تمر", "زبادي_يوناني", "فطور_الصباح"],
["رز", "دجاج"],
["خبز", "جبنة", "طماطم"],
["تفاح", "عنب", "رمان"],
["بقدونس", "قهوة", "شاي"],
["بصل", "ثوم", "دجاج"]
]

shell
Copy code

## Model Training
Word2Vec(
sentences=shopping_baskets,
vector_size=100,
window=3,
min_count=1,
workers=2,
sg=1
)

shell
Copy code

## Recommendation Function
recommend(item, top_n=4)

yaml
Copy code

## Sample Results
For "تمر": ثوم, عنب, دجاج, قهوة  
For "دجاج": طماطم, رز, عنب, خبز  

## Notes
Similarity scores are low due to the small dataset.  
Increasing the number of baskets and reducing vector size will improve results.

---
