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
# Animal Recommendation System using Word2Vec Embeddings

## Project Overview
This project demonstrates how embedding techniques can be applied to build a recommendation system using non-text data.
Instead of traditional NLP inputs, the system models:

- Animal names as tokens
- Animal groups (biological classes) as sentences

A Word2Vec model is trained to learn relationships between animals based on their co-occurrence within the same group.
The trained embeddings are then used to recommend animals that are similar to a given animal.

---

## Dataset
Source: Kaggle – Zoo Animal Classification Dataset  
File used: zoo.csv  

The dataset contains 101 animals classified into 7 biological categories.

Main columns:
- animal_name: Name of the animal
- class_type: Biological class label

Class types:
1. Mammal
2. Bird
3. Reptile
4. Fish
5. Amphibian
6. Bug
7. Invertebrate

---

## Data Preparation
The dataset is transformed into a format compatible with Word2Vec.

- Each biological class is treated as one sentence
- Each animal name is treated as a token


## Results
Animal embeddings were successfully trained

The model produced logical similarity-based recommendations

The project demonstrates the flexibility of Word2Vec beyond text data

