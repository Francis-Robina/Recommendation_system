#  Novel Recommendation System

##  Overview

A **content-based recommendation system** that suggests novels similar to a selected book title.
It uses **TF-IDF Vectorization** and **Cosine Similarity** to compare genres and descriptions to find the closest matches.

##  Features

* Recommends novels based on **genre** and **description**.
* Uses **TF-IDF** to convert text into numerical form.
* Computes similarity using **cosine similarity**.
* Loads dataset directly from **GitHub Raw link** (no manual uploads needed in Colab).


##  Dataset

**novels.csv** contains:

* **book\_id** → Unique ID
* **title** → Book title
* **author** → Author name
* **genre** → Book genre
* **description** → Short summary of the book


##  Technologies Used

* Python
* Pandas
* Scikit-learn

##  How to Run in Google Colab
import pandas as pd

url = "https://raw.githubusercontent.com/YourUsername/YourRepo/main/novels.csv"
books = pd.read_csv(url)
books.head()


##  Example Output

If you search for `"1984"`, you might get:


               title              author        genre
4    Brave New World    Aldous Huxley     Dystopian
1  Pride and Prejudice Jane Austen       Romance
0   The Great Gatsby   F. Scott Fitzgerald Classic
