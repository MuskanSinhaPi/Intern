from google.colab import drive
import random
import re
from nltk.stem.snowball import SnowballStemmer
import os
import gensim
from gensim.models import Doc2Vec
from nltk.corpus import stopwords
import nltk
from sklearn.cluster import KMeans
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pathlib import Path

# """### Downloading extra dependencies from NLTK"""
nltk.download('stopwords')
nltk.download('punkt')
# """### Getting stopwords customized to your problem statement"""
# Use this function to create custom list of stop_words for your Project

drive.mount('/content/drive')
path = '/content/drive/MyDrive/internship/stopwords.txt'

def get_stopwords(path):
    stopwords = nltk.corpus.stopwords.words('english')
    not_words = []
    with open(path, 'r') as f:
        not_words.append(f.readlines())
    not_words = [word.replace('\n', '') for words in not_words for word in words]
    not_words = set(not_words)
    stopwords = set(stopwords)
    customized_stopwords = list(stopwords - not_words)
    return stopwords, customized_stopwords


stop_words, customized_stopwords = get_stopwords(path)

# """### Tokenizing the document and filtering the tokens"""
def tokenize(train_texts):
    filtered_tokens = []
    tokens = [word for sent in nltk.sent_tokenize(train_texts) for word in nltk.word_tokenize(sent)]
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            if (('http' not in token) and ('@' not in token) and ('<.*?>' not in token) and token.isalnum() and (
            not token in stop_words)):
                filtered_tokens.append(token)
    return filtered_tokens

# """### Tokenizing and stemming using Snowball stemmer"""
def tokenize_stem(train_texts):
    tokens = tokenize(train_texts)
    stemmer = SnowballStemmer('english')
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return stemmed_tokens
# def load_data(path, seed):
#     train_texts = []
#     for file_path in sorted(path.glob('*.txt')):
#         with open(file_path, 'r') as f:
#             train_texts.append(f.read())
#     random.seed(seed)
#     random.shuffle(train_texts)
#     return train_texts

# """**Loading Data**"""

path = Path('/content/drive/MyDrive/internship/Articles')
# path = '/content/drive/MyDrive/internship/Articles'
# path = r'/Users/muskanshomef/Desktop/DRDO Intern/Document-Clustering-Doc2vec-master/Clustering/Articles'  # Add the path to Articles folder
seed = 137

def load_data(path, seed):
    train_texts = []
    for file_path in sorted(path.glob('*.txt')):
        with open(file_path, 'r', encoding='ISO-8859-1') as f:
            train_texts.append(f.read())
    random.seed(seed)
    random.shuffle(train_texts)
    return train_texts

# Call the load_data function with the updated path
train_texts = load_data(path, seed)


# def load_data(path, seed):
#     train_texts = []
#     for fname in sorted(os.listdir(path)):
#         if fname.endswith('.txt'):
#             with open(os.path.join(path, fname), 'r') as f:
#                 train_texts.append(f.read())
#     random.seed(seed)
#     random.shuffle(train_texts)
#     return train_texts


# train_texts = load_data(path, seed)

# """**Create a list of tagged emails. **"""
LabeledSentence1 = gensim.models.doc2vec.TaggedDocument
all_content = []
j = 0
k = 0

vocab_tokenized = []
vocab_stemmed = []
for text in train_texts:
    allwords_tokenized = tokenize(text)
    vocab_tokenized.append(allwords_tokenized)

    allwords_stemmed = tokenize_stem(text)
    vocab_stemmed.append(allwords_tokenized)

for text in vocab_tokenized:
    # add tokens to list
    if len(text) > 0:
        all_content.append(LabeledSentence1(text, [j]))
        j += 1

    k += 1

print("Number of emails processed: ", k)
print("Number of non-empty emails vectors: ", j)
"""**Create a model using Doc2Vec and train it**"""

d2v_model = Doc2Vec(all_content, vector_size=2000, min_count=5, dm=0,
                    alpha=0.0025, min_alpha=0.0001)
d2v_model.train(all_content, total_examples=d2v_model.corpus_count, epochs=50, start_alpha=0.002, end_alpha=-0.016)
# Imports the libraries and read the data files
!pip install --upgrade gensim
# Imports the libraries and read the data files
!pip install --upgrade gensim

# ... (rest of the code)

# Elbow Method
nc = range(1, 10)
kmeans = [KMeans(n_clusters=i, n_init=100, max_iter=500) for i in nc]

# Check if d2v_model.docvecs has a 'vectors' attribute
if hasattr(d2v_model.docvecs, 'vectors'):
    score = [-kmeans[i].fit(d2v_model.docvecs.vectors).score(d2v_model.docvecs.vectors) for i in range(len(kmeans))]
else:
    # Use the correct attribute name based on the updated Gensim documentation
    score = [-kmeans[i].fit(d2v_model.dv.vectors).score(d2v_model.dv.vectors) for i in range(len(kmeans))]
# # Elbow Method
# nc = range(1, 10)
# kmeans = [KMeans(n_clusters=i, n_init=100, max_iter=500) for i in nc]
# score = [-kmeans[i].fit(d2v_model.docvecs.vectors_docs).score(d2v_model.docvecs.vectors_docs) for i in range(len(kmeans))]  # Negate the score to make it positive
# Plot the elbow
plt.plot(nc, score)
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('Elbow Curve')
plt.show()
# Apply K-means clustering on the model
K_value = 4
kmeans_model = KMeans(n_clusters=K_value, init='k-means++', n_init=2000, max_iter=6000)
X = kmeans_model.fit(d2v_model.dv.vectors)
labels = kmeans_model.labels_.tolist()
clusters = kmeans_model.fit_predict(d2v_model.dv.vectors)
# PCA
l = kmeans_model.fit_predict(d2v_model.dv.vectors)
pca = PCA(n_components=2).fit(d2v_model.dv.vectors)
datapoint = pca.transform(d2v_model.dv.vectors)
# GRAPH
# """**Plot the clustering result**"""

plt.figure
label1 = ["#FFFF00", "#008000", "#0000FF", "#800080"]
color = [label1[i] for i in labels]
plt.scatter(datapoint[:, 0], datapoint[:, 1], c=color)

centroids = kmeans_model.cluster_centers_
centroidpoint = pca.transform(centroids)
plt.scatter(centroidpoint[:, 0], centroidpoint[:, 1], marker='^', s=150, c='#000000')
plt.show()
