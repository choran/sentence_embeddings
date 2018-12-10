# Overview

These are files to investigate the new sentence embeddings models published by google on [TF Hub](https://tfhub.dev/google/universal-sentence-encoder-large/3)
It is besed on the paper [Universal Sentence Encoder](https://arxiv.org/pdf/1803.11175.pdf)

# File Overview

## VectorSpaceExample.ipynb
```
Simple example of how to represent 2D vector on a vector space and how visually we can identify patterns such as clustering
```
## ReducedVocabularyEmbeddings.ipynb
```
Using Principal Component Analysis (PCA) on one hot encoding example to generate a 2D graph of 4D points
```

## GenerateEmbedding.ipynb
```
Example of how to use the USE module to generate a sentence embedding and visualize the range of the 512 dimensions
```

## BaselineTest.ipynb
```
Code to measure the similarity between a baseline of curated queries and 1000 questions from the quora dataset.
It goes through each question and finds the best match for that question from the quora list
```

## VisualComparison.ipynb
```
Code to visually compare sentence embedding pairs via scatter plot and bar chart
```

### quora_recomend.py
```
Code to find top five recommendation of similar sentences from quora test list.
This is an example of how sentence embeddings coould be used to create a some recommender for a chatbot
```
