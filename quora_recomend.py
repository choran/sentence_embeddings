import tensorflow as tf
import tensorflow_hub as hub
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import random
import argparse
from sklearn.decomposition import PCA

# Setup command line so that you can enter a new question
parser = argparse.ArgumentParser(description='Find the most similair quora questions to your query')
parser.add_argument('-q', dest='question', action='store',
                    help='New question')
parser.add_argument('-r', dest='recommend', action='store', type=int,
                    help='Number of recommended similar questions')
args = parser.parse_args()

# Import the Universal Sentence Encoder's TF Hub module
module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"
embed = hub.Module(module_url)

# Set the relative paths
use_path = "dataset/"
quora_file = "quora_sample_clusters.csv"

# Use a TF placeholder
sts_input1 = tf.placeholder(tf.string, shape=(None))
sts_input2 = tf.placeholder(tf.string, shape=(None))

# For evaluation we use exactly normalized rather than
# approximately normalized.
sts_encode1 = tf.nn.l2_normalize(embed(sts_input1), axis=1)
sts_encode2 = tf.nn.l2_normalize(embed(sts_input2), axis=1)

# Get cosine similarity for comparison
cosine_similarities = tf.reduce_sum(tf.multiply(sts_encode1, sts_encode2), axis=1)
clip_cosine_similarities = tf.clip_by_value(cosine_similarities, 0.0, 1.0)
sim_scores = 1.0 - tf.divide(tf.acos(clip_cosine_similarities), 3.14)

def get_quora_qs():
    quora_path = use_path + quora_file
    quora_qs = pd.read_csv(quora_path)
    return(quora_qs)

def get_scores(session, questions):
    """Returns the similarity scores"""
    emba, embb, scores = session.run(
        [sts_encode1, sts_encode2, sim_scores],
        feed_dict={
            sts_input1: questions['new_query'].tolist(),
            sts_input2: questions['query'].tolist()
        })
    return (emba, embb, scores)

def get_parameters(df):
    # Check if users entered any command line parameters
    if (args.question) is not None:
        test_q = args.question
        same_qs = [test_q] * len(df)
    else:
        # Select a random question from the list
        rand = random.randint(0, len(df)-1)
        same_qs = [df.iloc[rand]['query']] * len(df)        
    if (args.recommend) is not None:
        num = args.recommend
    else:
        num = 5
    return(same_qs, num)

def bar_scores(rec_df):
    objects = list(range(1, len(rec_df['query'].tolist()) +1))
    y_pos = np.arange(len(objects))
    performance = rec_df.index.values.tolist()
    plt.figure(figsize=(15,10))
    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel(rec_df['new_query'].tolist()[0])
    plt.title('Top 5 Recommendations')
    loc = -0.1
    for i,res in enumerate(rec_df['query'].tolist()):
        plt.text(-0.7, loc, '{0}: {1}'.format(i+1, res), fontsize=15)
        loc-=0.1
    plt.savefig("similar_qs.png", bbox_inches = "tight")
    plt.show()
        
def pca_transform(recs):
    # Convert the 512 dimensions into 2 so we can represent them in a graph
    pca = PCA(2)  # project from 512 to 2 dimensions
    queries = recs['query'].tolist()
    queries.append(recs['new_query'].tolist()[0])
    
    embeds1 = recs['emba'].tolist()
    embeds2 = recs['embb'].tolist()
    embeds2.append(embeds1[0])
    projected = pca.fit_transform(embeds2)
    plt.figure(figsize=(15,10))
    # Create a DF of groups of lablels
    # Get the 2D embeds from each group of similar labels
    x,y =zip(*projected.tolist())
    plt.scatter(x,y)
    # Set a limit so there is some room for the points
    plt.xlim(-0.8, 0.8)
    plt.ylim(-0.8, 0.8)
    for (i, (x,y)) in enumerate(zip(x,y)):
        plt.text(x,y,queries[i], ha='center')
    plt.xlabel(recs['new_query'].tolist()[0])
    plt.ylabel(recs['new_query'].tolist()[0])
    plt.savefig("example_clusters.png")   
    
with tf.Session() as session:
    qs_df = get_quora_qs()
    new_query, top_qs = get_parameters(qs_df)
    # Add new question column to DF
    qs_df["new_query"] = new_query 
    # Init the TF variables
    session.run(tf.global_variables_initializer())
    session.run(tf.tables_initializer())
    # Get the similarity score
    emba, embb, scores = get_scores(session, qs_df)
    # Add the similarity scores to the DF
    qs_df['sim_score'] =  scores
    # Add the embeddings to the DF
    qs_df['emba'] = np.array(emba).tolist()
    qs_df['embb'] = np.array(embb).tolist()
    # Now sort them so we can get the top five closest matches
    sort_by_most_similar = qs_df.sort_values('sim_score', ascending=False)
    for i, s in enumerate(sort_by_most_similar.round(4).head(n=top_qs).iterrows()):
        print('{:2}: {}'.format(i+1, s[1][0]))
    sort_by_most_similar = sort_by_most_similar.set_index('sim_score')
    (sort_by_most_similar.head(n=top_qs)[['new_query', 'query', 'answer_group']]).to_csv('recommend.csv', float_format='%.4f')
    pca_transform(sort_by_most_similar.head(n=top_qs))
    bar_scores(sort_by_most_similar.head(n=top_qs))
