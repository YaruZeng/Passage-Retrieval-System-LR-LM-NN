from gensim.models import word2vec
import pandas as pd
import task1_preprocess
import numpy as np
from tqdm import tqdm


def load_data(file): # load data for analysis

    data = pd.read_csv(file, sep='\t', header=0)
    sentences_collection = pd.concat([data['queries'], data['passage']])
    sentences_collection.to_csv('sentences_collection.txt', index = False, header = False)

    return data


def generate_embeddings(): # represent passages and query based on Word2Vec

    sentences = word2vec.LineSentence('sentences_collection.txt')
    model = word2vec.Word2Vec(sentences, hs=1, min_count=2, window=3, vector_size=100)
    model.wv.save_word2vec_format('word_embeddings.txt', binary=False)


def load_embeddings(): # load word embeddings data as dictionary

    word_vectors_dict = {}

    with open('word_embeddings.txt', 'r') as f:
        word_vector = f.readlines()
        
        for i in tqdm(range(len(word_vector))):
            if i == 0:
                continue
            word_vector_list = word_vector[i].split()
            word_vectors_dict[word_vector_list[0]] = list(map(float,word_vector_list[1:]))

    print("load_embeddings")
    return word_vectors_dict


def train_data_sample(data): # generate a train data sample

    train_data = data

    train_data_1 = train_data[train_data["relevancy"]==1]
    train_data_0 = train_data[train_data["relevancy"]==0]

    train_data_sample = pd.concat([train_data_1, train_data_0.iloc[:1000000,:]])

    return train_data_sample


def extract_tokens(data): # extract tokens for queries and passages

    queries = data['queries'].unique()
    passages = data['passage'].unique()

    queries_tokens = task1_preprocess.extract_terms(queries)
    passages_tokens = task1_preprocess.extract_terms(passages)

    return  queries_tokens, passages_tokens


def compute_avg_embeddings(doc_tokens, word_vectors_dict): # compute average embedding for each query and passsage
    
    num_docs = len(doc_tokens)
    doc_embeddings = np.zeros(shape=[num_docs,100])

    for i in tqdm(range(num_docs)):
        # ID = id_list[i]
        tokens = doc_tokens[i]
        word_num = len(tokens)
        vectors_array = np.zeros(shape=[word_num,100])
        
        index = 0
        for word in tokens:
            if word in word_vectors_dict.keys():
                vectors_array[index] = np.array(word_vectors_dict[word])
                index += 1

        avg_vector = np.mean(vectors_array, axis=0)
        doc_embeddings[i] = avg_vector
    
    doc_embeddings_table = pd.DataFrame(doc_embeddings)

    print("compute_avg_embeddings")
    return doc_embeddings_table


if __name__ == "__main__":

    file = 'train_data.tsv'
    data_set = load_data(file)
    generate_embeddings()
    word_vectors_dict = load_embeddings()

    train_data_sample = train_data_sample(data_set)

    queries_tokens, passages_tokens = extract_tokens(train_data_sample)
    queries_embeddings_table = compute_avg_embeddings(queries_tokens, word_vectors_dict)
    passages_embeddings_table = compute_avg_embeddings(passages_tokens, word_vectors_dict)

    #output queries_embeddings and passages_embeddings for later analysis
    queries_embeddings_table.to_csv('train_queries_embeddings.csv', index=False, header=False)
    passages_embeddings_table.to_csv('train_passages_embeddings.csv', index=False, header=False)
    train_data_sample.to_csv('train_data_sample.csv', index=True, header=True)
