import pandas as pd
from nltk.stem.porter import *
import string
from nltk.corpus import stopwords
from collections import Counter
import json


def prepare_data(filepath): # prepara validation data for evalution
    
    validation_data = pd.read_csv(filepath, sep='\t', header=0)
    
    validation_passages_distinct = validation_data.drop_duplicates(subset='pid', keep='first', inplace=False)
    pid_list = validation_passages_distinct['pid'].tolist()
    passages = validation_passages_distinct['passage'].tolist()

    validation_queries_distinct = validation_data.drop_duplicates(subset='qid', keep='first', inplace=False)
    qid_list = validation_queries_distinct['qid'].tolist()
    queries = validation_queries_distinct['queries'].tolist()

    qid_candidatepid = {}
    for qid in qid_list:
        qid_candidatepid[qid] = list(validation_data["pid"][validation_data["qid"]==qid])
    
    return pid_list, passages, qid_list, queries, qid_candidatepid


def extract_terms(file): # extract tokens of docs

    porter = PorterStemmer()
    
    tokens = []
    extra = string.punctuation + "‘" + "’" + "“" + "”"
    for line in file:
        line = line.lower() # lower all characters
         # tokenisation (1-grams)
        for c in extra :
            line = line.replace(c, " ")# remove punctuations

        line_new = line.split()

        line_token = []
        for token in line_new:
            stemmed_token = porter.stem(token.strip()) # stemming
            line_token.append(stemmed_token)

        tokens.append(line_token)
    
    return tokens


def get_inverted_index(pid_list, passages): #generate inverted index
    
    passages_tokens = extract_terms(passages)

    inverted_index = {}
    stop_words = stopwords.words('english')

    for line in passages_tokens:
        for word in line:
            if word not in stop_words: # remove stop words
                inverted_index[word] = {}
          
    # get inverted index
    for ind_token in range(len(passages_tokens)): 
        pid = pid_list[ind_token]
        for word in passages_tokens[ind_token]:
            if word not in stop_words:
                inverted_index[word][pid] = inverted_index[word].get(pid,0) + 1
    
    # store inverted_index dist for further analysis
    with open("inverted_index.json", "w", newline='', encoding="UTF-8") as file:
        json.dump(inverted_index, file)

    return inverted_index


def tf_passages(inverted_index): # compute TF for passages
    
    tf_passages = {}

    for word, pid_count in inverted_index.items():
        for pid, count in pid_count.items():
            tf_passages[pid] = {}
            
    for word, pid_count in inverted_index.items():
        for pid, count in pid_count.items(): 
            tf_passages[pid][word] = tf_passages[pid].get(pid,0) + inverted_index[word][pid]
            
        # output data for later analysis
    with open("tf_passages.json", "w", newline='', encoding="UTF-8") as file:
        json.dump(tf_passages, file)


def tf_queries(qid_list, queries): # compute TF for queries

    tf_queries = {}
    query_tokens = extract_terms(queries)

    for qid in qid_list:
        tf_queries[qid] = {}

    for ind_token in range(len(query_tokens)):
        word_count = dict(Counter(query_tokens[ind_token]))

        qid = qid_list[ind_token]
        tf_queries[qid] = word_count

    # output data for later analysis
    with open("tf_queries.json", "w", newline='', encoding="UTF-8") as file:
        json.dump(tf_queries, file)


if __name__ == "__main__":

    pid_list, passages, qid_list, queries, qid_candidatepid = prepare_data("validation_data.tsv")
    
    inverted_index = get_inverted_index(pid_list, passages)

    tf_passages(inverted_index)

    tf_queries(qid_list, queries)



