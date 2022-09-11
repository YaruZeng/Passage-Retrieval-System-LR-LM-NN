import pandas as pd
from nltk.stem.porter import *
import math
import task1_preprocess
import json


def load_data(): # load data for analysis

    with open("inverted_index.json") as file:
        inverted_index_list = json.load(file)
        inverted_index = dict(inverted_index_list)

    with open("tf_queries.json") as file:
        tf_queries_list = json.load(file)
        tf_queries = dict(tf_queries_list)

    with open("tf_passages.json") as file:
        tf_passages_list = json.load(file)
        tf_passages = dict(tf_passages_list)

    with open("f_queries.json") as file:
        f_queries_list = json.load(file)
        f_queries = dict(f_queries_list)
    
    return inverted_index, tf_queries, tf_passages, f_queries


def bm(inverted_index, qid_list, qid_candidatepid, f_queries): # compute BM25

    N = len(tf_passages)
    k1 = 1.2
    k2 = 100
    b = 0.75
    
    BM = {}
    n_queries = {}

    # construct the data structure
    for qid in qid_list:
        BM[qid] = {}
        n_queries[qid] = {}

    # coumpute dl and avdl
    dl = {}
    for pid, word_count in tf_passages.items():
        dl[int(pid)] = sum(word_count.values())

    avdl = sum(dl.values())/len(dl)

    # compoute BM25
    for qid, word_count in tf_queries.items():
        candidate_pid = qid_candidatepid[int(qid)]
        for pid in candidate_pid:
            BM[int(qid)][pid] = 0
            for word, count in word_count.items():

                passages_words = tf_passages[str(pid)].keys()

                if word in inverted_index.keys():
                    n = len(inverted_index[word]) 
                    if word in passages_words:
                        f = inverted_index[word][str(pid)]
                    else:
                        f = 0
                else:
                    n = 0

                if word in passages_words:
                    qf = tf_queries[qid][word]
                    K = k1*((1-b)+b*dl[pid]/avdl) 

                    item1 = (N-n+0.5)/(n+0.5)
                    item2 = ((k1+1)*f)/(K+f)
                    item3 = ((k2+1)*qf)/(k2+qf)

                    BM[int(qid)][pid] += math.log(item1+item2+item3)

    return BM


def output_data(data, qid_list): # output data

    qid_pid_score = pd.DataFrame() # create a dataframe to store data

    for qid in qid_list:
        pid_score = sorted(data[qid].items(),key=lambda x:x[1], reverse=True) # order by score reversely
        pid_score_table = pd.DataFrame(pid_score,columns = ["pid","score"])
        pid_score_table["qid"] = qid
        
        if len(pid_score_table)>=100:
            pid_score_table = pid_score_table.iloc[0:100,]
        
        qid_pid_score = pd.concat([qid_pid_score,pid_score_table])
        
    qid_pid_score = qid_pid_score[["qid","pid","score"]]
    
    return qid_pid_score


if __name__ == "__main__":

    pid_list, passages, qid_list, queries, qid_candidatepid = task1_preprocess.prepare_data("validation_data.tsv")
    inverted_index, tf_queries, tf_passages, f_queries = load_data()

    BM = bm(inverted_index, qid_list, qid_candidatepid, f_queries)

    bm_table = output_data(BM, qid_list)
    bm_table.to_csv("bm25.csv", index = False, header = False)




