import pandas as pd
import numpy as np
import math


def prepare_data(model_ranked_file): # prepare rankded data for evaluation

    validation_data = pd.read_csv('validation_data.tsv', sep='\t', header=0)

    rank = pd.read_csv(model_ranked_file, header=None)
    rank.columns = ['qid','pid','score']
    rank_rel = pd.merge(rank, validation_data, how = "left")
    rank_rel = rank_rel.drop(['queries'], axis=1)
    rank_rel = rank_rel.drop(['passage'], axis=1)

    return rank_rel


def compute_mAP(rank_rel, cutoff): # compute mean average precision

    avg_precision_list = []
    qid_list = rank_rel.loc[:,"qid"].unique()

    for qid in qid_list:
        data_per_query = rank_rel[rank_rel.loc[:,'qid']==qid]

        if len(data_per_query) <= cutoff:
            pass
        else:
            data_per_query = data_per_query.loc[:cutoff, :]

        data_per_query.index = range(len(data_per_query))
        
        for index in range(len(data_per_query)):
            if data_per_query.loc[index, 'relevancy'] == 0:
                data_per_query.loc[index,'precision'] = 0
            else:
                rank = index+1
                data_per_query.loc[index,'precision'] = data_per_query.loc[:(index+1), 'relevancy'].sum()/ rank
                
        rel_cnt = data_per_query.loc[:,'relevancy'].sum()
        if rel_cnt == 0:
            avg_precision = 0
        else:
            avg_precision = data_per_query.loc[:,'precision'].sum() / data_per_query.loc[:,'relevancy'].sum()
            
        avg_precision_list.append(avg_precision)
        
    mAP = np.average(avg_precision_list)

    return mAP


def compute_NDCG(rank_rel, cutoff): # compute NDCG

    NDCG_list = []
    qid_list = rank_rel.loc[:,"qid"].unique()

    for qid in qid_list:
        
        data_per_query = rank_rel[rank_rel.loc[:,'qid']==qid]
        
        if len(data_per_query) <= cutoff:
            pass
        else:
            data_per_query = data_per_query.loc[:cutoff, :]

        data_per_query.index = range(len(data_per_query))
        
        for index in range(len(data_per_query)):
            rank = index+1
            DCG_numerator = pow(2, data_per_query.loc[index,'relevancy']) - 1
            DCG_denominator = math.log((rank+1),2)
            data_per_query.loc[index,'DCG'] = DCG_numerator / DCG_denominator
            
            DCG = data_per_query.loc[:,'DCG'].sum()
            
        data_per_query.sort_values(by="relevancy", inplace=True, ascending=False)
        data_per_query.index = range(len(data_per_query))
        
        for index in range(len(data_per_query)):
            rank = index+1
            optDCG_numerator = pow(2, data_per_query.loc[index,'relevancy']) - 1
            optDCG_denominator = math.log((rank+1),2)
            data_per_query.loc[index,'optDCG'] = optDCG_numerator / optDCG_denominator
            
            optDCG = data_per_query.loc[:,'optDCG'].sum()
        
        if optDCG == 0:
            NDCG = 0
        else:
            NDCG = DCG/optDCG
            
        NDCG_list.append(NDCG)
        
    NDCG = np.average(NDCG_list)

    return NDCG


if __name__ == "__main__":

    # Compute the performance of using BM25 as the retrieval model using mAP and NDCG
    
    model_ranked_file = 'bm25.csv'
    rank_rel = prepare_data(model_ranked_file)
    mAP = compute_mAP(rank_rel, 100)
    NDCG = compute_NDCG(rank_rel, 100)

    print("Evaluation Result: ")
    print(f"The mAP value of the BM25 model is {mAP}.")
    print(f"The NDCG value of the BM25 model is {NDCG}.")







