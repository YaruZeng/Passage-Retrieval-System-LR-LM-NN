import numpy as np
import xgboost as xgb
import task1_bm25_retrieval
import task1_evaluation
import pandas as pd



def params_tuning(x_train, y_train): # tune parameters of LambdaMART model

    dtrain = xgb.DMatrix(x_train, label=y_train)

    # define parameters we tune
    gridsearch_params = [
        (max_depth, min_child_weight)
        for max_depth in range(2,6)
        for min_child_weight in range(1,5)
    ]

    # define initial params 
    params = {'max_depth': 2,
              'min_child_weight': 1,
              'objective':'rank:pairwise', 
              'eval_metric':["ndcg", "map"]}

    # define initial map and best params dict
    max_map = 0.0
    best_params = {'max_depth': 2,
             'min_child_weight': 1,
             'objective':'rank:pairwise', 
             'eval_metric':["ndcg", "map"]}

    for max_depth, min_child_weight in gridsearch_params:
        
        # update parameters of the model
        params['max_depth'] = max_depth
        params['min_child_weight'] = min_child_weight
        
        print(f"CV with max_depth={max_depth}, min_child_weight={min_child_weight}")
        
        # run model by CV
        cv_results = xgb.cv(
            params,
            dtrain,
        )

        tmp_map = cv_results['test-map-mean'].max()
        print("\tMAP: {}".format(tmp_map))
        
        if tmp_map > max_map: # update best params on max map
            max_map = tmp_map
            best_params['max_depth'] = max_depth
            best_params['min_child_weight'] = min_child_weight
            
    print("Best params: {}, {}, MAP: {}".format(best_params['max_depth'], best_params['min_child_weight'], max_map))

    # train to get the best model using the best parameters
    best_model = xgb.train(
        best_params,
        dtrain,
    )

    return best_model



def evaluate_model(sample_data, y_predict): # evaluate model with matrics mAP and NDCG from task1
    
    qid_candidatepid = {}
    sample_qid = sample_data["qid"].unique()
    for qid in sample_qid:
        qid_candidatepid[qid] = list(sample_data["pid"][sample_data["qid"]==qid])
        
    relevancy_predict = {}
    for qid in qid_candidatepid.keys():
        candidate_pid = qid_candidatepid[qid]
        relevancy_predict[qid] = {}
        for pid in candidate_pid:
            relevancy_predict[qid][pid] = {}
    
    ind = 0
    for qid in qid_candidatepid.keys():
        for pid in qid_candidatepid[qid]:
            relevancy_predict[qid][pid] = y_predict[ind]
            ind += 1
            
    qid_list = relevancy_predict.keys()
    LM_predict_rank = task1_bm25_retrieval.output_data(relevancy_predict, qid_list)
    LM_predict_rank.to_csv("LM_predict_rank.csv", index = False, header = False)
    
    LM_ranked_file = 'LM_predict_rank.csv'
    rank_rel = task1_evaluation.prepare_data(LM_ranked_file)
    mAP = task1_evaluation.compute_mAP(rank_rel, 100)
    NDCG = task1_evaluation.compute_NDCG(rank_rel, 100)

    return mAP, NDCG



if __name__ == "__main__":
    
    # prepare train data, validation data for XGBoost model

    x_train_LR = np.load('x_train.npy')
    y_train_LR = np.load('y_train.npy')

    x_valid = np.load('x_valid.npy')
    y_valid = np.load('y_valid.npy')

    # tune pramaters and get predicted result based on the best model
    
    best_model = params_tuning(x_train_LR, y_train_LR)
    dvalid = xgb.DMatrix(x_valid, label=y_valid)
    y_predict = best_model.predict(dvalid)

    # evaluate model performance using mAP and NDCG
    
    validation_data = pd.read_csv('validation_data.tsv', sep='\t', header=0)
    mAP, NDCG = evaluate_model(validation_data, y_predict)
    
    print("Evaluation Result: ")
    print(f"The mAP value of the LM model is {mAP}.")
    print(f"The NDCG value of the LM model is {NDCG}.")




