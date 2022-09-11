import pandas as pd
import numpy as np
import task1_evaluation
import task1_bm25_retrieval
import task2_embeddings
from tqdm import tqdm


def load_embeddings(sample_data, sample_queries_embeddings, sample_passages_embeddings): # prepare embeddings for training logistic regression model

    queries_embeddings_arr = np.array(sample_queries_embeddings)
    passages_embeddings_arr = np.array(sample_passages_embeddings)

    sample_qid = sample_data["qid"].unique()
    sample_pid = sample_data["pid"].unique()

    # get embeddings of each query/passage
    queries_embeddings_dict = {}
    passages_embeddings_dict = {}

    for i in tqdm(range(len(sample_qid))):
        qid = sample_qid[i]
        queries_embeddings_dict[qid] = queries_embeddings_arr[i]

    for i in tqdm(range(len(sample_pid))):
        pid = sample_pid[i]
        passages_embeddings_dict[pid] = passages_embeddings_arr[i]

    # get candidata pids for each qid
    qid_candidatepid = {}
    for qid in sample_qid:
        qid_candidatepid[qid] = list(sample_data["pid"][sample_data["qid"]==qid])
    
    return queries_embeddings_dict, passages_embeddings_dict, qid_candidatepid



def generate_relevance_vector(queries_embeddings_dict, passages_embeddings_dict, qid_candidatepid): # compute feature vectors of the logistic regression model by connect two vectors

    relevance_dict = {}
    for qid in qid_candidatepid.keys():
        candidate_pid = qid_candidatepid[qid]
        relevance_dict[qid] = {}

        for pid in candidate_pid:
            query_vector = queries_embeddings_dict[qid]
            passage_vector = passages_embeddings_dict[pid]
            relevance_vector = np.append(query_vector, passage_vector)
            relevance_dict[qid][pid] = relevance_vector
            
    return relevance_dict



def generate_x_y(relevance_dict, sample_data): # generate x and y for the LR model
    
    x = np.zeros(shape=[len(sample_data), 200])
    y = np.zeros(shape=[len(sample_data),1])

    ind = 0 
    for qid, pid_vector in tqdm(relevance_dict.items()):
        for pid, score in pid_vector.items():
            x[ind] = relevance_dict[qid][pid]
            y[ind] = int(sample_data["relevancy"][(sample_data["qid"]==qid)&(sample_data["pid"]==pid)])
            ind += 1

    return x, y



def logistic_loss(y, sigmoid): # define loss function of LR model
    loss = -np.mean(y * np.log(sigmoid) + (1 - y) * np.log(1 - sigmoid))
    return loss


def sigmoid(z): # define sigmoid function of LR model
    sigmoid = 1/(1 + np.exp(-z))
    return sigmoid


def logistic_regression(x_train, y_train, iteration_num, learning_rate): # train LR model

    #get data structure
    features_num = x_train.shape[0]
    weight_num = x_train.shape[1]

    #initialise parameters
    w = np.zeros(shape=[weight_num,1])
    b = np.zeros(shape=[1,1])

    #training model
    for count in tqdm(range(iteration_num)):
        z = np.matmul(x_train, w) + b
        sig = sigmoid(z)
        djdz = sig - y_train
        djdw = (1/features_num) * np.matmul(x_train.T, djdz)
        djdb = np.sum(djdz)
        w = w - learning_rate * djdw
        b = b - learning_rate * djdb
        
    loss = logistic_loss(y_train, sig)
            
    return w, b, loss



def prepare_evaluation_data(file): # prepare validation data for evaluation

    validation_data = task2_embeddings.load_data(file)
    task2_embeddings.generate_embeddings()
    word_vectors_dict = task2_embeddings.load_embeddings()

    queries_tokens, passages_tokens = task2_embeddings.extract_tokens(validation_data)
    valid_queries_embeddings = task2_embeddings.compute_avg_embeddings(queries_tokens, word_vectors_dict)
    valid_passages_embeddings = task2_embeddings.compute_avg_embeddings(passages_tokens, word_vectors_dict)
    
    valid_queries_embeddings = pd.DataFrame(valid_queries_embeddings)
    valid_passages_embeddings = pd.DataFrame(valid_passages_embeddings)
    queries_embeddings_dict, passages_embeddings_dict, qid_candidatepid_valid = load_embeddings(validation_data, valid_queries_embeddings, valid_passages_embeddings)
    
    return queries_embeddings_dict, passages_embeddings_dict, qid_candidatepid_valid



def evaluation(valid_queries_embeddings, valid_passages_embeddings, valid_qid_candidatepid, w, b):
    
    valid_relevance_dict = generate_relevance_vector(valid_queries_embeddings, valid_passages_embeddings, valid_qid_candidatepid)
    x_valid, y_valid = generate_x_y(valid_relevance_dict, validation_data)
    
    relevancy_predict = {}
    
    # construct structure
    for qid in valid_qid_candidatepid.keys():
        candidate_pid = valid_qid_candidatepid[qid]
        relevancy_predict[qid] = {}
        for pid in candidate_pid:
            relevancy_predict[qid][pid] = {}
       
    # evaluating
    z = np.matmul(x_valid, w) + b
    y_predict = sigmoid(z)
        
    ind = 0 
    for qid, pid_relevancy in relevancy_predict.items():
        for pid, relevancy in pid_relevancy.items():
            relevancy_predict[qid][pid] = y_predict[ind][0]
            ind += 1
            
    qid_list = relevancy_predict.keys()
    LR_predict_rank = task1_bm25_retrieval.output_data(relevancy_predict, qid_list)
    LR_predict_rank.to_csv("LR_predict_rank.csv", index = False, header = False)
    
    LR_ranked_file = 'LR_predict_rank.csv'
    rank_rel = task1_evaluation.prepare_data(LR_ranked_file)
    mAP = task1_evaluation.compute_mAP(rank_rel, 100)
    NDCG = task1_evaluation.compute_NDCG(rank_rel, 100)

    return mAP, NDCG, x_valid, y_valid


if __name__ == "__main__":
    
    # load data for LR model training and evaluating
    validation_data = pd.read_csv('validation_data.tsv', sep='\t', header=0)
    train_data = pd.read_csv("train_data_sample.csv")
    train_queries_embeddings_table = pd.read_csv('train_queries_embeddings.csv', header=None)
    train_passages_embeddings_table = pd.read_csv('train_passages_embeddings.csv', header=None)

    # generate x and y for training LR model
    train_queries_embeddings, train_passages_embeddings, train_qid_candidatepid = load_embeddings(train_data, train_queries_embeddings_table, train_passages_embeddings_table)
    train_relevance_dict = generate_relevance_vector(train_queries_embeddings, train_passages_embeddings, train_qid_candidatepid)
    x_train, y_train = generate_x_y(train_relevance_dict, train_data)

    # train model and observe the impact of learning rates on the model training loss
    iteration_num = 2000
    learning_rate_list = [0.005, 0.001, 0.0005, 0.0001]
    training_loss_list = []
    for learning_rate in learning_rate_list:
        w, b, training_loss = logistic_regression(x_train, y_train, iteration_num, learning_rate)
        training_loss_list.append(training_loss)
        print(f"learning_rate: {learning_rate}, training_loss: {training_loss}")


    # prepare data for evaluation
    file = 'validation_data.tsv'
    valid_queries_embeddings, valid_passages_embeddings, valid_qid_candidatepid = prepare_evaluation_data(file)

    # evaluate model with learning_rate = 0.0001 using validation data
    mAP, NDCG, x_valid, y_valid = evaluation(valid_queries_embeddings, valid_passages_embeddings, valid_qid_candidatepid, w, b)

    print("Evaluation Result: ")
    print(f"The mAP value of the LR model is {mAP}.")
    print(f"The NDCG value of the LR model is {NDCG}.")

    # save train and validation data for task4 model
    np.save('x_train.npy', x_train)
    np.save('y_train.npy', y_train)

    np.save('x_valid.npy', x_valid)
    np.save('y_valid.npy', y_valid)





