# Passage-Retrieval-System-Improvement
The project is based on the project named 'Passage-Retrieval-System' in my repository. It aims to develop information retrieval models to solve the problem of passage retrieval. 

## Project Introduction 
Techniques of information retrieval are applied widely on searching engines, question answering, and recommendations. To improve the basic models implemented in the first project, this project constructs Logistic Regression Model, LambdaMART Model, and Neural Network Model to predict the relevance of queries and passages. Finally, the performance of the models are evaluated by Mean Average Precision(MAP) and Normalized Discounted Cumulative Gain(NDCG).

## Data sources
### 1. test-queries.tsv (200 rows)
A tab separated file, where each row contains a test query identifier (qid) and the actual query text.

### 2. candidate-passages-top1000.tsv (189877 rows)
A tab separated file with an initial selection of at most 1000 passages for each of the queries in 'test-queries.tsv'. The format of this file is <qid pid query passage>, where pid is the identifier of the passage retrieved, query is the query text, and passage is the passage text (all tab separated). The passages contained in this file are the same as the ones in passage-collection.txt. 

### 3. train_data.tsv (1048576 rows) and validation_data.tsv (1048576 rows)
These are the datasets used for training and validation. The models are trained on the training set and their performances are evaluated on the validation set. In these datasets, additional relevance columns are given indicating the relevance of the passage to the query. The formats of both files are 'qid pid query passage relevancy'. 

## Deliveries

### 1. task1.py
The metrics of Mean Average Precision(MAP) and Normalized Discounted Cumulative Gain(NDCG) are computed to evaluate the performance of BM25 model implemented in the first project. 

### 2. task2.py
Word2Vec is used to generate embeddings of words, which then are utilised to compute query/passage embeddings by averaging embeddings of all the words in that query/passage. Then Logistic Regression(LR) is trained using feature vectors derived from query and passage embeddings. Finally, the performance of LR model is evaluated by the MAP and NDCG methods on task1.

### 3. task3.py
The LambdaMART(LM) learning-to-rank algorithm from XGBoost gradient boosting library is used to learn a model that can re-rank passages. 

### 4. task4.py
This part of work utilises Tensorflow to build a neural network based model with the ability of re-ranking passages. The performance of the model is evaluated by MAP and NDCG on the validation data set.


For more details of the input processing, features or representations chosen, hyper-parameter tuning method, and neural architecture chosen while implementing the models and modeling, please check on the 'Report.pdf' file. 
