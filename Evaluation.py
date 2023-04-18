
import pandas as pd
import pickle
from evaluation_metrics import *
from candidate_ranker import Candidate_Ranker
import sys

# Set path to the candidates data
candidates_path = 'data/candidates'

# Load the candidates data
with open (candidates_path, 'rb') as fp:
        candidates = pickle.load(fp)
print(len(candidates))
    
   
# Define a function to create true sets from a list of items 
def create_true_sets(items):
    vec = [0]*len(candidates)
    for item in items:
        if item in candidates:
            vec[candidates.index(item)] = 1
    return vec

# Define a function to create prediction sets from a list of items
def create_pred_sets(items):
    vec = [0]*len(candidates)
    for item in items:
        if item[0] in candidates:
            vec[candidates.index(item[0])] = item[1]
    return vec

if __name__=='__main__':

    model_name = 'models/'+sys.argv[1]
    num_epoch = sys.argv[2]
    
    print('Epoch ',num_epoch, '\n')
    # load testing data
    with open('./data/testing_new', 'rb') as fp:
         testing = pickle.load(fp)
    
    # current symptom matching model, bi_encoder + cross_encoder
    cls = Candidate_Ranker(model_path= model_name)
    
    def bi_encoder(query):
        predictions = cls.search_candidates(query, return_scores = True)
        return predictions
    

    true_sets = [item[1] for item in testing]
    golden_labels =  true_sets.copy()
    pred_sets = [bi_encoder(item[0]) for item in testing]
    predictions = pred_sets.copy()
    true_sets = np.array([create_true_sets(item) for item in true_sets])
    pred_sets = np.array([create_pred_sets(item) for item in pred_sets])

    # Calculate and print the ndcg score for the predicted sets
    ndcg = str(round(cal_ndcg(true_sets, pred_sets, topk = 5),4)*100)
    print('ndcg: ', ndcg)
    
    #golden_labels = d['label'].tolist()
    #predictions = d['pre'].tolist()
    predictions = [[item[0] for item in sublist] for sublist in predictions]
    true_set = [set(item) for item in golden_labels]
    pre_set = [set(item) for item in predictions]

    # Calculate the micro f1 score for the predicted sets and print it
    micro_f1 = str(cal_set_micro_f1(true_set, pre_set))
    print(micro_f1)

    result_save_path = 'results/'+sys.argv[1]+'_re.txt' #+model_name.replace("models/", "")+"_re.txt"
    textfile = open(result_save_path, "a")
    textfile.write('ndcg: '+ ndcg+ "\n")
    textfile.write(micro_f1 + "\n")
    textfile.close()
