# HiHDNCE.oy is the Python script for training our model Bi-hardNCE.
from torch.utils.data import DataLoader
import math
from sentence_transformers import SentenceTransformer,  LoggingHandler, losses, models, util
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import InputExample
import logging
import pandas as pd
import sys
import os

# OURS
from BidirectionalHardNegativesRankingLoss import BidirectionalHardNegativesRankingLoss # LOSS FUNCTION
from candidate_ranker import Candidate_Ranker # Loading the current model (or trained model at previous epoch)

# Configure logging
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

# Tune the beta value (threshold) for hard-negative mining (excluding possible false negatives).                   
def tune_beta(cls, validation, alpha):
    logging.info("Tuning beta: ")
    text1 = validation['text1'].tolist()
    text2 = validation['text2'].tolist()
    
    # Compute cosine similarities between text pairs
    similarities = cls.cos_sim(text1, text2)
    validation['cosine_similarities'] = similarities
    
    # Iterate over beta values until desired precision is reached
    temp = validation.copy()
    beta = 0.5
    while len(temp)>0:
          temp = temp[temp['cosine_similarities']>=beta] # predicted positives
          if len(temp)!=0 and len(temp[temp['label']==1])/len(temp) >= alpha:
              break
          else:
              beta = beta+0.015

    # Log the chosen beta value
    logging.info("Tuned beta is : "+ str(beta))
 
    # Write beta value to file
    result_save_path = 'results/'+sys.argv[1]+'_re.txt'
    textfile = open(result_save_path, "a")
    textfile.write("Epoch "+ sys.argv[2]+"\n")
    textfile.write('alpha: '+ str(alpha)+ "\n")
    textfile.write('beta: '+ str(beta)+ "\n")
    textfile.close()
    return beta

# Preprocess the dataset for training and validation
def preprocessing_dataset(cls, positive_threshold, beta, training, validation, train_batch_size, num_hard_negative_queries, num_hard_negative_candidates, sampling_method):

    logging.info("Pre-processing training/validation dataset: ")

    # Define a function to generate (positive, hard negative) pairs
    def bi_encoder(query, true_candidate):
        predictions_candidate = cls.search_hard_negatives(anchor = 'query', query = query, true_candidate = true_candidate, sampling_method = sampling_method, positive_threshold = positive_threshold, beta = beta, num_negatives = num_hard_negative_candidates)
        predictions_query = cls.search_hard_negatives(anchor = 'candidate', query = query, true_candidate = true_candidate, sampling_method = sampling_method, positive_threshold = positive_threshold, beta = beta, num_negatives = num_hard_negative_queries)
        
        return (predictions_candidate, predictions_query)

    # Define a function to create training examples        
    def contruct_training_instance(x):
        hard_negatives_sym, hard_negatives_query = bi_encoder(x['text1'], x['text2'])
        return InputExample(texts=[x['text1'], *hard_negatives_query, x['text2'], *hard_negatives_sym],label=float(x['label']))

    # Define a function to create validation examples
    def contruct_instance(x):
        return InputExample(texts=[x['text1'], x['text2']], label=float(x['label']))
    
    # Generate training examples    
    training['training_instances'] = training.apply(contruct_training_instance, axis = 1)
    train_examples = training['training_instances'].tolist()
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=train_batch_size)
    
    # Generate validation examples
    validation['validation_instances'] = validation.apply(contruct_instance, axis = 1)
    validation_examples = validation['validation_instances'].tolist()
    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(validation_examples, name='sym-dev')
    
    return (train_dataloader, evaluator)

# Constructing the model
def build_model(model_path, max_seq_length):
    logging.info("Build model from: " + model_path)
    word_embedding_model = models.Transformer(model_path,
                                              max_seq_length=max_seq_length)

    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                   pooling_mode_mean_tokens=True,
                                   pooling_mode_cls_token=False,
                                   pooling_mode_max_tokens=False)

    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    return model


# Training function 
def train_func(model, model_path, train_dataloader, evaluator, num_hard_negatives):
    logging.info("Training: ")
    
    # Use BidirectionalHardNegativesRankingLoss to train the model
    train_loss = BidirectionalHardNegativesRankingLoss(model=model, num_hard_negatives_query = num_hard_negatives)

    # Fit the model on the training datas
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              evaluator=evaluator,
              epochs=1,
              evaluation_steps=math.ceil(len(train_dataloader) * 0.25),
              warmup_steps=math.ceil(len(train_dataloader) * 0.1),
              output_path=model_path,
              show_progress_bar=True)



if __name__=='__main__':
   
   num_epoch = int(sys.argv[2])   
   model_path = 'models/'+sys.argv[1]

   logging.info("Loading datasets: ")
   training = pd.read_csv('data/training_ct.csv', sep = '\t')
   validation = pd.read_csv('data/validation.csv', sep = '\t')
   
   # hyper-parameters
   num_hard_negative_queries = 10
   num_hard_negative_candidates = 3
   train_batch_size = 32
   max_seq_length = 32
   positive_threshold = 0 if num_epoch == 0 else 0.5
   alpha = 0.8+ 0.015*num_epoch
   sampling_method = 'multinomial'
   #sampling_method: ['topK', 'topK_with_E-FN', 'larger_than_true', 'larger_than_true_with_E-FN', 'multinomial', 'multinomial_with_E-FN']
   
   # Loading the ranking function and tuning thresholds
   cls = Candidate_Ranker(model_path= model_path)
   beta = tune_beta(cls,validation, alpha)
   beta = 1 if num_epoch == 0 else beta
    
   # Generating hard negatives for training data 
   train_dataloader, evaluator = preprocessing_dataset(cls, positive_threshold, beta, training, validation, train_batch_size, num_hard_negative_queries, num_hard_negative_candidates, sampling_method)

   # Building the model and training the model
   model = build_model(model_path, max_seq_length)
   train_func(model, model_path, train_dataloader, evaluator, num_hard_negative_queries)
