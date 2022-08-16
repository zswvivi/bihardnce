from torch.utils.data import DataLoader
import math
from sentence_transformers import SentenceTransformer,  LoggingHandler, losses, models, util
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import InputExample
import logging
import pandas as pd
import sys
import os
from candidate_ranker import Candidate_Ranker

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
                    
def tune_beta(cls, validation, alpha):
    logging.info("Tuning beta: ")
    text1 = validation['text1'].tolist()
    text2 = validation['text2'].tolist()
    similarities = cls.cos_sim(text1, text2)
    validation['cosine_similarities'] = similarities
    #validation_data.apply(lambda x: cls.cos_sim(x['text1'], x['text2']), axis = 1)
    temp = validation.copy()
    beta = 0.5
    while len(temp)>0:
          temp = temp[temp['cosine_similarities']>=beta] # predicted positives
          if len(temp)!=0 and len(temp[temp['label']==1])/len(temp) >= alpha:
              break
          else:
              beta = beta+0.015
    logging.info("Tuned beta is : "+ str(beta))
    result_save_path = 'results/'+sys.argv[1]+'_re.txt'
    textfile = open(result_save_path, "a")
    textfile.write("Epoch "+ sys.argv[2]+"\n")
    textfile.write('alpha: '+ str(alpha) + "\n")
    textfile.write('beta: '+ str(beta)+ "\n")
    textfile.close()
    return beta


def preprocessing_dataset(cls, positive_threshold, beta, training, validation, train_batch_size, num_hard_negative_candidates):

    logging.info("Pre-processing training/validation dataset: ")
    def bi_encoder(query, true_candidate):
        predictions_candidate = cls.search_hard_negatives(anchor = 'query', query = query, true_candidate = true_candidate, sampling_method = 'multinomial_with_E-FN', positive_threshold = positive_threshold, beta = beta, num_negatives = num_hard_negative_candidates)
        return predictions_candidate
        
    def contruct_training_instance(x):
        if num_hard_negative_candidates == 0:
             return InputExample(texts=[x['text1'], x['text2']],label=float(x['label']))
        else:
             hard_negatives_sym = bi_encoder(x['text1'], x['text2'])
             return InputExample(texts=[x['text1'], x['text2'], *hard_negatives_sym],label=float(x['label']))

    def contruct_instance(x):
        return InputExample(texts=[x['text1'], x['text2']], label=float(x['label']))
        
    training['training_instances'] = training.apply(contruct_training_instance, axis = 1)
    train_examples = training['training_instances'].tolist()
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=train_batch_size)
    
    validation['validation_instances'] = validation.apply(contruct_instance, axis = 1)
    validation_examples = validation['validation_instances'].tolist()
    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(validation_examples, name='sym-dev')
    
    return (train_dataloader, evaluator)




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


def train_func(model, model_path, train_dataloader, evaluator):
    logging.info("Training: ")
    train_loss = losses.MultipleNegativesRankingLoss(model)
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
   
   #training = training.sample(n=100) 
   # hyper-parameters
   use_hard_negative = True
   num_hard_negative_candidates = 3 if use_hard_negative else 0
   train_batch_size = 32 if use_hard_negative else 64
   max_seq_length = 32
   positive_threshold = 0 if num_epoch == 0 else 0.5
   alpha = 0.8+ 0.015*num_epoch
   sampling_method = 'multinomial_with_E-FN'
   #sampling_method: ['topK', 'topK_with_E-FN', 'larger_than_true', 'larger_than_true_with_E-FN', 'multinomial', 'multinomial_with_E-FN']
   
   cls = Candidate_Ranker(model_path= model_path)
   beta = tune_beta(cls,validation, alpha) if use_hard_negative else 1
   beta = 1 if num_epoch == 0 else beta
   
   train_dataloader, evaluator = preprocessing_dataset(cls, positive_threshold, beta, training, validation, train_batch_size, num_hard_negative_candidates)
   model = build_model(model_path, max_seq_length)
   train_func(model, model_path, train_dataloader, evaluator)
