# This Python file uses the following encoding: utf-8

import os
from sentence_transformers import SentenceTransformer, util
import pickle
import torch
import random
import statistics

class Candidate_Ranker(object):
    
    def __init__(self, model_path = 'models'):
 
        self.candidates_path = 'data/candidates'
        with open (self.candidates_path, 'rb') as fp:
                self.candidates = pickle.load(fp)
        
        self.queries_path = 'data/queries'
        with open (self.queries_path, 'rb') as fp:
                self.queries = pickle.load(fp)
                
        print('#candidates: ', len(self.candidates))
        print('#queries: ', len(self.queries))
        
        self.bi_encoder = SentenceTransformer(model_path) #, device = 'cpu'
        self.candidates_embeddings = self.bi_encoder.encode(self.candidates, convert_to_tensor=True, device ='cuda')
        self.queries_embeddings = self.bi_encoder.encode(self.queries, convert_to_tensor=True, device ='cuda')
        
    # This function will search all standard symptoms for querys that
    # match the query
    def search_candidates(self, query, top_k = 5, return_scores= False, threshold = 0):
        question_embedding = self.bi_encoder.encode(query, convert_to_tensor=True, device ='cuda')
        hits = util.semantic_search(question_embedding, self.candidates_embeddings, top_k = top_k)
        hits = hits[0]
        if return_scores:
            hits = [(self.candidates[hit['corpus_id']],hit['score']) for hit in hits if hit['score']>threshold]
        else:
            hits = [self.candidates[hit['corpus_id']] for hit in hits if hit['score']>threshold]
        return hits
    
    def topk(self,ls, k):
        indices = range(len(ls))
        indices = sorted(indices, key = lambda x: ls[x], reverse = True)
        return indices[:k]

    def search_hard_negatives(self, anchor = 'query', query = None, true_candidate = None, sampling_method = 'topK', positive_threshold = 0.5, beta = 1,num_negatives = 1):
                                   
        if anchor == 'query':
           query_embeddings = self.bi_encoder.encode(query, convert_to_tensor=True, device ='cuda')
           hits = util.semantic_search(query_embeddings, self.candidates_embeddings, top_k=len(self.candidates))
           hits = hits[0]
           hits = [(self.candidates[hit['corpus_id']],min(max(hit['score'],0.001),1)) for hit in hits if self.candidates[hit['corpus_id']]!=true_candidate]
           hits_cp = [item for item in hits if item[1]>=positive_threshold] # positive predictions
        elif anchor == 'candidate':
           candidate_embedding = self.bi_encoder.encode(true_candidate, convert_to_tensor=True, device ='cuda')
           hits = util.semantic_search(candidate_embedding, self.queries_embeddings, top_k=len(self.queries))
           hits = hits[0]
           hits = [(self.queries[hit['corpus_id']],min(max(hit['score'],0.001),1)) for hit in hits if self.queries[hit['corpus_id']]!=query]
           hits_cp = [item for item in hits if item[1]>=positive_threshold] # positive predictions
        else:
           return None
                
        probs = [item[1] for item in hits_cp]
        candts = [item[0] for item in hits_cp]
        
        if sampling_method == 'topK':
           indices = self.topk(probs, num_negatives)
           
        elif sampling_method == 'topK_with_E-FN':
           hits_cp = [hit for hit in hits_cp if hit[1]<=beta]
           while len(hits_cp)<=num_negatives:
                   hits_cp.append(random.sample(hits,1)[0])
           probs = [item[1] for item in hits_cp]
           candts = [item[0] for item in hits_cp]
           indices = self.topk(probs, num_negatives)
           
        elif sampling_method == 'larger_than_true':
             score_true = self.cos_sim(query, true_candidate)[0]
             hits_cp = [hit for hit in hits_cp if hit[1]>=score_true]
             while len(hits_cp)<=num_negatives:
                   hits_cp.append(random.sample(hits,1)[0])
             probs = [item[1] for item in hits_cp]
             candts = [item[0] for item in hits_cp]
             indices = self.topk(probs, num_negatives)
             
        elif sampling_method == 'larger_than_true_with_E-FN':
             score_true = self.cos_sim(query, true_candidate)[0]
             hits_cp = [hit for hit in hits_cp if hit[1]>=score_true and hit[1]<=beta]
             while len(hits_cp)<=num_negatives:
                   hits_cp.append(random.sample(hits,1)[0])
             probs = [item[1] for item in hits_cp]
             candts = [item[0] for item in hits_cp]
             indices = self.topk(probs, num_negatives)
             
        elif sampling_method == 'multinomial':
           while len(hits_cp)<=num_negatives:
                   hits_cp.append(random.sample(hits,1)[0])
                   
           probs = [item[1] for item in hits_cp]
           candts = [item[0] for item in hits_cp]
           probs = torch.tensor(probs, dtype=torch.float)
           indices = torch.multinomial(probs, num_negatives, replacement=False)
           
        elif sampling_method == 'multinomial_with_E-FN':
           hits_cp = [hit for hit in hits_cp if hit[1]<=beta]
           while len(hits_cp)<=num_negatives:
                   hits_cp.append(random.sample(hits,1)[0])
           probs = [item[1] for item in hits_cp]
           candts = [item[0] for item in hits_cp]
           probs = torch.tensor(probs, dtype=torch.float)
           indices = torch.multinomial(probs, num_negatives, replacement=False)
           
        else:
           indices = random.sample(range(len(candts)), k = min(num_negatives,len(candts)))
           
        return [candts[i] for i in indices]

    def cos_sim(self, query, target):
        query_emb = self.bi_encoder.encode(query, convert_to_tensor=True, device ='cuda')
        target_emb = self.bi_encoder.encode(target, convert_to_tensor=True, device ='cuda')
        return util.cos_sim(query_emb,target_emb).tolist()[0]

if __name__=='__main__':
   cls = Candidate_Ranker()
   with open('data/testing_new','rb') as pf:
        test = pickle.load(pf)

   # 'testing search_candidates'
   print('Testing function search_candidates:')
   samples = [item[0] for item in test][:10]
   for t in samples:
       print(cls.search_candidates(t))
   
   print('\n')
   # 'testing search_hard_negatives'
   print('Testing function search_hard_negatives:')
   samples = [(item[0],item[[1][0]]) for item in test][:10]
   for t in samples:
       print(t)
       for hd_method in ['topK', 'topK_with_E-FN', 'larger_than_true', 'larger_than_true_with_E-FN', 'multinomial', 'multinomial_with_E-FN']:
           print('-----------------------------------------------------------------')
           print('Testing hd method: ',hd_method)
           print(cls.search_hard_negatives(anchor = 'query', query = t[0], true_candidate = t[1], sampling_method = hd_method,positive_threshold = 0.5, beta = 0.9,num_negatives = 3))
           print(cls.search_hard_negatives(anchor = 'candidate', query = t[0], true_candidate = t[1], sampling_method = hd_method,positive_threshold = 0.5, beta = 0.9,num_negatives = 3))
           print('-----------------------------------------------------------------')
       
   
   
   
