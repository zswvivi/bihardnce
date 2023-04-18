# This Python script is our loss function

import torch
from torch import nn, Tensor
from typing import Iterable, Dict
from sentence_transformers import SentenceTransformer
from sentence_transformers import util

class BidirectionalHardNegativesRankingLoss(nn.Module):
    
    def __init__(self, model: SentenceTransformer, scale: float = 20.0, similarity_fct = util.cos_sim, num_hard_negatives_query: int = 0):

        super(BidirectionalHardNegativesRankingLoss, self).__init__()
        self.model = model
        self.scale = scale
        self.similarity_fct = similarity_fct
        self.num_hard_negatives_query = num_hard_negatives_query
        self.cross_entropy_loss = nn.CrossEntropyLoss()


    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):

        # Get the sentence embeddings for each sentence feature using the SentenceTransformer model
        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]

        # Concatenate the anchor embeddings (first num_hard_negatives_query+1 embeddings) and candidate embeddings (remaining embeddings)   
        anchor = torch.cat(reps[0:self.num_hard_negatives_query+1])
        candidates = torch.cat(reps[self.num_hard_negatives_query+1:])

        # Compute the pairwise cosine similarity between the anchor and candidate embeddings and scale the result
        scores = self.similarity_fct(anchor, candidates) * self.scale

        # Create a tensor for labels with the length of the first sentence embeddings
        labels = torch.tensor(range(len(reps[0])), dtype=torch.long, device=scores.device)

        # Split scores to anchor_positive_scores and candidates_positive_scores
        anchor_positive_scores = scores[:, 0:len(reps[1])]
        candidates_positive_scores = scores[0:len(reps[0]),:]

        # Compute the forward and backward loss using the cross-entropy loss function
        forward_loss = self.cross_entropy_loss(candidates_positive_scores, labels)
        backward_loss = self.cross_entropy_loss(anchor_positive_scores.transpose(0, 1), labels)

        # Return the average of the forward and backward loss
        return (forward_loss + backward_loss) / 2

    def get_config_dict(self):

        # Return a dictionary containing the scale and name of the similarity function
        return {'scale': self.scale, 'similarity_fct': self.similarity_fct.__name__}





