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
        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
   
        anchor = torch.cat(reps[0:self.num_hard_negatives_query+1])
        candidates = torch.cat(reps[self.num_hard_negatives_query+1:])

        scores = self.similarity_fct(anchor, candidates) * self.scale
        labels = torch.tensor(range(len(reps[0])), dtype=torch.long, device=scores.device)

        anchor_positive_scores = scores[:, 0:len(reps[1])]
        candidates_positive_scores = scores[0:len(reps[0]),:]

        forward_loss = self.cross_entropy_loss(candidates_positive_scores, labels)
        backward_loss = self.cross_entropy_loss(anchor_positive_scores.transpose(0, 1), labels)
        return (forward_loss + backward_loss) / 2

    def get_config_dict(self):
        return {'scale': self.scale, 'similarity_fct': self.similarity_fct.__name__}





