
from sentence_transformers import SentenceTransformer, SentencesDataset, LoggingHandler, losses, InputExample
from torch.utils.data import DataLoader
import torch
from torch import Tensor
from .MultiHeadGeneralizedPooling import MultiHeadGeneralizedPooling

class GeneralizedSentenceTransormer(SentenceTransformer):

    def __init__(self, base_model):
        super(GeneralizedSentenceTransormer, self).__init__()
        self.model = base_model
        self.generalized_pooling = MultiHeadGeneralizedPooling()  # Include the pooling method

    def forward(self, input: dict[str, Tensor], **kwargs) -> dict[str, Tensor]:
        input = super().forward(input)
        pooled_output = self.generalized_pooling(input['token_embeddings'])  # Apply generalized pooling
        return {'sentence_embedding': pooled_output}