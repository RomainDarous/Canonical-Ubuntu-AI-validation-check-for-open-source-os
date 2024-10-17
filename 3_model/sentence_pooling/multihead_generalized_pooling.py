# Generalized pooling formula is taken from the following research paper :
# "Enhancing Sentence Embedding with Generalized Pooling" Qian Chen, Zhen-Hua Ling, Xiaodan Zhu. COLING (2018)
# https://aclanthology.org/C18-1154.pdf


# Module created following the Sentence Transformer documentation :
# https://sbert.net/docs/sentence_transformer/usage/custom_models.html


import torch
from torch import nn
import torch.nn.functional as F
import os
import json

class MultiHeadGeneralizedPooling(nn.Module):
    def __init__(self, token_dim: int = 768, sentence_dim: int = 768, num_heads: int = 8, mean_pooling_init=True):
        """
        Initialize the MultiHeadGeneralizedPooling class based on multi-head pooling formula. If mean_pooling_init is True, initialize the pooling mechanism
        to behave like mean pooling (i.e., equal weights for all tokens across heads).
        
        Args:
            embedding_dim (int): The dimension of the token embeddings (output of the transformer).
            hidden_dim (int): The size of the hidden layer used in each head for the pooling computation.
            num_heads (int): The number of attention heads (I in the formula).
        """
        super(MultiHeadGeneralizedPooling, self).__init__()
        
        self.num_heads = num_heads
        self.head_dim = int(sentence_dim / self.num_heads)
        self.sentence_dim = sentence_dim
        self.token_dim = token_dim
        self.hidden_dim = 4 * self.head_dim

        # Define learnable weights and biases for each head
        self.W1 = nn.ModuleList([nn.Linear(self.head_dim, self.hidden_dim) for _ in range(num_heads)])  # W1^i for each head
        self.W2 = nn.ModuleList([nn.Linear(self.hidden_dim, self.head_dim) for _ in range(num_heads)])  # W2^i for each head
        self.P = nn.ModuleList([nn.Linear(self.token_dim, self.head_dim) for _ in range(num_heads)]) # Projection matrices to apply
        # Optionally initialize to behave like mean pooling
        if mean_pooling_init:
            self.initialize_mean_pooling()

    def initialize_mean_pooling(self):
        """
        Initialize weights to simulate mean pooling by making the attention distribution uniform for each head.
        """
        # Initialize all heads with weights that simulate mean pooling
        for i in range(self.num_heads):
            nn.init.constant_(self.W1[i].weight, 0)  # Set W1 weights to 0
            nn.init.constant_(self.W1[i].bias, 0)    # Set W1 bias to 0
            nn.init.constant_(self.W2[i].weight, 0)  # Set W2 weights to 0
            nn.init.constant_(self.W2[i].bias, 1)    # Set W2 bias to 1, ensuring equal output for each token
            
            nn.init.constant_(self.P[i].weight, 0)   # Initialize weight to identity matrix
            nn.init.eye_(self.P[i].weight[:, self.head_dim * i : self.head_dim * (i + 1)]) # Initialize the projections to successively be a slice of the original embedding matrix
            nn.init.constant_(self.P[i].bias, 0)     # Set bias to 0
    def forward(self, features: dict[str, torch.Tensor], **kwargs) -> dict   [str, torch.Tensor]:
        """
        Perform multi-head generalized pooling on the token embeddings using the given formula.
        
        Args:
            token_embeddings (torch.Tensor): Token-level embeddings (batch_size, seq_len, token_dim).
            attention_mask (torch.Tensor): Mask to ignore padding tokens (batch_size, seq_len).
            
        Returns:
            torch.Tensor: The pooled sentence embeddings (batch_size, num_heads * embedding_dim).
        """
        attention_mask = features["attention_mask"].unsqueeze(-1)  # (batch_size, seq_len, 1)

        head_outputs = []  # To store output from each head
        
        for i in range(self.num_heads):
            # Linear transformation with ReLU for each head: W1^i * H^T + b1^i
            H = features["token_embeddings"] # (batch_size, seq_len, token_dim)
            H_i = self.P[i](H)
            A_i = self.W1[i](H_i)  # (batch_size, seq_len, hidden_dim) for head i
            A_i = F.relu(A_i)  # Apply ReLU activation

            # Second linear transformation: W2^i * ReLU(W1^i * H^T + b1^i)
            A_i = self.W2[i](A_i)  # (batch_size, seq_len, token_dim) for head i

            # Apply softmax to get attention weights for head i
            attention_mask_expanded = attention_mask.repeat(1, 1, self.head_dim)
            A_i = F.softmax(A_i + attention_mask_expanded.log(), dim=1)  # Softmax along seq_len
            

            # Apply attention weights to get the weighted sum of token embeddings for head i
            v_i = torch.sum(H_i * A_i, dim=1)  # Weighted sum over seq_len (batch_size, token_dim)
            
            head_outputs.append(v_i)  # Store the output of this head
        
        # Concatenate outputs from all heads along the embedding dimension
        pooled_output = torch.cat(head_outputs, dim=-1)  # (batch_size, num_heads * hidden_dim = self.token_dim)
        assert pooled_output.shape[1] == self.sentence_dim

        features["sentence_embedding"] = pooled_output
        return features  # Return the final multi-head pooled sentence embedding
    
    def get_config_dict(self) -> dict[str, float]:
        return {"sentence dimension": self.sentence_dim, "number of heads": self.num_heads, "hidden dimension": self.head_dim}

    def get_sentence_embedding_dimension(self) -> int:
        return self.sentence_dim

    def save(self, save_dir: str, **kwargs) -> None:
        with open(os.path.join(save_dir, "config.json"), "w") as fOut:
            json.dump(self.get_config_dict(), fOut, indent=4)

    def load(self, load_dir: str, **kwargs) -> "MultiHeadGeneralizedPooling":
        with open(os.path.join(load_dir, "config.json")) as fIn:
            config = json.load(fIn)

        return MultiHeadGeneralizedPooling(**config)
