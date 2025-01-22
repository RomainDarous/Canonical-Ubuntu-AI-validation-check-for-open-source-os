# Generalized pooling formula is taken from the following research paper :
# "Enhancing Sentence Embedding with Generalized Pooling" Qian Chen, Zhen-Hua Ling, Xiaodan Zhu. COLING (2018)
# https://aclanthology.org/C18-1154.pdf

# Module created and code adapted following the Sentence Transformer documentation :
# https://sbert.net/docs/sentence_transformer/usage/custom_models.html


import math
import torch
from torch import nn
import torch.nn.functional as F
import os
import json

class MultiHeadGeneralizedPooling(nn.Module):
    # Pooling type
    ADDITIVE = 0
    DOT_PRODUCT = 1

    # Wieght initialization for additive pooling
    MEAN = 0
    NOISED = 1
    RANDOM = 2


    def __init__(self, pooling_type:int, token_dim: int = 768, sentence_dim: int = 768, num_heads: int = 8, initialize: int=2) -> None:
        """
        Initialize the MultiHeadGeneralizedPooling class based on multi-head pooling formula. If mean_pooling_init is True, initialize the pooling mechanism
        to behave like mean pooling (i.e., equal weights for all tokens across heads).
        
        Args:
            embedding_dim (int): The dimension of the token embeddings (output of the transformer).
            hidden_dim (int): The size of the hidden layer used in each head for the pooling computation.
            num_heads (int): The number of attention heads (I in the formula).
            initialize (str): Sets the initialization methods for the weights : 'mean' (0), 'noised' (1) or 'random' (2)
        """
        super(MultiHeadGeneralizedPooling, self).__init__()
        
        self.num_heads = num_heads
        self.head_dim = int(sentence_dim / self.num_heads)
        self.sentence_dim = sentence_dim
        self.token_dim = token_dim
        self.hidden_dim = 4 * self.head_dim
        self.initialize = initialize
        self.pooling_type = pooling_type
        
        # Initialize pooling
        if pooling_type == self.ADDITIVE :
            self.initialize_additive_pooling()
        elif pooling_type == self.DOT_PRODUCT :
            self.initialize_dot_product_pooling()
        else :
            raise ValueError(f"Unsupported pooling type: {self.pooling_type}")
    
    def initialize_dot_product_pooling(self) -> None :
        # Define learnable weights and biases for each head
        self.Q = nn.ModuleList([nn.Linear(self.head_dim, 1) for _ in range(self.num_heads)]) # Projection matrices to apply
        self.P_K = nn.ModuleList([nn.Linear(self.token_dim, self.head_dim) for _ in range(self.num_heads)]) # Projection matrices to apply

        # Initialize weights using Xavier initialization
        for i in range(self.num_heads):
            nn.init.xavier_uniform_(self.Q[i].weight)
            nn.init.zeros_(self.Q[i].bias)
            nn.init.xavier_uniform_(self.P_K[i].weight)
            nn.init.zeros_(self.P_K[i].bias)

    
    def forward(self, features: dict[str, torch.Tensor], **kwargs) -> dict   [str, torch.Tensor]:
        if self.pooling_type == self.ADDITIVE :
            return self.forward_additive(features, **kwargs)
        else : 
            return self.forward_dot_product(features, **kwargs)

    def forward_dot_product(self, features: dict[str, torch.Tensor], **kwargs) -> dict   [str, torch.Tensor]:
        attention_mask = features["attention_mask"].unsqueeze(-1)  # (batch_size, seq_len, 1)

        head_outputs = []  # To store output from each head
        H = features["token_embeddings"] # (batch_size, seq_len, token_dim)
        for i in range(self.num_heads):

            K_i = self.P_K[i](H) # (batch_size, seq_len, head_dim) for head i
            Q_i = self.Q[i]  # (batch_size, head_dim, 1) for head i
            A_i = Q_i(K_i) / math.sqrt(self.head_dim)  # (batch_size, seq_len, 1)
            A_i = F.softmax(A_i  + attention_mask.log(), dim=1)  # (batch_size, seq_len)

            # Apply attention weights
            v_i = torch.sum(K_i * A_i, dim=1)  # Weighted sum (batch_size, head_dim)
            head_outputs.append(v_i)  # Store the output of this head
        
        # Concatenate outputs from all heads along the embedding dimension
        pooled_output = torch.cat(head_outputs, dim=-1)  # (batch_size, num_heads * hidden_dim = self.token_dim)
        assert pooled_output.shape[1] == self.sentence_dim

        features["sentence_embedding"] = pooled_output
        return features  # Return the final multi-head pooled sentence embedding
    

    def initialize_additive_pooling(self) -> None:
        """
        Initialize weights to simulate mean pooling by making the attention distribution uniform for each head.
        """
        # Define learnable weights and biases for each head
        self.P = nn.ModuleList([nn.Linear(self.token_dim, self.head_dim) for _ in range(self.num_heads)]) # Projection matrices to apply
        self.W1 = nn.ModuleList([nn.Linear(self.head_dim, self.hidden_dim) for _ in range(self.num_heads)])  # W1^i for each head
        self.W2 = nn.ModuleList([nn.Linear(self.hidden_dim, self.head_dim) for _ in range(self.num_heads)])  # W2^i for each head

        if self.initialize == self.MEAN or self.initialize == self.NOISED :
            # Initialize all heads with weights that simulate mean pooling
            for i in range(self.num_heads):
                nn.init.constant_(self.W1[i].weight, 0)  # Set W1 weights to 0
                nn.init.constant_(self.W1[i].bias, 0)    # Set W1 bias to 0
                nn.init.constant_(self.W2[i].weight, 0)  # Set W2 weights to 0
                nn.init.constant_(self.W2[i].bias, 1)    # Set W2 bias to 1, ensuring equal output for each token
                
                nn.init.constant_(self.P[i].weight, 0)   # Initialize weight to identity matrix
                nn.init.eye_(self.P[i].weight[:, self.head_dim * i : self.head_dim * (i + 1)]) # Initialize the projections to successively be a slice of the original embedding matrix
                nn.init.constant_(self.P[i].bias, 0)     # Set bias to 0

                if self.initialize == self.NOISED :       
                    # Add small random perturbations
                    with torch.no_grad():
                        self.W1[i].weight.add_(torch.randn_like(self.W1[i].weight) * 0.01)
                        self.W1[i].bias.add_(torch.randn_like(self.W1[i].bias) * 0.01)
                        self.W2[i].weight.add_(torch.randn_like(self.W2[i].weight) * 0.01)
                        self.W2[i].bias.add_(torch.randn_like(self.W2[i].bias) * 0.01)
                        self.P[i].weight.add_(torch.randn_like(self.P[i].weight) * 0.01)
                        self.P[i].bias.add_(torch.randn_like(self.P[i].bias) * 0.01)

        elif self.initialize == self.RANDOM :
            # Initialize weights randomly
            for i in range(self.num_heads):
                nn.init.kaiming_uniform_(self.W1[i].weight, a=0)
                nn.init.zeros_(self.W1[i].bias)
                nn.init.kaiming_uniform_(self.W2[i].weight, a=0)
                nn.init.zeros_(self.W2[i].bias)
                nn.init.kaiming_uniform_(self.P[i].weight, a=0)
                nn.init.zeros_(self.P[i].bias)
        else :
            raise ValueError(f"Unsupported initialization type: {self.initialize}")

    def forward_additive(self, features: dict[str, torch.Tensor], **kwargs) -> dict   [str, torch.Tensor]:
        """
        Perform multi-head generalized pooling on the token embeddings using the given formula.
        
        Args:
            features (dict[str, torch.Tensor]): A dictionary containing:
            - "token_embeddings" (torch.Tensor): Token-level embeddings (batch_size, seq_len, token_dim).
            - "attention_mask" (torch.Tensor): Mask to ignore padding tokens (batch_size, seq_len).
            
            dict[str, torch.Tensor]: A dictionary with the pooled sentence embeddings under the key "sentence_embedding" (batch_size, num_heads * embedding_dim).
        Returns:
            torch.Tensor: The pooled sentence embeddings (batch_size, num_heads * embedding_dim).
        """
        attention_mask = features["attention_mask"].unsqueeze(-1)  # (batch_size, 1, seq_len)

        head_outputs = []  # To store output from each head
        
        for i in range(self.num_heads):

            H = features["token_embeddings"] # (batch_size, seq_len, token_dim)
            H_i = self.P[i](H) # Projecting H in a lower dimension
            A_i = self.W1[i](H_i)  # (batch_size, seq_len, hidden_dim) for head i
            A_i = F.relu(A_i)  # Apply ReLU activation

            # Second linear transformation: W2^i * ReLU(W1^i * H^T + b1^i)
            A_i = self.W2[i](A_i)  # (batch_size, seq_len, token_dim) for head i

            # Apply softmax to get attention weights for head i
            attention_mask_expanded = attention_mask.expand(-1, -1, self.head_dim)
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
        return {"sentence_dim": self.sentence_dim, "token_dim": self.token_dim, "num_heads": self.num_heads, "initialize": self.initialize, "pooling_type" : self.pooling_type}

    def get_sentence_embedding_dimension(self) -> int:
        return self.sentence_dim

    def save(self, save_dir: str, **kwargs) -> None:
        with open(os.path.join(save_dir, "config.json"), "w") as fOut:
            json.dump(self.get_config_dict(), fOut, indent=4)
    
    @staticmethod
    def load(load_dir: str, **kwargs) -> "MultiHeadGeneralizedPooling":
        with open(os.path.join(load_dir, "config.json")) as fIn:
            config = json.load(fIn)

        return MultiHeadGeneralizedPooling(**config)
