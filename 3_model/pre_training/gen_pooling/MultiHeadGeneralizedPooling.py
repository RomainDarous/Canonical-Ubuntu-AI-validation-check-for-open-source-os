import torch
from torch import nn
import torch.nn.functional as F

class MultiHeadGeneralizedPooling(nn.Module):
    def __init__(self, embedding_dim: int = 768, hidden_dim: int = 512, num_heads: int = 8):
        """
        Initialize the MultiHeadGeneralizedPooling class based on multi-head pooling formula.
        
        Args:
            embedding_dim (int): The dimension of the token embeddings (output of the transformer).
            hidden_dim (int): The size of the hidden layer used in each head for the pooling computation.
            num_heads (int): The number of attention heads (I in the formula).
        """
        super(MultiHeadGeneralizedPooling, self).__init__()
        
        self.num_heads = num_heads
        
        # Define learnable weights and biases for each head
        self.W1 = nn.ModuleList([nn.Linear(embedding_dim, hidden_dim) for _ in range(num_heads)])  # W1^i for each head
        self.W2 = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(num_heads)])  # W2^i for each head

    def forward(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Perform multi-head generalized pooling on the token embeddings using the given formula.
        
        Args:
            token_embeddings (torch.Tensor): Token-level embeddings (batch_size, seq_len, embedding_dim).
            attention_mask (torch.Tensor): Mask to ignore padding tokens (batch_size, seq_len).
            
        Returns:
            torch.Tensor: The pooled sentence embeddings (batch_size, num_heads * embedding_dim).
        """
        # Ensure attention_mask is of the correct shape (batch_size, seq_len, 1) and broadcastable
        attention_mask = attention_mask.unsqueeze(-1).float()  # (batch_size, seq_len, 1)

        head_outputs = []  # To store output from each head
        
        for i in range(self.num_heads):
            # Linear transformation with ReLU for each head: W1^i * H^T + b1^i
            H_T = token_embeddings.transpose(1, 2)  # (batch_size, embedding_dim, seq_len)
            H_T = self.W1[i](H_T)  # (batch_size, hidden_dim, seq_len) for head i
            H_T = F.relu(H_T)  # Apply ReLU activation

            # Second linear transformation: W2^i * ReLU(W1^i * H^T + b1^i)
            A_i = self.W2[i](H_T).squeeze(-1)  # (batch_size, seq_len) for head i

            # Apply softmax to get attention weights for head i
            A_i = F.softmax(A_i + attention_mask.log(), dim=-1)  # Softmax along seq_len

            # Apply attention weights to get the weighted sum of token embeddings for head i
            A_i = A_i.unsqueeze(-1)  # (batch_size, seq_len, 1) for broadcasting
            v_i = torch.sum(token_embeddings * A_i, dim=1)  # Weighted sum over seq_len (batch_size, embedding_dim)
            
            head_outputs.append(v_i)  # Store the output of this head
        
        # Concatenate outputs from all heads along the embedding dimension
        pooled_output = torch.cat(head_outputs, dim=-1)  # (batch_size, num_heads * embedding_dim)

        return pooled_output  # Return the final multi-head pooled sentence embedding
