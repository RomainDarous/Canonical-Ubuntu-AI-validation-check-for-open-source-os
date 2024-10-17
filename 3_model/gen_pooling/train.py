import torch
from sentence_transformers import SentenceTransformer, models
from .MultiHeadGeneralizedPooling import MultiHeadGeneralizedPooling

# Step 1: Load the existing SentenceTransformer model
existing_model = SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased-v2")

# Extract the transformer and the last dense layer
transformer = existing_model[0]  # Transformer (DistilBERT)
dense_layer = existing_model[2]   # Dense layer


# Step 3: Create the new SentenceTransformer model
generalized_pooling = MultiHeadGeneralizedPooling(token_dim=transformer.get_word_embedding_dimension())

# Build the new model using the reused components
model = SentenceTransformer(modules=[transformer, generalized_pooling, dense_layer])

# Step 4: Print the new model architecture
print(model)


import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler
from sentence_transformers import SentenceTransformer
from torch.optim import Adam

# Assuming you have a custom Dataset
class CustomDataset(Dataset):
    def __init__(self, sentences, labels):
        self.sentences = sentences
        self.labels = labels

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx], self.labels[idx]

# Example data
sentences = ["Hello, world!", "This is a test sentence."]
labels = torch.tensor([[1.0], [0.0]])  # Example target labels

# Create dataset and dataloader
dataset = CustomDataset(sentences, labels)
dataloader = DataLoader(dataset, batch_size=2, sampler=RandomSampler(dataset))

# Initialize the model
optimizer = Adam(model.parameters(), lr=1e-5)
loss_fn = CustomLoss()

# Step 4: Training Loop
model.train()
for epoch in range(num_epochs):
    for batch_sentences, batch_labels in dataloader:
        optimizer.zero_grad()

        # Forward pass
        embeddings = model.encode(batch_sentences, convert_to_tensor=True)

        # Compute the loss
        loss = loss_fn(embeddings, batch_labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")
