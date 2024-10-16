class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, embeddings, labels):
        # Example: using cosine similarity for a supervised task
        loss = 1 - nn.functional.cosine_similarity(embeddings, labels)
        return loss.mean()
