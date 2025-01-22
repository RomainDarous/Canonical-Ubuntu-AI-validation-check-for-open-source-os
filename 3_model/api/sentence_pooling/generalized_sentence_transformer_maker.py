from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Transformer, Pooling
from sentence_pooling import MultiHeadGeneralizedPooling

class GeneralizedSentenceTransformerMaker:
    def __init__(self, model_name: SentenceTransformer, pooling_type, initalize, device: str = 'cpu'):
        """A small class to build a generalized multilingual sentence embedding model.

        Args:
            model_name (SentenceTransformer): the original base model to build upon.
            device (str): the device to train the model on
        """
        # Step 1: Load the existing SentenceTransformer model
        self.existing_model = model_name
        self.device = device
        
        # Step 2: Extract the transformer and the dense layer
        self.transformer = self.existing_model[0]  # Transformer
        self.dense_layer = self.existing_model[2]  # Compression dense layer

        # Step 3: Initialize the custom pooling layer
        self.pooling = MultiHeadGeneralizedPooling(pooling_type, token_dim=self.transformer.get_word_embedding_dimension(), initialize=initalize)

        # Step 4: Build the new SentenceTransformer model with modified architecture
        self.new_model = SentenceTransformer(modules=[
            self.transformer, 
            self.pooling, 
            self.dense_layer
        ], device=device)

    def __str__(self) -> str:
        """Prints the architecture of the newly built model.

        Returns:
            str: A string describing the model architecture.
        """
        return str(self.new_model)
    
    def get_model(self) -> SentenceTransformer :
        """Returns the created model.

        Returns:
            SentenceTransformer: The created model.
        """
        return self.new_model