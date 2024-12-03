
# Semantic Text Similarity fine-tuning
On two different models, using two different types of pooling

### Mean Pooling
Mean Pooling + a Dense Layer
The method used to perform the sentence embedding is the mean pooling

## MultiHead Generalized Pooling
Mean Pooling leads to a non-neglitcible information loss, as some parts of the sentence are more meaningful than others.
Hence, we try to fine-tune a model by replacing the mean pooling by a trainable pooling using multi-head attetnion, initialized by mean pooling.

- Only changing the pooling parameters
- Also fine-tuning the LLM parameters 

To decide

## Fine-tuning process
- Fine-tuning performed in the research paper
- Second fine-tuning with the specific dataset

1. One with mean pooling
2. One with generalized pooling fine-tuning everything
3. Ine with generalized pooling fine-tuning only the pooling