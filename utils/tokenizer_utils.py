import torch
from transformers import RobertaTokenizer, RobertaModel

def initialize_pretrained_tokenizer_and_model(model_name="microsoft/codebert-base"):
    """
    Generate Tokenizer and Model for tokenize code.
    
    Args:
        model_name (str): Name of the pre-trained model to use

    Returns:
        (tokenizer, model) (PreTrainedTokenizerBase, PreTrainedModel): Tokenizer, Model of pre-trained model with pre-defined name
    """

    # Initialize tokenizer and model
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaModel.from_pretrained(model_name)
    
    # Convert to evaluation mode
    model.eval()
    return tokenizer, model

def get_code_token_embeddings(code_tokens, tokenizer, model):
    """
    Generate embeddings for code tokens using CodeBERT.
    
    Args:
        code_tokens (List[str]): List of code tokens
    
    Returns:
        torch.Tensor: Token embeddings of shape [num_tokens, embedding_dim]
    """
    embeddings = []
    with torch.no_grad():
        for token in code_tokens:
            # Tokenize with special tokens
            encoded = tokenizer.encode(token, 
                                     add_special_tokens=True,
                                     return_tensors='pt')
            
            # Get embeddings from model
            outputs = model(encoded)
            
            # Use CLS token embedding as token representation
            token_embedding = outputs.last_hidden_state[:, 0, :]  # [1, embedding_dim]
            embeddings.append(token_embedding)
    
    # Stack all embeddings
    return torch.cat(embeddings, dim=0)  # [num_tokens, embedding_dim]

    