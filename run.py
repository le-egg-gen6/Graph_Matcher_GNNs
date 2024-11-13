from model import (
    CodeCloneDetection
)

from utils import (
    parsers
)

import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from torch.nn import CrossEntropyLoss, MSELoss
import numpy as np
import logging
import os
import json
import random
from tqdm import tqdm
import logging

import multiprocessing

cpu_count = multiprocessing.cpu_count()

# Cài đặt logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

torch.cuda.empty_cache()
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def collate_features(batch):
    """
    Custom collate function for InputFeature objects
    Args:
        batch: List of InputFeature objects
    Returns:
        Dictionary containing batched data
    """
    return {
        'code1': [item.code1 for item in batch],
        'code2': [item.code2 for item in batch],
        'labels': torch.tensor([item.label for item in batch], dtype=torch.long),
        'lang': [item.lang for item in batch]
    }

class InputFeature:
    def __init__(self, code1, code2, label, lang="python"):
        self.code1 = code1
        self.code2 = code2
        self.label = label
        self.lang = lang

class CodePairDataset(Dataset):
    def __init__(self, args, file_path='dataset'):
        self.examples = []
        self.args = args
        
        logger.info("Creating features from index file at %s", file_path)
        
        # Load code data
        data_file = os.path.join(os.path.dirname(file_path), 'data.jsonl')
        url_to_code = {}
        
        try:
            with open(data_file) as f:
                for line in f:
                    try:
                        line = line.strip()
                        js = json.loads(line)
                        url_to_code[js['idx']] = js['func']
                    except json.JSONDecodeError:
                        logger.warning(f"Skipping invalid JSON line: {line}")
                        continue
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {data_file}")
            
        logger.info(f"Loaded {len(url_to_code)} code snippets")
        
        # Load and process pairs
        try:
            with open(file_path) as f:
                for line in f:
                    try:
                        line = line.strip()
                        url1, url2, label = line.split('\t')
                        
                        # Skip if either code is missing
                        if url1 not in url_to_code or url2 not in url_to_code:
                            continue
                            
                        # Convert label to integer
                        label = int(label)
                        
                        # Check if label is valid
                        if label in [0, 1, 2, 3]:
                            example = InputFeature(
                                code1=url_to_code[url1],
                                code2=url_to_code[url2],
                                label=label,
                                lang="java"  # Assuming Python, adjust if needed
                            )
                            self.examples.append(example)
                        else:
                            logger.warning(f"Skipping invalid label: {label}")
                    except ValueError:
                        logger.warning(f"Skipping malformed line: {line}")
                        continue
        except FileNotFoundError:
            raise FileNotFoundError(f"Index file not found: {file_path}")
            
        # Validate dataset size
        if len(self.examples) == 0:
            raise ValueError(
                "No valid examples found in dataset. "
                "Please check your data files and make sure they contain valid examples."
            )
            
        logger.info(f"Loaded {len(self.examples)} valid example pairs")
        
        # Subsample validation set if needed
        if 'valid' in file_path:
            sample_size = int(len(self.examples) * 0.1)
            self.examples = random.sample(self.examples, sample_size)
            logger.info(f"Subsampled to {len(self.examples)} validation examples")

    def __len__(self):
        return len(self.examples)
        
    def __getitem__(self, idx):
        return self.examples[idx]

class CloneDetectionTrainer:
    def __init__(
        self,
        model: CodeCloneDetection,
        args,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.device = device
        self.model = model.to(self.device)
        self.args = args
        
        self.optimizer = Adam(
            self.model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        
        # Loss functions
        self.similarity_loss = MSELoss()
        self.clone_type_loss = CrossEntropyLoss()
        
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> dict:
        self.model.train()
        total_loss = 0
        correct_predictions = 0
        total_samples = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')
        
        for batch_idx, batch in enumerate(progress_bar):
            self.optimizer.zero_grad()
            
            batch_losses = []
            batch_size = len(batch)
            for i in range(batch_size):
                similarity, attention, pred_type = self.model(
                    code1=batch['code1'][i],
                    code2=batch['code2'][i],
                    lang=batch['lang'][i]
                )
                
                # Chuyển tensors lên device
                similarity = similarity.to(self.device)
                attention = attention.to(self.device)
                target_similarity = torch.tensor(1.0).to(self.device)
                label = batch['labels'][i].to(self.device)
                
                # Calculate losses
                sim_loss = self.similarity_loss(similarity, target_similarity)
                
                clone_logits = self.model.clone_classifier(
                    torch.cat([similarity.unsqueeze(0), attention.mean(dim=0)])
                )
                type_loss = self.clone_type_loss(clone_logits.unsqueeze(0), label)
                
                total_loss = sim_loss + type_loss
                batch_losses.append(total_loss)
                
                if pred_type == label.item():
                    correct_predictions += 1
                total_samples += 1
                
            # Calculate mean loss and backpropagate
            loss = torch.stack(batch_losses).mean()
            loss.backward()
            
            # Clip gradients
            if self.args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.args.max_grad_norm
                )
                
            self.optimizer.step()
            
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{correct_predictions/total_samples:.4f}'
            })
            
        return {
            'loss': total_loss / len(train_loader),
            'accuracy': correct_predictions / total_samples
        }

def train(args, model, train_dataset, eval_dataset):
    """Main training function"""
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_features
    )
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_features
    )
    
    # Initialize trainer
    trainer = CloneDetectionTrainer(
        model=model,
        args=args,
        device=args.device
    )
    
    # Training loop
    best_eval_acc = 0
    early_stopping_counter = 0
    
    for epoch in range(args.num_train_epochs):
        # Train
        train_metrics = trainer.train_epoch(train_loader, epoch)
        
        # Evaluate
        eval_metrics = trainer.evaluate(eval_loader)
        
        # Logging
        logger.info(f"Epoch {epoch+1}/{args.num_train_epochs}")
        logger.info(f"Train Loss: {train_metrics['loss']:.4f}")
        logger.info(f"Train Accuracy: {train_metrics['accuracy']:.4f}")
        logger.info(f"Eval Loss: {eval_metrics['eval_loss']:.4f}")
        logger.info(f"Eval Accuracy: {eval_metrics['eval_accuracy']:.4f}")
        
        # Save best model
        if eval_metrics['eval_accuracy'] > best_eval_acc:
            best_eval_acc = eval_metrics['eval_accuracy']
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': trainer.optimizer.state_dict(),
                    'best_acc': best_eval_acc,
                },
                os.path.join(args.output_dir, 'best_model.pt')
            )
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            
        # Early stopping
        if early_stopping_counter >= args.patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break
            
    logger.info(f"Training finished. Best validation accuracy: {best_eval_acc:.4f}")

class Args:
    def __init__(self):
        self.learning_rate = 1e-4
        self.weight_decay = 1e-5
        self.train_batch_size = 32
        self.eval_batch_size = 32
        self.num_train_epochs = 10
        self.max_grad_norm = 1.0
        self.num_workers = min(4, cpu_count)
        self.output_dir = 'output'
        self.patience = 3  # Early stopping patience

def main():
    args = Args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize datasets
    train_dataset = CodePairDataset(args, file_path='dataset/train_100.txt') 
    eval_dataset = CodePairDataset(args, file_path='dataset/train_100.txt')
    
    # Initialize model with device
    model = CodeCloneDetection(
        parsers=parsers,
        hidden_channels=128,
        out_channels=96,
        device=device
    )
    
    # Train
    args.device = device
    train(args, model, train_dataset, eval_dataset)
    

if __name__ == "__main__":
    main()