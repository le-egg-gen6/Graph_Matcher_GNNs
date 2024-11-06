import torch
import torch.nn as nn

class CloneDetectionLoss(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        self.weights = weights if weights is not None else torch.ones(4)
        
    def forward(self, predictions, targets):
        # Cross entropy for multi-class classification
        ce_loss = nn.CrossEntropyLoss(weight=self.weights)(predictions, targets)
        
        # Add hierarchical constraints
        hierarchy_loss = 0.0
        probs = torch.softmax(predictions, dim=1)
        
        # Type-1 should have highest probability when present
        hierarchy_loss += torch.relu(probs[:, 1:] - probs[:, 0]).mean()
        
        # Type-2 should have higher probability than Type-3 and Type-4 when present
        hierarchy_loss += torch.relu(probs[:, 2:] - probs[:, 1]).mean()
        
        # Type-3 should have higher probability than Type-4 when present
        hierarchy_loss += torch.relu(probs[:, 3] - probs[:, 2]).mean()
        
        return ce_loss + 0.1 * hierarchy_loss