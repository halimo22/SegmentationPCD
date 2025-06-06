from torchvision.models import efficientnet_b7
from torch import nn

class EfficientNetB7Classifier(nn.Module):
    def __init__(self, num_classes=8):
        super(EfficientNetB7Classifier, self).__init__()
        self.model = efficientnet_b7()
        num_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
            
        )
        self.class_names = ['Beam','Ceiling','Column', 'Floor', 'Pipe','Stairs', 'Wall']

    
    def forward(self, x):
        return self.model(x)
