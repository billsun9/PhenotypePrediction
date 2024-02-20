import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer  # Assuming you use Hugging Face Transformers


class DNABertLSTMModel(nn.Module):
    def __init__(self, pretrained_model_name, num_classes):
        super(DNABertLSTMModel, self).__init__()

        # Load pretrained DNA-BERT model
        self.dna_bert = AutoModel.from_pretrained(pretrained_model_name, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name, trust_remote_code=True)

        for param in self.dna_bert.parameters():
            param.requires_grad = True

        # Add three linear layers on top
        self.lstm = nn.LSTM(self.dna_bert.config.hidden_size, 256, batch_first=True)
        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Linear(128, num_classes)
        )

    def forward(self, input_seqs):
        # Forward pass through the DNA-BERT model
        inputs = self.tokenizer(input_seqs, return_tensors="pt", padding=True, truncation=True)
        outputs = self.dna_bert(**inputs)

        # Extract the last hidden states
        last_hidden_states = outputs[0]

        # Apply linear layers
        x = torch.mean(last_hidden_states, dim=1)  # Pooling (you may choose a different pooling strategy)
        x, _ = self.lstm(x)
        # Take the output from the last time step
        # x = x[:, -1, :]
        x = self.fc(x)

        return x

class DNABertMLPModel(nn.Module):
    def __init__(self, pretrained_model_name, num_classes, layer_size = 128):
        super(DNABertMLPModel, self).__init__()

        # Load pretrained DNA-BERT model
        self.dna_bert = AutoModel.from_pretrained(pretrained_model_name, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name, trust_remote_code=True)

        for param in self.dna_bert.parameters():
            param.requires_grad = True

        # Add three linear layers on top
        self.linear1 = nn.Linear(self.dna_bert.config.hidden_size, layer_size)
        self.linear2 = nn.Linear(layer_size, layer_size)
        self.linear3 = nn.Linear(layer_size, num_classes)

    def forward(self, input_seqs):
        # Forward pass through the DNA-BERT model
        inputs = self.tokenizer(input_seqs, return_tensors="pt", padding=True, truncation=True)
        outputs = self.dna_bert(**inputs)

        # Extract the last hidden states
        last_hidden_states = outputs[0]

        # Apply linear layers
        x = torch.mean(last_hidden_states, dim=1)  # Pooling (you may choose a different pooling strategy)
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = self.linear3(x)

        return x

