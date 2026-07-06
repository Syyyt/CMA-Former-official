import torch
import torch.nn as nn


class MultitaskAutoencoder(nn.Module):
    """Multitask Autoencoder for spatial filtering"""
    def __init__(self, input_dim, hidden_dim, num_decoders):
        super(MultitaskAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_decoders = num_decoders
        
        # Encoder (single layer, linear)
        self.encoder = nn.Linear(input_dim, hidden_dim)
        
        # Multiple decoders (one for each spatial subregion)
        self.decoders = nn.ModuleList([
            nn.Linear(hidden_dim, input_dim) for _ in range(num_decoders)
        ])
    
    def forward(self, x):
        # Encode
        encoded = self.encoder(x)
        
        # Decode with multiple decoders
        decoded_outputs = []
        for decoder in self.decoders:
            decoded = decoder(encoded)
            decoded_outputs.append(decoded)
        
        return decoded_outputs

class DOAClassifier(nn.Module):
    """Multilayer classifier for DOA estimation in each subregion"""
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(DOAClassifier, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.Tanh())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.classifier = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.classifier(x)

class DNNDOAEstimator(nn.Module):
    """Complete DNN framework for DOA estimation"""
    def __init__(self, input_dim, hidden_dim, num_subregions, grids_per_subregion, classifier_hidden_dims):
        super(DNNDOAEstimator, self).__init__()
        
        self.num_subregions = num_subregions
        self.grids_per_subregion = grids_per_subregion
        
        # Multitask autoencoder
        self.autoencoder = MultitaskAutoencoder(input_dim, hidden_dim, num_subregions)
        
        # Parallel classifiers for each subregion
        self.classifiers = nn.ModuleList([
            DOAClassifier(input_dim, classifier_hidden_dims, grids_per_subregion) 
            for _ in range(num_subregions)
        ])
    
    def forward(self, x):
        # Get decoder outputs from autoencoder
        decoder_outputs = self.autoencoder(x)
        
        # Process each decoder output with corresponding classifier
        classifier_outputs = []
        for i, decoder_output in enumerate(decoder_outputs):
            classifier_output = self.classifiers[i](decoder_output)
            classifier_outputs.append(classifier_output)
        
        # Concatenate all classifier outputs to form spatial spectrum
        spatial_spectrum = torch.cat(classifier_outputs, dim=1)
        
        return spatial_spectrum, decoder_outputs



