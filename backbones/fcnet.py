import torch
from torch import nn as nn
import torch.nn.functional as F

from backbones.blocks import full_block, full_block_fw

class FCNet(nn.Module):
    fast_weight = False  # Default

    def __init__(self, x_dim, layer_dim=[64, 64], dropout=0.2, fast_weight=False):
        super(FCNet, self).__init__()
        self.fast_weight = fast_weight

        layers = []
        in_dim = x_dim
        for dim in layer_dim:
            if self.fast_weight:
                layers.append(full_block_fw(in_dim, dim, dropout))
            else:
                layers.append(full_block(in_dim, dim, dropout))
            in_dim = dim

        self.encoder = nn.Sequential(*layers)
        self.final_feat_dim = layer_dim[-1]

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)


class EnFCNet(nn.Module):

    def __init__(self, x_dim, go_mask, hid_dim=64, z_dim=64, dropout=0.2):
        super(EnFCNet, self).__init__()

        # self.go_mask = generate_simple_go_mask(x_dim=x_dim, num_GOs=3) # for testing
        self.go_mask = go_mask

        self.num_GOs = len(self.go_mask)
        self.masks = None
        self.z_dim = z_dim

        self.conv1 = nn.Conv1d(x_dim, hid_dim, 1, bias=True)
        self.conv2 = nn.Conv1d(hid_dim, z_dim, 1, bias=True)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.bn1 = nn.BatchNorm1d(self.num_GOs + 1)
        self.bn2 = nn.BatchNorm1d(self.num_GOs + 1)
        self.final_feat_dim = z_dim

    def generate_masks(self, x):
        batch, num_genes = x.shape
        self.masks = torch.zeros(self.num_GOs + 1, batch, num_genes)
        for i, genes in enumerate(self.go_mask):
            self.masks[i, :, genes] = 1
        selected_genes = torch.sum(self.masks[:, 0, :], axis=0)
        self.masks[-1, :, :] = 1

    def forward(self, x):
        batch, num_genes = x.shape
        # need to generate masks if the batch size change or
        if self.masks is None or self.masks.shape[1] != batch:
            self.generate_masks(x)
            self.masks = self.masks.to(x.device)
        # x before applying mask: (batch, numGenes)
        x = x.view(1, batch, -1)
        # x after applying mask: (numGOs, batch, numGenes)
        x = self.masks * x
        # change to (batch, numGOs, numGenes)
        x = x.permute(1, 2, 0)
        x = self.conv1(x)
        x = x.permute(0, 2, 1)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)
        x = self.conv2(x)
        x = x.permute(0, 2, 1)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)

        return x

class FCNetRN(nn.Module):
    """ Class for the backbone of the RelationNet relation module."""
    fast_weight = False # Default

    def __init__(self, x_dim, hidden_dim, layer_dim=[64, 64], dropout=0.2, fast_weight=False):
        """
        Initialization of the FCNetRN backcbone. 
        
        Args:
            x_dim (int): second dimension of the input x
            hidden_dim (int): second dimension of the first hidden layer
            layer_dim (list, optional): dimensions of the hidden layer, defaults to [64, 64]
            dropout (float, optional): dropout probability, defaults to 0.2.
            fast_weight (bool, optional): fast weights parameter, defaults to False.

        """"
        super(FCNetRN, self).__init__()
        self.fast_weight = fast_weight

        layers = []
        in_dim = x_dim
        for dim in layer_dim:
            if self.fast_weight:
                layers.append(full_block_fw(in_dim, dim, dropout))
            else:
                layers.append(full_block(in_dim, dim, dropout))
            in_dim = dim
      
        self.encoder = nn.Sequential(*layers)

        # Following architeture of the original paper
        # First FC layer for Relation Module of RN
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        # Second FC layer 
        self.fc2 = nn.Linear(hidden_dim, 1)

        self.final_feat_dim = 1

    def forward(self, x):
        x = self.encoder(x)
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        
        return x