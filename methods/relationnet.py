# This code is implemented by Marija Zelic and Elena Mrdja as a part of Deep Learning in Biomedicine project
# The code is adjusted version of # This code is modified from https://github.com/floodsung/LearningToCompare_FSL 

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from methods.meta_template import MetaTemplate

class RelationNet(MetaTemplate):
    """Creates a class for RelationNet model."""
    
    def __init__(self, backbone, n_way, n_support, backbone_relation, is_rn=True):
        """
        
        Initialization of the RelationNet model. Most of the parameters are inherited from the Meta Template base class.
        Args:
            backbone (FCNet): backbone used for the feature embeddings module
            n_way (int): number of classes in the classification problem
            n_support (int): number of support samples for each class
            backbone_relation (FCNetRN): backbone used for the relation module
            is_rn (bool, optional): boolean indicating that this model is RelationNet, defaults to True
            
        """
        super(RelationNet, self).__init__(backbone, n_way, n_support, backbone_relation, is_rn=True)
        # As in the original paper, loss function is MSE
        self.loss_fn = nn.MSELoss()

    def set_forward(self, x, is_feature=False):
        
        # For computing embeddings, we are using inherited parse_function
        z_support, z_query = self.parse_feature(x, is_feature)
        
        # We are going to calculate relation scores differently if we have one-shot or K-shot situation
        if self.n_support == 1: # one-shot
            # Repeat support embeddings so we can do concatenation
            z_support = torch.squeeze(z_support, 1)
        else: # K-shot
            z_support = torch.sum(z_support, 1)
        
        # Repeat support embeddings, so we can do concatenation
        z_support = z_support.repeat(self.n_way * self.n_query, 1)

        # Repeat_iterleave query embeddings so we can do concatenation
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)
        z_query = torch.repeat_interleave(z_query, self.n_way, 0)

        # Concatenation - input for relation module
        input_relation_module = torch.cat((z_support, z_query), 1)

        # Apply bacbone_relation to the input_relation_module
        output_relation_module = self.relation_score(input_relation_module) # shape: n_way^2 * n_query
        output_relation_module = output_relation_module.view(-1, self.n_way) # shape: n_way * n_query, n_way

        return output_relation_module
    
    def set_forward_loss(self, x):
        
        # Compute relation scores between support and query samples
        relation_scores = self.set_forward(x)

        # Create category labels for the queries
        y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query)).unsqueeze(-1)
        y_query_ohe = torch.zeros(self.n_way * self.n_query, self.n_way).scatter_(1, y_query, 1)
        y_query_ohe = Variable(y_query_ohe.cuda())

        # Compute the loss
        loss = self.loss_fn(relation_scores, y_query_ohe)

        return loss