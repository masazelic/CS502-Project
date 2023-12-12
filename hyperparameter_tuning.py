# This code is implemented by Marija Zelic and Elena Mrdja
# With the purpose of hyperparameter tuning for RelationNet model

from methods.relationnet import RelationNet
from datasets.prot.swissprot import *
from datasets.cell.tabula_muris import *
from backbones.fcnet import FCNet, FCNetRN

def get_dataset(n_way, n_support, n_query, dataset='swissprot'):
    """
    Get specific dataset for tuning task.
    
    Args:
        n_way (int): number of classes for the classification problem
        n_support (int): number of samples in the support set 
        n_query (int): number of samples in the query set
        dataset (str, optional): dataset we are tuning on, defaults to 'swissprot'

    Returns:
        train_dataset : 600 episodes from training dataset
        val_dataset: 600 episodes from validation dataset
        train_loader: DataLoader for train_dataset
        val_loader: DataLoader for validation dataset
        
    """
    
    if dataset == 'swissprot':
        train_dataset = SPSetDataset(n_way=n_way, n_support=n_support, n_query=n_query, n_episode=600, mode='train')
        val_dataset = SPSetDataset(n_way=n_way, n_support=n_support, n_query=n_query, mode='val')
    else:
        train_dataset = TMSetDataset(n_way=n_way, n_support=n_support, n_query=n_query, n_episode=600, mode='train')
        val_dataset = TMSetDataset(n_way=n_way, n_support=n_support, n_query=n_query, mode='val')
    
    train_loader = train_dataset.get_data_loader()
    val_loader = val_dataset.get_data_loader()
    
    return train_dataset, val_dataset, train_loader, val_loader

def initialize_model(learning_rate, backbone_layers, backbone_relation_layers, hidden_dim, train_dataset, n_way, n_support):
    """
    Initialize the model with the given parameters. 
    
    Args:
        learning_rate (float): learning rate for the optimizer
        backbone_layers (list of ints): number of nodes in the feature module layers
        backbone_relation_layers (list of ints): number of nodes in the relation module layers
        hidden_dim (int): second dimension of the Linear layer in the relation module
        train_dataset: train dataset obtained from get_dataset function
        n_way (int): number of classes in the classification problem
        n_support (int): number of support samples

    Returns:
        model (RelationNet): initialized RelationNet model
        optimizer (Adam): optimizer
        
    """
    backbone = FCNet(layer_dim=backbone_layers, x_dim=train_dataset.x_dim)
    backbone_relation = FCNetRN(layer_dim=backbone_relation_layers, hidden_dim=hidden_dim, x_dim=backbone_layers[-1]*2)
    model = RelationNet(n_way=n_way, n_support=n_support, backbone=backbone, backbone_relation=backbone_relation)
    optimizer = torch.optim.Adam(lr=learning_rate, params=model.parameters())
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    return model, optimizer

def train(train_loader, epoch, model, optimizer):
    """
    Train loop.
    
    Args:
        train_loader: train loader obtained from the get_dataset function
        epoch (int): number of epochs
        model (RelationNet): RelationNet model under training
        optimizer (Adam): optimizer used
        
    """
    # Run train_loop from MetaLearning template
    for i in range(epoch):
        model.train()
        model.train_loop(epoch, train_loader, optimizer)

def hyperparameter_tuning(n_way, n_support, n_query, dataset='swissprot'):
    """
    Hyperparameter tuning function.
    
    Args:
        n_way (int): number of classes in the classification problem
        n_support (int): number of support samples
        n_query (int): number of query samples
        dataset (str, optional): dataset we are tuning, defaults to 'swissprot'
    
    """
    
    # Parameters we are going to vary 
    learning_rate = [0.01, 0.01, 0.0001]
    epochs = [40, 60, 80]
    backbone_layers = [[512, 512], [512], [128], [64, 64]]
    backbone_relation_layers = [[256, 256], [256, 64], [128]]
    hidden_dim = [128, 64, 32]
    
    best_mean_acc = -1
    best_lr = -1
    best_epoch = -1
    best_bb_lyr = []
    best_bb_rel_lyr = []
    best_hd = -1
    
    # Performing grid-search
    for lr in learning_rate:
        for epoch in epochs:
            for bb_lyr in backbone_layers:
                for bb_rel_lyr in backbone_relation_layers:
                    for hd in hidden_dim:
                        
                        # Load train and validation dataset
                        train_dataset, _, train_loader, val_loader = get_dataset(n_way, n_support, n_query, dataset)
                        print("Curently evaluating these parameters:")
                        print("Learning rate:", lr)
                        print("Epochs:", epoch)
                        print("Backbone layers:", bb_lyr)
                        print("Backbone relation layers:", bb_rel_lyr)
                        print("Hidden dim:", hd)
                        
                        # Initialize the model  
                        model, optimizer = initialize_model(lr, bb_lyr, bb_rel_lyr, hd, train_dataset, n_way, n_support)
                            
                        # First, we are going to train the model on one training dataset
                        train(train_loader, epoch, model, optimizer)
                            
                        # Then, we are going to evaluate it on one validation dataset
                        acc_mean, acc_std = model.test_loop(val_loader, return_std=True)
                        
                        # If we have better average mean accuracy we are going to update best parameters
                        if acc_mean > best_mean_acc:
                            
                            best_mean_acc = acc_mean
                            
                            # Saving the parameters of the best model till now
                            best_lr = lr
                            best_epoch = epoch
                            best_bb_lyr = bb_lyr
                            best_bb_rel_lyr = bb_rel_lyr
                            best_hd = hd
                            
    print("Best hyperparameters:")
    print("Learning rate:", lr)
    print("Epochs:", epoch)
    print("Backbone layers:", bb_lyr)
    print("Backbone relation layers:", bb_rel_lyr)
    print("Hidden dim:", hd)
    
if __name__ == '__main__':
    hyperparameter_tuning(n_way=5, n_support=5, n_query=15, dataset='swissprot')
                        
                       
                            
                            
                            
                            
        
    
    
    
    
    