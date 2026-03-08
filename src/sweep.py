import wandb
import argparse
from train import train

sweep_config = {
    'method': 'bayes',
    'metric': {'name': 'val_accuracy', 'goal': 'maximize'},
    'parameters': {
        'epochs':         {'values': [10, 15, 20]},
        'batch_size':     {'values': [32, 64, 128]},
        'loss':           {'values': ['cross_entropy', 'mse']},
        'optimizer':      {'values': ['sgd', 'momentum', 'nag', 'rmsprop']},
        'learning_rate':  {'distribution': 'log_uniform_values', 'min': 1e-4, 'max': 1e-1},
        'weight_decay':   {'values': [0.0, 0.0005, 0.001, 0.005]},
        'num_layers':     {'values': [2, 3, 4, 5]},
        'hidden_size':    {'values': [32, 64, 128]},
        'activation':     {'values': ['sigmoid', 'tanh', 'relu']},
        'weight_init':    {'values': ['random', 'xavier', 'zeros']},
    }
}

def sweep_train():
    wandb.init()
    cfg = wandb.config
    
    # Create an args namespace
    args = argparse.Namespace(
        dataset='mnist',
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        loss=cfg.loss,
        optimizer=cfg.optimizer,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        num_layers=cfg.num_layers,
        hidden_size=cfg.hidden_size,
        activation=cfg.activation,
        weight_init=cfg.weight_init,
        wandb_project='da6401_assignment_1',
        wandb_entity=None,
        no_wandb=False, # We want it to log to the sweep run
        val_split=0.1,
        seed=42,
        save_dir='src/',
        model_path='src/best_model.npy',
        config_path='src/best_config.json'
    )
    
    train(args)

if __name__ == '__main__':
    sweep_id = wandb.sweep(sweep_config, project='da6401_assignment_1')
    wandb.agent(sweep_id, sweep_train, count=100)
