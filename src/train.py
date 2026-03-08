import os
import json
import argparse
import numpy as np
import wandb
from sklearn.metrics import f1_score, accuracy_score

from ann.neural_network import NeuralNetwork
from ann.optimizers import get_optimizer, NAG
from ann.objective_functions import compute_loss
from utils.data_loader import load_dataset, preprocess, get_batches, get_samples_for_logging

CLASS_NAMES = {
    'mnist': [str(i) for i in range(10)],
    'fashion_mnist': ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
}

def resolve_hidden_sizes(num_layers, hidden_size_input):
    """
    num_layers: total layers (hidden + 1 output)
    hidden_size_input: int or list
    Returns: list of hidden sizes (length = num_layers - 1)
    """
    num_hidden = num_layers - 1
    if isinstance(hidden_size_input, int):
        return [hidden_size_input] * num_hidden
    elif isinstance(hidden_size_input, list):
        if len(hidden_size_input) == 1:
            return [hidden_size_input[0]] * num_hidden
        return hidden_size_input
    return [128] * num_hidden

def train(args):
    # Set seed
    np.random.seed(args.seed)
    
    # 1. Load + preprocess data
    X_tr_raw, y_tr_raw, X_te_raw, y_te_raw = load_dataset(args.dataset)
    (X_tr, y_tr_oh, X_val, y_val_oh, X_te, y_te_oh, 
     y_tr_int, y_val_int, y_te_int) = preprocess(
         X_tr_raw, y_tr_raw, X_te_raw, y_te_raw, 
         val_split=args.val_split, seed=args.seed
     )
    
    # 2. Build NeuralNetwork
    hidden_sizes = resolve_hidden_sizes(args.num_layers, args.hidden_size)
    model = NeuralNetwork(
        input_size=784, 
        hidden_sizes=hidden_sizes, 
        num_classes=10, 
        activation=args.activation, 
        weight_init=args.weight_init
    )
    
    # 3. Get optimizer
    optimizer = get_optimizer(args.optimizer, args.learning_rate)
    
    # 4. wandb.init
    if not args.no_wandb:
        run_name = f"{args.optimizer}_{args.activation}_lr{args.learning_rate}_nhl{args.num_layers}"
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=run_name, config=args)
        
        # Q2.1 Data Exploration Table
        table = wandb.Table(columns=['image', 'label', 'class_name'])
        samples = get_samples_for_logging(X_tr_raw, y_tr_raw, num_per_class=5)
        ds_name = args.dataset.lower()
        for img, label in samples:
            table.add_data(wandb.Image(img), label, CLASS_NAMES[ds_name][label])
        wandb.log({'sample_images': table})

    best_val_f1 = -1.0
    best_weights = None
    
    # 5. Training loop
    iteration = 0
    for epoch in range(1, args.epochs + 1):
        batch_losses = []
        for X_batch, y_batch_oh in get_batches(X_tr, y_tr_oh, args.batch_size):
            iteration += 1
            
            if isinstance(optimizer, NAG):
                optimizer.apply_lookahead(model.layers)
                
            probs = model._probs if model.forward(X_batch) is not None else None # forward returns logits, sets _probs
            probs = model._probs
            loss_val = compute_loss(probs, y_batch_oh, args.loss, args.weight_decay, model.layers)
            batch_losses.append(loss_val)
            
            model.backward(X_batch, y_batch_oh, loss=args.loss, weight_decay=args.weight_decay)
            
            if isinstance(optimizer, NAG):
                optimizer.restore(model.layers)
            
            optimizer.step(model.layers)
            
            # Q2.9 Weight Initialization & Symmetry (first 50 iters)
            if iteration <= 50 and not args.no_wandb:
                layer0_neurons = model.layers[0].out_features
                n = min(5, layer0_neurons)
                neuron_grads = model.get_neuron_gradients(layer_idx=0, neuron_indices=list(range(n)))
                log_dict = {f'neuron_grad/layer1_neuron_{j}': float(neuron_grads[j]) 
                            for j in range(len(neuron_grads))}
                log_dict['iteration'] = iteration
                wandb.log(log_dict)

        # Epoch end metrics
        train_loss = np.mean(batch_losses)
        
        # Validation
        val_logits = model.forward(X_val)
        val_probs = model._probs
        val_preds = np.argmax(val_logits, axis=1)
        val_acc = accuracy_score(y_val_int, val_preds)
        val_f1 = f1_score(y_val_int, val_preds, average='macro')
        
        # Explicitly log both loss types for Q2.6
        val_ce = compute_loss(val_probs, y_val_oh, 'cross_entropy', 0.0)
        val_mse = compute_loss(val_probs, y_val_oh, 'mse', 0.0)
        
        # Dead neuron analysis (Q2.5)
        dead_logs = {}
        for i, layer in enumerate(model.layers[:-1]):
            acts = layer._A
            dead_frac = np.mean(np.mean(np.abs(acts), axis=0) < 1e-6)
            dead_logs[f'dead_neurons/layer_{i+1}'] = dead_frac
            if not args.no_wandb:
                dead_logs[f'activation_dist/layer_{i+1}'] = wandb.Histogram(acts.flatten())
        
        # Gradient Norms (Q2.4)
        grad_norms = model.get_gradient_norms()
        grad_logs = {f'grad_norm/layer_{k}': v for k, v in grad_norms.items()}
        
        # Optional: Train accuracy on sample for Q2.7
        sample_idx = np.random.choice(len(X_tr), min(2000, len(X_tr)), replace=False)
        tr_sample_preds = model.predict(X_tr[sample_idx])
        tr_sample_acc = accuracy_score(y_tr_int[sample_idx], tr_sample_preds)
        
        if not args.no_wandb:
            log_data = {
                'epoch': epoch,
                'max_epochs': args.epochs,
                'train_loss': train_loss,
                'val_accuracy': val_acc,
                'val_f1': val_f1,
                'val_cross_entropy': val_ce,
                'val_mse': val_mse,
                'train_accuracy': tr_sample_acc
            }
            log_data.update(dead_logs)
            log_data.update(grad_logs)
            wandb.log(log_data)
            
        print(f"Epoch {epoch}/{args.epochs} - Loss: {train_loss:.4f} - Val Acc: {val_acc:.4f} - Val F1: {val_f1:.4f}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_weights = model.get_weights()

    # 6. Save best model
    os.makedirs(args.save_dir, exist_ok=True)
    best_model_path = os.path.join(args.save_dir, 'best_model.npy')
    best_config_path = os.path.join(args.save_dir, 'best_config.json')
    
    np.save(best_model_path, best_weights)
    with open(best_config_path, 'w') as f:
        json.dump(best_weights['_config'], f, indent=4)
        
    # Also copy to models/
    os.makedirs('models', exist_ok=True)
    np.save('models/best_model.npy', best_weights)
        
    # 7. Final Test Evaluation
    model.set_weights(best_weights)
    test_preds = model.predict(X_te)
    test_acc = accuracy_score(y_te_int, test_preds)
    test_f1 = f1_score(y_te_int, test_preds, average='macro')
    
    if not args.no_wandb:
        wandb.run.summary['best_val_f1'] = best_val_f1
        wandb.log({'test_accuracy': test_acc, 'test_f1': test_f1})
        # Q2.8 Confusion Matrix
        wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(
            probs=None,
            y_true=y_te_int,
            preds=test_preds,
            class_names=CLASS_NAMES[args.dataset.lower()]
        )})
        
    return {'accuracy': test_acc, 'f1': test_f1}

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, choices=['mnist', 'fashion_mnist'], default='mnist')
    parser.add_argument('-e', '--epochs', type=int, default=15)
    parser.add_argument('-b', '--batch_size', type=int, default=64)
    parser.add_argument('-l', '--loss', type=str, choices=['cross_entropy', 'mse', 'mean_squared_error'], default='cross_entropy')
    parser.add_argument('-o', '--optimizer', type=str, choices=['sgd', 'momentum', 'nag', 'rmsprop'], default='rmsprop')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.0)
    parser.add_argument('-nhl', '--num_layers', type=int, default=3)
    parser.add_argument('-sz', '--hidden_size', type=int, nargs='+', default=[128, 128])
    parser.add_argument('-a', '--activation', type=str, choices=['sigmoid', 'tanh', 'relu'], default='relu')
    parser.add_argument('-wi', '--weight_init', type=str, choices=['random', 'xavier', 'zeros'], default='xavier')
    parser.add_argument('-wp', '--wandb_project', type=str, default='da6401_assignment_1')
    parser.add_argument('-we', '--wandb_entity', type=str, default=None)
    parser.add_argument('--no_wandb', action='store_true', default=False)
    parser.add_argument('--val_split', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='src/')
    parser.add_argument('--model_path', type=str, default='src/best_model.npy')
    parser.add_argument('--config_path', type=str, default='src/best_config.json')
    return parser.parse_args()

def parse_args():
    return parse_arguments()

if __name__ == '__main__':
    args = parse_arguments()
    train(args)
