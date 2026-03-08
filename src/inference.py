import os
import json
import argparse
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from ann.neural_network import NeuralNetwork
from utils.data_loader import load_dataset, preprocess

def load_model(model_path):
    """
    Load trained model from disk.
    """
    data = np.load(model_path, allow_pickle=True).item()
    return data

def run_inferences(args):
    # 1. Load best_config.json
    with open(args.config_path, 'r') as f:
        config = json.load(f)
    
    # 2. Load dataset specified in config (or args.dataset)
    dataset_name = config.get('dataset', args.dataset)
    X_tr_raw, y_tr_raw, X_te_raw, y_te_raw = load_dataset(dataset_name)
    
    # 3. Preprocess -> get X_te, y_te_int
    (_, _, _, _, X_te, _, _, _, y_te_int) = preprocess(
        X_tr_raw, y_tr_raw, X_te_raw, y_te_raw, 
        val_split=args.val_split, seed=args.seed
    )
    
    # 4. Build NeuralNetwork from config
    model = NeuralNetwork(
        input_size=config['input_size'],
        hidden_sizes=config['hidden_sizes'],
        num_classes=config['num_classes'],
        activation=config['activation'],
        weight_init=config['weight_init']
    )
    
    # 5. model.set_weights
    weights = load_model(args.model_path)
    model.set_weights(weights)
    
    # 6. forward -> argmax -> preds
    preds = model.predict(X_te)
    
    # 7. Compute metrics
    acc = accuracy_score(y_te_int, preds)
    prec = precision_score(y_te_int, preds, average='macro')
    rec = recall_score(y_te_int, preds, average='macro')
    f1 = f1_score(y_te_int, preds, average='macro')
    
    # 8. PRINT
    print("-" * 30)
    print(f"Inference Results on {dataset_name}:")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("-" * 30)
    
    return {'accuracy': acc, 'f1': f1}

def parse_arguments():
    """
    Parses CLI arguments. Must be identical to train.py for autograder.
    """
    parser = argparse.ArgumentParser(description='Inference with saved MLP')
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
    run_inferences(args)
