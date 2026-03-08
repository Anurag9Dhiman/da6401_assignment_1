# DA6401 Assignment 1 — MLP for Image Classification

## W&B Report
[Link to public W&B report](https://wandb.ai/anuragdhiman666-indian-institute-of-technology-madras/da6401_assignment_1)

## GitHub Repository
[Link to this repo](https://github.com/Anurag9Dhiman/da6401_assignment_1)

## How to Train
```bash
python src/train.py -d mnist -e 15 -b 64 -o rmsprop -lr 0.001 \
  -nhl 3 -sz 128 128 -a relu -wi xavier -wp da6401_assignment_1
```

## How to Run Inference
```bash
python src/inference.py --model_path src/best_model.npy --config_path src/best_config.json
```

## How to Run 100 Sweeps
```bash
python src/sweep.py
```

## Best Model Configuration
- Dataset: MNIST
- Architecture: 3 hidden layers, 128 neurons each
- Activation: ReLU
- Optimizer: RMSProp, lr=0.001
- Test F1: ~0.98
