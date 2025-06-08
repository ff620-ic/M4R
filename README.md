# Enhancing the Construction of Combinatorial Counterexamples Using Graph Neural Networks
The complete algorithm is inspired by [Wagner's paper](https://arxiv.org/pdf/2104.14516). We mainly changed the model and training function while kept others the same as [Wagner's code](https://github.com/zawagner22/cross-entropy-for-combinatorics/blob/main/demos/cem_binary_conj21.py).
## Dependencies
The project dependencies are specified in `requirements.txt`. The implementation relies heavily on PyTorch Geometric (`torch_geometric`) for Graph Neural Network construction and training. To set up the conda environment run:

```
conda create --name <env> --file requirements.txt
```

## Outputs
The program logs performance metrics in real-time and saves all iteration data in both pickle (`.pkl`) and text (`.txt`) formats to the script directory. Due to fast convergence, data is recorded at every iteration for complete trajectory analysis.

| File | Content |
| -- | -- |
| reinforce_loss.txt | Reinforcement loss value per iteration  |
| best_species_pickle.txt | Best super sessions saved using `pickle`  | 
| best_species_txt.txt | Best graph per iteration |
| best_species_rewards.txt | Reward for the best graph per iteration |
| best_10_percent_rewards.txt | Mean reward of 10% of best graphs per iteration |
| best_elite_rewards.txt | Mean reward of super sessions for each iteration |
| worst_reward.txt | Reward of worst session per iteration |
| worst_actions.txt | Corresponding graph of worst session per iteration |
| counterexample.txt | the final counterexample graph construction |

## Executing
The code is compatible with standard personal computers, though GPU acceleration is recommended for performance speed-up. Using the default hyperparameters, convergence typically occurs within several hours to discover counterexamples.

| Hyperparameter | Value|
| --- | --- |
| Learning rate | 0.001 |
| Weight decay | 1e-5 |
| Patience of scheduler | 5 |
| Percentage of elite sessions (100-α) | 90 |
| Percentage of super sessions (100-γ) | 94 |
| Number of sessions | 1000 |
| Output dimension of Embedding | 64 |
| Hidden dimension of GNNStack | 128 |
| Output dimension of GNNStack | 64 |
| Number of layers of GNNSatck | 4 |
| Dropout probability | 0.1 |

 


