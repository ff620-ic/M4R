# M4R--GNN

## Dependencies
The project dependencies are specified in `requirements.txt`. The implementation relies heavily on PyTorch Geometric (`torch_geometric`) for Graph Neural Network construction and training.

## Outputs
The program logs performance metrics in real-time and saves all iteration data in both pickle (`.pkl`) and text (`.txt`) formats to the script directory. Due to fast convergence, data is recorded at every iteration for complete trajectory analysis.

## Executing
The code is compatible with standard personal computers, though GPU acceleration is recommended for performance speed-up. Using the default hyperparameters, convergence typically occurs within several hours to discover counterexamples.


