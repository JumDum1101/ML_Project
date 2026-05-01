# Usage Guide
## Traditional ML Models
### Decision Trees
How to Run:
1. Open Decision Tree Final.ipynb in Jupyter Notebook  or VS Code (with the Jupyter extension installed)
2. Run all cells from top to bottom

What the Notebook Does:
- Imports required libraries
- Loads the Covertype dataset using ucimlrepo
- Splits data into train/validation/test (70/10/20, stratified)
- Trains a baseline Decision Tree model
- Performs 5-fold cross-validation
- Tunes hyperparameters using GridSearchCV
- Trains the final optimized model

Outputs:
- Accuracy, precision, recall, and F1-score
- Confusion matrix
- Inference timing results

## Deep Learning Models
### Multi-Layer Perceptrons
How to Run
1. Open Decision Tree Final.ipynb in Jupyter Notebook  or VS Code (with the Jupyter extension installed)
2. Run all cells from top to bottom

What the Notebook Does
- Imports required libraries
- Loads the Covertype dataset
- Splits data into train/validation/test (70/10/20, stratified)
- Applies feature scaling using StandardScaler (via pipeline)
- Trains a baseline MLP model
- Performs hyperparameter tuning using GridSearchCV (5-fold CV)
- Trains the final optimized model

Outputs
- Accuracy and evaluation metrics
- Confusion matrix
- Tuned model results

Note
- A convergence warning may appear during training; this is expected and does not affect final results.
