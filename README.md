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

### Logistic Regression

How to Run:

- Open logistic_regression.ipynb in Jupyter Notebook or VS Code (with the Jupyter extension installed)
- Run all cells from top to bottom
- Ensure dataset path (covertype/covtype.data.gz) is correct before running

What the Notebook Does:

- Imports required libraries
- Loads the Covertype dataset
- Assigns feature names to the dataset
- Splits data into train/validation/test (70/10/20, stratified)
- Applies feature scaling using StandardScaler
- Trains a standard logistic regression model
- Trains a class-balanced logistic regression model
- Tunes the regularization parameter (C) using validation performance
- Evaluates both models on validation and test sets

Outputs:

- Accuracy (train, validation, test)
- Precision, recall, and F1-score
- Balanced accuracy (for balanced model)
- Confusion matrix
- Comparison of standard vs balanced model performance

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

### Large Language Model (LLM)

How to Run:

- Open llm.ipynb in Jupyter Notebook or VS Code (with the Jupyter extension installed)
- Ensure required libraries are installed (transformers, torch, sklearn, pandas, etc.)
- Run all cells from top to bottom
- A GPU is recommended for training due to memory requirements
- Ensure dataset path (covertype/covtype.data.gz) is correct before running

What the Notebook Does:

- Imports required libraries
- Loads and preprocesses the Covertype dataset
- Splits data into train/validation/test (70/10/20, stratified)
- Converts tabular data into structured text prompts
- Creates a custom PyTorch dataset for LLM training
- Loads and configures a pretrained DistilGPT-2 model
- Adds special tokens for class labels (<C1>–<C7>)
- Trains the model on the full dataset
- Generates predictions using text generation
- Evaluates the model on validation and test sets

Outputs:

- Accuracy
- Precision, recall, and F1-score
- Confusion matrix
- Training performance observations

Note:

- Training requires significant memory; evaluation during training may be disabled due to hardware constraints, training will also take a while