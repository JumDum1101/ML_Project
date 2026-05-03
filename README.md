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

### Gaussian Naive Bayes
How to Run:
1. Open the Gaussian Naive Bayes notebook in Jupyter Notebook or VS Code.
2. Ensure the required libraries are installed:
- numpy
- pandas
- matplotlib
- scikit-learn
- Ucimlrepo
3. Run all cells from top to bottom.
4. The dataset is loaded directly from the UCI repository using fetch_ucirepo.
  
What the Notebook Does:
- Imports required libraries.
- Loads the Covertype dataset using ucimlrepo.
- Converts the target labels into a usable one-dimensional format.
- Selects the 10 continuous terrain-related features
- Shows class distribution using a bar chart.
- Splits the dataset into train/validation/test sets: 70% training, 10% validation, 20% testing
- Uses stratified splitting to preserve class proportions.
- Applies PowerTransformer with the Yeo-Johnson method to make continuous features more Gaussian-like.
- Trains a Gaussian Naive Bayes model.
- Tunes the var_smoothing hyperparameter using np.logspace(-12, 1, 20)
- Selects the best model based on validation macro F1-score.
- Evaluates the model on validation and test data.
- Generates visualizations and evaluation metrics

Output:
- Validation accuracy
- Test accuracy
- Precision, recall, and F1-score
- Log loss
- Confusion matrix
- Hyperparameter tuning plot for var_smoothing


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

### Convolutional Neural Network
How to Run:
1. Open the CNN notebook in Jupyter Notebook or VS Code.
2. Ensure the required libraries are installed:
- tensorflow
- keras
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- ucimlrepo
3. Run all notebook cells from top to bottom.
4. Ensure TensorFlow is properly installed before running.

What the Notebook Does:
- Imports required libraries.
- Loads the Forest CoverType dataset using `ucimlrepo`.
- Assigns feature names to the dataset.
- Splits the dataset into train/validation/test sets: 70% training, 10% validation, 20% testing- Uses stratified splitting to preserve class proportions.
- Applies feature scaling using `StandardScaler`.
- Reshapes the tabular data into a format compatible with `Conv1D` layers.
- One-hot encodes the target class labels.
- Builds a 1D Convolutional Neural Network architecture using TensorFlow/Keras.
- Trains the CNN model using:
  * Adam optimizer
  * categorical cross-entropy loss
  * batch normalization
  * dropout regularization
  * early stopping
- Tracks training and validation performance across epochs.
- Evaluates the trained model on the test dataset.
- Generates visualizations and evaluation metrics.

Output:
- Training and validation accuracy curves
- Training and validation loss curves
- Test accuracy and test loss
- Precision, recall, and F1-score
- Confusion matrix
- Per-class metric visualizations

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
