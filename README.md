Arrhythmia Classification with SVM
A Python script for classifying cardiac arrhythmias using a Support Vector Machine (SVM) algorithm. The script automatically handles missing data, scales features, and performs a grid search to find the optimal hyperparameters for the highest classification accuracy.

Features
Data Imputation: Replaces missing values with the column mean using SimpleImputer.

Feature Scaling: Standardizes the input features with StandardScaler to properly handle the imbalanced nature of the dataset.

Hyperparameter Tuning: Iterates through multiple combinations of C (Cost) and gamma to find the best configuration for the RBF kernel.

Visualization: Generates a confusion matrix heatmap for the optimal model using seaborn and matplotlib.

Requirements
Python 3.x with the following libraries:

numpy

pandas

scikit-learn

matplotlib

seaborn
Usage
Ensure the arrhythmia.data file is placed in the same directory as the script. Run the script directly from your terminal by typing:

python script_name.py

The script will print the accuracy for each parameter combination, announce the maximum accuracy achieved with its corresponding parameters, and open a graphical window displaying the confusion matrix for the optimal model.
