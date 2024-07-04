# Cancer-Classification
This repository contains a Jupyter Notebook file for a cancer classification model. The model is developed using the Breast Cancer Wisconsin dataset.

Dataset
he dataset used for this model is the Breast Cancer Wisconsin (Diagnostic) dataset. It contains 569 instances of breast tumor data with 30 features each, computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. The features describe characteristics of the cell nuclei present in the image.

The dataset is included in the repository in the file data.csv.

Model
The model developed in the Jupyter Notebook is a classification model that predicts whether a tumor is malignant or benign based on the features in the dataset. The model is developed using the scikit-learn library in Python.

The Jupyter Notebook contains code for data preprocessing, feature selection, model training, and evaluation. Many Models were trained using the Naive Bayes, KNN Model, Logistic Regression, Random Forest algorithms, and achieves an accuracy of 96.5% on the test set with Random Forest Algorithm.

Dependencies
The following libraries are required to run the Jupyter Notebook:

pandas
numpy
scikit-learn
matplotlib
seaborn
You can install these dependencies using pip:

pip install pandas numpy scikit-learn matplotlib seaborn
Usage
To run the Jupyter Notebook, clone the repository and open the file `Cancer_classification_model.ipynb` in Jupyter Notebook. Run the code cells in order to preprocess the data, train the model, and evaluate its performance.
