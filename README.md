1.Title:
HYLGAM: Hybrid Linguistic and Geometric Feature Augmentation with Machine Learning for Robust AI Generated Text Classification

2.Description :
This project presents a Hybrid Linguistic and graph-geometric feature-augmented machine learning framework for binary classification. It evaluates the effectiveness of combining conventional numerical features with additional graph-based neighborhood characteristics to improve classification performance.

The framework generates three graph-geometric features from the relationships between samples in the feature space: Graph Neighborhood Purity, Graph Local Density (Average Distance), and Graph Cluster Boundary Radius. These features are combined with the original dataset features and used to train and evaluate multiple machine learning classifiers.

3.Dataset Information:
File format: CSV (.csv)
Target variable: A binary classification label.
Target column identification: The code searches for a target column named label, target, or class.
Input features: Numerical or convertible feature columns.
Target classes: Binary labels encoded as 0 and 1.
Number of original features: 38 Linguistic and 38 stylistic features.
Additional features: Three graph-geometric features are generated during processing.
The generated features are:
1. Graph_Neighborhood_Purity: Measures the proportion of neighboring samples belonging to the majority class.
2. Graph_Local_Density_AvgDist: Represents the average distance between a sample and its selected nearest neighbors.
3. Graph_Cluster_Boundary_Radius: Represents the distance to the farthest selected neighbor within the local neighborhood.
The final feature dimension is \(d+3\), where \(d\) is the number of original input features.


4. Code Information
The code implements an end-to-end classification benchmarking pipeline. Its main components include:
- Automatic identification of the target column.
- Removal of records with missing target labels.
- Handling of missing feature values using training-data-derived feature medians.
- Feature standardization based on training-set statistics.
- Construction of nearest-neighbor relationships using Euclidean distance.
- Extraction of three graph-geometric features.
- Feature augmentation by combining original and generated features.
- Training and evaluation of multiple machine learning classifiers.
- Stratified five-fold cross-validation.
- Performance evaluation and statistical summarization.
- Computational time and error analysis.
- Generation of performance plots and result files.
The implemented classification models are:
1. Logistic Regression
2. Gaussian Naive Bayes
3. Random Forest
4. XGBoost
5. LightGBM
6. CatBoost
7. Multi-Layer Perceptron (MLP)
The framework records classification performance, prediction probabilities, computational time, and fold-wise evaluation results.

5.Usage Instructions
5.1 Dataset preparation
1. Prepare the dataset in CSV format.
2. Ensure that the dataset contains a binary target column named label, target, or class.
3. Place the dataset in the appropriate working directory.
4. Update the input CSV path in the code to point to the dataset location.
5.2 Running the code
1. Install the required Python libraries.
2. Open the Python script in a compatible Python environment.
3. Set the input dataset path.
4. Execute the script.
The pipeline automatically identifies the target column, prepares the data, generates graph-geometric features, and performs model evaluation.
5.3 Output files
The code generates result files and visualizations, including:
- Classification performance summaries.
- Cross-validation results.
- Model error-analysis results.
- Computational time comparisons.
- Feature inventory information.
- ROC curves.
- Precision–Recall curves.
- False Positive Rate (FPR) and False Negative Rate (FNR) comparisons.
- Fold-wise error-rate boxplots.
- Accuracy comparison plots.
- Training-time comparison plots.
- MLP confusion matrix.
- CSV and Excel result files, where the required Excel-writing support is available.

6. Requirements
The project requires Python packages:

numpy: 2.4.6
pandas: 3.0.5
scipy: 1.17.1
torch: 2.11.0+cu128
transformers: 5.17.0
tokenizers: 0.23.2
datasets: Not installed
matplotlib: 3.11.1
psutil: 7.2.2
tqdm: 4.70.0
scikit_learn: 1.9.0
torch_geometric: 2.8.0.post1


The code also uses Python standard-library modules for file handling, timing, and system operations.

7. Methodology
1. Load the dataset from the CSV file.
2. Identify the target column and remove records with missing target labels.
3. Divide the dataset into five stratified folds for cross-validation.
4. Handle missing feature values using the median calculated from the training data.
5. Standardize the features using the training data statistics.
6. Construct a nearest-neighbor graph using Euclidean distance.
7. Determine the neighborhood size based on the square root of the number of training samples.
8. Extract three graph-geometric features: neighborhood purity, average neighborhood distance, and boundary radius.
9. Combine the three generated features with the original features to create an augmented dataset.
10. Train seven classification models using the augmented training data.
11. Evaluate each model on the corresponding validation fold.
12. Calculate classification performance metrics, including accuracy, precision, recall, F1-score, ROC-AUC, and PR-AUC.
13. Analyze classification errors and computational time.
14. Aggregate the results across all five folds using mean, standard deviation, and confidence intervals.
15. Generate comparison plots and export the final evaluation results.

Citations:
HC3 [Data set]: Hello-SimpleAI, HC3: Human ChatGPT Comparison Corpus Dataset, Hugging Face Datasets, 2023. Available: https://huggingface.co/datasets/Hello-SimpleAI/HC3.  Last Accessed: Feb. 8, 2026.
DAIGT V2 Train Dataset [Data set]: Kaggle, 2023.  https://www.kaggle.com/datasets/thedrcat/daigt-v2-train-dataset.  Last Accessed: Feb. 8, 2026.

License


Materials & Methods
Computing infrastructure (operating system, hardware, etc)
Component	Configuration
OS	Windows 10 64-bit
Python	3.11.9
PyTorch	2.11.0+cu128
GPU	NVIDIA RTX PRO 4000 Blackwell
VRAM	~24 GB
CUDA	Available
CPU	24 physical / 32 logical cores
System RAM	128 GB

Dataset
HC3 [Data set]: Hello-SimpleAI, HC3: Human ChatGPT Comparison Corpus Dataset, Hugging Face Datasets, 2023. Available: https://huggingface.co/datasets/Hello-SimpleAI/HC3.  Last Accessed: Feb. 8, 2026.
DAIGT V2 Train Dataset [Data set]: Kaggle, 2023.  https://www.kaggle.com/datasets/thedrcat/daigt-v2-train-dataset.  Last Accessed: Feb. 8, 2026.

