# Integrating Domain Knowledge into Transformer-based Approaches to Vulnerability Detection

TransVulDet is a Transformer-based Language Model for Vulnerability Detection aiming to better domain knowledge integration (e.g. CWE hierarchy) with source code datasets.

### Dataset
* Big-Vul dataset 
  * 91 CWE
  * https://dl.acm.org/doi/10.1145/3379597.3387501
  * https://github.com/ZeoVan/MSR_20_Code_vulnerability_CSV_Dataset
    * MSR_data_cleaned.csv(10.9GB):
      * Downloaded the cleaned version of split functions(CSV format)
      *  https://drive.google.com/file/d/1-0VhnHBp9IGh90s2wCNjeCMuy70HPl8X/view?usp=sharing
* CVEfixes Dataset
  * 209 CWEs
  * The latest version 'CVEfixes_v1.0.7' 
  * https://zenodo.org/record/7029359
 
#### Data Preprocessing & Visualization
* For MSR dataset,
  * `data_preprocessing/MSR_preprocessing.ipynb`
* For CVEfixes dataset, data collection by sql query and then preprocessing it
 * download 'CVEfixes_v1.0.7' and put `CVEfixes_preprocessing.py` in CVEfixes_v1.0.7/Examples and execute it. (data_preprocessing/CVEfixes_preprocessing.py)
* For combining two dataset and reassign CWE IDs and split them into train/validation/test dataset as well as balanced validation dataset,
  * `data_preprocessing/assign_cwes_and_split_datasets.ipynb`
* To create Directed Acyclic Graph (DAG) for CWE Hierarchy,
  * `data_preprocessing/preprocessing_paths_to_JSON.py` : convert given cwe node paths to json file
  * `src/graph.py` : greate/plot the graph (DAG) and save figure 
    
### Model
Pre-trained Transformer-based Language Models
* CodeBERT
* GraphCodeBERT


### Experiments
HPO with `main_hpo_sqlite.py`
#### Model Configurations
| Model                             | Loss Function                | Loss Weights                              | Classification Type |
| --------------------------------- | ---------------------------- | ------------------------------------------ | ------------------- |
| CodeBERT                          | Cross Entropy, Focal Loss     | Default, Class Weights                   | Non-Hierarchical    |
| GraphCodeBERT                     | Cross Entropy, Focal Loss     | Default, Class Weights                   | Non-Hierarchical    |
| CodeBERT with Hierarchical Classifier | BCE (per node)              | Default, Equalize, Descendants, Reachable Leaf Nodes | Hierarchical       |
| GraphCodeBERT with Hierarchical Classifier | BCE (per node)          | Default, Equalize, Descendants, Reachable Leaf Nodes | Hierarchical       |


The CodeBERT/GraphCodeBERT with Hierarchical Classifier will be called 'hCodeBERT'/'hGraphCodeBERT' in Result section.

### Fine-tuning
With `main_train.py`

 
### Result
#### Evaluation Metrics for Binary/Multiclass Classification Tasks
*  run `load_best_model_and_compute_metric.py`for both binary/multiclass classification measures at once
* Binary Classification Metrcs
 * Accuracy
 * Precision
 * Recall
 * F1-Score
* Multiclass Classification Metrics
  * Accuracy
  * Balanced Accuracy
  * Weighted F1-Score
  * Macro F1-Score
  * 
 #### Binary Classification Results
 | Model                           | Accuracy | Precision | Recall | F1-Score |
|---------------------------------|----------|-----------|--------|----------|
| CodeBERT-CE                     | 26.08    | 0.00      | 0.00   | 0.00     |
| GraphCodeBERT-CE                | 26.08    | 0.00      | 0.00   | 0.00     |
| CodeBERT-FL                     | **73.36**    | 75.92     | **93.67**  | **83.87**    |
| GraphCodeBERT-FL                | **73.81**    | 76.54     | **93.10**  | **84.01**    |
| hCodeBERT-default               | 69.62    | **80.00**     | 78.52  | 79.26    |
| hGraphCodeBERT-default          | 71.74    | 78.18     | 85.68  | 81.76    |
| hCodeBERT-equalize              | 70.39    | 78.17     | 83.17  | 80.59    |
| hGraphCodeBERT-equalize         | 69.01    | 78.11     | 80.70  | 79.38    |
| hCodeBERT-descendants           | 70.26    | 79.93     | 79.81  | 79.87    |
| hGraphCodeBERT-descendants      | 70.16    | 79.20     | 80.87  | 80.03    |
| hCodeBERT-reachable_leaf_nodes  | **73.79**    | **80.43**     | 85.30  | 82.79    |
| hGraphCodeBERT-reachable_leaf_nodes | 69.14    | 79.22     | 78.96  | 79.09    |

 
 #### Multiclass Classification Results
 | Model                            | Accuracy | Balanced Accuracy | Macro F1-Score | Weighted F1-Score |
|----------------------------------|----------|-------------------|----------------|-------------------|
| **CodeBERT-CE**                  | **26.08**| 4.76              | 1.97           | 10.79             |
| **GraphCodeBERT-CE**             | **26.08**| 4.76              | 1.97           | 10.79             |
| CodeBERT-FL                      | 15.80    | 13.38             | 11.31          | 18.39             |
| GraphCodeBERT-FL                 | 18.33    | **16.36**         | **13.36**      | 20.53             |
| hCodeBERT-default                | 20.85    | 11.10             | 11.44          | 20.86             |
| hGraphCodeBERT-default           | 18.91    | 8.47              | 9.59           | 20.07             |
| hCodeBERT-equalize               | 20.02    | 12.19             | 11.76          | 19.86             |
| hGraphCodeBERT-equalize          | 18.37    | 8.50              | 7.51           | 18.05             |
| **hCodeBERT-descendants**        | **25.34**| **15.05**         | **13.84**      | **23.54**         |
| hGraphCodeBERT-descendants       | 20.86    | 10.01             | 10.33          | 20.87             |
| hCodeBERT-reachable_leaf_nodes   | 22.14    | 12.60             | 12.45          | **23.10**         |
| hGraphCodeBERT-reachable_leaf_nodes | 21.22 | 12.14             | 11.64          | 20.84             |
