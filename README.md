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
  * https://zenodo.org/record/7029359

### Model
Pre-trained Transformer-based Language Models
* CodeBERT
* GraphCodeBERT

### Model Configurations

| Model                             | Loss Function                | Loss Weights                              | Classification Type |
| --------------------------------- | ---------------------------- | ------------------------------------------ | ------------------- |
| CodeBERT                          | Cross Entropy, Focal Loss     | Default, Class Weights                   | Non-Hierarchical    |
| GraphCodeBERT                     | Cross Entropy, Focal Loss     | Default, Class Weights                   | Non-Hierarchical    |
| CodeBERT with Hierarchical Classifier | BCE (per node)              | Default, Equalize, Descendants, Reachable Leaf Nodes | Hierarchical       |
| GraphCodeBERT with Hierarchical Classifier | BCE (per node)          | Default, Equalize, Descendants, Reachable Leaf Nodes | Hierarchical       |


### Evaluation
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
 
### Result
