# Integrating Domain Knowledge into Transformer-based Approaches to Vulnerability Detection

TransVulDet is a Transformer-based Language Model for Vulnerability Detection aiming to better domain knowledge integration (e.g. CWE hierarchy) with source code datasets.

### Dataset
* Multiclass Vulnerability Dataset (MVD)
  * 40 CWEs, multiclass
  * https://github.com/muVulDeePecker/muVulDeePecker
  * mvd.txt(256.3MB)
* Big-Vul dataset 
  * 91 CWE
  * https://dl.acm.org/doi/10.1145/3379597.3387501
  * https://github.com/ZeoVan/MSR_20_Code_vulnerability_CSV_Dataset
    * MSR_data_cleaned.csv(10.9GB): Downloaded the cleaned version of split functions(CSV format)
* CVEfixes Dataset
  * 209 CWEs
  * https://zenodo.org/record/7029359

### Model
Pre-trained Transformer-based Language Models
* CodeBERT
* GraphCodeBERT
* BERT-based (BERT, RoBERTa, DistillBERT, etc)
* T5

### Evaluation

### Result
