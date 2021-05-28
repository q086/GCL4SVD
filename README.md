# GCL4SVD
Graph Confident Learning for Software Vulnerability Detection

## Environment Setup
Python library dependencies:

* tensorflow -v : 1.14.0
* numpy -v : 1.19.3
* gensim -v : 3.8.1
* scipy -v : 1.4.1
* cleanlab -v : 0.1.1
* others: sklearn

## Data Preprocessing
Zhou, Y., Liu, S., Siow, J., Du, X., \& Liu, Y. Devign: Effective vulnerability identification by learning comprehensive program semantics via graph neural networks. Conference on Neural Information Processing Systems. (2019), 10197-10207

We also uploaded the corresponding datasets.
## GCL
Denoise the data of training set to get “datasetname_ train_ clean.jsonl” file.

If you want to replace the dataset, you can use the variable dataset = 'dataset name' in the related files. And you can modify the parameters in the files to suit your needs.
* Dataset format conversion and processing: 

  ```
   python data_process.py
  ```
 
* Composition preparation:

   ```
   python remove_words.py
  ```
  
* Code composition:

  ```
  python build_graph.py
  ```
  
* GCL denoising：
  
  ```
   python train_CL.py
  ```
  
## SVD
Put the obtained file named “datasetname_train_clean.jsonl”, test set and validation set in data_ process folder as input, then GGNN is retrained to get the prediction results.

If you want to replace the dataset, you can use the variable dataset = 'dataset name' in the related files. And you can modify the parameters in the files to suit your needs.
* Dataset format conversion and processing:

  ```
   python data_process.py
  ```
  
* Composition preparation:

  ```
  python remove_words.py
  ```
  
* Code composition:
    
   ```
  python build_graph.py
  ```
  
* Training and testing:

  ```
  python train.py
  ```
  
## Baselines
We choose Transformer, LSTM, Devign, GGNN as our baseline model. The code of GGNN is the same as SVD part, which uses the original datasets. For the rest of baselines, their code is in the Baselines folder.
