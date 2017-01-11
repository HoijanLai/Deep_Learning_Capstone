# Deepgits
A Jupyter Notebook Series that Implements Digit Recognition on the SVHN Dataset 

## Getting The Data
Go to http://ufldl.stanford.edu/housenumbers/ and download the format 2 data in （\*.mat）format.

## Run 
1. Run cells in /model/preprocess_mat.ipynb  **in order** and you will get three (\*.pickle) files, named "train", "validation" and "test" respectively.
2. Run all cells in /model/cnn_mat.ipynb, this is my final model.
3. To view the experiments, checkout /model/experiment.ipynb
4. Running model/cnn_with_summary.py will implement my final model and record summary for tensorboard usage. Before running this script, you should have the pickle data

## Details
The model is built using CPU version tensorflow 
