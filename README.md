# Bacteria_Classifier
Species identification of bacterial BSI pathogens from the three-dimensional refractive index tomography

# Trained model
Download models from [Google Drive](https://drive.google.com/drive/folders/1q0Cdj6WODSJ5cd0q3H7AXFGhPfeQWSRh?usp=sharing)

# How to inference
First install packages
```bash
pip install -r requirements.txt
```

And run `main.py` while specifying the location of the data, the location of the model, and the name of the model parameter file.

```bash
cd src
python main.py --data_dir ../dataset/example_inference/ --load_fname ${Download_from_GDrive}
```

# Inference result
A folder whose name is identical to the model's filename is generated.

The folder contains two files: `result_test.mat` and `test_confusion.npy`

- `result_test.mat` contains the paths of the inference data, the scores for each species, and the target class.

- `test_confusion.npy` contains the confusion matrix, which can also be generated from the contents of "result_test.mat"
