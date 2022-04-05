# Bacteria_Classifier
Species identification of bacterial BSI pathogens from the three-dimensional refractive index tomography

# Trained model
Download models from https://drive.google.com/drive/folders/1q0Cdj6WODSJ5cd0q3H7AXFGhPfeQWSRh?usp=sharing

# How to infer
run main.py while specifying the location of the data, the location of the model, and the name of the model parameter file.

ex) python main.py --data_dir /dataset/example_inference/ --model_dir /model/ --load_fname mdl01_train[0.9830]_valid[0.8237]_test[0.8145].pth.tar

# Inference result
A folder whose name is identical to the model's filename is generated.

The folder contains two files: "result_test.ma"t and "test_confusion.npy"

"result_test.mat" contains the paths of the inference data, the scores for each species, and the target class.

"test_confusion.npy" contains the confusion matrix, which can also be generated from the contents of "result_test.mat"
