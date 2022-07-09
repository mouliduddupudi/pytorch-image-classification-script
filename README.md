# Pytorch Scripting
We are all very comfortable with Notebooks for implementing our projects but in reality they are slow and not adaptable. In this tutorial I have implemented
a basic neural network architecture, LeNet, for image classification using scipting. 

After having a good knowledge on what has to be done, scripting gives us more robustness over the implementation, one of the main advantage of scripting is 
debugging. 

To run the model:

```
python train.py
```

* Change the path of your file in train.py and subsequently do any modifications necessary on the basis of data format in the folder.
* Evaluate your model on the test data using
* ``` python evaluate.py ```


# Dataset 

My data consists of train and test image names along with label in csv files. I need
to collect my train and test images paths seperately from images dataset given.

```data.py``` takes image names and image paths, which I have seperated from the images folder,
and transformations. Since each image name has a label, I have specified it in __getitem__.


**If you find any mistakes or disagree with any of the explanations, please do not hesitate to submit an issue. I welcome any feedback, positive or negative!**
