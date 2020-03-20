# Detecting covid-19 using Xray images

**Collecting data**

[This](https://github.com/ieee8023/covid-chestxray-dataset) repository contains a database of COVID-19 cases with chest X-ray or CT images as well as MERS, SARS, and ARDS. Clone this repository in order t proceed with building the dataset.

Using the *metadata.csv*, extract all images which are covid-positive using the [build_dataset.py](https://github.com/yashk2000/covid-detection/blob/master/build_dataset.py) script. 

```python
python build_dataset.py -m <path to the cloned dataset folder> -o <path to where you want the output stored>
```

Next inorder to get some xrays of healthy patients/covid-negetive patients, we can randomly take images from the [Kaggle chest xray dataset](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia). To prevent class imbalance, keep the number of images of covid-positive and covid-negetive same. Store this in a folder called *normal* in your dataset.

The dataset should have a structure like the one in this repo. 

My [dataset](https://github.com/yashk2000/covid-detection/tree/master/dataset) consists of 68 [covid-positive](https://github.com/yashk2000/covid-detection/tree/master/dataset/covid) images and 70 [normal](https://github.com/yashk2000/covid-detection/tree/master/dataset/normal) images.

**Training the model**

The model is trained using VGGNet which is pre-trained on the imagenet dataset. To train obtain a model and save it, run the [trainModel.py](https://github.com/yashk2000/covid-detection/blob/master/trainModel.py) script. 

```python
python trainModel.py -d <path to dataset>
```

This will save the model in the same folder as the  script as a file, `[model.h5](https://github.com/yashk2000/covid-detection/blob/master/model.h5)`. A [plot](https://github.com/yashk2000/covid-detection/blob/master/plot.png) showing accuracy and loss curves will also be generated.  

![plot](https://user-images.githubusercontent.com/41234408/77198206-88f8ae80-6b0c-11ea-8f46-7dd74b4c5a78.png)

This is the plot generated while I was training my model. 

**Classifying more images using the trained model**

Now using the model obtained by running the above script, we can classify our own images as covid-positive or covid-negetive. To do this, run the [detectCovid.py](https://github.com/yashk2000/covid-detection/blob/master/detectCovid.py) script. 

```python
python detectCovid.py -i <path to input image> -m <path to the trained model>
```

This will give an output with image labeled as covid-positive or covid-negetive.

**Sample Output**

There are a few sample images in the [test-data](https://github.com/yashk2000/covid-detection/tree/master/test-data) folder. 

For exmaple, for the image [covid1.jpeg](https://github.com/yashk2000/covid-detection/blob/master/test-data/covid1.jpeg), which is covid-positive, I got the following output:

```python
python detectCovid.py -i test-data/covid1.jpeg -m model.h5
```

![Screenshot from 2020-03-21 00-48-17](https://user-images.githubusercontent.com/41234408/77198931-f0632e00-6b0d-11ea-8233-1c52cad25b30.png)


For the image [normal1.jpeg](https://github.com/yashk2000/covid-detection/blob/master/test-data/normal1.jpeg), which is covid-negetive, I got the following output:

```python
python3 detectCovid.py -i test-data/normal1.jpeg -m model.h5
```

![Screenshot from 2020-03-21 00-51-53](https://user-images.githubusercontent.com/41234408/77199111-4fc13e00-6b0e-11ea-95a5-224b0e78d2fa.png)

Cheers to [this](https://www.pyimagesearch.com/2020/03/16/detecting-covid-19-in-x-ray-images-with-keras-tensorflow-and-deep-learning/) blog post by Adrian Rosebrock which was an invaluable resource. 
