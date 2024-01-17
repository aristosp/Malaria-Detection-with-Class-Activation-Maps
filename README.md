# Malaria Detection with Class Activation Maps
This project was developed during my efforts to self-learn Pytorch. This repository is based on Chapter 6 of [Modern Computer Vision with Pytorch]. Malaria Detetion consists of detecting whether a mosquito was infected or not when it bit a person. The dataset used can be found in the following [link]. Another idea that was implemented here was the Class Activation Maps (CAMs), displaying the activations that led to the classification results. In the following image, predictions of the model can be seen:

![image](https://github.com/aristosp/Malaria-Detection-with-Class-Activation-Maps/assets/62808962/f00b948a-6c2d-4b47-8785-cd6de9680b15)

While in the next image the class activation maps can be seen for two infected samples:
![image](https://github.com/aristosp/Malaria-Detection-with-Class-Activation-Maps/assets/62808962/e7620e19-99b7-472b-a7d2-fec1bc2f8704)
![image](https://github.com/aristosp/Malaria-Detection-with-Class-Activation-Maps/assets/62808962/b9be82db-8b86-4b84-8329-8130956d4fc5)

The brighter area in the right images is the original image placed on top of the activations, with different level of brightness.






[Modern Computer Vision with Pytorch]: https://www.oreilly.com/library/view/modern-computer-vision/9781839213472/
[link]:https://www.kaggle.com/datasets/miracle9to9/files1
