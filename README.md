# Gestures-Alive
An Gesture Recognition Project using `OpenCV 3.3.0`, `Numpy 1.13.3` and `Python 3.6.3` for detecting and marking pre-defined hand gestures using various image processing techniques such as Background Subtraction, Colorspace Conversion, Gaussian and Median Blur Smoothening, Histogram Projection, Thresholding, Contour and Convex Hull Finding and more! 

## Installation
For a one stop solution, I always use `conda`. First install `conda` and create you own environment as:
```
conda create --name myenv
```
where `myenv` is the environment name (which can be changed at will). Then activate the environment by:
```
source activate myenv
```
and then install `OpenCV` and `Python` using a single command:
```
conda install -c conda-forge opencv
```

## Running
You can execute the project using the following command in the `src` folder:
```
python3 main.py
```

## Notes
Currently, the project is in a standalone working phase (but without a number of possible optimisations). This project is the culmination of more than 6 months of brainstorming (read as 20 core days) through various OpenCV tutorials and the [library documentation](https://docs.opencv.org/3.3.0/), auditing freely available lectures on image processing (thanks to [Prof. Ranga Rodrigo](http://www.ent.mrt.ac.lk/~ranga/) of University of Moratuwa, Sri Lanka), losing and rediscovering motivation time and again and making short progresses, and looking up at different codebases for tackling some minor problems.

The project currently works in HSV(Hue-Saturation-Value) colorspace for optimal results under thresholding and background subtraction.

Any suggestions and PRs are always welcome!

## Screenshots
(Please ignore the awkward camera angles. It is a well documented camera placement error on the XPS 15 9560.)

![](/screenshots/0.jpeg)

![](/screenshots/1.jpeg)

![](/screenshots/2.jpeg)

![](/screenshots/3.jpeg)

![](/screenshots/4.jpeg)
