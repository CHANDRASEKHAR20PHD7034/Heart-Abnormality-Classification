import tensorflow as tf
import ipython

print("TensorFlow version:", tf.__version__)
import pkg_resources

version = pkg_resources.get_distribution("tensorflow-estimator").version
print("TensorFlow Estimator version:", version)
