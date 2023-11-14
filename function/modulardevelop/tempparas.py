import tensorflow as tf
import new_evaluation as eva
md = 'resnet'
dmin = 2.75
dmax = 4.0
wmin = 18
wmax = 26
x_train, y_train, x_test, y_test = eva.load_data()
epochs = 1
