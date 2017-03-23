import numpy as np
from keras.models import Model
from keras.layers import Dense, Activation, Flatten, Input
import math
import cv2


def build_label(class_idx, n_classes, DOO):
    # returns the target for a training instance
    label = np.zeros((n_classes * DOO, ))
    for ii in range(DOO):
        label[ii * n_classes + class_idx] = 1.0 / DOO
    return label


def infer(probs, DOO, n_classes):
    # infer from a test instance
    out = np.ones((n_classes,))
    for ii in range(DOO):
        for jj in range(n_classes):
            out[jj] = out[jj] * probs[jj + ii * n_classes] * DOO
    return out

batch_size = 128
nb_classes = 3
nb_epoch = 20

DOO = 16

input_layer = Input(shape=(2,))
x = Dense(64, activation='relu')(input_layer)
x = Dense(128, activation='relu')(x)
x = Dense(128, activation='relu')(x)
x = Dense(128, activation='relu')(x)
out = Dense(DOO * nb_classes, activation='softmax')(x)

model = Model(inputs=input_layer, outputs=out)

model.compile(loss='categorical_crossentropy',
              optimizer='adam')


# Create the training data
X = []
Y = []

rads = [1.0, 0.5, 0.25]

for i in np.linspace(0, 2*math.pi, 1000):
    for ix, rad in enumerate(rads):
        x = rad * math.cos(i)
        y = rad * math.sin(i)
        X.append([x, y])
        Y.append(build_label(ix, nb_classes, DOO))

X_train = np.array(X)
Y_train = np.array(Y)

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1)

# plot the results
x_min, x_max = -2, 2
y_min, y_max = x_min, x_max
h = 0.01
num_px = int((x_max - x_min) / h)

xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))


Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
P = np.zeros((num_px*num_px, 3))

for i in range(Z.shape[0]):
    P[i, :] = infer(Z[i, :], DOO, nb_classes)

P = np.array(P)
P = np.reshape(P, xx.shape + (nb_classes,))

img = (255*P).astype(np.uint8)

scale = 0.25 * (x_max - x_min) / h
for rad in rads:
    cv2.circle(img, (num_px/2, num_px/2), int(rad * scale), (255,255,255), thickness=2)

cv2.imshow('preds', img)
cv2.waitKey(0)
