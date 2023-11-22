from keras import models
from keras import layers
from keras import regularizers
from keras import initializers
from keras import activations
from keras import optimizers
from keras import losses
from keras import metrics
from keras import callbacks
from keras.preprocessing import image

model = models.Sequential()

model.add(layers.Conv2D(
    32, (5, 5),
    input_shape = (64, 64, 3),
    activation = 'relu'
))
model.add(layers.MaxPooling2D(
    pool_size = (2, 2)
))

model.add(layers.Conv2D(
    16, (5, 5),
    input_shape = (30, 30, 3),
    activation = 'relu'
))
model.add(layers.MaxPooling2D(
    pool_size = (2, 2)
))

model.add(layers.Conv2D(
    4, (5, 5),
    input_shape = (13, 13, 3),
    activation = 'relu'
))
model.add(layers.MaxPooling2D(
    pool_size = (2, 2)
))

model.add(layers.Flatten())

model.add(layers.Dense(256,
  kernel_regularizer = regularizers.L2(1e-4),
    kernel_initializer=initializers.GlorotNormal(),
  bias_initializer = initializers.Zeros()
))
model.add(layers.Dropout(0.2))
model.add(layers.Activation(activations.relu))


model.add(layers.Dense(64,
  kernel_regularizer = regularizers.L2(1e-4),
    kernel_initializer=initializers.GlorotNormal(),
  bias_initializer = initializers.Zeros()
))
model.add(layers.Activation(activations.relu))


model.add(layers.Dense(64,
  kernel_regularizer = regularizers.L2(1e-4),
    kernel_initializer=initializers.GlorotNormal(),
  bias_initializer = initializers.Zeros()
))
model.add(layers.Activation(activations.relu))

model.add(layers.Dense(2,
    kernel_initializer=initializers.GlorotNormal(),
  bias_initializer = initializers.Zeros()
))
model.add(layers.Activation(activations.softmax))

model.compile(
    optimizer=optimizers.Adam(),
    loss=losses.BinaryCrossentropy(),
    metrics=[metrics.CategoricalAccuracy(), metrics.Precision()]
)


dataGen = image.ImageDataGenerator(
    rescale = 1.0/255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True,
    vertical_flip = False,
    validation_split = 0.2
)

X_train = dataGen.flow_from_directory(
    './PetImages',
    target_size = (64,64),
    batch_size  = 32,
    class_mode = 'categorical',
    subset ='training'
)

X_tests = dataGen.flow_from_directory(
    './PetImages',
    target_size = (64,64),
    batch_size  = 32,
    class_mode = 'categorical',
    subset ='validation'
)

model.fit(X_train, 
          steps_per_epoch=500,
          epochs=50,
          validation_data=X_tests,
          validation_steps=100,
          callbacks=[
              callbacks.EarlyStopping(patience=4),
              callbacks.ModelCheckpoint(filepath = 'model.{epoch:02d}-{val_loss:.2f}.h5')
          ]
)

model.save('model')
