import tensorflow as tf
from keras.src.legacy.preprocessing.image import ImageDataGenerator

IMG_SIZE = (224,224)

generator = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_data = generator.flow_from_directory(
    "dataset",
    target_size=IMG_SIZE,
    batch_size=16,  
    class_mode="binary",
    subset="training"
)

val_data = generator.flow_from_directory(
    "dataset",
    target_size=IMG_SIZE,
    batch_size=16,
    class_mode="binary",
    subset="validation"
)

model = tf.keras.Sequential([

    tf.keras.layers.Conv2D(
        32,
        (3,3),
        activation='relu',
        input_shape=(224,224,3)
    ),

    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(
        64,
        (3,3),
        activation='relu'
    ),

    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(
        64,
        activation='relu'
    ),

    tf.keras.layers.Dense(
        1,
        activation='sigmoid'
    )

])

model.compile(

    optimizer='adam',

    loss='binary_crossentropy',

    metrics=['accuracy']

)

model.fit(

    train_data,

    validation_data=val_data,

    epochs=10

)

model.save("grooming_model.h5")