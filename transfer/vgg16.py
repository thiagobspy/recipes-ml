from keras.applications import VGG16
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import preprocess_input
from keras_preprocessing.image import load_img, img_to_array

model = VGG16()
print(model.summary())


def predict(file):
    image = load_img(file, target_size=(224, 224))
    array_image = img_to_array(image)
    array_image = array_image.reshape((1, array_image.shape[0], array_image.shape[1], array_image.shape[2]))

    # Normalizacao
    x = preprocess_input(array_image)
    y = model.predict(x)

    label = decode_predictions(y)
    print(label)


predict('./carro.jpg')
predict('./gatinho.jpg')
