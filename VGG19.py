from keras.applications import VGG19
from keras.layers.experimental.preprocessing import Rescaling
from keras import layers
from keras.models import Sequential


class VGG19CNN:

    def __init__(self, image_size, batch_size = 1, nb_epochs = 10, trainable=False):
        self.image_size = image_size
        self.base_model = VGG19(include_top = False, weights = 'imagenet', input_shape = image_size)
        self.base_model.trainable = trainable
        self.batch_size = batch_size
        self.nb_epochs = nb_epochs
        self.model = self.build_model()

    def build_model(self):

        model = Sequential(
            [
                Rescaling(1./255, input_shape = self.image_size),
                self.base_model,
                layers.Flatten(),
                layers.Dense(256, activation='relu'),
                layers.Dropout(0.2),
                layers.Dense(128, activation='relu'),
                layers.Dropout(0.2),
                layers.Dense(1, activation='sigmoid'),
            ]
        )

        return model

    def train_model(self, X, y):

        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        history = self.model.fit(X, y, batch_size=self.batch_size, epochs=self.nb_epochs)

        return history

    def evaluate_model(self, X, y):
        score = self.model.evaluate(X, y, batch_size=self.batch_size)
        return score

    def predict(self, image):
        predictions = self.model.predict(image, batch_size=self.batch_size)
        return predictions

    def model_summary(self):
        self.model.summary()
