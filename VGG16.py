from keras.models import Sequential
from keras import layers
import matplotlib.pyplot as plt


class VGG16CNN:

    def __init__(self, image_size, batch_size = 1, nb_epochs = 10):
        self.image_size = image_size
        self.model = self.build_model()
        self.batch_size = batch_size
        self.nb_epochs = nb_epochs

    def build_model(self):
        model = Sequential(
            [
                layers.ZeroPadding2D((1,1), input_shape=self.image_size),
                layers.Conv2D(filters=64, kernel_size=3, activation='relu'),
                layers.ZeroPadding2D((1,1)),
                layers.Conv2D(filters=64, kernel_size=3, activation='relu'),
                layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)),

                layers.ZeroPadding2D((1, 1)),
                layers.Conv2D(filters=128, kernel_size=3, activation='relu'),
                layers.ZeroPadding2D((1, 1)),
                layers.Conv2D(filters=128, kernel_size=3, activation='relu'),
                layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

                layers.ZeroPadding2D((1, 1)),
                layers.Conv2D(filters=256, kernel_size=3, activation='relu'),
                layers.ZeroPadding2D((1, 1)),
                layers.Conv2D(filters=256, kernel_size=3, activation='relu'),
                layers.ZeroPadding2D((1, 1)),
                layers.Conv2D(filters=256, kernel_size=3, activation='relu'),
                layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

                layers.ZeroPadding2D((1, 1)),
                layers.Conv2D(filters=512, kernel_size=3, activation='relu'),
                layers.ZeroPadding2D((1, 1)),
                layers.Conv2D(filters=512, kernel_size=3, activation='relu'),
                layers.ZeroPadding2D((1, 1)),
                layers.Conv2D(filters=512, kernel_size=3, activation='relu'),
                layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

                layers.Flatten(),

                layers.Dense(units=4096, activation='relu'),
                layers.Dropout(0.5),
                layers.Dense(units=4096, activation='relu'),
                layers.Dropout(0.5),
                layers.Dense(1, activation='sigmoid')
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

    def save_model(self):
        self.model.save("./")

    def model_summary(self):
        self.model.summary()


