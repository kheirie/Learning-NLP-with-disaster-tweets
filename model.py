from tensorflow.keras.layers import Input, Dense, Flatten, Bidirectional, concatenate, Dropout
from tensorflow.keras import Model
import tensorflow_hub as hub
import matplotlib.pyplot as plt


def embed_list(x):
    hub_model = "https://tfhub.dev/google/nnlm-en-dim50/2"
    embed = hub.load(hub_model)
    return embed(x)


def model(input_shapes):
    inputs = []
    models = []
    for i in range(len(input_shapes)):
        input_ = Input(shape=(input_shapes[i],))
        inputs.append(input_)
        input_model = Flatten()(input_)
        input_model = Dense(1024, activation='relu')(input_model)
        input_model = Dropout(0.5)(input_model)
        models.append(input_model)

    merged = concatenate(models, axis=1)
    merged = Dense(1024, activation='relu')(merged)
    merged = Dropout(0.5)(merged)
    final = Dense(1, activation='sigmoid')(merged)

    final = Model(inputs=inputs, outputs=final)

    return final


def train_model(final_model, x, y, epochs, loss, opt, val_split, bs, earlystop=None):

    final_model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])

    if earlystop:
        history = final_model.fit(
            x=x,
            y=y,
            batch_size=bs,
            epochs=epochs,
            validation_split=val_split,
            callbacks=[earlystop]
        )
    else:
        history = final_model.fit(
            x=x,
            y=y,
            batch_size=bs,
            epochs=epochs,
            validation_split=val_split
        )

    return history


def plot_acc_loss(history):

    acc = history.history["accuracy"]
    loss = history.history["loss"]

    val_acc = history.history["val_accuracy"]
    val_loss = history.history["val_loss"]

    epochs = range(len(acc))

    plt.figure()

    plt.plot(epochs, acc, label="training accuracy")
    plt.plot(epochs, val_acc, label="validation accuracy")
    plt.title("training and validation accuracy")
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, label="training loss")
    plt.plot(epochs, val_loss, label="validation loss")
    plt.title("training and validation loss")
    plt.legend()

    plt.show()


def predict(final_model, x):
    predictions = final_model.predict(x).tolist()
    return [1 if result[0] > 0.5 else 0 for result in predictions]