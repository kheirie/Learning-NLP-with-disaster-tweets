import pandas as pd
import numpy as np
import preprocessing as pr
from sklearn.utils import shuffle
from model import embed_list, model, train_model, plot_acc_loss
import tensorflow as tf


def main():
    train = pd.read_csv("./data/train.csv")
    test = pd.read_csv("./data/test.csv")

    # Preprocess the train data
    train["keyword"] = train["keyword"].apply(lambda x: pr.to_lower(x) if pd.isna(x) is not True else x)
    train["location"] = train["location"].apply(lambda x: pr.to_lower(x) if pd.isna(x) is not True else x)
    train["text"] = train["text"].apply(lambda x: pr.to_lower(x))
    train["keyword"] = train["keyword"].apply(lambda x: pr.strip_all_entities(x) if pd.isna(x) is not True else x)
    train["location"] = train["location"].apply(lambda x: pr.strip_all_entities(x) if pd.isna(x) is not True else x)
    train["text"] = train["text"].apply(lambda x: pr.strip_all_entities(x))
    train["keyword"] = train["keyword"].apply(lambda x: pr.remove_tags(x) if pd.isna(x) is not True else x)
    train["location"] = train["location"].apply(lambda x: pr.remove_tags(x) if pd.isna(x) is not True else x)
    train["text"] = train["text"].apply(lambda x: pr.remove_tags(x) if pd.isna(x) is not True else x)
    train["keyword"] = train["keyword"].apply(lambda x: pr.remove_punc(x) if pd.isna(x) is not True else x)
    train["location"] = train["location"].apply(lambda x: pr.remove_punc(x) if pd.isna(x) is not True else x)
    train["text"] = train["text"].apply(lambda x: pr.remove_punc(x) if pd.isna(x) is not True else x)
    train["keyword"] = train["keyword"].apply(lambda x: pr.spelling_correction(x) if pd.isna(x) is not True else x)
    train["location"] = train["location"].apply(lambda x: pr.spelling_correction(x) if pd.isna(x) is not True else x)
    train["text"] = train["text"].apply(lambda x: pr.spelling_correction(x) if pd.isna(x) is not True else x)
    train["keyword"] = train["text"].apply(lambda x: pr.extract_keywords(doc=x, no_of_keywords=1)[0] if pd.isna(x) is True else x)
    train["location"] = train["text"].apply(lambda x: pr.get_location(doc=x)[0] if pd.isna(x) is True else x)
    train["keyword"] = train["keyword"].apply(lambda x: pr.lemmatize(x) if pd.isna(x) is not True else x)
    train["location"] = train["location"].apply(lambda x: pr.lemmatize(x) if pd.isna(x) is not True else x)
    train["text"] = train["text"].apply(lambda x: pr.lemmatize(x) if pd.isna(x) is not True else x)
    train["text"] = train["text"].apply(lambda x: pr.remove_stopwords(x) if pd.isna(x) is not True else x)

    # Preprocess the test data
    test["keyword"] = test["keyword"].apply(lambda x: pr.to_lower(x) if pd.isna(x) is not True else x)
    test["location"] = test["location"].apply(lambda x: pr.to_lower(x) if pd.isna(x) is not True else x)
    test["text"] = test["text"].apply(lambda x: pr.to_lower(x) if pd.isna(x) is not True else x)
    test["keyword"] = test["keyword"].apply(lambda x: pr.strip_all_entities(x) if pd.isna(x) is not True else x)
    test["location"] = test["location"].apply(lambda x: pr.strip_all_entities(x) if pd.isna(x) is not True else x)
    test["text"] = test["text"].apply(lambda x: pr.strip_all_entities(x) if pd.isna(x) is not True else x)
    test["keyword"] = test["keyword"].apply(lambda x: pr.remove_tags(x) if pd.isna(x) is not True else x)
    test["location"] = test["location"].apply(lambda x: pr.remove_tags(x) if pd.isna(x) is not True else x)
    test["text"] = test["text"].apply(lambda x: pr.remove_tags(x) if pd.isna(x) is not True else x)
    test["keyword"] = test["keyword"].apply(lambda x: pr.remove_punc(x) if pd.isna(x) is not True else x)
    test["location"] = test["location"].apply(lambda x: pr.remove_punc(x) if pd.isna(x) is not True else x)
    test["text"] = test["text"].apply(lambda x: pr.remove_punc(x) if pd.isna(x) is not True else x)
    test["keyword"] = test["keyword"].apply(lambda x: pr.spelling_correction(x) if pd.isna(x) is not True else x)
    test["location"] = test["location"].apply(lambda x: pr.spelling_correction(x) if pd.isna(x) is not True else x)
    test["text"] = test["text"].apply(lambda x: pr.spelling_correction(x) if pd.isna(x) is not True else x)
    test["keyword"] = test["text"].apply(lambda x: pr.extract_keywords(doc=x, no_of_keywords=1)[0] if pd.isna(x) is True else x)
    test["location"] = test["text"].apply(lambda x: pr.get_location(doc=x) if pd.isna(x) is True else x)
    test["keyword"] = test["keyword"].apply(lambda x: pr.lemmatize(x) if pd.isna(x) is not True else x)
    test["location"] = test["location"].apply(lambda x: pr.lemmatize(x) if pd.isna(x) is not True else x)
    test["text"] = test["text"].apply(lambda x: pr.lemmatize(x) if pd.isna(x) is not True else x)
    test["text"] = test["text"].apply(lambda x: pr.remove_stopwords(x) if pd.isna(x) is not True else x)

    train = shuffle(train, random_state=42).reset_index(drop=True)  # shuffle dataset
    y = np.array(train["target"].tolist())  # convert the target column into a numpy array

    key_embed = embed_list(train.keyword.to_list())  # keyword embeddings
    loc_embed = embed_list(train.location.to_list())  # location embeddings
    text_embed = embed_list(train.text.to_list())  # text embeddings

    final_model = model([key_embed.shape[1], loc_embed.shape[1], text_embed.shape[1]])

    lr = 0.1  # learning rate
    epochs = 100  # number of epochs
    opt = tf.keras.optimizers.SGD(lr=lr, momentum=0.8, decay=lr / epochs)  # optimizer
    loss = 'binary_crossentropy'

    earlystop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', mode='min', patience=10, verbose=1
    )

    history = train_model(final_model, [key_embed, loc_embed, text_embed], y, epochs, loss, opt, 0.1, 32, earlystop)

    plot_acc_loss(history)


if __name__ == '__main__':
    main()