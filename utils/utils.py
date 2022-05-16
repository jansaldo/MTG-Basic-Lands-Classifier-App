import altair as alt
import numpy as np
import os
import pandas as pd
import streamlit as st
import subprocess
import tensorflow as tf
from io import BytesIO
from PIL import Image
from requests import get
from sklearn.manifold import TSNE


@st.cache(suppress_st_warning=True)
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(
        self,
        df,
        batch_size=4,
        preprocessing_function=tf.keras.applications.vgg16.preprocess_input
    ):
        self.df = df
        self.indices = np.arange(self.df.shape[0])
        self.uris = self.df['art_crop_uri'].values
        self.batch_size = batch_size
        self.preprocessing_function = preprocessing_function
        

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))


    def __getitem__(self, idx):
        def img_from_uri_to_array(uri):
            img = Image.open(BytesIO(get(uri).content))
            img = img.resize((224,224))
            return tf.keras.preprocessing.image.img_to_array(img)
        
        inds = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        uris = self.uris[inds]
        x = []
        for uri in uris:
            x.append(img_from_uri_to_array(uri))

        return self.preprocessing_function(np.array(x))


def decrease_id():
    st.session_state["id"] -= 1


def display_img(df, idx):
    img = (
        df.loc[
            idx,
            ["art_crop_uri", "name", "collector_number",
             "set_name", "artist", "released_at"]
        ].values
    )
    uri = img[0]
    caption = f"{img[1]} #{img[2]} - {img[3]} - {img[4]} ({img[5][:4]})"
    return uri, caption


@st.cache(show_spinner=False)
def download_model():
    if not os.path.exists("model/model.h5"):
        url = "https://www.dropbox.com/s/d7i5tdnf8le44h1/model.zip?dl=1"
        downloaded_file = get(url)
        with open("model.zip", "wb") as out_file:
            out_file.write(downloaded_file.content)
        subprocess.run("unzip -o model.zip", shell=True)
        os.remove("model.zip")


def extract_features(feature_extractor, data):
    features = feature_extractor.predict(data)
    features = pd.DataFrame(features)
    if not os.path.exists("data"):
        os.mkdir("data")
    features.to_csv("data/img_feats.csv", index=False)
    return features


def filter(df, ms_sets, ms_types):
    if len(ms_sets) > 0 and len(ms_types) == 0:
        df_filtered = df.loc[
            df["set_name"].isin(ms_sets)
        ].reset_index(drop=True)
    elif len(ms_sets) > 0 and len(ms_types) > 0:
        df_filtered = df.loc[
            (df["set_name"].isin(ms_sets)) &
            (df["name"].isin(ms_types))
        ].reset_index(drop=True)
    elif len(ms_sets) == 0 and len(ms_types) > 0:
        df_filtered = df.loc[
            df["name"].isin(ms_types)
        ].reset_index(drop=True)
    else:
        df_filtered = df
    
    return df_filtered


def header(st):
    st.markdown(
        """
        <style>
        img {
            vertical-align: sub;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    header_html = f'<img src="https://emojipedia-us.s3.dualstack.us-west-1.amazonaws.com/thumbs/160/microsoft/106/mage_1f9d9.png" height="50" width="50"/>'
    st.markdown(f"# {header_html} MTG Basic Lands Classifier", unsafe_allow_html=True)


def increase_id():
    st.session_state["id"] += 1


def get_prediction(model, img, from_uri=True):
    labels = {
        0: "Forest",
        1: "Island",
        2: "Mountain",
        3: "Plains",
        4: "Swamp"
    }

    if from_uri:
        img = Image.open(BytesIO(get(img).content))
        img = img.resize((224, 224))
    else:
        img = img.resize((224, 224))
    
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = tf.keras.applications.vgg19.preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    prediction_probability = np.max(prediction)
    predicted_label = np.argmax(prediction)
    title = (
        "Predicted label: {} - Probability = {:.2f} %".format(
            labels[predicted_label], prediction_probability*100
        )
    )

    source = pd.DataFrame({
        "Land type": labels.values(),
        "Probability": prediction.ravel()
    })
    chart = alt.Chart(source, height=200, width=560, title=title).mark_bar().encode(
        x="Probability",
        y=alt.Y("Land type", sort="-x")
    )
    bar_chart = st.altair_chart(chart)

    return bar_chart


@st.cache(allow_output_mutation=True, show_spinner=False)
def get_tsne_embeddings(features, metadata, idx_cols):
    X = pd.concat(
        [metadata[idx_cols].reset_index(drop=True), features],
        axis=1
    ).set_index(idx_cols)
    
    tsne = TSNE(random_state=420, n_jobs=-1)
    tsne_embeddings = tsne.fit_transform(X)
    tsne_embeddings = pd.DataFrame(
        tsne_embeddings,
        columns=['tsne_0', 'tsne_1'],
        index=X.index
    ).reset_index()
    
    return tsne_embeddings


@st.cache(show_spinner=False)
def load_data():
    response = get("https://api.scryfall.com/bulk-data")
    download_uri = response.json()["data"][1]["download_uri"]
    response = get(download_uri)
    df = pd.DataFrame(response.json())
    df = df.loc[df["name"].isin(["Forest", "Island", "Mountain", "Plains", "Swamp"]),
                ["id", "name", "released_at", "image_uris", "set_name", "collector_number", "artist"]] \
           .sort_values(["released_at", "set_name", "collector_number"])
    df["art_crop_uri"] = df["image_uris"].apply(lambda x: x["art_crop"])
    df.drop("image_uris", axis=1, inplace=True)
    return df


@st.cache(allow_output_mutation=True, show_spinner=False)
def load_model(feature_extractor=False):
    model = tf.keras.models.load_model("model/model.h5")
    if not feature_extractor:
        return model
    else:
        inpt = tf.keras.layers.Input(shape=(224, 224, 3), name='input_img')
        conv_base = model.get_layer('vgg16')(inpt)
        flatten = model.get_layer('flatten')(conv_base)
        outpt = model.get_layer('dense')(flatten)
        feature_extractor = tf.keras.models.Model(inpt, outpt, name='feature_extractor')
        feature_extractor.make_predict_function()
        return feature_extractor


def plot_tsne_embeddings(tsne_embeddings):
    tsne_embeddings["released_at"] = (
        tsne_embeddings["released_at"].map(lambda x: x[:4])
    )

    tsne_embeddings = tsne_embeddings.rename(
        columns={
            "art_crop_uri": "image",
            "name": "Land type",
            "collector_number": "Collector number",
            "set_name": "Set",
            "artist": "Artist",
            "released_at": "Year"
        }
    )
    names = ["Forest", "Island", "Mountain", "Plains", "Swamp"]
    rng = ["green", "blue", "red", "yellow", "black"]
    selection_interval = alt.selection_interval(bind="scales")
    selection_multi = alt.selection_multi(fields=["Land type"], bind="legend")

    chart = (
        alt.Chart(tsne_embeddings)
        .mark_circle(size=200)
        .encode(
            x=alt.X("tsne_0", axis=None),
            y=alt.Y("tsne_1", axis=None),
            color=alt.Color(
                "Land type",
                scale=alt.Scale(domain=names, range=rng),
                legend=alt.Legend(title="Basic land type")
            ),
            tooltip=["Land type", "Collector number", "Set", "Artist", "Year", "image"]
        )
        .properties(width=1100, height=550)
        .configure_mark(opacity=0.5)
        .configure_axis(grid=False)
        .add_selection(selection_interval)
        .add_selection(selection_multi)
        .transform_filter(selection_multi)
    )

    st.markdown("<style>#vg-tooltip-element{z-index: 1000051}</style>",
                unsafe_allow_html=True)

    return st.altair_chart(chart, use_container_width=True)


def upload_img():
    img = st.file_uploader(
        """
        Try out the classifier uploading any image you like,
        e.g., nonbasic land artworks or real photos resembling Magic lands!
        """,
        type= ["png", "jpg", "jpeg"]
    )
    if img is not None:
        img = Image.open(img)
        return img
