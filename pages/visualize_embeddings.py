import pandas as pd
from utils import (
    DataGenerator, extract_features,
    get_tsne_embeddings,
    load_data, load_model,
    plot_tsne_embeddings
)


def visualize_embeddings(st, **state):    
    basics = load_data()
    idx_cols = ["art_crop_uri", "name", "collector_number", "set_name", "artist", "released_at"]
    datagen = DataGenerator(basics, batch_size=4)
    feature_extractor = load_model(feature_extractor=True)

    try:
        features = pd.read_csv("data/img_feats.csv")
        assert len(features) == len(basics)
    except:
        with st.spinner("Extracting features from images, this may take a while..."):
            features = extract_features(feature_extractor, datagen)
    
    with st.spinner("Generating embeddings..."):
        tsne_embeddings = get_tsne_embeddings(features, basics, idx_cols)
    plot_tsne_embeddings(tsne_embeddings)
