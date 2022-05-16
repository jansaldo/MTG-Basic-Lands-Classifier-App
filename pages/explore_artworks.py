from utils import (
    decrease_id, display_img, download_model, filter,
    get_prediction, increase_id, load_data, load_model
)


def explore_artworks(st, **state):
    with st.spinner(text="Loading data and model..."):
        basics = load_data()
        sets = list(basics["set_name"].unique())
        types = list(basics["name"].unique())
        download_model()
    model = load_model()
       
    ms_sets = st.sidebar.multiselect("Select sets", sets, default=None)
    if "ms_sets" not in st.session_state:
        st.session_state["ms_set"] = ms_sets
    if "id" not in st.session_state or ms_sets != st.session_state["ms_sets"]:
        st.session_state["id"] = 0
        st.session_state["ms_sets"] = ms_sets
    ms_types = st.sidebar.multiselect("Select basic land types", types, default=types)

    basics_filtered = filter(basics, ms_sets, ms_types)
    n = len(basics_filtered)
    
    try:
        assert n > 0
    except:
        st.warning("No artwork could be found with the applied filters. Try selecting other options.")
        st.stop()
    
    col1, col2 = st.sidebar.columns(2)
    if  n > 1 and st.session_state["id"] != 0:
        with col1:
            st.button("Previous artwork", on_click=decrease_id)
    if n > 1 and st.session_state["id"] < n-1:
        with col2:
            st.button("Next artwork", on_click=increase_id)
    
    try:
        uri, caption = display_img(basics_filtered, st.session_state["id"])
    except:
        st.session_state["id"] = 0
        uri, caption = display_img(basics_filtered, st.session_state["id"])
    
    st.image(uri, caption=caption)
    get_prediction(model, uri)
