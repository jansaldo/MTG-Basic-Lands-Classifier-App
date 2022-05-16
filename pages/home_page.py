def home_page(st, **state):
    st.subheader("A Magic: The Gathering basic lands artwork classifier")
    panorama = '<img src="https://pbs.twimg.com/media/DfHqk95U0AAUElL?format=jpg&name=large" width="589" height="86"/>'
    st.markdown(f"{panorama}<br>", unsafe_allow_html=True)
    st.write(
        "This app was developed by [jansaldo](mailto:julianansaldo@gmail.com). " +
        "The source code can be found in this [GitHub repo](https://github.com/jansaldo/MTG-Basic-Lands-Classifier-App)."
    )
    star = '<img src="https://github.githubassets.com/images/icons/emoji/unicode/1f31f.png" height="20" width="20"/>'
    st.markdown(f"Feel free to leave a {star} if you liked it!", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(
        "Artwork in this app is copyrighted by Wizards of the Coast, LLC, a subsidiary of Hasbro, Inc. " +
        "This app is not produced by, endorsed by, supported by, or affiliated with Wizards of the Coast."
    )
