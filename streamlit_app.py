import streamlit as st
from streamlit_multipage import MultiPage
from pages import pages
from utils import header


st.set_page_config(
    page_title = "MTG Basic Lands Classifier",
    page_icon="https://emojipedia-us.s3.dualstack.us-west-1.amazonaws.com/thumbs/160/microsoft/106/mage_1f9d9.png"
)


def main():
    app = MultiPage(hide_menu=False)
    app.st = st
    app.header = header
    app.hide_navigation = True
    app.navbar_name = "<img src='http://media.wizards.com/2016/images/daily/MM20161114_Wheel.png' height='150' width='150'/>"
    app.navbar_style = "SelectBox"
    for app_name, app_function in pages.items():
        app.add_app(app_name, app_function)
    app.run()


if __name__ == "__main__":
    main()
