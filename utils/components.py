import streamlit as st


def hero(title: str, subtitle: str):

    st.markdown(
        f"""
<div style="
background: linear-gradient(135deg,#4F8BF9,#7B61FF);
padding:40px;
border-radius:20px;
margin-bottom:25px;
box-shadow:0 8px 20px rgba(0,0,0,.35);
">

<h1 style="
color:white;
margin:0;
font-size:42px;
">

{title}

</h1>

<p style="
color:white;
font-size:18px;
margin-top:10px;
">

{subtitle}

</p>

</div>
""",
        unsafe_allow_html=True,
    )


def section(title):
    st.subheader(title)


def metric_row(items):

    cols = st.columns(len(items))

    for col, item in zip(cols, items):

        col.metric(item[0], item[1])


def footer():

    st.divider()

    st.caption("Interactive Machine Learning Laboratory")