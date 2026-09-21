import sys

# If executed via Streamlit CLI (e.g. `streamlit run app.py`), run full Streamlit dashboard
if any("streamlit" in str(arg).lower() for arg in sys.argv):
    import streamlit_app
else:
    # Vercel Serverless Function entry point
    from api.index import handler
    app = handler
    application = handler

    # Top-level handler definition fallback
    if __name__ == "__main__":
        import streamlit_app
