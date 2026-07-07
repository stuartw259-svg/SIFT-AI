FROM python:3.12-slim

# Non-root user: the container process should never run as root
RUN useradd --create-home --uid 10001 sift
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY sift_core.py train.py app.py ./
COPY .streamlit ./.streamlit

# Train at build time so the image ships with a freshly generated model
# (no pickled artifact is ever committed or transported)
RUN python train.py && chown -R sift:sift /app

USER sift

EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
