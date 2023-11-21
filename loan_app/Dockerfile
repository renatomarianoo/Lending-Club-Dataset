FROM python:3.11
EXPOSE 8080
WORKDIR /loan_app
COPY . ./
ADD requirements.txt requirements.txt
RUN pip install -r requirements.txt
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]