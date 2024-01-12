FROM python:3.7
WORKDIR app
COPY requirements.txt .
RUN pip install tensorflow
RUN pip install -r requirements.txt
COPY . .
CMD /bin/bash
