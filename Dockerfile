FROM python:3.8

WORKDIR /app
COPY . /app
# create a new environment
RUN conda create -n audiogpt python=3.8

#  prepare the basic environments
RUN pip install -r requirements.txt

# download the foundation models you need
RUN bash download.sh

# prepare your private openAI private key
RUN export OPENAI_API_KEY={Your_Private_Openai_Key}

# Start AudioGPT !
RUN python audio-chatgpt.py
