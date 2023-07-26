# Run AudioGPT
```bash
# create a new environment
conda create -n audiogpt python=3.8
conda activate audiogpt

# prepare the basic environments
pip install -r requirements.txt

# download the foundation models you need
bash download.sh

# prepare your private openAI private key
export OPENAI_API_KEY={Your_Private_Openai_Key}

# Start AudioGPT !
python audio-chatgpt.py

# [Optional] deactivate after running
conda deactivate
```
