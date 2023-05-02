# Run AudioGPT

- Check if Conda and Python are installed
- Download and install Conda: https://www.anaconda.com/products/individual
- Check if Python is installed: Open a terminal/cmd and type ```python --version```

### Create a new virtual environment using Conda 
```conda create -n audiogpt python=3.8```

### Activate the virtual environment 
```conda activate audiogpt```

### Prepare the basic environments/ Install the neccessary dependencies
```pip install -r requirements.txt```

### Download the foundation models required for AudioGPT
```bash download.sh``` 

Make sure the download.sh script exists and downloads the correct models

### Prepare your private openAI private key
```export OPENAI_API_KEY={Your_Private_Openai_Key}```

### Start AudioGPT !
```python audio-chatgpt.py```



