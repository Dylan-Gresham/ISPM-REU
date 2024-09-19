# Required libraries:
# llama-cpp-python
# huggingface_hub
# Easy installation command (make sure you're in your virtual environment first)
# pip install --upgrade --upgrade-strategy eager --force-reinstal --no-cache-dir llama-cpp-python huggingface_hub
# You'll need to create a HuggingFace account and create a token under your account Settings
# You'll also need some sort of C compiler, Visual Studio for Windows, gcc or clang for Linux and Git LFS
#   If you're unable to get a C compiler up and running, there's a pre-built wheel version of llama-cpp-python:
#   pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
# Once you've installed all the packages successfully, run the commands:
# git lfs install
# huggingface-cli login
#
# The huggingface-cli login command will require you to paste your HuggingFace token in, this allows you to pull models
# from HuggingFace's model database for use in Python code

# Import llama-cpp-python and the built-in timeit module
from llama_cpp import Llama
from timeit import default_timer as timer

print('Starting to get the LLM...')
# Get start time for getting the model
llm_start = timer()

# Pull down the model from HuggingFace
# The first time this runs, you'll need an internet connection to actually download the model itself.
# After the first time, the huggingface-cli tool will manage the models for you, stored locally in the HuggingFace
# cache directory.
# Substantial speed-up after running it once
llm = Llama.from_pretrained(
    # Specify which HuggingFace repository the model is in
    repo_id='QuantFactory/Meta-Llama-3-8B-Instruct-GGUF-v2',
    # Specify the name of the model file to download
    filename='Meta-Llama-3-8B-Instruct-v2.Q6_K.gguf',
    verbose=False
)
# You can also download the model in advance and tell llama-cpp-python to just pull it from a local file
# llm = Llama(
#     model_path="relative/file/path/to/model"
# )

# Get inference end time
llm_end = timer()
print('LLM acquired!')

# Compute time taken to acquire the model
llm_elapsed_time = llm_end - llm_start
llm_mins, llm_secs = divmod(llm_elapsed_time, 60)
llm_hours, llm_mins = divmod(llm_mins, 60)

print('Starting to perform inference...')
# Get start time for inference
inference_start = timer()

# Start the inference using the high-level API provided by llama-cpp-python
output = llm.create_chat_completion(
    # Define the message template for the conversation
    messages=[
        # Define how the system (the LLM) is to act
        {
            "role": "system",
            "content": "You are an assistant who perfectly describes large language models imitating the speech style of pirates."
        },
        # Define what the user's prompt is for the LLM.
        {
            "role": "user",
            "content": "Tell me what a LLM is."
        }
    ]
)

# Get end time for inference
inference_end = timer()
print('Inference completed!')

# Compute time taken for inference
inference_elapsed_time = inference_end - inference_start
inference_mins, inference_secs = divmod(inference_elapsed_time, 60)
inference_hours, inference_mins = divmod(inference_mins, 60)

# Printing results
print(f"Acquiring the model took: {llm_hours:.0f} hours, {llm_mins:.0f} minutes, and {llm_secs:.0f} seconds")
print(f"Performing inference took: {inference_hours:.0f} hours, {inference_mins:.0f} minutes, and {inference_secs:.0f} seconds")
print()
print(output["choices"][0]["message"]["content"])
