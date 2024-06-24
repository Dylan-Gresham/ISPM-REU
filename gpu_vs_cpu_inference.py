# Most importantly, make sure you're in a Python virtual environment first, have
# a Nvidia GPU, and have the CUDA toolkit installed.

# Prior to installing `llama-cpp-python`, we'll need to do some extra steps. All
# you'll need to do is set some additional CLI environment variables before 
# executing the pip command to install libraries.


# Windows

# The commands to set the environment variables are:

# - `$env:CMAKE_ARGS="-DLLAMA_CUBLAS=on"`
# - `$env:FORCE_CMAKE=1`
# - `$env:CUDACXX="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v<version_number>\bin\nvcc.exe"`
#   - Change `<version_number>` to whichever version of the CUDA toolkit you have installed.
#   - For instructions on downloading and installing the CUDA toolkit on 
#     Windows, visit
#     https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html.

# Linux

# The commands to set the environment variables are:

# - `export CMAKE_ARGS="-DLLAMA_CUBLAS=on"`
# - `export FORCE_CMAKE=1`
# - `export CUDACXX="<path_to_nvcc_executable>"`
#   - Where `<path_to_nvcc_executable>` is your installation path to the 
#     executable for your Linux distribution's CUDA Toolkit installation.
#   - For instructions on downloading and installing the CUDA toolkit on 
#     Windows, visit
#     https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html.

# Once the environment variables have been set, execute the following pip 
# command to download and install the appropriately compiled `llama-cpp-python`
# and other necessary packages.

# pip install --upgrade --force-reinstall --no-cache-dir llama-cpp-python jupyter huggingface_hub

# HuggingFace Hub Setup

# You'll need to create a HuggingFace account and access token:

# 1. Create an account on [HuggingFace](https://huggingface.co).
# 2. Once logged into your account, click your profile picture in the upper 
#    right corner and navigate to Settings > Access Tokens.
# 3. Click New Token and generate a new token, I made mine a "Write" token but 
#    it shouldn't matter if it's a "Read" or "Write" token for this script.
# 4. Run the command `huggingface-cli login` and paste in the access token you 
#    created in step 3.

# Git & Git LFS Setup

# 1. Download [git](https://git-scm.com/downloads) if you don't already have it 
#    installed.
#    1. Setup your git account and verify that you're able to clone a private 
#       repo (doesn't matter if the repo actually has anything in it, just need 
#       to make sure that you're able to use git properly).
#    2. Follow the git-lfs install guide 
#       https://github.com/git-lfs/git-lfs?utm_source=gitlfs_site&utm_medium=installation_link&utm_campaign=gitlfs#installing.
#    3. If you didn't run it in the guide, run the command `git lfs install` 
#       after getting git setup and git-lfs installed.

# Import libraries
import os
from timeit import default_timer as timer
from llama_cpp import Llama

# Get model with GPU offloading
model = Llama.from_pretrained(
    repo_id="QuantFactory/Meta-Llama-3-8B-Instruct-GGUF-v2",
    filename="Meta-Llama-3-8B-Instruct-v2.Q6_K.gguf",
    n_gpu_layers=-1,  # Offload as much as possible
    n_threads=os.cpu_count(),
    n_ctx=8192,
    verbose=False,
)

# Define the entire chat to pass for output generation
chat = [
    {
        'role': 'system',
        'content': 'You are an assistant who perfectly describes large language models while imitating the speech style of pirates.'
    },
    {
        'role': 'user',
        'content': 'Tell me what a LLM is.'
    }
]

# Get GPU start time
gpu_start = timer()

# Get GPU output
gpu_output = model.create_chat_completion(messages=chat)

# Get GPU end time
gpu_end = timer()

# Free up memory for the CPU inference
del model

# Get model with no GPU offloading
model = Llama.from_pretrained(
    repo_id="QuantFactory/Meta-Llama-3-8B-Instruct-GGUF-v2",
    filename="Meta-Llama-3-8B-Instruct-v2.Q6_K.gguf",
    n_threads=os.cpu_count(),
    n_ctx=8192,
    verbose=False,
)

# Get CPU start time
cpu_start = timer()

# Get CPU output
cpu_output = model.create_chat_completion(messages=chat)

# Get CPU end time
cpu_end = timer()

# Calculate runtimes
gpu_elapsed_time = gpu_end - gpu_start
gpu_mins, gpu_secs = divmod(gpu_elapsed_time, 60)
gpu_hours, gpu_mins = divmod(gpu_mins, 60)

cpu_elapsed_time = cpu_end - cpu_start
cpu_mins, cpu_secs = divmod(cpu_elapsed_time, 60)
cpu_hours, cpu_mins = divmod(cpu_mins, 60)

# Calculate difference in runtimes
gpu_faster = True
diff_hours = 0.0
diff_mins = 0.0
diff_secs = 0.0

if cpu_elapsed_time > gpu_elapsed_time:
    diff_elapsed = cpu_elapsed_time - gpu_elapsed_time
    diff_mins, diff_secs = divmod(diff_elapsed, 60)
    diff_hours, diff_mins = divmod(diff_mins, 60)
    percentage_difference = ((cpu_elapsed_time - gpu_elapsed_time) / gpu_elapsed_time) * 100
else:
    gpu_faster = False
    diff_elapsed = gpu_elapsed_time - cpu_elapsed_time
    diff_mins, diff_secs = divmod(diff_elapsed, 60)
    diff_hours, diff_mins = divmod(diff_mins, 60)
    percentage_difference = ((gpu_elapsed_time - cpu_elapsed_time) / cpu_elapsed_time) * 100

# Print results
print(f"CPU-only took: {cpu_hours:.0f} hours, {cpu_mins:.0f} minutes, and {cpu_secs:.2f} seconds for inference.")
print(f"With GPU offloading it took: {gpu_hours:.0f} hours, {gpu_mins:.0f} minutes, and {gpu_secs:.2f} seconds for inference.")
print("CPU Output:")
print()
print(cpu_output['choices'][0]['message']['content'])
print()
print('GPU Output:')
print()
print(gpu_output['choices'][0]['message']['content'])
print()

if gpu_faster:
    print(f"GPU Inference time was {percentage_difference:.2f}% faster.")
    print(f"GPU inference time was faster by {diff_hours:.0f} hours, {diff_mins:.0f} minutes, and {diff_secs:.2f} seconds.")
else:
    print(f"CPU Inference time was {percentage_difference:.2f}% faster.")
    print(f"CPU inference time was faster by {diff_hours:.0f} hours, {diff_mins:.0f} minutes, and {diff_secs:.2f} seconds.")
