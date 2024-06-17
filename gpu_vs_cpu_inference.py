# Import libraries
import os
from timeit import default_timer as timer
from llama_cpp_python import Llama

# Installing libraries:
#   Most importantly, make sure you're in a Python virtual environment first, have an Nvidia GPU, and have
#   the CUDA toolkit installed.
#
#   When installing llama-cpp-python, you'll need to do some extra steps beforehand.
#   All you'll need to do is set some additional CLI args before executing the pip command. These args are:
#
#   - CMAKE_ARGS="-DLLAMA_CUBLAS=on"
#   - FORCE_CMAKE=1
#   - CUDACXX="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v<version_number>\bin\nvcc.exe"
#     - Change path to wherever your installation is.
#
#   Then execute the following pip command to install the appropriately compiled llama-cpp-python.
#
#   pip install --upgrade --force-reinstall --no-cache-dir llama-cpp-python

# Get model with GPU offloading
model = Llama.from_pretrained(
    repo_id="",
    filename="",
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
    repo_id="",
    filename="",
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
else:
    gpu_faster = False
    diff_elapsed = gpu_elapsed_time - cpu_elapsed_time
    diff_mins, diff_secs = divmod(diff_elapsed, 60)
    diff_hours, diff_mins = divmod(diff_mins, 60)

# Print results
print(f"CPU-only took: {cpu_hours:.0f} hours, {cpu_mins:.0f} minutes, and {cpu_secs:.0f} seconds for inference.")
print(f"With GPU offloading it took: {gpu_hours:.0f} hours, {gpu_mins:.0f}, and {gpu_secs:.0f} seconds for inference.")
print("CPU Output:")
print()
print(cpu_output['choices'][0]['message']['content'])
print()
print('GPU Output:')
print()
print(gpu_output['choices'][0]['message']['content'])
print()

if gpu_faster:
    print(f"GPU inference time was faster by {diff_hours:.0f} hours, {diff_mins:.0f} minutes, and {diff_secs:.0f} seconds.")
else:
    print(f"CPU inference time was faster by {diff_hours:.0f} hours, {diff_mins:.0f} minutes, and {diff_secs:.0f} seconds.")

