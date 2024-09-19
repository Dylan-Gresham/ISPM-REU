# ISPM-REU

A collection of relevant files for simple local LLM deployment.

## Installing Packages

The required packages for all the Python scripts in this repository are the following:

1. `llama-cpp-python`
2. `huggingface_hub`

And if you want to run the notebooks as well, the only other package required is:

1. `jupyter` (only if you want to run the notebooks and not just the scripts)

### Pre-Install Steps

1. Download a C/C++ compiler.

Llama-cpp-python provides Python bindings for a C/C++ library called llama-cpp. In order to successfully use this 
library, you will need to have a C/C++ compiler installed prior to trying to install llama-cpp-python.

For Linux, either gcc or clangd are recommended.

For macOS, having Xcode installed will automatically download and install a compatible C/C++ compiler.

For Windows, use Microsoft Visual Studio (note, not VS Code) or directly download the [build tools for Visual Studio](https://visualstudio.microsoft.com/downloads/)
from the bottom of that URL.

### Installing for CPU-only Inference

No additional options or environment variables need to be set.

The full command can be found below:

```bash
pip install llama-cpp-python huggingface_hub jupyter
```

Alternatively, executing the `install-cpu.sh` bash script will install the exact versions used in this repository.

This script will install into an existing Python virtual environment, *but* it **will not** create a new virtual 
environment for you. This script will only work if a virtual environment exists. Additionally, upon exiting the script,
your shell environment will change from when you executed the script. I.e., if you hadn't sourced the Python virtual
environment before executing the script, you will still not be sourced into the virtual environment when the script 
finishes. The packages installed packages will still only be installed to the virtual environment.

### Installing for CPU and/or GPU Inference

To allow for `llama-cpp-python` to utilize the GPU three conditions need to be met.

1. You need to own a CUDA compatible GPU
2. You need to have the Nvidia CUDA GPU Computing Toolkit installed
3. `llama-cpp-python` needs to be installed with certain environment variables set

The aforementioned variables are:

```bash
FORCE_CMAKE=1
CMAKE_ARGS="-DGGML_CUDA=on"
CUDACXX="/path/to/your/nvcc/executable"
```

On Linux and macOS, you can determine your `nvcc` executable location by executing the following command in a terminal: 
`which nvcc`.

On Windows, you can determine your `nvcc` executable by using the `Get-Command` PowerShell cmdlet or if you don't use 
PowerShell, the `where.exe` program.

In PowerShell:

```powershell
Get-Command nvcc
```

In Windows cmd:

```console
where nvcc
```

Once the path has been obtained, you can execute the following command, replacing `/your/path/to/nvcc` with your path to 
the `nvcc` executable that you just found, in your operating system's file path syntax, i.e., Linux using `./` and 
Windows using `.\`.

```bash
FORCE_CMAKE=1 CMAKE_ARGS="-DGGML_CUDA=on" CUDACXX="/your/path/to/nvcc" pip install llama-cpp-python jupyter huggingface_hub
```

Alternatively, executing the `install-cpu.sh` bash script will install the exact versions used in this repository with
the environment flags set automatically for you.

This script will install into an existing Python virtual environment, *but* it **will not** create a new virtual
environment for you. This script will only work if a virtual environment exists. Additionally, upon exiting the script,
your shell environment will change from when you executed the script. I.e., if you hadn't sourced the Python virtual
environment before executing the script, you will still not be sourced into the virtual environment when the script
finishes. The packages installed packages will still only be installed to the virtual environment.

### What to do if Your Pip Install Failed 

Fix the issue and execute the command again, this time with the following flags as well:

```bash
<ENVIRONMENT_VARIABLES> pip install --upgrade --force-reinstall --no-cache-dir <PACKAGES>
```

Some of the common issues I've seen and/or ran into myself have been misspelling the CMAKE_ARGS and putting in DDGGML or
DGML or even forgetting the '-' at the start. Another potential issue that I haven't seen personally would be having an
outdated version of the CUDA Toolkit installed. I'm aware there's a minimum version, but it seems like you would have to
have not used CUDA for several years and didn't allow auto-updating, in which case your best bet is to reinstall the 
toolkit entirely, or having installed a previous version of the toolkit for another use case, in which case I imagine
you're more proficient with the CUDA Toolkit than I am and are aware of how to switch to a more recent version.

If all else fails, the GitHub for llama-cpp-python is usually more than happy to help users get started. Make sure to 
take a look previous installation issues before posting your own in case someone else has already posted the same 
problem and a solution already exists.

## Git and Git Large File Storage (LFS) Setup

1. Download [git](https://git-scm.com/downloads) if you don't already have it installed.
    1. Set up your git account and verify that you're able to clone a private repo (doesn't matter if the repo actually 
    has anything in it, just need to make sure that you're able to use git properly).
    2. Follow the official git-lfs install guide [git-lfs](https://github.com/git-lfs/git-lfs?utm_source=gitlfs_site&utm_medium=installation_link&utm_campaign=gitlfs#installing).
    3. If you didn't run it in the guide, run the command `git lfs install` after getting git setup and git-lfs installed.

## HuggingFace Hub and CLI Setup

1. You'll need to create a HuggingFace account and access token:
    1. Create an account on [HuggingFace](https://huggingface.co).
    2. Once logged into your account, click your profile picture in the upper right corner and navigate to 
    Settings > Access Tokens.
    3. Click New Token and generate a new token, I made mine a "Write" token, but it shouldn't matter if it's a "Read" 
    or "Write" token for the purposes of this repository.
2. Copy your access token that you made in Step 1 to your clipboard.
3. Run the command `huggingface-cli login` and paste your access token when prompted.
4. I said yes to add the token to my git credentials, I don't think this is necessary though.

## `local_llm_deployment.md`

Markdown note file describing the typical process for deploying a LLM for local usage.

## `cpu_only.ipynb` & `cpu_only.py`

A Jupyter notebook detailing the setup and running of inference using the Llama 3 8b model from Meta.

The `Requirements` section in this notebook is probably the most detailed one.

`cpu_only.py` is the same code as `cpu_only.ipynb` just in a Python script format instead of a Jupyter Notebook.

## `conversations.ipynb`

A Jupyter notebook detailing the setup and running of conversational inference using the Llama 3 8b model from Meta.

Very similar to `cpu_only` except the previous chats are kept track of and passed to the model at each inference 
generation step.

## `cpu_vs_gpu.ipynb`

A Jupyter notebook where inference is run on both the GPU and CPU to compare runtimes.

## `llagua2-llama3.ipynb`

A Jupyter notebook detailing a simple sample process for using Llama Guard 2 as a companion model for another model, in 
this case Llama 3 8b from Meta.

## A Note on Purple Llama

Llama Guard and Llama Guard 2 are fine-tuned models (often referred to as derived or children models) of Llama 2 7b and 
Llama 3 8b respectively.

CodeShield is a Python script that aims to analyze code (only a [small subset of all programming languages](https://github.com/meta-llama/PurpleLlama/blob/main/CodeShield/insecure_code_detector/README.md#languages-supported) are 
supported as of now) that's either generated by or passed into a LLM for vulnerabilities and potentially insecure 
portions of code.

- Llama Guard/Llama Guard 2 and CodeShield are ***NOT*** meant to be used on their own. They're meant to be used as
- an additional layer on top of whatever other security practices are implemented in your environment.

## A Note on LLMs in General

You'll find that in the code examples provided, I only used `llama-cpp-python` as the library for inference. This is due
to its ability to run on either the CPU or GPU or on both. This allows running LLMs locally on virtually any machine no 
matter if they have a GPU or not.

Of course, it's very much possible to use other libraries, most others simply didn't perform as well as 
`llama-cpp-python` did in my testing or I was unable to get them properly setup to run with quantized models.

### Runtime

CPU only inference time will always be longer than GPU inference time. This is due to GPU's inherent capability to 
perform matrix/tensor computations at a faster rate than a CPU.

GPU inference time (and also CPU inference time) is heavily dependent on the amount of VRAM (GPUs) and RAM (CPUs, 
optionally alongside GPUs) that the host machine has available.

When inference is being done, the model parameters are loaded into the RAM/VRAM spaces while the input is cached. The 
cache is then taken up (for the remainder of the inference time) with the previous/current layers' outputs and the 
remaining parameters until there are no more parameters left. The last step is un-tokenizing the output into 
human-readable text for output.

Although it may not initially be clear, the primary limiting factor for inference time is the speed of data transfer 
between Disk -> RAM/VRAM -> cache while the secondary limiting factor is the actual speed of the CPU/GPU that is running 
the computations.

- More often than not, the slowdown is due to the CPU/GPU computing faster than the data can be loaded into the relevant
memory.

