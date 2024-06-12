# Deploying a LLM Locally

No matter what model you're working with there's always three steps to deploying it locally.

1. Determining the model.
2. Pulling the model into your local environment.
3. Setting up an interface for interacting with the model.

## Determining the Model

This is arguably the most important step. There's hundreds of thousands of models to choose from on HuggingFace and other sites from base models like Llama 3 8b to child models like Llama 3 8b Instruct Gradient 1048k GGUF from bartowski.

Learning to choose a model that best fits your use case is more of a learned skill from experimentation than something that can be easily taught. My best advice for choosing a model that is appropriate for your use case is to take a brokendown approach to it.

1. Determine what your use case is.
  1. Are you trying to generate images, text, video, or something else?
  2. Do you want the generated content to be new synthetic content (meaning it's generated based off the inputs and what the model "knows") or do you want it to be summarized information based on what the model "knows" or is given to work with?
2. Determine how much resources you're willing to allocate to the model.
  1. Do you have a GPU? If so, how much VRAM does it have?
    1. Computation time is mostly limited by the amount of memory available to work with. The computation speed is incredibly fast on GPUs but this brings the problem of getting data into the GPU quick enough. For more general use cases like video games or simpler vector arithmetic, this doesn't really come into play but with LLMs, the entire GPU's VRAM is typically filled with the parameters and inputs are fed into memory, used in computations, and then used parameters are dumped out of the GPU and the remainnig parameters are brought in. This leads to a bottleneck for GPUs (and CPUs too) where you have to be able to get information to the processing unit fast enough to maximize efficiency. Thus, larger memory sizes will naturally perform better than smaller ones as less transfers need to be made.
  2. How much time are you willing to wait for generations?
3. Find the models that fit your use case and can be run with the resources you have available.

## Pulling the Model Into Your Local Environment

This is probably the simplest part.

There's typically two common approaches that are taken when pulling models down from the Internet.

1. Helper method
2. Manual downloading & referencing

### Helper Method

Libraries like HuggingFace's `transformers` library have helper methods for this exact situation.

Almost all popular ML libraries will have a method to pull a model down, i.e., `from_pretrained`, `AutoModelForCausalLM`, `Llama`, and sometimes you'll have to use another libraries method to get the model before using it in your desired library.

### Manual Download and Referencing

Occasionally, the model you'll want to work with won't be easily accessible. Instead what you may need to do is manually download the files and store them locally and then in your code provide the relative path to the file.

The most common reason this approach might be used is for quantized models. They're easier to run and smaller in size but have a different format than standard LLMs which means they need different methods for working with them.

## Interfacing With the Model

This is another part that's dependent on your use case. If you're building a web app you may want to host it on a server in the cloud and use HTTP requests to interact with it.

If you're building a desktop app you may want to package the model with the app itself (large executable) and be able to query it whenever and from wherever. Or, you may want to host it somewhere else and use an API to access it.

Sometimes you're not even building an app and are fine to just use the CLI or interact with it through Python code directly.

