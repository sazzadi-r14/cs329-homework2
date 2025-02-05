# cs329-homework2
Here we are collecting a set of tools to make it easier to evaluate tool-use and agentic LLM techniques.

## Setup

```bash
1. Download anaconda from https://www.anaconda.com/ and install
2. Create a new environment, and install the repo:

```bash
conda create -n cs329a-hw2 python=3.10 -y
conda activate cs329a-hw2
pip install -e .  # Run this from root of the repo.
```

This will make sure the package is installed with requirements, so you can import functionality from `cs329_hw2`. For instance, you can use the `get_sampler` function to get a sampler.

Make sure you have the `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `TOGETHER_API_KEY`, `GOOGLE_API_KEY`, and `ALPHA_VANTAGE_KEY` environment variables set.