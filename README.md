# Research Agent  

## Configure Python Environment  

Go to the project's root directory.  
First, [install uv](https://docs.astral.sh/uv/getting-started/installation/) if you haven't already.  
Install Python via uv:  
```bash
uv python install 3.12
```  
Install project dependencies:  
```bash
uv sync --all-packages --all-extras
```
Python virtual env `.venv` should appear in the project's root directory.  

```sh
uv run python main.py
```  
