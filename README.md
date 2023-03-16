[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
[![PyPI Version](https://img.shields.io/pypi/v/codeproject-ai-api.svg)](https://pypi.org/project/codeproject-ai-api/)

# codeproject-ai-api
A simple Python SDK for working with [CodeProject.AI Server](https://codeproject.com/ai). This SDK provides classes for making requests to the object detection & face detection/recognition endpoints, as well as helper functions for processing the results. See the Jupyter notebooks for usage.

This work is a direct port of Robin Cole's original [deepstack-python](https://github.com/robmarkcole/deepstack-python) project.

Run CodeProject.AI Server (CPU mode):
```
docker run --name CodeProject.AI-Server -d -p 32168:32168 ^
 --mount type=bind,source=C:\ProgramData\CodeProject\AI\docker\data,target=/etc/codeproject/ai ^
 --mount type=bind,source=C:\ProgramData\CodeProject\AI\docker\modules,target=/app/modules ^
   codeproject/ai-server
```
and ensure you have an ObjectDetection module and a Face Processing module installed.

Check CodeProject.AI Server is running by opening the dashboard at http://localhost:32168

## Development
* Create venv -> `python3.7 -m venv venv`
* Use venv -> `source venv/bin/activate`
* `pip3 install -r requirements.txt` and `pip3 install -r requirements-dev.txt`
* Run tests with `venv/bin/pytest tests/*`
* Black format with `venv/bin/black .`

## Jupyter
* Docs are created using Jupyter notebooks
* Install in venv with -> `pip3 install jupyterlab`
* Run -> `venv/bin/jupyter lab`
