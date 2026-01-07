import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO,format='[%(asctime)s]: %(message)s:')

project_name = "app"

list_of_files = [
    f"{project_name}/api/__init__.py",
    f"{project_name}/api/v1/__init__.py",
    f"{project_name}/api/v1/endpoints/__init__.py",
    f"{project_name}/data/__init__.py",
    f"{project_name}/model/__init__.py",
    f"{project_name}/schemas/__init__.py",
    f"{project_name}/services/__init__.py",
    f"{project_name}/utils/__init__.py",
    f"{project_name}/__init__.py",
    f"{project_name}/config.py",
    f"{project_name}/main.py",
    ".env",
    ".gitignore",
    "README.md",
    "requirements.txt",
]


for filepath in list_of_files:
    filepath = Path(filepath)

    dir,file=os.path.split(filepath)

    if dir != "":
        os.makedirs(dir,exist_ok=True)
        logging.info(f"Directory{dir} created for file{file} successfully")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath)==0):
        with open(filepath, "w") as f:
            logging.info(f"Creating files: {filepath} successfully")
    else:
        logging.info("Files{file} already exists")
 