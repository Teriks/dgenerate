import toml
import os

poetry_file = os.path.join(os.path.dirname(__file__), '..', '..', 'poetry', 'pyproject.toml')

with open(poetry_file, "r") as file:  # Use "rb" since tomllib requires bytes
    data = toml.load(file)

version = data["tool"]["poetry"]["dependencies"].get("opencv-python-headless")
print(version.lstrip('^'))
