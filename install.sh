!#/bin/bash

# Install dependencies
echo "Installing dependencies"
python3 -m venv .env
source .env/bin/activate
pip install -r requirements.txt
echo "Dependencies installed"