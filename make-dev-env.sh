#!/bin/bash

set -e

echo "Setting up dgenerate development environment..."

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is not installed or not in PATH"
    exit 1
fi

# Run the Python setup script
python3 make_dev_env.py

if [ $? -eq 0 ]; then
    echo ""
    echo "Setup completed successfully!"
else
    echo ""
    echo "Setup failed! Please check the error messages above."
    exit 1
fi