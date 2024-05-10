#!/bin/bash

# Create the .kaggle directory if it doesn't exist
mkdir -p ~/.kaggle

# Copy kaggle.json to the .kaggle directory
cp kaggle.json ~/.kaggle/

# Set permissions for kaggle.json
chmod 600 ~/.kaggle/kaggle.json
