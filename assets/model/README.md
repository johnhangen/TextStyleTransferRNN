# Model Files

Place your ONNX model files in this directory.

## Required Files

1. `model.onnx` - The ONNX model file converted from your PyTorch GRU model
2. `char_encoding.json` - The character encoding mapping file

## Converting Your Model

You can export your model directly from your Python code using the integrated export functionality:

```python
from src.model.model import Model
from config.configs import Config

# Load your configuration
config = Config()

# Create and load your model
model = Model(config)
model.load()

# Export to ONNX format (saves to this directory by default)
model.export_to_onnx()
```

This will generate both the ONNX model file and the character encoding JSON file in this directory.

## Demo Mode

If no model file is present, the application will run in "Demo Mode" which simulates text generation without using an actual model. 