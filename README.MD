# Text Style Transfer with GRU

A web application that uses a GRU (Gated Recurrent Unit) neural network to generate text in a specific style. This app runs entirely in your browser using ONNX Runtime Web.

## Live Demo

Visit the live demo at: [https://yourusername.github.io/TextStyleTransferRNN](https://yourusername.github.io/TextStyleTransferRNN)

## Features

- Generate text in the style of your trained model
- Adjust generation length and temperature
- Runs entirely in the browser - no server required
- Works offline once loaded

## How to Use

1. Enter some seed text in the input box
2. Adjust the generation length (how many characters to generate)
3. Adjust the temperature (higher = more creative, lower = more predictable)
4. Click "Generate Text"
5. The model will generate text continuing from your seed text

## Technical Details

This application uses:

- A GRU (Gated Recurrent Unit) neural network with attention mechanism
- ONNX Runtime Web for in-browser inference
- Character-level text generation

## Setup Instructions

### Converting Your Model

To use your own trained model, you can export it directly from your Python code:

```python
from src.model.model import Model
from config.configs import Config

# Load your configuration
config = Config()

# Create and load your model
model = Model(config)
model.load()

# Export to ONNX format (saves to assets/model/ by default)
model.export_to_onnx()
```

This will generate both the ONNX model file and the character encoding JSON file in the `assets/model/` directory.

### Model Dimensions

The web application automatically detects the dimensions of your model (hidden size and number of layers) from the exported ONNX model. If you encounter dimension mismatch errors, make sure:

1. The model is exported correctly with the `export_to_onnx()` method
2. The `model_info.json` file contains the correct dimensions:

```json
{
  "hidden_size": 512,  // Replace with your model's hidden size
  "num_layers": 2,     // Replace with your model's number of layers
  "vocab_size": 78     // Replace with your vocabulary size
}
```

### Deploying to GitHub Pages

1. Fork or clone this repository
2. Add your model files to the `assets/model/` directory
3. Push to GitHub
4. Enable GitHub Pages in your repository settings

## Development

To run locally:

1. Clone the repository
2. Open `index.html` in a web browser

## Troubleshooting

### Dimension Mismatch Errors

If you see errors like:

```
Error: failed to call OrtRun(). ERROR_CODE: 2, ERROR_MESSAGE: Got invalid dimensions for input: hidden
```

This indicates a mismatch between the dimensions expected by the model and those provided by the JavaScript code. To fix this:

1. Make sure you've installed the required packages: `pip install onnx onnxruntime`
2. Re-export your model using the `export_to_onnx()` method
3. Check that the `model_info.json` file has the correct dimensions
4. If needed, manually update the `modelConfig` values in `assets/js/app.js`

## Limitations

- The model runs in the browser, so performance depends on the user's device
- Large models may be slow on mobile devices
- The initial load may take some time as the model is downloaded

## License

MIT License

## Acknowledgments

- This project uses [ONNX Runtime Web](https://github.com/microsoft/onnxruntime) for in-browser inference
