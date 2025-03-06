// Global variables
let session;
let charEncoding;
let modelLoaded = false;
let modelConfig = {
    hiddenSize: 512,  // Default hidden size, will be updated after model inspection
    numLayers: 2      // Default number of layers, will be updated after model inspection
};

// Initialize the application
async function init() {
    try {
        // Update UI to show loading state
        const generateBtn = document.getElementById('generate-btn');
        generateBtn.textContent = 'Loading model...';
        
        // Load character encoding
        console.log('Loading character encoding...');
        const response = await fetch('model/char_encoding.json');
        if (!response.ok) {
            throw new Error(`Failed to load character encoding: ${response.status} ${response.statusText}`);
        }
        charEncoding = await response.json();
        console.log('Character encoding loaded successfully');
        
        // Try to load model info if available
        try {
            const modelInfoResponse = await fetch('assets/model/model_info.json');
            if (modelInfoResponse.ok) {
                const modelInfo = await modelInfoResponse.json();
                modelConfig.hiddenSize = modelInfo.hidden_size || modelConfig.hiddenSize;
                modelConfig.numLayers = modelInfo.num_layers || modelConfig.numLayers;
                console.log('Model info loaded successfully:', modelInfo);
            }
        } catch (modelInfoError) {
            console.warn('Could not load model info:', modelInfoError);
            console.warn('Using default model configuration');
        }
        
        // Set up ONNX Runtime options
        console.log('Setting up ONNX Runtime...');
        const ort = window.ort;
        const options = {
            executionProviders: ['wasm'],
            graphOptimizationLevel: 'all'
        };
        
        try {
            // Try to load the model
            console.log('Loading model...');
            session = await ort.InferenceSession.create('model/model.onnx', options);
            console.log('Model loaded successfully');
            
            // Inspect model to determine hidden size and number of layers if not already loaded from model_info.json
            try {
                const inputInfo = session.inputNames.map(name => {
                    const info = session.inputIndices[name];
                    return { name, info };
                });
                console.log('Model input info:', inputInfo);
                
                // Try to extract hidden size from model metadata
                if (session.inputNames.includes('hidden')) {
                    const hiddenInfo = session._inputs.find(input => input.name === 'hidden');
                    if (hiddenInfo && hiddenInfo.dims && hiddenInfo.dims.length >= 3) {
                        modelConfig.numLayers = hiddenInfo.dims[0];
                        modelConfig.hiddenSize = hiddenInfo.dims[2];
                        console.log(`Detected model config: layers=${modelConfig.numLayers}, hiddenSize=${modelConfig.hiddenSize}`);
                    }
                }
            } catch (inspectError) {
                console.warn('Could not inspect model details:', inspectError);
                console.warn('Using previously loaded or default model configuration');
            }
            
            // Enable the generate button
            generateBtn.disabled = false;
            generateBtn.textContent = 'Generate Text';
            
            modelLoaded = true;
        } catch (modelError) {
            console.error('Error loading model:', modelError);
            
            // If we can't load the model, update UI to show demo mode
            generateBtn.textContent = 'Generate Text (Demo Mode)';
            generateBtn.disabled = false;
            
            // We'll use demo mode without the actual model
            modelLoaded = false;
        }
        
        // Set up event listeners
        generateBtn.addEventListener('click', generateText);
        
        // Set up temperature slider
        const tempSlider = document.getElementById('temperature');
        const tempValue = document.getElementById('temp-value');
        tempSlider.addEventListener('input', () => {
            tempValue.textContent = tempSlider.value;
        });
        
    } catch (error) {
        console.error('Initialization error:', error);
        document.getElementById('generate-btn').textContent = 'Error loading app';
        
        // Show error message to user
        const output = document.getElementById('output');
        output.textContent = `Error initializing the app: ${error.message}\n\nPlease check the console for more details.`;
        output.style.color = 'red';
    }
}

// Text generation function
async function generateText() {
    const seedText = document.getElementById('seed-text').value.trim();
    if (!seedText) {
        alert('Please enter some seed text');
        return;
    }
    
    const length = parseInt(document.getElementById('length').value);
    const temperature = parseFloat(document.getElementById('temperature').value);
    
    // Show loading indicator
    document.getElementById('loading').classList.remove('hidden');
    document.getElementById('output').textContent = '';
    document.getElementById('generate-btn').disabled = true;
    
    try {
        let result;
        
        if (modelLoaded && session) {
            // Real model inference
            result = await generateWithModel(seedText, length, temperature);
        } else {
            // Demo mode - fake generation with a delay
            result = await demoGeneration(seedText, length, temperature);
        }
        
        // Display the result
        document.getElementById('output').textContent = result;
    } catch (error) {
        console.error('Error during text generation:', error);
        document.getElementById('output').textContent = `Error generating text: ${error.message}`;
        document.getElementById('output').style.color = 'red';
    } finally {
        // Hide loading indicator and re-enable button
        document.getElementById('loading').classList.add('hidden');
        document.getElementById('generate-btn').disabled = false;
    }
}

// Generate text using the ONNX model
async function generateWithModel(seedText, length, temperature) {
    // Process the seed text
    let result = seedText;
    let currentChar = seedText.slice(-1);
    
    // Initialize hidden state (zeros)
    const hiddenSize = modelConfig.hiddenSize;
    const numLayers = modelConfig.numLayers;
    console.log(`Using model config: layers=${numLayers}, hiddenSize=${hiddenSize}`);
    
    // Create hidden state with the correct dimensions
    let hidden = new Float32Array(numLayers * 1 * hiddenSize); // num_layers * batch_size * hidden_size
    hidden.fill(0);
    
    // Generate text character by character
    for (let i = 0; i < length; i++) {
        // Get the index for the current character
        const charIndex = charEncoding.char_to_idx[currentChar] || 0;
        
        // Create input tensor (shape: [1, 1])
        const inputTensor = new ort.Tensor('int64', [BigInt(charIndex)], [1, 1]);
        
        // Create hidden tensor with the correct shape
        const hiddenTensor = new ort.Tensor('float32', hidden, [numLayers, 1, hiddenSize]);
        
        try {
            // Run inference
            const outputs = await session.run({
                'input': inputTensor,
                'hidden': hiddenTensor
            });
            
            // Get output and next hidden state
            const outputData = outputs.output.data;
            hidden = outputs.next_hidden.data;
            
            // Apply temperature and sample
            const logits = Array.from(outputData);
            const probs = softmax(logits, temperature);
            const nextCharIndex = sampleFromDistribution(probs);
            const nextChar = charEncoding.idx_to_char[nextCharIndex.toString()] || ' ';
            
            // Update result and current character
            result += nextChar;
            currentChar = nextChar;
            
            // Update UI every few characters for responsiveness
            if (i % 5 === 0) {
                document.getElementById('output').textContent = result;
                // Allow UI to update
                await new Promise(resolve => setTimeout(resolve, 0));
            }
        } catch (inferenceError) {
            console.error('Inference error:', inferenceError);
            throw new Error(`Error during text generation: ${inferenceError.message}`);
        }
    }
    
    return result;
}

// Demo generation function for when the model isn't available
async function demoGeneration(seedText, length, temperature) {
    // This is a simple markov-chain-like demo generator
    // It doesn't use the actual model but provides a reasonable demo
    
    // Define some common patterns based on the seed text style
    const result = seedText;
    let demoText = '';
    
    // Simple demo generation logic
    const isUpperCase = (char) => char === char.toUpperCase() && char !== char.toLowerCase();
    const hasPunctuation = /[.,!?;:]/.test(seedText);
    const wordLength = seedText.split(/\s+/).map(w => w.length);
    const avgWordLength = wordLength.reduce((a, b) => a + b, 0) / wordLength.length;
    
    // Get some statistics from the seed text to mimic its style
    const charFreq = {};
    for (let i = 0; i < seedText.length; i++) {
        const char = seedText[i];
        charFreq[char] = (charFreq[char] || 0) + 1;
    }
    
    // Sort characters by frequency
    const sortedChars = Object.keys(charFreq).sort((a, b) => charFreq[b] - charFreq[a]);
    
    // Generate text with a delay to simulate processing time
    let currentText = seedText;
    
    for (let i = 0; i < length; i++) {
        // Simulate processing delay
        await new Promise(resolve => setTimeout(resolve, 10));
        
        // Simple character prediction based on seed text statistics
        let nextChar;
        
        if (Math.random() < 0.7) {
            // 70% chance to use a character from the seed text
            const randomIndex = Math.floor(Math.random() * sortedChars.length);
            nextChar = sortedChars[randomIndex];
        } else {
            // 30% chance to use a random character
            const chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,;:!?-_\"'()[] \n";
            nextChar = chars[Math.floor(Math.random() * chars.length)];
        }
        
        // Add the character to the result
        demoText += nextChar;
        
        // Update UI every few characters
        if (i % 5 === 0) {
            document.getElementById('output').textContent = currentText + demoText;
        }
    }
    
    return currentText + demoText;
}

// Helper function: softmax with temperature
function softmax(logits, temperature = 1.0) {
    // Apply temperature
    const scaled = logits.map(l => l / temperature);
    
    // Subtract max for numerical stability
    const maxLogit = Math.max(...scaled);
    const exps = scaled.map(l => Math.exp(l - maxLogit));
    
    // Normalize
    const sumExps = exps.reduce((a, b) => a + b, 0);
    return exps.map(e => e / sumExps);
}

// Helper function: sample from probability distribution
function sampleFromDistribution(probs) {
    const rand = Math.random();
    let sum = 0;
    for (let i = 0; i < probs.length; i++) {
        sum += probs[i];
        if (rand < sum) return i;
    }
    return probs.length - 1;
}

// Initialize when the page loads
document.addEventListener('DOMContentLoaded', init); 