# Quantum Ollama Terminal

A powerful quantum computing interface combining classical AI (Ollama) with quantum computing capabilities (D-Wave and Cirq).

## Features

- **Quantum Computing Integration**
  - D-Wave quantum annealing support
  - Cirq quantum circuit simulation
  - Hybrid quantum-classical computation

- **AI Capabilities**
  - Local LLM integration via Ollama
  - Quantum-enhanced AI problem solving
  - Real-time chat interface

- **System Features**
  - Python script execution
  - System command execution
  - Web search integration
  - Data visualization

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/OneNessQBAI/QuantumOllama.git
   cd QuantumOllama
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install Ollama:
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ```

4. Pull desired Ollama models:
   ```bash
   ollama pull llama2
   ```

5. Configure D-Wave:
   - Obtain API token from [D-Wave Leap](https://cloud.dwavesys.com/leap/)
   - Set environment variable:
     ```bash
     export DWAVE_API_TOKEN="your-token-here"
     ```

## Usage

Start the application:
```bash
streamlit run app.py
```

### Key Features

1. **Quantum Problem Solving**
   - Formulate QUBO problems
   - Run on D-Wave hardware or Cirq simulator
   - Visualize results

2. **AI Chat Interface**
   - Local LLM integration
   - Quantum-enhanced responses
   - Context-aware conversations

3. **System Integration**
   - Execute Python scripts
   - Run system commands
   - Web search capabilities

## Configuration

### Ollama Models
- Default model: llama2
- Available models can be listed and selected in the sidebar
- New models can be pulled using:
  ```bash
  ollama pull <model_name>
  ```

### D-Wave Settings
- API token configuration
- Hardware mode selection (QPU/Hybrid)
- Real-time hardware monitoring

## Troubleshooting

### Common Issues

1. **Ollama Model Pulling Errors**
   - Ensure Ollama is installed and running
   - Check network connectivity
   - Verify sufficient disk space

2. **D-Wave Connection Issues**
   - Verify API token is valid
   - Check D-Wave Leap status
   - Ensure proper environment variables

3. **Quantum Simulation Errors**
   - Verify Cirq installation
   - Check system resources
   - Validate problem formulation

## License

MIT License

Copyright (c) 2024 Oneness Blockchain AI (www.onenesscan.com)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

For support, contact: support@onenessblockchain.ai
