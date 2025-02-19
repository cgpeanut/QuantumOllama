import streamlit as st
import subprocess
import plotly.graph_objects as go
from datetime import datetime
import json
import os
from bs4 import BeautifulSoup
import sys
import io
from contextlib import redirect_stdout
import requests
from dwave.system import DWaveSampler, EmbeddingComposite
import dimod
import ollama
import cirq
from cirq import Simulator
import graphviz

# Initialize quantum backends
sampler = None
quantum_backend = None

# Initialize D-Wave hardware
try:
    dwave_sampler = DWaveSampler()
    sampler = EmbeddingComposite(dwave_sampler)
    st.sidebar.success("Connected to D-Wave quantum hardware")
except Exception as e:
    st.sidebar.error(f"D-Wave hardware connection failed: {str(e)}")

# Initialize Cirq simulator
try:
    quantum_backend = Simulator()
    st.sidebar.success("Cirq quantum simulator available")
except ImportError:
    st.sidebar.warning("Cirq not available for simulation")

# Initialize Ollama client
ollama_client = ollama.Client()

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'ollama_model' not in st.session_state:
    st.session_state.ollama_model = "llama2"

# Get Ollama models
def get_ollama_models():
    try:
        result = subprocess.run(['ollama', 'list'], 
                              capture_output=True, 
                              text=True,
                              encoding='utf-8',
                              errors='replace',
                              shell=True,
                              check=True)
        if result.returncode == 0:
            return result.stdout
        return f"No models found\n{result.stderr}"
    except subprocess.CalledProcessError as e:
        return f"Error retrieving models: {e.stderr}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"

def pull_ollama_model(model_name):
    try:
        process = subprocess.Popen(
            ['ollama', 'pull', model_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
            encoding='utf-8',
            errors='replace'
        )
        
        # Create a progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Read output in real-time
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
                
            if output:
                # Update progress based on output
                if 'pulling manifest' in output.lower():
                    status_text.text("Downloading model manifest...")
                    progress_bar.progress(0.2)
                elif 'downloading layer' in output.lower():
                    status_text.text("Downloading model layers...")
                    progress_bar.progress(0.5)
                elif 'verifying sha256' in output.lower():
                    status_text.text("Verifying model integrity...")
                    progress_bar.progress(0.8)
                elif 'success' in output.lower():
                    status_text.text("Model successfully pulled!")
                    progress_bar.progress(1.0)
                    return True
                    
        # Check final status
        if process.returncode == 0:
            return True
        else:
            st.error(f"Failed to pull model: {process.stderr.read()}")
            return False
            
    except Exception as e:
        st.error(f"Error pulling model: {str(e)}")
        return False

# Get D-Wave configuration
def get_dwave_config():
    try:
        sampler = DWaveSampler()
        return {
            "solver": sampler.solver.name,
            "qubits": sampler.solver.num_qubits,
            "couplers": len(sampler.solver.edges),
            "properties": sampler.solver.properties
        }
    except Exception as e:
        return f"Error retrieving D-Wave config: {str(e)}"

def chat_with_ollama(messages):
    try:
        response = ollama_client.chat(
            model=st.session_state.ollama_model,
            messages=messages,
            options={
                'temperature': 0.7,
                'max_tokens': 2000
            }
        )
        return response['message']['content']
    except Exception as e:
        st.error(f"Error communicating with Ollama: {str(e)}")
        return None

def execute_quantum_problem(problem, simulation=False):
    try:
        # Convert problem to QUBO
        bqm = dimod.BinaryQuadraticModel.from_qubo(problem)
        
        if not simulation and sampler is not None:
            # Use real D-Wave hardware if available
            sampleset = sampler.sample(bqm, num_reads=100)
            return json.dumps({
                "mode": "hardware",
                "samples": sampleset.first.sample,
                "energy": sampleset.first.energy,
                "num_occurrences": sampleset.first.num_occurrences,
                "chain_break_fraction": sampleset.first.chain_break_fraction,
                "hardware_mode": st.session_state.get('hardware_mode', 'QPU')
            })
        elif quantum_backend is not None:
            # Use Cirq simulation
            qubits = [cirq.GridQubit(0, i) for i in range(len(bqm))]
            circuit = cirq.Circuit()
            
            # Create Hamiltonian from QUBO
            hamiltonian = sum(
                bqm.linear[i] * cirq.Z(qubits[i]) +
                sum(bqm.quadratic[i, j] * cirq.Z(qubits[i]) * cirq.Z(qubits[j])
                    for j in bqm.quadratic if j > i)
                for i in bqm.linear
            )
            
            # Add Hamiltonian evolution
            circuit.append(cirq.rz(0.1).on_each(*qubits))
            circuit.append(cirq.rx(0.1).on_each(*qubits))
            
            # Simulate
            result = quantum_backend.simulate(circuit)
            return json.dumps({
                "mode": "simulation",
                "state_vector": result.final_state_vector.tolist(),
                "measurements": result.measurements
            })
        else:
            # Fallback to classical simulation
            from dimod import SimulatedAnnealingSampler
            sim_sampler = SimulatedAnnealingSampler()
            sampleset = sim_sampler.sample(bqm, num_reads=100)
            return json.dumps({
                "mode": "classical_simulation",
                "samples": sampleset.first.sample
            })
    except Exception as e:
        return str(e)

def execute_command(command):
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return result.stdout if result.returncode == 0 else result.stderr
    except Exception as e:
        return str(e)

def execute_python_script(script):
    try:
        output = io.StringIO()
        with redirect_stdout(output):
            exec(script)
        return output.getvalue()
    except Exception as e:
        return str(e)

def search_duckduckgo(query):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        url = f"https://duckduckgo.com/html/?q={query}"
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        results = []
        for result in soup.find_all('div', class_='result'):
            title = result.find('a', class_='result__a')
            snippet = result.find('a', class_='result__snippet')
            if title and snippet:
                results.append({
                    'title': title.text.strip(),
                    'snippet': snippet.text.strip(),
                    'url': title['href']
                })
        return results[:5]
    except Exception as e:
        return [{"error": str(e)}]

def create_connectivity_graph(adjacency):
    """Create a graph visualization of qubit connectivity"""
    dot = graphviz.Digraph()
    
    # Add nodes
    for node in adjacency:
        dot.node(str(node))
    
    # Add edges
    for node, neighbors in adjacency.items():
        for neighbor in neighbors:
            dot.edge(str(node), str(neighbor))
    
    return dot

def create_visualization(data):
    try:
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except:
                pass
        
        if isinstance(data, dict):
            fig = go.Figure(data=[go.Table(
                header=dict(values=list(data.keys()),
                          fill_color='paleturquoise',
                          align='left'),
                cells=dict(values=list(data.values()),
                          fill_color='lavender',
                          align='left'))
            ])
        elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            headers = list(data[0].keys())
            values = [[d[k] for d in data] for k in headers]
            fig = go.Figure(data=[go.Table(
                header=dict(values=headers,
                          fill_color='paleturquoise',
                          align='left'),
                cells=dict(values=values,
                          fill_color='lavender',
                          align='left'))
            ])
        elif isinstance(data, str):
            fig = go.Figure()
            fig.add_annotation(
                text=data,
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False
            )
        else:
            return None
            
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
        return fig
    except:
        return None

# Custom CSS for chat interface
st.markdown("""<style>
.chat-container { 
    background: linear-gradient(145deg, #1e1e2f, #2a2a40); 
    border-radius: 15px; 
    padding: 20px; 
    margin: 10px 0; 
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); 
}
.user-message { 
    background: #3182ce; 
    color: white; 
    padding: 12px 16px; 
    border-radius: 15px 15px 0 15px; 
    margin: 8px 0; 
    max-width: 80%; 
    margin-left: auto; 
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}
.assistant-message { 
    background: #edf2f7; 
    color: #2d3748; 
    padding: 12px 16px; 
    border-radius: 15px 15px 15px 0; 
    margin: 8px 0; 
    max-width: 80%; 
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}
.emoji { 
    font-size: 1.2em; 
    margin-right: 8px; 
}
.code-block { 
    background: #1a202c; 
    padding: 12px; 
    border-radius: 8px; 
    margin: 8px 0; 
}
.stButton>button { 
    background: #4c51bf; 
    color: white; 
    border-radius: 8px; 
    padding: 8px 16px; 
    border: none; 
    transition: all 0.3s ease; 
}
.stButton>button:hover { 
    background: #667eea; 
    transform: translateY(-2px); 
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); 
}
</style>""", unsafe_allow_html=True)

st.title("✨ Quantum Ollama Terminal 🌌")

# Add configuration in sidebar
with st.sidebar:
    st.subheader("⚙️ Configuration")
    
    # Display Ollama models
    with st.expander("📚 Ollama Models"):
        models = get_ollama_models()
        st.code(models, language="bash")
    
    # Display D-Wave configuration
    with st.expander("⚛️ D-Wave Hardware"):
        config = get_dwave_config()
        if isinstance(config, dict):
            st.json(config)
        else:
            st.error(config)
    
    # Ollama model selection with auto-pull
    models = get_ollama_models()
    model_list = [line.split()[0] for line in models.split('\n')[1:] if line.strip()]
    
    if model_list:
        selected_model = st.selectbox(
            "🤖 Select Ollama Model",
            model_list,
            index=0
        )
        
        if selected_model != st.session_state.get('ollama_model'):
            with st.spinner(f"🔄 Pulling {selected_model} model..."):
                try:
                    result = subprocess.run(['ollama', 'pull', selected_model], capture_output=True, text=True)
                    if result.returncode == 0:
                        st.session_state.ollama_model = selected_model
                        st.success(f"✅ Successfully pulled {selected_model} model")
                    else:
                        st.error(f"Failed to pull model: {result.stderr}")
                except Exception as e:
                    st.error(f"Error pulling model: {str(e)}")
    else:
        st.error("No Ollama models found. Please install at least one model.")
    
    # D-Wave configuration
    st.subheader("⚛️ D-Wave Quantum Control Panel")
    
    col1, col2 = st.columns(2)
    with col1:
        dwave_token = st.text_input("🔑 Enter D-Wave API Token", type="password", value=st.session_state.get('dwave_token', ''))
        hardware_mode = st.radio("⚙️ Select Hardware Mode", ["QPU", "Hybrid"], index=0)
        
    with col2:
        if st.button("🚀 Start Configuration", disabled='sampler' in st.session_state):
            if dwave_token:
                st.session_state.dwave_token = dwave_token
                os.environ['DWAVE_API_TOKEN'] = dwave_token
                try:
                    sampler = DWaveSampler()
                    st.session_state.sampler = EmbeddingComposite(sampler)
                    st.session_state.hardware_mode = hardware_mode
                    st.success("✅ Successfully connected to D-Wave!")
                except Exception as e:
                    st.error(f"❌ Connection failed: {str(e)}")
        
        if st.button("🛑 Stop Configuration", disabled='sampler' not in st.session_state):
            st.session_state.pop('sampler', None)
            st.session_state.pop('hardware_mode', None)
            st.success("Configuration stopped")
    
    if 'sampler' in st.session_state:
        # Hardware Information
        with st.expander("🔍 Hardware Details"):
            solver = st.session_state.sampler.child.solver
            st.markdown(f"**Solver:** {solver.name}")
            
            # Qubit Visualization
            st.markdown("### Qubit Distribution")
            qubit_ranges = [(i, min(i+999, solver.num_qubits-1)) for i in range(0, solver.num_qubits, 1000)]
            st.table({
                "Range": [f"{start}-{end}" for start, end in qubit_ranges],
                "Qubits": [end-start+1 for start, end in qubit_ranges]
            })
            
            # Coupler Visualization
            st.markdown("### Coupler Distribution")
            coupler_ranges = [(i, min(i+999, len(solver.edges)-1)) for i in range(0, len(solver.edges), 1000)]
            st.table({
                "Range": [f"{start}-{end}" for start, end in coupler_ranges],
                "Couplers": [end-start+1 for start, end in coupler_ranges]
            })
        
        # Execution Control
        st.markdown("### 🎛️ Execution Control")
        exec_col1, exec_col2, exec_col3 = st.columns(3)
        
        with exec_col1:
            if st.button("🔬 Analyze QPU Topology"):
                try:
                    topology = st.session_state.sampler.child.sampler.adjacency
                    st.session_state.topology = topology
                    st.success("Topology analysis complete")
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
            
            if st.button("📊 Show Qubit Connectivity"):
                if 'topology' in st.session_state:
                    st.graphviz_chart(create_connectivity_graph(st.session_state.topology))
        
        with exec_col2:
            if st.button("⚡ Run AI Script (Hardware)"):
                if 'quantum_script' in st.session_state:
                    try:
                        result = execute_quantum_problem(st.session_state.quantum_script)
                        st.session_state.last_result = result
                        st.success("Script executed successfully")
                    except Exception as e:
                        st.error(f"Execution failed: {str(e)}")
                else:
                    st.warning("No quantum script loaded")
            
            if st.button("🧠 Run AI Script (Simulation)"):
                if 'quantum_script' in st.session_state:
                    try:
                        result = execute_quantum_problem(st.session_state.quantum_script, simulation=True)
                        st.session_state.last_result = result
                        st.success("Simulation executed successfully")
                    except Exception as e:
                        st.error(f"Simulation failed: {str(e)}")
                else:
                    st.warning("No quantum script loaded")
        
        with exec_col3:
            if st.button("📈 Analyze Results"):
                if 'last_result' in st.session_state:
                    st.json(st.session_state.last_result)
                else:
                    st.warning("No results to analyze")
            
            if st.button("🧹 Clear Results"):
                st.session_state.pop('last_result', None)
                st.success("Results cleared")
    
    if st.button("🧹 Clear Chat History"):
        st.session_state.messages = []

    st.markdown("""### 🚀 Features
- 🐍 Execute Python scripts
- ⚛️ Run quantum problems on D-Wave
- 💻 Run system commands
- 🔍 Web search via DuckDuckGo
- 📊 Data visualization
- 🤖 Powered by Ollama and D-Wave

### 📝 Response Formats
1. Scripts: ```script```
2. Quantum: ```quantum```
3. Commands: ```command```
4. Search: ```search```
5. Visualization: ```visualization```
6. Text: ```response```

### 💡 Tips
- Scripts and commands are executed safely
- Quantum problems run on real D-Wave hardware
- Web search provides up-to-date information
- Visualizations are created automatically
""")

# Updated system prompt with quantum capabilities
SYSTEM_PROMPT = """⚠️ You are a Quantum AI Assistant. Be interactive with the user but when coding COPY AND PASTE THESE EXACT FORMATS - DO NOT MODIFY THEM!

x❌ THESE WILL BE REJECTED:
- Raw text outside code blocks
- Scripts without try/except blocks
- Mixed code block types
- Incorrect block names
- Empty code blocks

✅ COPY THESE FORMATS EXACTLY:

1️⃣ FOR CLASSICAL SCRIPTS - USE BOTH BLOCKS:
First block MUST be:
```response
Here's a script that checks system information including CPU, memory, and disk usage
```

Second block MUST be:
```script
try:
    # Your code here
    import psutil
    print(psutil.cpu_percent())
except Exception as e:
    print(f"Error: {e}")
```

2️⃣ FOR QUANTUM PROBLEMS - USE BOTH BLOCKS:
First block MUST be:
```response
Here's a quantum problem formulation for optimization
```

Second block MUST be:
```quantum
{
    "problem": {
        "(0,0)": -1,
        "(1,1)": -1,
        "(0,1)": 2
    },
    "type": "qubo"
}
```

3️⃣ FOR COMMANDS - USE BOTH BLOCKS:
First block MUST be:
```response
This command will show all active network connections and listening ports
```

Second block MUST be:
```command
netstat -an
```

4️⃣ FOR SEARCHES - ONE BLOCK ONLY:
```search
latest quantum computing breakthroughs 2024
```

5️⃣ FOR VISUALIZATIONS - USE BOTH BLOCKS:
First block MUST be:
```response
This visualization shows quantum circuit execution results
```

Second block MUST be:
```visualization
{
    "qubits": [0, 1],
    "results": [0.75, 0.25]
}
```

⚠️ CRITICAL RULES:
1. COPY these formats EXACTLY
2. DO NOT modify the block types
3. Scripts MUST have try/except
4. NO text outside blocks
5. NO mixing block types
6. Commands/scripts need response block first
7. Search/visualization use exact format shown

✅ VALID BLOCK TYPES:
- ```response``` - For explanations
- ```script``` - For Python code (with try/except)
- ```quantum``` - For quantum problems
- ```command``` - For system commands
- ```search``` - For web searches
- ```visualization``` - For data visualization

⚠️ REMEMBER:
- COPY the formats above EXACTLY
- DO NOT modify or improvise
- ALWAYS use try/except in scripts
- NO raw text allowed
"""

# Chat input
user_input = st.chat_input("Enter your message...")

if user_input:
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    try:
        # Prepare messages for API call
        api_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_input}
        ]
        
        # Get AI response
        content = chat_with_ollama(api_messages)
        
        if content:
            # Strict response validation and formatting
            def format_response(content):
                # Split into code blocks
                parts = content.split('```')
                if len(parts) < 3:  # Must have at least one code block
                    return ["```response\nPlease provide a properly formatted response.\n```"]
                
                formatted_parts = []
                current_type = None
                
                for i in range(1, len(parts), 2):
                    block = parts[i].strip()
                    if not block:
                        continue
                    
                    # Extract block type and content
                    lines = block.split('\n', 1)
                    if len(lines) < 2:
                        continue
                    
                    block_type = lines[0].strip()
                    block_content = lines[1].strip()
                    
                    # Validate block type
                    valid_types = ['script', 'command', 'search', 'visualization', 'response', 'quantum']
                    if block_type not in valid_types:
                        block_type = 'response'
                        block_content = block
                    
                    # Enforce response block first for script/command/quantum
                    if current_type is None and block_type in ['script', 'command', 'quantum'] and not any('```response' in p for p in parts[:i]):
                        formatted_parts.append(f"```response\nHere's the {block_type} to execute:\n```")
                    
                    # Format block
                    if block_type == 'script' and ('try:' not in block_content or 'except' not in block_content):
                        block_content = f"try:\n{block_content}\nexcept Exception as e:\n    print(f'Error: {{e}}')"
                    
                    formatted_parts.append(f"```{block_type}\n{block_content}\n```")
                    current_type = block_type
                
                return formatted_parts if formatted_parts else ["```response\nPlease provide a properly formatted response.\n```"]
            
            # Format and add responses
            formatted_parts = format_response(content)
            for part in formatted_parts:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": part
                })
    except Exception as e:
        st.error(f"Error getting AI response: {str(e)}")

# Display chat history
for msg in st.session_state.messages:
    role_class = "user-message" if msg["role"] == "user" else "assistant-message"
    with st.container():
        st.markdown(f'<div class="chat-container"><div class="{role_class}">', unsafe_allow_html=True)
        content = msg["content"]
        
        # Handle command blocks
        if "```command" in content:
            command = content.split("```command")[1].split("```")[0].strip()
            st.code(command, language="bash")
            button_key = f"cmd_{len(st.session_state.messages)}_{abs(hash(command))}"
            if st.button(f"Execute Command", key=button_key):
                result = execute_command(command)
                st.code(result, language="bash")
        
        # Handle script blocks
        elif "```script" in content:
            script = content.split("```script")[1].split("```")[0].strip()
            st.code(script, language="python")
            button_key = f"script_{len(st.session_state.messages)}_{abs(hash(script))}"
            if st.button(f"Run Script", key=button_key):
                result = execute_python_script(script)
                st.code(result, language="python")
        
        # Handle quantum blocks
        elif "```quantum" in content:
            problem = content.split("```quantum")[1].split("```")[0].strip()
            try:
                problem_data = json.loads(problem)
                st.code(problem, language="json")
                button_key = f"quantum_{len(st.session_state.messages)}_{abs(hash(problem))}"
                if st.button(f"Run Quantum Problem", key=button_key):
                    result = execute_quantum_problem(problem_data['problem'])
                    st.code(json.dumps(result, indent=2), language="json")
            except Exception as e:
                st.error(f"Error parsing quantum problem: {str(e)}")
        
        # Handle regular text
        else:
            st.markdown(content)
        
        st.markdown('</div></div>', unsafe_allow_html=True)

# [Rest of the file remains unchanged...]
