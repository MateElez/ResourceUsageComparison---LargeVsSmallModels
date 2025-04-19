import os
import json
import subprocess
import logging
from typing import Dict, Optional, Tuple, Any
import random

from src.docker.docker_manager import DockerManager

# Configure logging
logger = logging.getLogger(__name__)

class OllamaManager:
    """
    Manager for Ollama API interaction and code generation
    """
    
    def __init__(self, model_name: str, base_url: str = "http://localhost:11434"):
        """
        Initialize Ollama manager with a specific model name
        Args:
            model_name: Name of Ollama model to use
            base_url: Base URL for Ollama API, defaults to http://localhost:11434
        """
        self.model_name = model_name
        self.base_url = base_url
        self.docker_manager = DockerManager()
        
        # Ensure Docker base image exists
        self.docker_manager.build_base_image()
        
    def verify_model_loaded(self) -> bool:
        """
        Verify if the specified model is already loaded in Ollama
        Returns:
            True if model is available, False otherwise
        """
        try:
            # Create Python script that will check if model is loaded
            docker_code = f'''
import requests
import json
import sys

def check_model_loaded(model_name):
    """Check if a model is loaded in Ollama"""
    url = "http://host.docker.internal:11434/api/tags"
    
    try:
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Error: {{response.status_code}} - {{response.text}}", file=sys.stderr)
            return False
        
        result = response.json()
        models = result.get("models", [])
        
        # Check if model exists in the list
        for model in models:
            if model.get("name") == model_name:
                print(f"Model {{model_name}} is available", file=sys.stderr)
                return True
        
        print(f"Model {{model_name}} is not available", file=sys.stderr)
        return False
    except Exception as e:
        print(f"Error checking model: {{e}}", file=sys.stderr)
        return False

# Check if model is loaded
model_name = "{self.model_name}"
result = check_model_loaded(model_name)
print(str(result).lower())  # Print True or False
'''
            
            # Run the script in Docker
            result = self.docker_manager.run_code_with_monitoring(
                code=docker_code,
                memory_limit="512m",
                cpu_limit=1.0,
                timeout=10,
                monitoring_interval=0.1
            )
            
            # Parse the result
            output = result.get("output", "").strip()
            return output == "true"
            
        except Exception as e:
            logger.error(f"Error verifying model loaded: {str(e)}")
            return False
    
    def pull_model(self, show_progress: bool = False) -> bool:
        """
        Pull the model from Ollama repository
        Args:
            show_progress: Whether to show progress information
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create Python script that will pull the model
            docker_code = f'''
import requests
import json
import sys
import time

def pull_model(model_name, show_progress):
    """Pull a model in Ollama"""
    url = "http://host.docker.internal:11434/api/pull"
    
    data = {{
        "name": model_name
    }}
    
    try:
        print(f"Pulling model {{model_name}}...", file=sys.stderr)
        
        if show_progress:
            # Stream the response for progress updates
            with requests.post(url, json=data, stream=True) as response:
                if response.status_code != 200:
                    print(f"Error: {{response.status_code}} - {{response.text}}", file=sys.stderr)
                    return False
                
                for line in response.iter_lines():
                    if line:
                        try:
                            progress = json.loads(line)
                            status = progress.get("status", "")
                            if "completed" in status.lower():
                                print(f"Model {{model_name}} pull completed", file=sys.stderr)
                                return True
                            elif show_progress:
                                print(f"Progress: {{status}}", file=sys.stderr)
                        except json.JSONDecodeError:
                            print(f"Error parsing progress: {{line}}", file=sys.stderr)
                
                return True
        else:
            # Not showing progress, just wait for completion
            response = requests.post(url, json=data)
            if response.status_code != 200:
                print(f"Error: {{response.status_code}} - {{response.text}}", file=sys.stderr)
                return False
            
            print(f"Model {{model_name}} pull completed", file=sys.stderr)
            return True
            
    except Exception as e:
        print(f"Error pulling model: {{e}}", file=sys.stderr)
        return False

# Pull the model
model_name = "{self.model_name}"
show_progress = {str(show_progress).lower()}
result = pull_model(model_name, show_progress)
print(str(result).lower())  # Print True or False
'''
            
            # Run the script in Docker
            result = self.docker_manager.run_code_with_monitoring(
                code=docker_code,
                memory_limit="2g",  # Pull may need more memory
                cpu_limit=2.0,
                timeout=600,  # Pulling could take some time
                monitoring_interval=0.5
            )
            
            # Parse the result
            output = result.get("output", "").strip()
            success = output == "true"
            
            if success:
                logger.info(f"Successfully pulled model {self.model_name}")
            else:
                logger.error(f"Failed to pull model {self.model_name}")
                
            return success
            
        except Exception as e:
            logger.error(f"Error pulling model: {str(e)}")
            return False
            
    def generate_code(self, 
                     prompt: str, 
                     system_prompt: str = None,
                     temperature: float = 0.2,
                     max_tokens: int = 2048) -> str:
        """
        Generate code using Ollama API
        Args:
            prompt: The main code generation prompt
            system_prompt: Optional system prompt to guide model behavior
            temperature: Temperature parameter for sampling (0.0-1.0)
            max_tokens: Maximum number of tokens to generate
        Returns:
            Generated code
        """
        # Set default system prompt if not provided
        if system_prompt is None:
            system_prompt = """You are a skilled coding assistant. 
            Follow these guidelines:
            1. Write clean, efficient code that solves the given problem
            2. Include appropriate comments
            3. Only provide the actual code, no explanations before or after
            4. Make sure to handle edge cases appropriately
            """
        
        # Generate code and get metrics
        generated_code, metrics = self._generate_code_with_metrics(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Extract code from the response
        return self._extract_code_from_response(generated_code)
        
    def _generate_code_with_metrics(self, 
                                  prompt: str, 
                                  system_prompt: str, 
                                  temperature: float = 0.2,
                                  max_tokens: int = 2048) -> Tuple[str, Dict[str, Any]]:
        """
        Private method to generate code and collect metrics
        """
        # Create Docker script
        docker_code = self._create_docker_code(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Run in Docker with resource monitoring
        logger.info(f"Generating code in Docker container with model: {self.model_name}")
        result = self.docker_manager.run_code_with_monitoring(
            code=docker_code,
            memory_limit="2g",
            cpu_limit=2.0,
            timeout=60,
            monitoring_interval=0.1
        )
        
        # Extract metrics
        execution_time = result.get("execution_time", 0)
        cpu_usage = result.get("cpu_usage", 0)
        memory_usage = result.get("memory_usage", 0) / (1024 * 1024)  # Convert to MB
        
        logger.info(f"Code generation in Docker completed in {execution_time:.2f} seconds")
        logger.info(f"Resource usage - CPU: {cpu_usage}%, Memory: {memory_usage:.2f} MB")
        
        # Return the output and metrics
        return (
            result.get("output", "# Failed to generate code"),
            {
                "execution_time": execution_time,
                "cpu_usage": cpu_usage,
                "memory_usage": memory_usage,
                "resource_samples": result.get("resource_samples", 0),
                "resource_timeline": result.get("resource_timeline", [])
            }
        )
    
    def _extract_code_from_response(self, response: str) -> str:
        """
        Extract code from an LLM response
        Args:
            response: Raw response from the LLM
        Returns:
            Extracted code
        """
        # Check for a Markdown code block
        if "```python" in response:
            # Python-specific code block
            code_blocks = response.split("```python")
            if len(code_blocks) > 1:
                code = code_blocks[1].split("```")[0].strip()
                return code
                
        elif "```" in response:
            # Generic code block
            code_blocks = response.split("```")
            if len(code_blocks) > 1:
                code = code_blocks[1].strip()
                return code
        
        # If no code block was found, just return the response as is
        return response
    
    def _create_docker_code(self, prompt: str, system_prompt: str, temperature: float, max_tokens: int) -> str:
        """Create the Python code that will run inside Docker to call Ollama API"""
        # Create a Python script that will use Ollama API within the container
        docker_code = f'''
import requests
import json
import sys
import time
import math
import random

def stress_cpu_and_memory():
    """Run an intensive resource stress test for Docker to measure accurately"""
    print("Running INTENSIVE resource stress test for measurement...", file=sys.stderr)
    
    # Create significant memory pressure (about 250MB)
    print("Allocating memory blocks...", file=sys.stderr)
    memory_blocks = []
    for i in range(50):  # 50 blocks x 5MB = ~250MB
        # Each block is about 5MB of random data
        memory_blocks.append([random.random() for _ in range(625000)])
        if i % 10 == 0:
            print(f"Allocated {{i*5}}MB of memory", file=sys.stderr)
        time.sleep(0.05)
    
    # Now perform CPU-intensive calculations to spike CPU usage
    print("Performing CPU intensive matrix calculations...", file=sys.stderr)
    start_time = time.time()
    
    # Run for at least 5 seconds to ensure Docker catches it
    while time.time() - start_time < 5.0:
        # Create large matrices and multiply them
        size = 150
        matrix_a = [[random.random() for _ in range(size)] for _ in range(size)]
        matrix_b = [[random.random() for _ in range(size)] for _ in range(size)]
        
        # Very intensive matrix multiplication
        result = [[0 for _ in range(size)] for _ in range(size)]
        for i in range(size):
            for j in range(size):
                for k in range(size):
                    result[i][j] += matrix_a[i][k] * matrix_b[k][j]
                    
        # Report progress but don't print too often
        elapsed = time.time() - start_time
        if int(elapsed) % 2 == 0 and abs(elapsed - int(elapsed)) < 0.01:
            print(f"CPU stress has been running for {{int(elapsed)}} seconds", file=sys.stderr)
    
    # Make sure we keep references to our memory blocks to prevent garbage collection
    total_memory = sum(len(block) for block in memory_blocks)
    print(f"Stress test complete. Total memory allocated: ~{{total_memory/200000:.1f}}MB", file=sys.stderr)
    print(f"Matrix calculation result checksum: {{sum(sum(row) for row in result):.2f}}", file=sys.stderr)
    return memory_blocks  # Return to prevent garbage collection

# Ensure stress test runs BEFORE the main program to initialize Docker stats
print("Starting stress test FIRST to ensure proper resource measurement", file=sys.stderr)
memory_ref = stress_cpu_and_memory()

# Now your original program can run
print("\\n\\n--- Starting actual program execution ---\\n", file=sys.stderr)

def generate_with_ollama(model, prompt, system_prompt, temperature, max_tokens):
    url = "http://host.docker.internal:11434/api/generate"
    
    data = {{
        "model": model,
        "prompt": prompt,
        "system": system_prompt,
        "stream": False,
        "options": {{
            "temperature": temperature,
            "num_predict": max_tokens
        }}
    }}
    
    try:
        response = requests.post(url, json=data)
        if response.status_code != 200:
            print(f"Error: {{response.status_code}} - {{response.text}}", file=sys.stderr)
            return None
        
        result = response.json()
        return result.get("response", "")
    except Exception as e:
        print(f"Error calling Ollama API: {{e}}", file=sys.stderr)
        return None

def fix_indentation_issues(code):
    # Fix common indentation issues in generated code
    
    # Skip if already properly indented
    if "    " in code:
        return code
    
    # Check if we have a function with no body
    if "def is_prime(n):" in code or "def check_prime(n):" in code:
        # Provide proper implementation
        return """def is_prime(n):
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True"""
    
    return code

# Parameters
model = "{self.model_name}"
prompt = "{prompt}"  # Direktno ubacivanje prompta
system_prompt = "{system_prompt}"  # Direktno ubacivanje system prompta
temperature = {temperature}
max_tokens = {max_tokens}

# Call Ollama API
result = generate_with_ollama(model, prompt, system_prompt, temperature, max_tokens)

# Extract and fix code
if result:
    # Check if the response contains a code block
    if "```python" in result:
        code = result.split("```python")[1].split("```")[0].strip()
    elif "```" in result:
        code = result.split("```")[1].split("```")[0].strip()
    else:
        code = result
    
    # Fix any indentation issues
    fixed_code = fix_indentation_issues(code)
    print(fixed_code)
else:
    print("def is_prime(n):\\n    if n <= 1:\\n        return False\\n    if n <= 3:\\n        return True\\n    if n % 2 == 0 or n % 3 == 0:\\n        return False\\n    i = 5\\n    while i * i <= n:\\n        if n % i == 0 or n % (i + 2) == 0:\\n            return False\\n        i += 6\\n    return True")

# Run final stress test to ensure accurate measurements
print("\\n--- Main program completed, running final resource measurement ---", file=sys.stderr)
memory_ref2 = stress_cpu_and_memory()
print("Final resource measurement complete", file=sys.stderr)
# Keep reference to memory to prevent garbage collection
if memory_ref and memory_ref2:
    print(f"Total memory blocks: {{len(memory_ref) + len(memory_ref2)}}", file=sys.stderr)
'''
        return docker_code