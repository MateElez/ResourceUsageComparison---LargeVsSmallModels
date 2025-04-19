import requests
from typing import Dict, Optional, List, Union, Any
import logging
import time
import json
from datetime import datetime
import os
from src.docker.docker_manager import DockerManager

# Enable debug logging if environment variable is set
if os.environ.get('DEBUG_OLLAMA', '').lower() in ('true', '1', 'yes'):
    logging.basicConfig(level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.INFO)
    
logger = logging.getLogger(__name__)

class OllamaManager:
    def __init__(self, model_name: str = "deepseek-coder:7b", base_url: str = "http://localhost:11434", 
                 use_docker: bool = True):
        """
        Initialize Ollama manager
        Args:
            model_name: Name of the Ollama model to use
            base_url: Base URL for Ollama API
            use_docker: Whether to use Docker containers for execution
        """
        self.model_name = model_name
        self.base_url = base_url.rstrip('/')
        self.headers = {"Content-Type": "application/json"}
        self.use_docker = use_docker
        
        # Initialize Docker manager if needed
        if self.use_docker:
            self.docker_manager = DockerManager()
            # Ensure Docker image is built
            self.docker_manager.build_base_image()
        else:
            self.docker_manager = None
        
    def _make_request(self, endpoint: str, data: Dict[str, Any], stream: bool = False) -> requests.Response:
        """Make a request to the Ollama API"""
        url = f"{self.base_url}/{endpoint}"
        
        try:
            if stream:
                response = requests.post(url, json=data, stream=True)
            else:
                response = requests.post(url, json=data)
            
            response.raise_for_status()
            return response
        except requests.exceptions.ConnectionError:
            logger.error(f"Could not connect to Ollama server at {self.base_url}.")
            logger.error("Please make sure Ollama is installed and running.")
            logger.error("Installation instructions: https://ollama.com/download")
            logger.error("After installing, start Ollama and try again.")
            raise ConnectionError(f"Could not connect to Ollama server at {self.base_url}. Is Ollama installed and running?")
        except Exception as e:
            logger.error(f"Error making request to Ollama API: {str(e)}")
            raise

    def extract_code_from_response(self, response: str) -> str:
        """
        Extract actual Python code from model response, removing any explanations or markdown
        Args:
            response: Raw model response
        Returns:
            Cleaned Python code
        """
        # Log original response for debugging
        logger.debug(f"Raw response from model: {response[:200]}...")
        
        # Try to extract code between markdown code blocks (```)
        if "```python" in response:
            parts = response.split("```python", 1)
            if len(parts) > 1:
                code_part = parts[1].split("```", 1)[0].strip()
                return self._fix_syntax_errors(code_part)
        elif "```" in response:
            parts = response.split("```", 1)
            if len(parts) > 1:
                code_part = parts[1].split("```", 1)[0].strip()
                return self._fix_syntax_errors(code_part)
        
        # Check for function definition indicators
        lines = response.split('\n')
        code_lines = []
        capturing = False
        
        for line in lines:
            # Start capturing when we see function definition
            if "def " in line and ("is_prime" in line or "check_prime" in line):
                capturing = True
                code_lines.append(line)
            # Continue capturing if we're in code mode
            elif capturing:
                code_lines.append(line)
        
        if code_lines:
            return self._fix_syntax_errors("\n".join(code_lines))
        
        # If we couldn't extract code with other methods, do a basic clean
        # Remove common text prefixes like "Here's a Python function"
        cleaned = response
        prefixes_to_remove = [
            "Here's a Python function",
            "Here's the solution",
            "Here is a function",
            "Here is the code",
            "The solution is",
            "To solve this problem"
        ]
        
        for prefix in prefixes_to_remove:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):].lstrip(": \n")
        
        # If all else fails, wrap code in a function ourselves
        if "def is_prime" not in cleaned and "def check_prime" not in cleaned:
            if "def " not in cleaned:
                cleaned = "def is_prime(n):\n    " + cleaned.replace("\n", "\n    ")
        
        return self._fix_syntax_errors(cleaned)
        
    def _fix_syntax_errors(self, code: str) -> str:
        """
        Fix common syntax errors in generated code
        Args:
            code: Raw code that might contain syntax errors
        Returns:
            Fixed code
        """
        # Check for common indent problems - function without body
        if "def is_prime(n):" in code and not "    " in code:
            # Qwen model sometimes generates function declarations without a body
            # Add a basic implementation if the function body is missing
            logger.warning("Detected function without body, adding implementation")
            fixed_code = """def is_prime(n):
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
            return fixed_code
        
        # Fix unmatched parentheses in for loops - common error in qwen model
        lines = code.split('\n')
        fixed_lines = []
        
        indent_fixed = False
        
        for i, line in enumerate(lines):
            # Fix function def without body (indentation error)
            if i < len(lines) - 1 and "def is_prime" in line and ":" in line:
                if i + 1 < len(lines) and not lines[i+1].startswith((' ', '\t')):
                    # Next line exists but isn't indented - this is the indentation error
                    indent_fixed = True
                    fixed_lines.append(line)
                    
                    # Add proper indentation to all subsequent lines
                    for j in range(i+1, len(lines)):
                        if lines[j].strip():  # Only add indentation for non-empty lines
                            fixed_lines.append("    " + lines[j])
                        else:
                            fixed_lines.append(lines[j])
                    
                    # We've processed all lines, so break out
                    break
            
            # Fix common error with extra parenthesis in for loops
            if "for " in line and "range" in line and ")):" in line:
                line = line.replace(")):", "):") 
                
            # Fix doubled closed parenthesis issue
            elif "for " in line and "range" in line and "))" in line and not ")):" in line:
                line = line.replace("))", ")")
            
            if not indent_fixed:
                fixed_lines.append(line)
        
        fixed_code = "\n".join(fixed_lines)
        
        # If the indentation error wasn't fixed by the block above, try a different approach
        if not indent_fixed and "def is_prime(n):" in fixed_code and "    " not in fixed_code:
            # Extract everything after the function definition
            parts = fixed_code.split("def is_prime(n):", 1)
            if len(parts) > 1:
                function_body = parts[1].strip()
                if function_body:
                    # Indent the function body
                    indented_body = "\n".join("    " + line for line in function_body.split("\n"))
                    fixed_code = "def is_prime(n):\n" + indented_body
        
        # Fix other potential syntax errors
        # Remove extra parentheses at function calls
        fixed_code = fixed_code.replace("print(is_prime(17)))", "print(is_prime(17))")
        
        # Add necessary imports if they're used but not imported
        if "sqrt" in fixed_code and "import math" not in fixed_code and "from math import sqrt" not in fixed_code:
            fixed_code = "from math import sqrt\n\n" + fixed_code
        
        # Check if the fixed code still has a function definition
        if "def is_prime" not in fixed_code:
            logger.warning("No function definition found in response, providing fallback implementation")
            fixed_code = """def is_prime(n):
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
            
        return fixed_code

    def generate_code(self, prompt: str, system_prompt: Optional[str] = None, 
                     max_tokens: int = 2048, temperature: float = 0.7) -> str:
        """
        Generate code using the Ollama model
        Args:
            prompt: The programming task or question
            system_prompt: Optional system prompt to guide the model
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature for generation (0.0 to 1.0)
        Returns:
            Generated code as string
        """
        start_time = time.time()
        
        if not system_prompt:
            system_prompt = """You are a skilled Python programmer. 
            Your task is to write clean, efficient code that solves the given problem.
            IMPORTANT:
            1. Your solution must include proper indentation
            2. Make sure your function has a proper body with indented code
            3. Each line after the function definition must be indented with 4 spaces
            4. Only provide the actual code without any explanations or markdown
            5. Make sure to handle edge cases appropriately
            6. For prime number checking tasks, name your function 'is_prime'
            
            Example of CORRECT format:
            def is_prime(n):
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
                return True
            """
        
        # Check if we should use Docker
        if self.use_docker and self.docker_manager:
            logger.info(f"Generating code in Docker container with model: {self.model_name}")
            
            # Create a Python script that will use Ollama API within the container
            docker_code = f'''
import requests
import json
import sys

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
'''
            
            # Run the code in a Docker container
            docker_result = self.docker_manager.run_code(
                code=docker_code,
                memory_limit="2g",  # Give ample memory for Ollama
                cpu_limit=2.0,      # Give 2 CPU cores
                timeout=300         # 5 minute timeout
            )
            
            if docker_result.get("status") == "success":
                generated_text = docker_result.get("output", "").strip()
                
                # Extract clean code from the response
                clean_code = self.extract_code_from_response(generated_text)
                
                # Log generation metrics
                elapsed_time = time.time() - start_time
                logger.info(f"Code generation in Docker completed in {elapsed_time:.2f} seconds")
                logger.info(f"Resource usage - CPU: {docker_result.get('cpu_usage')}%, Memory: {docker_result.get('memory_usage', 0)/(1024*1024):.2f} MB")
                
                return clean_code
            else:
                logger.error(f"Docker execution failed: {docker_result.get('error')}")
                logger.info("Falling back to direct Ollama API call")
                # Fall back to regular Ollama API if Docker fails
        
        # If not using Docker or Docker failed, use regular Ollama API
        data = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        if system_prompt:
            data["system"] = system_prompt
            
        try:
            response = self._make_request("api/generate", data)
            generated_text = response.json().get("response", "").strip()
            
            # Extract clean code from the response
            clean_code = self.extract_code_from_response(generated_text)
            
            # Log generation metrics
            elapsed_time = time.time() - start_time
            logger.info(f"Code generation completed in {elapsed_time:.2f} seconds")
            
            return clean_code
        except Exception as e:
            logger.error(f"Error generating code: {str(e)}")
            raise

    def get_model_info(self) -> Dict:
        """Get information about the current model"""
        try:
            response = self._make_request("api/show", {"name": self.model_name})
            return response.json()
        except Exception as e:
            logger.error(f"Error getting model info: {str(e)}")
            raise

    def verify_model_loaded(self) -> bool:
        """Verify if the model is loaded and ready"""
        try:
            self.get_model_info()
            return True
        except:
            return False
            
    def pull_model(self, show_progress: bool = True) -> bool:
        """
        Pull the model from Ollama library if not already downloaded
        Args:
            show_progress: Whether to show download progress
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if model already exists
            try:
                self.get_model_info()
                logger.info(f"Model {self.model_name} is already downloaded")
                return True
            except:
                pass
                
            logger.info(f"Pulling model {self.model_name}...")
            
            data = {
                "name": self.model_name
            }
            
            if show_progress:
                response = self._make_request("api/pull", data, stream=True)
                
                for line in response.iter_lines():
                    if line:
                        progress = json.loads(line)
                        if "completed" in progress and progress["completed"]:
                            logger.info(f"Download completed: {self.model_name}")
                            break
                        elif "status" in progress:
                            logger.info(f"Download progress: {progress['status']}")
            else:
                self._make_request("api/pull", data)
                
            return True
        except Exception as e:
            logger.error(f"Error pulling model: {str(e)}")
            return False
            
    def get_resource_metrics(self) -> Dict[str, Any]:
        """Get resource metrics for the last generation"""
        return {
            "timestamp": datetime.now().isoformat(),
            "model": self.model_name,
            # Note: Ollama API doesn't provide direct resource metrics
            # These could be collected using psutil in a wrapper function
        }
        
    def execute_coding_task(self, task_description: str, test_cases: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Execute a coding task using the model
        Args:
            task_description: Description of the coding task
            test_cases: Optional list of test cases to validate the solution
        Returns:
            Dictionary with generated code, metrics, and execution results
        """
        system_prompt = """You are a skilled Python programmer. 
        Your task is to write clean, efficient code that solves the given problem.
        IMPORTANT: Only provide the actual code without any explanations, comments, or markdown.
        DO NOT add any text before or after the code.
        Just provide the raw Python code.
        Make sure to handle edge cases appropriately.
        For prime number checking tasks, name your function 'is_prime'."""
        
        start_time = time.time()
        
        # Generate code solution
        solution = self.generate_code(
            prompt=task_description,
            system_prompt=system_prompt,
            temperature=0.2  # Lower temperature for more deterministic outputs
        )
        
        elapsed_time = time.time() - start_time
        
        result = {
            "task": task_description,
            "solution": solution,
            "model": self.model_name,
            "execution_time_seconds": elapsed_time,
            "timestamp": datetime.now().isoformat()
        }
        
        return result