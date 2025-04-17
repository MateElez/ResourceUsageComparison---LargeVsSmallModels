import requests
from typing import Dict, Optional, List, Union, Any
import logging
import time
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OllamaManager:
    def __init__(self, model_name: str = "deepseek-coder:7b", base_url: str = "http://localhost:11434"):
        """
        Initialize Ollama manager
        Args:
            model_name: Name of the Ollama model to use
            base_url: Base URL for Ollama API
        """
        self.model_name = model_name
        self.base_url = base_url.rstrip('/')
        self.headers = {"Content-Type": "application/json"}
        
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
            
            # Log generation metrics
            elapsed_time = time.time() - start_time
            logger.info(f"Code generation completed in {elapsed_time:.2f} seconds")
            
            return generated_text
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
        system_prompt = """You are a skilled coding assistant. 
        Follow these guidelines:
        1. Write clean, efficient code that solves the given problem
        2. Include appropriate comments
        3. Only provide the actual code, no explanations before or after
        4. Make sure to handle edge cases appropriately
        """
        
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