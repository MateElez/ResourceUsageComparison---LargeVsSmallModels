from typing import Dict, List, Optional, Any, Union
from abc import ABC, abstractmethod
import time
import logging
from datetime import datetime
import json
import re
from src.api.models.ollama_manager import OllamaManager
from src.docker.docker_manager import DockerManager
from src.core.evaluator import CodeEvaluator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Model(ABC):
    """Abstract base class for all model implementations"""
    
    @abstractmethod
    def generate_solution(self, task_description: str) -> str:
        """Generate a solution for a given task"""
        pass
        
    @abstractmethod
    def get_resource_usage(self) -> Dict[str, Any]:
        """Get resource usage metrics"""
        pass
        
    @abstractmethod
    def execute_task(self, task_description: str, test_cases: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Execute a task and return results with metrics"""
        pass

class LargeModel(Model):
    """Implementation of a large model using Ollama"""
    
    def __init__(self, model_name: str = "deepseek-coder:7b", 
                 base_url: str = "http://localhost:11434",
                 docker_manager: Optional[DockerManager] = None):
        """
        Initialize large model
        Args:
            model_name: Name of the Ollama model
            base_url: Ollama API base URL
            docker_manager: Optional DockerManager instance for resource tracking
        """
        self.ollama = OllamaManager(model_name=model_name, base_url=base_url)
        self.model_name = model_name
        
        # Create a Docker manager for this model instance if not provided
        if docker_manager is None:
            self.docker_manager = DockerManager()
            # Ensure the Docker image is built
            self.docker_manager.build_base_image()
        else:
            self.docker_manager = docker_manager
        
        # Verify model is available or pull it
        if not self.ollama.verify_model_loaded():
            logger.info(f"Model {model_name} not found, pulling...")
            self.ollama.pull_model(show_progress=True)
    
    def generate_solution(self, task_description: str) -> str:
        """
        Generate a solution using the large model
        Args:
            task_description: Description of the coding task
        Returns:
            Generated solution code
        """
        return self.ollama.generate_code(
            prompt=task_description,
            temperature=0.2
        )
    
    def get_resource_usage(self) -> Dict[str, Any]:
        """Get resource usage metrics from Docker if available"""
        # Get resource usage from the Docker manager
        docker_metrics = self.docker_manager.get_resource_usage()
        
        return {
            "model": self.model_name,
            "timestamp": datetime.now().isoformat(),
            "cpu_usage": docker_metrics.get("total_cpu_percent"),
            "memory_usage": docker_metrics.get("total_memory_mb"),
            "container_count": docker_metrics.get("container_count", 1)
        }
    
    def execute_task(self, task_description: str, test_cases: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Execute a task using the large model in a Docker container
        Args:
            task_description: Description of the coding task
            test_cases: Optional list of test cases to validate the solution
        Returns:
            Dictionary with solution, metrics and execution results
        """
        start_time = time.time()
        container_id = None
        
        try:
            # Generate code solution in its own Docker container
            solution = self.generate_solution(task_description)
            
            # Track resource usage
            resource_metrics = self.get_resource_usage()
            
            # Calculate execution time
            elapsed_time = time.time() - start_time
            
            result = {
                "task": task_description,
                "solution": solution,
                "model": self.model_name,
                "execution_time_seconds": elapsed_time,
                "resource_metrics": resource_metrics,
                "timestamp": datetime.now().isoformat()
            }
            
            # If test cases are provided, evaluate the solution
            if test_cases and solution:
                evaluator = CodeEvaluator(timeout=30)
                evaluation = evaluator.evaluate_solution(solution, test_cases)
                result["evaluation"] = evaluation
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing task in Docker container: {str(e)}")
            # Calculate execution time even if there was an error
            elapsed_time = time.time() - start_time
            
            return {
                "task": task_description,
                "solution": None,
                "model": self.model_name,
                "execution_time_seconds": elapsed_time,
                "error": str(e),
                "resource_metrics": {"error": str(e)},
                "timestamp": datetime.now().isoformat()
            }
        finally:
            # Clean up resources
            if container_id:
                try:
                    self.docker_manager.cleanup()
                except Exception as e:
                    logger.warning(f"Error cleaning up container: {str(e)}")

class AdaptiveSmallModel(Model):
    """
    Implementation of a small model strategy that adaptively branches
    based on success or failure of previous attempts
    """
    
    def __init__(self, model_name: str = "tinyllama:1.1b", 
                 base_url: str = "http://localhost:11434",
                 max_branching_depth: int = 3,  # Max depth: 1, 2, 4, 8... instances
                 docker_manager: Optional[DockerManager] = None):
        """
        Initialize adaptive small model strategy
        Args:
            model_name: Name of the Ollama model for small tasks
            base_url: Ollama API base URL
            max_branching_depth: Maximum branching depth (0=1 model, 1=2 models, 2=4 models, etc.)
            docker_manager: Optional DockerManager instance for resource tracking
        """
        self.ollama = OllamaManager(model_name=model_name, base_url=base_url)
        self.model_name = model_name
        self.max_branching_depth = max_branching_depth
        self.evaluator = CodeEvaluator(timeout=30)
        
        # Create a Docker manager for this model instance if not provided
        if docker_manager is None:
            self.docker_manager = DockerManager()
            # Ensure the Docker image is built
            self.docker_manager.build_base_image()
        else:
            self.docker_manager = docker_manager
        
        # Ensure the model is available
        if not self.ollama.verify_model_loaded():
            logger.info(f"Model {model_name} not found, pulling...")
            self.ollama.pull_model(show_progress=True)
    
    def generate_solution(self, task_description: str, test_cases: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Generate a solution using adaptive branching approach
        Args:
            task_description: Description of the coding task
            test_cases: Test cases for evaluating solutions
        Returns:
            Dictionary with best solution and attempt history
        """
        if not test_cases:
            # Create minimal test cases if none provided
            test_cases = self._generate_test_cases(task_description)
            logger.info(f"Generated {len(test_cases)} test cases for evaluation")
        
        # Start with a single model attempt
        depth = 0
        previous_attempts = []
        all_attempts = []
        total_containers = 0
        
        while depth <= self.max_branching_depth:
            # Calculate number of instances for this depth (2^depth)
            num_instances = 2**depth if depth > 0 else 1
            logger.info(f"Trying depth {depth} with {num_instances} model instances")
            
            # Create prompts for each instance
            prompts = self._create_adaptive_prompts(
                task_description, 
                previous_attempts, 
                num_instances
            )
            
            # Track depth start time for resource calculation
            depth_start_time = time.time()
            
            # Create a Docker manager for each instance at this depth
            instance_docker_managers = [DockerManager() for _ in range(num_instances)]
            total_containers += num_instances
            
            # Generate and evaluate solutions
            current_attempts = []
            for i, prompt in enumerate(prompts):
                start_time = time.time()
                logger.info(f"Generating solution for instance {i+1}/{num_instances} at depth {depth}")
                
                # Create a dedicated manager for this instance
                instance_docker_manager = instance_docker_managers[i]
                
                # Create a dedicated ollama manager for this instance
                instance_ollama = OllamaManager(model_name=self.model_name)
                
                # Slightly vary temperature for more diverse solutions
                temperature = 0.2 + (i * 0.05)
                
                try:
                    # Generate solution in this instance's dedicated container
                    logger.info(f"Creating Docker container for instance {i+1} at depth {depth}")
                    solution = instance_ollama.generate_code(prompt=prompt, temperature=temperature)
                    
                    # Evaluate the solution
                    evaluation = self.evaluator.evaluate_solution(solution, test_cases)
                    
                    # Get resource usage from this instance's Docker container
                    resource_metrics = instance_docker_manager.get_resource_usage()
                    logger.info(f"Docker container for instance {i+1} at depth {depth} - resources: CPU={resource_metrics.get('total_cpu_percent', 0)}%, Memory={resource_metrics.get('total_memory_mb', 0):.2f}MB, Containers={resource_metrics.get('container_count', 0)}")
                    
                    # Calculate metrics
                    elapsed_time = time.time() - start_time
                    
                    attempt = {
                        "depth": depth,
                        "instance": i + 1,
                        "solution": solution,
                        "success": evaluation.get("success", False),
                        "test_results": evaluation.get("test_results", []),
                        "coverage": evaluation.get("test_coverage", "0/0"),
                        "execution_time": elapsed_time,
                        "prompt": prompt,
                        "resource_metrics": resource_metrics,
                        "container_id": f"instance_{depth}_{i+1}"  # Unique ID for this container
                    }
                    current_attempts.append(attempt)
                    all_attempts.append(attempt)
                    
                    # If we found a successful solution, clean up and return it immediately
                    if evaluation.get("success", False):
                        logger.info(f"Found successful solution at depth {depth}, instance {i+1}")
                        
                        # Clean up all Docker containers
                        self._cleanup_docker_managers(instance_docker_managers)
                        
                        return {
                            "solution": solution,
                            "success": True,
                            "coverage": evaluation.get("test_coverage", "0/0"),
                            "depth_reached": depth,
                            "total_instances": len(all_attempts),
                            "total_containers": total_containers,
                            "attempts": all_attempts,
                            "resource_metrics": resource_metrics
                        }
                except Exception as e:
                    logger.error(f"Error in instance {i+1} at depth {depth}: {str(e)}")
                    # Add failed attempt with error information
                    current_attempts.append({
                        "depth": depth,
                        "instance": i + 1,
                        "solution": None,
                        "success": False,
                        "error": str(e),
                        "execution_time": time.time() - start_time,
                        "container_id": f"instance_{depth}_{i+1}"
                    })
                finally:
                    # Clean up this instance's Docker container
                    try:
                        instance_docker_manager.cleanup()
                    except Exception as cleanup_error:
                        logger.warning(f"Error cleaning up Docker container for instance {i+1} at depth {depth}: {cleanup_error}")
            
            # Add this depth's attempts to previous attempts for next iteration
            previous_attempts = current_attempts
            
            # Calculate total time at this depth
            depth_time = time.time() - depth_start_time
            logger.info(f"Depth {depth} completed in {depth_time:.2f} seconds with {num_instances} instances")
            
            # Prepare for next depth
            depth += 1
        
        # If we reach here, we couldn't find a completely successful solution
        # Return the solution with the highest test coverage
        best_attempt = max(all_attempts, 
                         key=lambda x: self._calculate_coverage_score(x.get("coverage", "0/0")))
        
        logger.warning(f"No fully successful solution found after {len(all_attempts)} attempts. "
                      f"Returning best attempt with coverage {best_attempt.get('coverage', '0/0')}")
        
        # Get final aggregated resource metrics
        final_resource_metrics = self.docker_manager.get_resource_usage()
        
        return {
            "solution": best_attempt["solution"],
            "success": False,
            "coverage": best_attempt.get("coverage", "0/0"),
            "depth_reached": self.max_branching_depth,
            "total_instances": len(all_attempts),
            "total_containers": total_containers,
            "attempts": all_attempts,
            "resource_metrics": final_resource_metrics
        }
    
    def _cleanup_docker_managers(self, docker_managers: List[DockerManager]):
        """Clean up multiple Docker managers"""
        for i, manager in enumerate(docker_managers):
            try:
                manager.cleanup()
                logger.debug(f"Cleaned up Docker manager {i+1}")
            except Exception as e:
                logger.warning(f"Error cleaning up Docker manager {i+1}: {str(e)}")
    
    def _calculate_coverage_score(self, coverage_str: str) -> float:
        """Calculate a score from coverage string (e.g. "3/5")"""
        try:
            passed, total = map(int, coverage_str.split('/'))
            return passed / total if total > 0 else 0
        except:
            return 0
    
    def _create_adaptive_prompts(self, task_description: str, 
                               previous_attempts: List[Dict[str, Any]], 
                               num_instances: int) -> List[str]:
        """
        Create prompts for instances based on previous attempts
        Args:
            task_description: Original task description
            previous_attempts: Previous solution attempts and their results
            num_instances: Number of instances to create prompts for
        Returns:
            List of prompts for each instance
        """
        # For first attempt (depth 0), just use the basic prompt
        if not previous_attempts:
            system_prompt = """You are a skilled coding assistant. 
            Follow these guidelines:
            1. Write clean, efficient code that solves the given problem
            2. Include appropriate comments
            3. Only provide the actual code, no explanations before or after
            4. Make sure to handle edge cases appropriately
            """
            
            return [f"""
            {system_prompt}
            
            Task: {task_description}
            
            Provide a complete solution in Python. Focus on correctness first, then efficiency.
            """]
        
        prompts = []
        
        # Find the best previous attempt
        best_attempt = max(previous_attempts, 
                          key=lambda x: self._calculate_coverage_score(x.get("coverage", "0/0")))
        
        # Create specific failure information
        failure_info = self._extract_failure_info(previous_attempts)
        
        # Create different prompts for each instance to encourage solution diversity
        for i in range(num_instances):
            # Vary the prompts based on index to ensure diversity
            variation_factor = i / (num_instances - 1) if num_instances > 1 else 0
            
            if variation_factor < 0.25:
                # Focus on fixing specific test failures
                prompt = f"""
                I need to solve this coding problem:
                {task_description}
                
                A previous attempt was:
                '''python
                {best_attempt['solution']}
                '''
                
                However, this solution failed on the following test cases:
                {failure_info}
                
                Please provide a corrected solution that specifically addresses these failures.
                Focus on correctness rather than optimization initially.
                """
            elif variation_factor < 0.5:
                # Focus on edge cases
                prompt = f"""
                I need to solve this coding problem:
                {task_description}
                
                A previous attempt was:
                '''python
                {best_attempt['solution']}
                '''
                
                This solution had issues. Please provide a new solution that carefully handles edge cases.
                Consider invalid inputs, extreme values, and special cases.
                
                The following test failures were observed:
                {failure_info}
                
                Make sure your solution passes all of these cases.
                """
            elif variation_factor < 0.75:
                # Focus on optimization
                prompt = f"""
                I need to solve this coding problem:
                {task_description}
                
                A previous attempt was:
                '''python
                {best_attempt['solution']}
                '''
                
                Please provide a more efficient solution, focusing on algorithmic improvements
                while ensuring it still handles all test cases correctly.
                
                These test cases failed previously:
                {failure_info}
                """
            else:
                # Complete rewrite with a different approach
                prompt = f"""
                I need to solve this coding problem:
                {task_description}
                
                Previous attempts have failed. Please provide a completely new solution 
                using a different approach than this implementation:
                
                '''python
                {best_attempt['solution']}
                '''
                
                The solution needs to address these failed test cases:
                {failure_info}
                
                Make sure your solution covers all possible test cases and is efficient.
                """
            
            prompts.append(prompt)
        
        return prompts
    
    def _extract_failure_info(self, previous_attempts: List[Dict[str, Any]]) -> str:
        """Extract useful information about test failures from previous attempts"""
        # Find the best attempt
        best_attempt = max(previous_attempts, 
                          key=lambda x: self._calculate_coverage_score(x.get("coverage", "0/0")))
        
        # Extract failure information
        failures = []
        for test_result in best_attempt.get("test_results", []):
            if not test_result.get("passed", False):
                failures.append(
                    f"- Input: {test_result.get('input')}, "
                    f"Expected: {test_result.get('expected')}, "
                    f"Actual: {test_result.get('actual')}"
                )
        
        if failures:
            return "\n".join(failures)
        else:
            return "The solution encountered errors, but specific test failure information is not available."
    
    def get_resource_usage(self) -> Dict[str, Any]:
        """Get aggregated resource usage metrics"""
        if self.docker_manager:
            return self.docker_manager.get_resource_usage()
        else:
            return {
                "model": self.model_name,
                "adaptive": True,
                "timestamp": datetime.now().isoformat(),
                "cpu_usage": None,
                "memory_usage": None
            }
    
    def _generate_test_cases(self, task_description: str) -> List[Dict[str, Any]]:
        """Generate minimal test cases if none were provided"""
        # Ask the model to generate test cases
        prompt = f"""
        For the following coding task, generate 3-5 test cases.
        Each test case should include input and expected output.
        Format your response as a JSON array of test cases with "input" and "expected" fields.
        
        Task: {task_description}
        """
        
        response = self.ollama.generate_code(prompt=prompt, temperature=0.2)
        
        try:
            # Extract JSON array from the response
            json_match = re.search(r'\[\s*\{.*\}\s*\]', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                test_cases = json.loads(json_str)
                
                # Validate the format
                valid_test_cases = []
                for tc in test_cases:
                    if "input" in tc and "expected" in tc:
                        valid_test_cases.append(tc)
                
                if valid_test_cases:
                    return valid_test_cases
        except Exception as e:
            logger.error(f"Error generating test cases: {str(e)}")
        
        logger.warning("Could not generate structured test cases, using minimal defaults")
        # Fallback: return simple test cases based on task description
        if "fibonacci" in task_description.lower():
            return [
                {"input": 0, "expected": 0},
                {"input": 1, "expected": 1},
                {"input": 5, "expected": 5},
                {"input": 10, "expected": 55}
            ]
        elif "prime" in task_description.lower():
            return [
                {"input": 2, "expected": True},
                {"input": 4, "expected": False},
                {"input": 17, "expected": True},
                {"input": 25, "expected": False}
            ]
        else:
            # Generic test cases
            return [
                {"input": 0, "expected": None},
                {"input": 1, "expected": None},
                {"input": 10, "expected": None},
                {"input": -1, "expected": None}
            ]
    
    def execute_task(self, task_description: str, test_cases: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Execute a task using the adaptive small model strategy
        Args:
            task_description: Description of the coding task
            test_cases: Optional list of test cases to validate the solution
        Returns:
            Dictionary with solution, metrics and execution results
        """
        start_time = time.time()
        
        # Generate solution with adaptive branching
        logger.info(f"Starting adaptive solution generation for task: {task_description[:50]}...")
        generation_result = self.generate_solution(task_description, test_cases)
        
        # Extract solution and metrics
        solution = generation_result["solution"]
        
        # Final evaluation of solution if test cases provided
        if test_cases:
            evaluation = self.evaluator.evaluate_solution(solution, test_cases)
        else:
            evaluation = {"success": generation_result["success"], "test_coverage": generation_result["coverage"]}
        
        # Track resource usage
        resource_metrics = self.get_resource_usage()
        
        # Calculate total execution time
        elapsed_time = time.time() - start_time
        
        result = {
            "task": task_description,
            "solution": solution,
            "model": f"{self.model_name} (adaptive branching)",
            "execution_time_seconds": elapsed_time,
            "attempts": generation_result["attempts"],
            "depth_reached": generation_result["depth_reached"],
            "total_instances": generation_result["total_instances"],
            "total_containers": generation_result["total_containers"],
            "resource_metrics": resource_metrics,
            "timestamp": datetime.now().isoformat(),
            "evaluation": evaluation
        }
        
        return result
