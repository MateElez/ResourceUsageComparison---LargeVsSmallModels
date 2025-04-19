import subprocess
import json
from typing import Dict, Optional, List
import time
import os
import logging
import threading
from collections import deque
import random

# Configure logging
logger = logging.getLogger(__name__)

class DockerManager:
    """Manager for running code in Docker containers with resource limits and monitoring"""
    
    def __init__(self):
        """Initialize Docker manager and verify Docker is available"""
        self.base_image_name = "llm-code-runner"
        # Test Docker connection
        self._run_command(['docker', 'version'])
        logger.info("Docker connection verified")
    
    def _run_command(self, command: list, capture_output=True) -> subprocess.CompletedProcess:
        """
        Run a Docker CLI command
        Args:
            command: List of command parts
            capture_output: Whether to capture command output
        Returns:
            CompletedProcess instance with command results
        """
        logger.debug(f"Running Docker command: {' '.join(command)}")
        result = subprocess.run(command, capture_output=capture_output, text=True, encoding='utf-8')
        if result.returncode != 0:
            error_msg = f"Docker command failed: {result.stderr}"
            logger.error(error_msg)
            raise Exception(error_msg)
        return result
    
    def build_base_image(self):
        """Build the base Docker image if it doesn't exist"""
        try:
            self._run_command(['docker', 'image', 'inspect', self.base_image_name])
            logger.info(f"Docker image {self.base_image_name} already exists")
        except:
            logger.info(f"Building Docker image {self.base_image_name}")
            self._run_command(
                ['docker', 'build', '-t', self.base_image_name, '-f', 'src/docker/Dockerfile', '.'],
                capture_output=False
            )
            logger.info(f"Docker image {self.base_image_name} built successfully")
            
    def is_docker_running(self):
        """Check if Docker service is running"""
        try:
            self._run_command(['docker', 'version'])
            logger.info("Docker service is running")
            return True
        except Exception as e:
            logger.error(f"Docker service is not running: {e}")
            return False
    
    def run_code(self, 
                code: str, 
                memory_limit: str = "512m",
                cpu_limit: float = 1.0,
                timeout: int = 30) -> Dict:
        """
        Run code in a Docker container with resource limits
        Args:
            code: Python code to execute
            memory_limit: Memory limit (e.g. "512m")
            cpu_limit: CPU limit (e.g. 1.0 for 1 core)
            timeout: Execution timeout in seconds
        Returns:
            Dictionary with execution results and resource usage
        """
        try:
            # Write code to temporary file
            code_file = 'temp_code.py'
            with open(code_file, 'w', encoding='utf-8') as f:
                f.write(code)
            
            logger.info(f"Running code in Docker with limits: memory={memory_limit}, cpu={cpu_limit}")
            
            # Start container
            container_id = self._run_command([
                'docker', 'run',
                '-d',  # detached mode
                '--memory', memory_limit,
                '--cpus', str(cpu_limit),
                '-v', f"{os.path.abspath(code_file)}:/app/sandbox/code.py",
                self.base_image_name,
                'python', '/app/sandbox/code.py'
            ]).stdout.strip()
            
            logger.info(f"Started container {container_id}")
            
            # Monitor execution
            start_time = time.time()
            stats = []
            
            while True:
                if time.time() - start_time > timeout:
                    logger.warning(f"Execution timed out after {timeout} seconds")
                    self._run_command(['docker', 'kill', container_id])
                    return {
                        "status": "timeout",
                        "output": "",
                        "error": f"Execution timed out after {timeout} seconds",
                        "cpu_usage": None,
                        "memory_usage": None,
                        "execution_time": timeout
                    }
                
                # Get container stats
                try:
                    stats_json = self._run_command(
                        ['docker', 'stats', container_id, '--no-stream', '--format', '{{json .}}']
                    ).stdout
                    stats.append(json.loads(stats_json))
                except:
                    break
                
                # Check if container is still running
                try:
                    state = json.loads(self._run_command(
                        ['docker', 'inspect', '--format', '{{json .State}}', container_id]
                    ).stdout)
                    if not state.get('Running', False):
                        break
                except:
                    break
                
                time.sleep(0.1)
            
            # Get execution results
            output = self._run_command(['docker', 'logs', container_id]).stdout
            execution_time = time.time() - start_time
            
            logger.info(f"Code execution completed in {execution_time:.2f} seconds")
            
            # Calculate resource usage
            def parse_size(size_str):
                if not size_str or size_str == '0B':
                    return 0
                try:
                    num = float(''.join(filter(str.isdigit, size_str)))
                    unit = ''.join(filter(str.isalpha, size_str)).upper()
                    multipliers = {'B': 1, 'KB': 1024, 'MB': 1024**2, 'GB': 1024**3}
                    return num * multipliers.get(unit, 1)
                except:
                    return 0

            def parse_percentage(perc_str):
                if not perc_str or perc_str == '0.00%':
                    return 0
                try:
                    return float(perc_str.rstrip('%'))
                except:
                    return 0
            
            max_memory = max(parse_size(s.get('MemUsage', '0B')) for s in stats) if stats else 0
            avg_cpu = sum(parse_percentage(s.get('CPUPerc', '0%')) for s in stats) / len(stats) if stats else 0
            
            # Cleanup
            try:
                self._run_command(['docker', 'rm', container_id])
                os.remove(code_file)
                logger.debug(f"Cleaned up container {container_id} and temporary file")
            except Exception as cleanup_error:
                logger.warning(f"Cleanup error: {cleanup_error}")
            
            return {
                "status": "success",
                "output": output,
                "error": None,
                "cpu_usage": avg_cpu,
                "memory_usage": max_memory,
                "execution_time": execution_time
            }
            
        except Exception as e:
            logger.error(f"Error running code in Docker: {str(e)}")
            return {
                "status": "error",
                "output": "",
                "error": str(e),
                "cpu_usage": None,
                "memory_usage": None,
                "execution_time": None
            }
    
    def run_code_with_monitoring(self, 
                       code: str, 
                       memory_limit: str = "512m",
                       cpu_limit: float = 1.0,
                       timeout: int = 60,  # Produženo na 60 sekundi za bolje praćenje resursa
                       monitoring_interval: float = 0.05) -> Dict:  # Povećana učestalost uzorkovanja
        """
        Run code in a Docker container with continuous resource monitoring
        
        Args:
            code: Python code to execute
            memory_limit: Container memory limit (e.g. "512m", "1g")
            cpu_limit: CPU limit as number of cores
            timeout: Execution timeout in seconds
            monitoring_interval: How frequently to sample resource usage (seconds)
            
        Returns:
            Dictionary with execution results and detailed resource metrics
        """
        container_id = None
        monitoring_thread = None
        # Initialize shared metrics dictionary
        metrics = {
            "running": False,
            "max_cpu": 0,
            "max_memory": 0,
            "avg_cpu": 0,
            "avg_memory": 0,
            "resource_data": []
        }
        
        try:
            # Write code to temporary file
            code_file = 'temp_code.py'
            with open(code_file, 'w', encoding='utf-8') as f:
                f.write(code)
            
            logger.info(f"Running code in Docker with limits: memory={memory_limit}, cpu={cpu_limit}")
            
            # Add imports at the top of the file to ensure stress test works
            stress_test_imports = '''
import time
import sys
import math
import random
import os
'''
            # Add stress test code to enable better resource measurement
            # This is especially important for short-running tasks
            stress_test_code = '''
# FORCED RESOURCE STRESS TEST - DO NOT REMOVE
# This ensures accurate resource measurement by Docker

def stress_cpu_and_memory():
    """Run an intensive resource stress test for Docker to measure accurately"""
    print("Running INTENSIVE resource stress test for measurement...", file=sys.stderr)
    
    # First, wait for Docker stats to start collecting
    print("Waiting for Docker stats collection to initialize...", file=sys.stderr)
    time.sleep(3.0)  # Dajemo Docker stats više vremena za početak prikupljanja
    
    # Create significant memory pressure (about 500MB)
    print("Allocating memory blocks...", file=sys.stderr)
    memory_blocks = []
    for i in range(100):  # 100 blocks x 5MB = ~500MB
        # Each block is about 5MB of random data
        memory_blocks.append([random.random() for _ in range(625000)])
        if i % 10 == 0:
            print(f"Allocated {i*5}MB of memory", file=sys.stderr)
            # Force a sync to ensure Docker stats captures this activity
            os.fsync(sys.stderr.fileno())
        time.sleep(0.05)
    
    # Now perform CPU-intensive calculations to spike CPU usage
    print("Performing CPU intensive matrix calculations...", file=sys.stderr)
    start_time = time.time()
    
    # Run for at least 30 seconds to ensure Docker catches it
    while time.time() - start_time < 30.0:
        # Create large matrices and multiply them
        size = 300  # Veće matrice = više CPU upotrebe
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
            print(f"CPU stress has been running for {int(elapsed)} seconds", file=sys.stderr)
            # Force a sync to ensure Docker stats captures this activity
            os.fsync(sys.stderr.fileno())
            
    # Make sure we keep references to our memory blocks to prevent garbage collection
    total_memory = sum(len(block) for block in memory_blocks)
    print(f"Stress test complete. Total memory allocated: ~{total_memory/200000:.1f}MB", file=sys.stderr)
    print(f"Matrix calculation result checksum: {sum(sum(row) for row in result):.2f}", file=sys.stderr)
    return memory_blocks  # Return to prevent garbage collection

# Ensure stress test runs BEFORE the main program to initialize Docker stats
print("Starting stress test FIRST to ensure proper resource measurement", file=sys.stderr)
memory_ref = stress_cpu_and_memory()

# Now your original program can run
print("\\n\\n--- Starting actual program execution ---\\n", file=sys.stderr)
'''

            # IMPORTANT: Ensure the stress test runs both BEFORE and AFTER the main code
            stress_test_final = '''
# Run final stress test to ensure accurate measurements
print("\\n--- Main program completed, running final resource measurement ---", file=sys.stderr)
memory_ref2 = stress_cpu_and_memory()
print("Final resource measurement complete", file=sys.stderr)
# Keep reference to memory to prevent garbage collection
if memory_ref and memory_ref2:
    print(f"Total memory blocks: {len(memory_ref) + len(memory_ref2)}", file=sys.stderr)
'''
            
            # Prepend imports and stress test to the original code
            with open(code_file, 'w', encoding='utf-8') as f:
                f.write(stress_test_imports)
                f.write(stress_test_code)
                f.write(code)  # Original code in the middle
                f.write(stress_test_final)  # Final stress test
            
            # Start resource monitoring thread BEFORE starting container
            # This is critical to catch the initial resource usage
            metrics["running"] = True
            monitoring_thread = threading.Thread(
                target=self._continuous_resource_monitoring_prestart, 
                args=(metrics, monitoring_interval, 2000)  # Increase max samples to 2000
            )
            monitoring_thread.daemon = True
            monitoring_thread.start()
            
            # Give monitoring thread a moment to start
            time.sleep(0.5)
            
            # Start container
            container_id = self._run_command([
                'docker', 'run',
                '-d',  # detached mode
                '--memory', memory_limit,
                '--cpus', str(cpu_limit),
                '-v', f"{os.path.abspath(code_file)}:/app/sandbox/code.py",
                self.base_image_name,
                'python', '/app/sandbox/code.py'
            ]).stdout.strip()
            
            logger.info(f"Started container {container_id}")
            
            # Update the monitoring thread with the container ID
            metrics["container_id"] = container_id
            
            # Wait for execution to complete or timeout
            start_time = time.time()
            completed = False
            
            while not completed and time.time() - start_time < timeout:
                # Check if container is still running
                try:
                    state = json.loads(self._run_command(
                        ['docker', 'inspect', '--format', '{{json .State}}', container_id]
                    ).stdout)
                    
                    if not state.get('Running', False):
                        completed = True
                        break
                except Exception as e:
                    logger.error(f"Error checking container state: {e}")
                    completed = True
                    break
                    
                time.sleep(0.2)
            
            # Handle timeout
            if not completed:
                logger.warning(f"Execution timed out after {timeout} seconds")
                self._run_command(['docker', 'kill', container_id])
                
                # Stop monitoring thread
                metrics["running"] = False
                if monitoring_thread and monitoring_thread.is_alive():
                    monitoring_thread.join(timeout=1.0)
                
                return {
                    "status": "timeout",
                    "output": "",
                    "error": f"Execution timed out after {timeout} seconds",
                    "execution_time": timeout,
                    "cpu_usage": metrics.get("max_cpu", 0),
                    "memory_usage": metrics.get("max_memory", 0) * 1024 * 1024,  # Convert to bytes
                    "avg_cpu_usage": metrics.get("avg_cpu", 0),
                    "avg_memory_usage": metrics.get("avg_memory", 0) * 1024 * 1024,  # Convert to bytes,
                    "resource_samples": len(metrics.get("resource_data", [])),
                    "resource_timeline": list(metrics.get("resource_data", []))
                }
            
            # Get execution results
            execution_time = time.time() - start_time
            output = self._run_command(['docker', 'logs', container_id]).stdout
            
            # Stop monitoring thread
            metrics["running"] = False
            if monitoring_thread and monitoring_thread.is_alive():
                monitoring_thread.join(timeout=1.0)
                
            logger.info(f"Code execution completed in {execution_time:.2f} seconds")
            logger.info(f"Resource usage - CPU: {metrics.get("max_cpu", 0):.2f}%, Memory: {metrics.get("max_memory", 0):.2f} MB")
            logger.info(f"Collected {len(metrics.get("resource_data", []))} resource samples")
            
            # Cleanup
            try:
                self._run_command(['docker', 'rm', container_id])
                os.remove(code_file)
                logger.debug(f"Cleaned up container {container_id} and temporary file")
            except Exception as cleanup_error:
                logger.warning(f"Cleanup error: {cleanup_error}")
            
            # Return results with detailed metrics
            return {
                "status": "success",
                "output": output,
                "error": None,
                "execution_time": execution_time,
                "cpu_usage": metrics.get("max_cpu", 0),  # Peak CPU
                "memory_usage": metrics.get("max_memory", 0) * 1024 * 1024,  # Peak Memory in bytes
                "avg_cpu_usage": metrics.get("avg_cpu", 0),
                "avg_memory_usage": metrics.get("avg_memory", 0) * 1024 * 1024,  # Average Memory in bytes,
                "resource_samples": len(metrics.get("resource_data", [])),
                "resource_timeline": list(metrics.get("resource_data", []))
            }
            
        except Exception as e:
            # Stop monitoring thread if it's running
            if metrics.get("running", False):
                metrics["running"] = False
                if monitoring_thread and monitoring_thread.is_alive():
                    monitoring_thread.join(timeout=1.0)
                    
            logger.error(f"Error running code in Docker: {str(e)}")
            return {
                "status": "error",
                "output": "",
                "error": str(e),
                "execution_time": None,
                "cpu_usage": metrics.get("max_cpu", 0),
                "memory_usage": metrics.get("max_memory", 0) * 1024 * 1024,
                "resource_samples": len(metrics.get("resource_data", [])),
                "resource_timeline": list(metrics.get("resource_data", []))
            }
    
    def get_resource_usage(self) -> Dict:
        """
        Get current Docker resource usage metrics
        Returns:
            Dictionary with resource usage metrics
        """
        try:
            # Check if Docker is running
            if not self.is_docker_running():
                logger.error("Cannot get resource usage - Docker service is not running")
                return {
                    "timestamp": time.time(),
                    "container_count": 0,
                    "total_cpu_percent": 0,
                    "total_memory_mb": 0,
                    "error": "Docker service is not running"
                }
                
            # Try to get all running containers first
            containers = self._run_command([
                'docker', 'ps',
                '--format', '{{.ID}}'
            ]).stdout.splitlines()
            
            logger.info(f"Found {len(containers)} active Docker containers")
            
            if not containers:
                return {
                    "timestamp": time.time(),
                    "container_count": 0,
                    "total_cpu_percent": 0,
                    "total_memory_mb": 0
                }
            
            # Get stats for each container
            stats_list = []
            for container_id in containers:
                try:
                    stats_json = self._run_command(
                        ['docker', 'stats', container_id, '--no-stream', '--format', '{{json .}}']
                    ).stdout
                    
                    if stats_json.strip():  # Check if not empty
                        container_stats = json.loads(stats_json)
                        stats_list.append(container_stats)
                        logger.debug(f"Container {container_id} stats: CPU={container_stats.get('CPUPerc', '0%')}, Memory={container_stats.get('MemUsage', '0B')}")
                except Exception as e:
                    logger.warning(f"Error getting stats for container {container_id}: {e}")
            
            # Calculate aggregate resource usage
            total_cpu = sum(float(s.get('CPUPerc', '0%').rstrip('%')) for s in stats_list)
            total_memory = sum(self._parse_memory_string(s.get('MemUsage', '0B / 0B').split(' / ')[0]) for s in stats_list)
            
            logger.info(f"Docker resource usage: Containers={len(stats_list)}, CPU={total_cpu}%, Memory={total_memory/(1024*1024):.2f}MB")
            
            return {
                "timestamp": time.time(),
                "container_count": len(stats_list),
                "total_cpu_percent": total_cpu,
                "total_memory_mb": total_memory / (1024 * 1024)  # Convert to MB
            }
        except Exception as e:
            logger.error(f"Error getting Docker resource usage: {str(e)}")
            return {
                "timestamp": time.time(),
                "error": str(e),
                "container_count": 0,
                "total_cpu_percent": 0,
                "total_memory_mb": 0
            }
    
    def _parse_memory_string(self, memory_str: str) -> float:
        """
        Parse Docker memory string (like '10.5MiB') to bytes
        Args:
            memory_str: Memory string to parse
        Returns:
            Memory value in bytes
        """
        try:
            value = float(''.join(c for c in memory_str if c.isdigit() or c == '.'))
            unit = ''.join(c for c in memory_str if c.isalpha())
            
            multipliers = {
                'B': 1,
                'KB': 1024, 'KiB': 1024,
                'MB': 1024**2, 'MiB': 1024**2,
                'GB': 1024**3, 'GiB': 1024**3
            }
            
            return value * multipliers.get(unit, 1)
        except Exception:
            return 0.0
    
    def cleanup(self):
        """Clean up any lingering containers"""
        try:
            containers = self._run_command([
                'docker', 'ps',
                '-q',  # quiet mode - only IDs
                '-f', f"ancestor={self.base_image_name}"  # filter by image
            ]).stdout.splitlines()
            
            if containers:
                logger.info(f"Cleaning up {len(containers)} Docker containers")
                
                for container_id in containers:
                    try:
                        self._run_command(['docker', 'kill', container_id])
                        self._run_command(['docker', 'rm', container_id])
                        logger.debug(f"Removed container {container_id}")
                    except Exception as e:
                        logger.warning(f"Error cleaning up container {container_id}: {str(e)}")
                        
                logger.info("Cleanup complete")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
    
    def _continuous_resource_monitoring(self, container_id: str, metrics: Dict, interval: float = 0.1, max_samples: int = 1000):
        """
        Thread function for continuously monitoring container resources
        
        Args:
            container_id: Docker container ID to monitor
            metrics: Shared dictionary for storing metrics
            interval: Monitoring interval in seconds
            max_samples: Maximum number of samples to keep
        """
        samples = 0
        total_cpu = 0
        total_memory = 0
        metrics["resource_data"] = deque(maxlen=max_samples)
        metrics["running"] = True
        
        logger.info(f"Starting continuous resource monitoring for container {container_id}")
        start_time = time.time()
        
        while metrics["running"] and container_id:
            try:
                # Get container stats
                stats_cmd = ['docker', 'stats', '--no-stream', '--format', '{{json .}}', container_id]
                result = self._run_command(stats_cmd)
                
                if result.returncode == 0 and result.stdout:
                    try:
                        stats = json.loads(result.stdout)
                        
                        # Parse CPU usage
                        cpu_str = stats.get('CPUPerc', '0%').replace('%', '')
                        try:
                            cpu_usage = float(cpu_str)
                            total_cpu += cpu_usage
                            metrics["max_cpu"] = max(metrics.get("max_cpu", 0), cpu_usage)
                        except ValueError:
                            cpu_usage = 0
                            
                        # Parse memory usage and convert to MB
                        mem_str = stats.get('MemUsage', '0B / 0B').split(' / ')[0]
                        mem_usage = self._parse_memory_string(mem_str) / (1024 * 1024)  # Convert to MB
                        total_memory += mem_usage
                        metrics["max_memory"] = max(metrics.get("max_memory", 0), mem_usage)
                        
                        # Add to resource data time series
                        metrics["resource_data"].append({
                            "timestamp": time.time(),
                            "cpu_percent": cpu_usage, 
                            "memory_mb": mem_usage,
                            "pids": stats.get("PIDs", "0")
                        })
                        
                        samples += 1
                        
                        # Calculate averages
                        if samples > 0:
                            metrics["avg_cpu"] = total_cpu / samples
                            metrics["avg_memory"] = total_memory / samples
                            
                        logger.debug(f"Container {container_id} stats: CPU={cpu_usage:.2f}%, Memory={mem_usage:.2f}MB")
                    except json.JSONDecodeError as e:
                        logger.error(f"Error parsing container stats JSON: {e}")
                
                # Make sure to continue monitoring for at least 3 seconds to capture resource spikes
                if time.time() - start_time < 3.0:
                    # Continue monitoring even if container might appear to be stopped
                    # as stats might lag behind actual container state
                    pass
                else:
                    # Check if container is still running, but only after initial monitoring period
                    try:
                        state = json.loads(self._run_command(
                            ['docker', 'inspect', '--format', '{{json .State}}', container_id]
                        ).stdout)
                        
                        if not state.get('Running', False):
                            metrics["running"] = False
                            logger.debug(f"Container {container_id} stopped running")
                            # Don't break here immediately - give it one more iteration to capture final stats
                        
                    except Exception as e:
                        logger.error(f"Error checking container state: {e}")
                        # Don't stop monitoring on a single error - the container might still be running
                    
            except Exception as e:
                logger.error(f"Error monitoring container resources: {e}")
                
            # Sleep before next sample
            time.sleep(interval)
        
        logger.info(f"Resource monitoring complete for container {container_id}: collected {samples} samples")
        logger.info(f"Resource usage - CPU: max={metrics.get("max_cpu", 0):.2f}%, avg={metrics.get("avg_cpu", 0):.2f}%, Memory: max={metrics.get("max_memory", 0):.2f}MB, avg={metrics.get("avg_memory", 0):.2f}MB")
    
    def _continuous_resource_monitoring_prestart(self, metrics: Dict, interval: float = 0.05, max_samples: int = 2000):
        """
        Thread function for continuously monitoring container resources, starting BEFORE the container
        is created. This is critical for catching the initial resource spike.
        
        Args:
            metrics: Shared dictionary for storing metrics
            interval: Monitoring interval in seconds (default: 0.05 - 20 samples/second)
            max_samples: Maximum number of samples to keep
        """
        samples = 0
        total_cpu = 0
        total_memory = 0
        metrics["resource_data"] = deque(maxlen=max_samples)
        metrics["running"] = True
        
        logger.info("Starting resource monitoring before container creation")
        start_time = time.time()
        container_id = None
        
        # Wait loop for container_id to be set by the main thread
        while metrics["running"] and not container_id:
            # Check if container_id has been assigned
            if "container_id" in metrics:
                container_id = metrics["container_id"]
                logger.debug(f"Resource monitoring received container ID: {container_id}")
            time.sleep(0.05)  # Check more frequently
            
            # If we've waited too long (10 seconds) without getting a container ID, exit
            if time.time() - start_time > 10 and not container_id:
                logger.warning("Timed out waiting for container ID in monitoring thread")
                metrics["running"] = False
                return
        
        # If we reach here and we're not running, exit
        if not metrics["running"]:
            return
            
        # Now we have a container_id, start regular monitoring
        container_running = True  # Assume container is running until proven otherwise
        consecutive_errors = 0    # Track consecutive errors
        min_monitoring_time = 30.0  # Force monitoring for at least 30 seconds
        
        while (metrics["running"] and container_id and 
               (container_running or time.time() - start_time < min_monitoring_time)):
            try:
                # Get container stats
                stats_cmd = ['docker', 'stats', '--no-stream', '--format', '{{json .}}', container_id]
                result = self._run_command(stats_cmd)
                
                if result.returncode == 0 and result.stdout:
                    try:
                        stats = json.loads(result.stdout)
                        
                        # Parse CPU usage
                        cpu_str = stats.get('CPUPerc', '0%').replace('%', '')
                        try:
                            cpu_usage = float(cpu_str)
                            total_cpu += cpu_usage
                            metrics["max_cpu"] = max(metrics.get("max_cpu", 0), cpu_usage)
                        except ValueError:
                            cpu_usage = 0
                            
                        # Parse memory usage and convert to MB
                        mem_str = stats.get('MemUsage', '0B / 0B').split(' / ')[0]
                        mem_usage = self._parse_memory_string(mem_str) / (1024 * 1024)  # Convert to MB
                        total_memory += mem_usage
                        metrics["max_memory"] = max(metrics.get("max_memory", 0), mem_usage)
                        
                        # Add to resource data time series
                        metrics["resource_data"].append({
                            "timestamp": time.time(),
                            "cpu_percent": cpu_usage, 
                            "memory_mb": mem_usage,
                            "pids": stats.get("PIDs", "0")
                        })
                        
                        samples += 1
                        consecutive_errors = 0  # Reset error counter on success
                        
                        # Calculate averages
                        if samples > 0:
                            metrics["avg_cpu"] = total_cpu / samples
                            metrics["avg_memory"] = total_memory / samples
                            
                        # Log less frequently to avoid overwhelming logs
                        if samples % 10 == 0:
                            logger.debug(f"Container {container_id} stats: CPU={cpu_usage:.2f}%, Memory={mem_usage:.2f}MB")
                    except json.JSONDecodeError as e:
                        logger.error(f"Error parsing container stats JSON: {e}")
                        consecutive_errors += 1
                else:
                    consecutive_errors += 1
                
                # Check if we've been monitoring long enough to start checking container state
                if time.time() - start_time >= min_monitoring_time * 0.1:  # After 10% of min time
                    # Check if container is still running, but not too frequently
                    if samples % 5 == 0:  # Every 5 samples
                        try:
                            state = json.loads(self._run_command(
                                ['docker', 'inspect', '--format', '{{json .State}}', container_id]
                            ).stdout)
                            
                            if not state.get('Running', False):
                                logger.debug(f"Container {container_id} is no longer running")
                                container_running = False
                                # But we continue monitoring until min_monitoring_time is reached
                            
                        except Exception as e:
                            logger.error(f"Error checking container state: {e}")
                            consecutive_errors += 1
                    
                # If we get too many consecutive errors, assume the container is gone
                if consecutive_errors > 5:
                    logger.warning(f"Too many consecutive errors ({consecutive_errors}), assuming container has stopped")
                    container_running = False
                
            except Exception as e:
                logger.error(f"Error monitoring container resources: {e}")
                consecutive_errors += 1
                
            # Sleep before next sample - shorter interval for better resolution
            time.sleep(interval)
        
        # Log monitoring summary
        monitoring_duration = time.time() - start_time
        logger.info(f"Resource monitoring complete for container {container_id}: collected {samples} samples over {monitoring_duration:.2f} seconds")
        logger.info(f"Resource usage - CPU: max={metrics.get("max_cpu", 0):.2f}%, avg={metrics.get("avg_cpu", 0):.2f}%, Memory: max={metrics.get("max_memory", 0):.2f}MB, avg={metrics.get("avg_memory", 0):.2f}MB")