import subprocess
import json
from typing import Dict, Optional
import time
import os
import logging

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
        result = subprocess.run(command, capture_output=capture_output, text=True)
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
            with open(code_file, 'w') as f:
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