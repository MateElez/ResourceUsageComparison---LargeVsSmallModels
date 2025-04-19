import logging
import time
import json
from src.core.models import LargeModel, SmallModel
from src.docker.docker_manager import DockerManager
from src.database.db import DatabaseConnection
import os
import psutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def monitor_resources():
    """Monitor system resources during execution"""
    process = psutil.Process(os.getpid())
    return {
        "cpu_percent": psutil.cpu_percent(),
        "memory_usage_mb": process.memory_info().rss / (1024 * 1024),
        "available_memory_mb": psutil.virtual_memory().available / (1024 * 1024)
    }

def test_large_model():
    """Test the large model (deepseek-coder) with a coding task"""
    logger.info("Starting large model test with deepseek-coder:7b")
    
    # Example coding task
    task = """
    Write a Python function to check if a number is prime. 
    The function should efficiently handle larger numbers 
    and include proper validation for input.
    Return True if the number is prime, False otherwise.
    """

    # Initialize model
    model = LargeModel(
        model_name="deepseek-coder:7b",
        base_url="http://localhost:11434"
    )
    
    # Execute task and measure performance
    start_time = time.time()
    resources_before = monitor_resources()
    
    logger.info("Executing task with large model...")
    result = model.execute_task(task)
    
    resources_after = monitor_resources()
    total_time = time.time() - start_time
    
    # Log performance metrics
    logger.info(f"Large model execution completed in {total_time:.2f} seconds")
    
    # Calculate resource usage
    resource_delta = {
        "cpu_usage_percent": resources_after["cpu_percent"] - resources_before["cpu_percent"],
        "memory_usage_mb": resources_after["memory_usage_mb"] - resources_before["memory_usage_mb"]
    }
    
    # Update result with additional metrics
    result["system_resource_usage"] = resource_delta
    
    # Save result to database
    try:
        db = DatabaseConnection()
        db.connect()
        
        # Create document for MongoDB
        document = {
            "task_description": task,
            "model_type": "large",
            "model_name": "deepseek-coder:7b",
            "solution": result["solution"],
            "execution_time_seconds": result["execution_time_seconds"],
            "resource_metrics": result["resource_metrics"] if "resource_metrics" in result else {},
            "system_resource_usage": resource_delta,
            "timestamp": result["timestamp"]
        }
        
        # Insert into database
        result_id = db.insert_result(document)
        logger.info(f"Results saved to database with ID: {result_id}")
        
        # Disconnect from database
        db.disconnect()
    except Exception as e:
        logger.error(f"Error saving to database: {e}")
    
    # Print the solution
    print("\n---- Large Model Solution ----")
    print(result["solution"])
    print("\n---- Performance Metrics ----")
    print(f"Execution time: {result['execution_time_seconds']:.2f} seconds")
    print(f"CPU usage: {resource_delta['cpu_usage_percent']:.2f}%")
    print(f"Memory usage: {resource_delta['memory_usage_mb']:.2f} MB")
    
    return result

def test_small_model():
    """Test the small model approach with a coding task"""
    logger.info("Starting small model approach test")
    
    # Same task as large model for comparison
    task = """
    Write a Python function to check if a number is prime. 
    The function should efficiently handle larger numbers 
    and include proper validation for input.
    Return True if the number is prime, False otherwise.
    """

    # Initialize small model approach
    model = SmallModel(
        model_name="tinyllama:1.1b", # Using a much smaller model
        base_url="http://localhost:11434",
        max_branches=4  # Break down into 4 sub-tasks
    )
    
    # Execute task and measure performance
    start_time = time.time()
    resources_before = monitor_resources()
    
    logger.info("Executing task with small model approach...")
    result = model.execute_task(task)
    
    resources_after = monitor_resources()
    total_time = time.time() - start_time
    
    # Log performance metrics
    logger.info(f"Small model execution completed in {total_time:.2f} seconds")
    
    # Calculate resource usage
    resource_delta = {
        "cpu_usage_percent": resources_after["cpu_percent"] - resources_before["cpu_percent"],
        "memory_usage_mb": resources_after["memory_usage_mb"] - resources_before["memory_usage_mb"]
    }
    
    # Update result with additional metrics
    result["system_resource_usage"] = resource_delta
    
    # Save result to database
    try:
        db = DatabaseConnection()
        db.connect()
        
        # Create document for MongoDB
        document = {
            "task_description": task,
            "model_type": "small_branched",
            "model_name": "tinyllama:1.1b",
            "branches": 4,
            "solution": result["solution"],
            "execution_time_seconds": result["execution_time_seconds"],
            "sub_tasks": result["sub_tasks"] if "sub_tasks" in result else [],
            "resource_metrics": result["resource_metrics"] if "resource_metrics" in result else {},
            "system_resource_usage": resource_delta,
            "timestamp": result["timestamp"]
        }
        
        # Insert into database
        result_id = db.insert_result(document)
        logger.info(f"Results saved to database with ID: {result_id}")
        
        # Disconnect from database
        db.disconnect()
    except Exception as e:
        logger.error(f"Error saving to database: {e}")
    
    # Print the solution
    print("\n---- Small Model Solution ----")
    print(result["solution"])
    print("\n---- Performance Metrics ----")
    print(f"Execution time: {result['execution_time_seconds']:.2f} seconds")
    print(f"CPU usage: {resource_delta['cpu_usage_percent']:.2f}%")
    print(f"Memory usage: {resource_delta['memory_usage_mb']:.2f} MB")
    
    # Print sub-task breakdown
    print("\n---- Sub-Task Breakdown ----")
    for i, sub_task in enumerate(result["sub_tasks"]):
        print(f"Sub-task {i+1}: {sub_task['sub_task'][:50]}...")
        print(f"Execution time: {sub_task['execution_time']:.2f} seconds")
    
    return result

if __name__ == "__main__":
    print("=== Testing Models for Resource Comparison ===")
    print("1. Testing Large Model (deepseek-coder:7b)")
    large_result = test_large_model()
    
    print("\n2. Testing Small Model Approach (branches)")
    small_result = test_small_model()
    
    # Compare results
    print("\n=== Comparison ===")
    large_time = large_result["execution_time_seconds"]
    small_time = small_result["execution_time_seconds"]
    time_diff = large_time - small_time
    time_percent = (time_diff / large_time) * 100 if large_time > 0 else 0
    
    print(f"Time difference: {abs(time_diff):.2f} seconds ({abs(time_percent):.2f}% {'faster' if time_diff > 0 else 'slower'} with {'small' if time_diff > 0 else 'large'} model approach)")
    
    # Compare resource usage
    large_memory = large_result["system_resource_usage"]["memory_usage_mb"]
    small_memory = small_result["system_resource_usage"]["memory_usage_mb"]
    memory_diff = large_memory - small_memory
    memory_percent = (memory_diff / large_memory) * 100 if large_memory > 0 else 0
    
    print(f"Memory difference: {abs(memory_diff):.2f} MB ({abs(memory_percent):.2f}% {'less' if memory_diff > 0 else 'more'} with {'small' if memory_diff > 0 else 'large'} model approach)")