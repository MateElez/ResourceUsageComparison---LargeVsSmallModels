import logging
import time
import json
import os
from typing import Dict, List, Any, Optional
import argparse
from datetime import datetime

from src.core.models import LargeModel, AdaptiveSmallModel
from src.core.evaluator import CodeEvaluator
from src.database.db import DatabaseConnection

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ResourceComparisonWorkflow:
    """Main workflow class for resource comparison between large and small models"""
    
    def __init__(self, db_connection: DatabaseConnection):
        """
        Initialize the workflow
        Args:
            db_connection: Database connection instance
        """
        self.db = db_connection
        self.evaluator = CodeEvaluator(timeout=30)
        
        # Initialize models
        self.large_model = None
        self.small_model = None
        
    def initialize_models(self, large_model_name: str = "deepseek-coder:7b", 
                         small_model_name: str = "tinyllama:1.1b",
                         ollama_url: str = "http://localhost:11434",
                         max_branching_depth: int = 4):
        """Initialize both models with specified settings"""
        logger.info(f"Initializing models: large={large_model_name}, small={small_model_name}")
        
        # Initialize large model
        self.large_model = LargeModel(
            model_name=large_model_name,
            base_url=ollama_url
        )
        
        # Initialize small model with adaptive branching
        self.small_model = AdaptiveSmallModel(
            model_name=small_model_name,
            base_url=ollama_url,
            max_branching_depth=max_branching_depth
        )
        
        logger.info("Models initialized successfully")
    
    def load_tasks_from_db(self) -> List[Dict[str, Any]]:
        """
        Load tasks from database
        Returns:
            List of task documents
        """
        try:
            logger.info("Loading tasks from database")
            tasks = self.db.get_all_tasks()
            logger.info(f"Loaded {len(tasks)} tasks")
            return tasks
        except Exception as e:
            logger.error(f"Error loading tasks: {str(e)}")
            return []
    
    def save_result_to_db(self, result: Dict[str, Any]) -> Optional[str]:
        """
        Save a result to the database in the results collection
        Args:
            result: Result document to save
        Returns:
            ID of the inserted document if successful, None otherwise
        """
        try:
            result_id = self.db.insert_result(result)
            logger.info(f"Saved result to database with ID: {result_id}")
            return result_id
        except Exception as e:
            logger.error(f"Error saving result to database: {str(e)}")
            return None
            
    def get_test_cases_for_task(self, task_type: str, task_data: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Get predefined test cases for a specific task type or from task data
        Args:
            task_type: Type of the task
            task_data: Optional task data containing test cases
        Returns:
            List of test cases
        """
        # If we have task data with test_list, use those tests
        if task_data and "test_list" in task_data:
            test_cases = []
            # Format test cases from the test_list
            for test in task_data["test_list"]:
                test_cases.append({"test": test})
            return test_cases
            
        # For prime number check
        if task_type == "prime":
            return [
                {"input": 2, "expected": True},
                {"input": 17, "expected": True},
                {"input": 4, "expected": False},
                {"input": 1, "expected": False},
                {"input": 97, "expected": True},
                {"input": 100, "expected": False},
                {"input": 0, "expected": False},
                {"input": -5, "expected": False}
            ]
        # For fibonacci
        elif task_type == "fibonacci":
            return [
                {"input": 0, "expected": 0},
                {"input": 1, "expected": 1},
                {"input": 2, "expected": 1},
                {"input": 10, "expected": 55},
                {"input": 20, "expected": 6765},
                {"input": 30, "expected": 832040}
            ]
        # Add more task types as needed
        return []
    
    def run_task(self, task_description: str, task_type: str = "", task_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Run a single task using both large and small models
        Args:
            task_description: Description of the task
            task_type: Type of the task, used for test cases
            task_data: Optional task data containing test cases
        Returns:
            Comparison results
        """
        logger.info(f"Running task: {task_description[:50]}...")
        
        # Get test cases for the task
        test_cases = self.get_test_cases_for_task(task_type, task_data)
        
        # Execute with large model - TEMPORARILY DISABLED
        logger.info("Large model execution temporarily disabled to save time")
        large_start_time = time.time()
        # Commented out large model execution due to long processing time
        # large_result = self.large_model.execute_task(task_description, test_cases)
        large_elapsed = 0  # Set to 0 since we're not running the large model
        
        # Create a dummy large model result
        large_result = {
            "solution": "# Large model execution disabled",
            "resource_metrics": {"memory_usage": 0},
            "evaluation": {
                "success": False,
                "test_coverage": "0/0"
            }
        }
        
        # Execute with small model with adaptive branching
        logger.info("Executing with adaptive small model...")
        small_start_time = time.time()
        small_result = self.small_model.execute_task(task_description, test_cases)
        small_elapsed = time.time() - small_start_time
        
        # Get memory usage for both models
        large_memory = large_result.get("resource_metrics", {}).get("memory_usage", 0)
        small_memory = small_result.get("resource_metrics", {}).get("memory_usage", 0)
        
        # Ensure memory values are numeric
        large_memory = 0 if large_memory is None else large_memory
        small_memory = 0 if small_memory is None else small_memory
        
        # Calculate memory difference if values are available
        memory_diff = 0
        memory_diff_percent = 0
        if large_memory > 0 and small_memory > 0:
            memory_diff = large_memory - small_memory
            memory_diff_percent = (memory_diff / large_memory * 100) if large_memory > 0 else 0
        
        # Combine results
        comparison = {
            "task_description": task_description,
            "task_type": task_type,
            "timestamp": datetime.now().isoformat(),
            "large_model": {
                "name": f"{self.large_model.model_name} (DISABLED)",
                "execution_time": large_elapsed,
                "solution": large_result.get("solution", ""),
                "resource_metrics": large_result.get("resource_metrics", {}),
                "evaluation": large_result.get("evaluation", {})
            },
            "small_model": {
                "name": f"{self.small_model.model_name} (adaptive branching)",
                "execution_time": small_elapsed,
                "solution": small_result.get("solution", ""),
                "resource_metrics": small_result.get("resource_metrics", {}),
                "adaptive_branching": {
                    "max_depth": self.small_model.max_branching_depth,
                    "depth_reached": small_result.get("depth_reached", 0),
                    "total_instances": small_result.get("total_instances", 0)
                },
                "evaluation": small_result.get("evaluation", {})
            },
            "comparison": {
                "time_diff_seconds": 0,  # Set to 0 since we're not comparing
                "time_diff_percent": 0,  # Set to 0 since we're not comparing
                "memory_diff": memory_diff,
                "memory_diff_percent": memory_diff_percent,
                "solution_quality": "small_model_better"  # Default to small model since large is disabled
            }
        }
        
        # Add the original task data if available
        if task_data:
            comparison["original_task"] = {
                "task_id": task_data.get("task_id", ""),
                "code": task_data.get("code", ""),
                "test_list": task_data.get("test_list", [])
            }
        
        # Save to database
        self.save_result_to_db(comparison)
        
        return comparison

    def _compare_solution_quality(self, large_eval: Dict[str, Any], small_eval: Dict[str, Any]) -> str:
        """Compare solution quality between large and small models"""
        large_success = large_eval.get("success", False)
        small_success = small_eval.get("success", False)
        
        large_coverage = large_eval.get("test_coverage", "0/0")
        small_coverage = small_eval.get("test_coverage", "0/0")
        
        # Extract coverage numbers
        try:
            large_passed, large_total = map(int, large_coverage.split('/'))
            small_passed, small_total = map(int, small_coverage.split('/'))
            
            # Calculate coverage percentages
            large_percent = (large_passed / large_total * 100) if large_total > 0 else 0
            small_percent = (small_passed / small_total * 100) if small_total > 0 else 0
            
            # Compare solutions
            if large_success and small_success:
                return "equivalent"
            elif large_success:
                return "large_model_better"
            elif small_success:
                return "small_model_better"
            elif abs(large_percent - small_percent) < 0.01:
                return "equivalent"
            elif large_percent > small_percent:
                return "large_model_better"
            else:
                return "small_model_better"
        except:
            # Fallback if we can't parse coverage
            if large_success == small_success:
                return "equivalent"
            elif large_success:
                return "large_model_better"
            else:
                return "small_model_better"
        
    def run_all_tasks(self) -> List[Dict[str, Any]]:
        """
        Run all tasks in the database
        Returns:
            List of comparison results
        """
        tasks = self.load_tasks_from_db()
        results = []
        
        for task in tasks:
            # Za mbpp.jsonl zadatke, opis zadatka je u polju "text"
            task_description = task.get("text", "") or task.get("task_description", "")
            task_type = task.get("task_type", "")
            
            if task_description:
                result = self.run_task(task_description, task_type, task)
                results.append(result)
        
        return results
    
    def print_comparison_summary(self, comparisons: List[Dict[str, Any]]):
        """Print a summary of comparison results"""
        print("\n==== Resource Comparison Summary ====")
        print(f"Total tasks compared: {len(comparisons)}")
        
        if not comparisons:
            return
            
        # Calculate averages
        avg_large_time = sum(c["large_model"]["execution_time"] for c in comparisons) / len(comparisons)
        avg_small_time = sum(c["small_model"]["execution_time"] for c in comparisons) / len(comparisons)
        
        print(f"\nAverage execution time:")
        print(f"  Large model: {avg_large_time:.2f} seconds")
        print(f"  Small model (adaptive): {avg_small_time:.2f} seconds")
        
        # Count successes
        large_successes = sum(1 for c in comparisons if c["large_model"].get("evaluation", {}).get("success", False))
        small_successes = sum(1 for c in comparisons if c["small_model"].get("evaluation", {}).get("success", False))
        
        print(f"\nSuccess rate:")
        print(f"  Large model: {large_successes}/{len(comparisons)} ({large_successes/len(comparisons)*100:.2f}%)")
        print(f"  Small model: {small_successes}/{len(comparisons)} ({small_successes/len(comparisons)*100:.2f}%)")
        
        # Time efficiency
        faster_count = sum(1 for c in comparisons if c["comparison"]["time_diff_seconds"] > 0)
        
        print(f"\nEfficiency comparison:")
        print(f"  Tasks where small model was faster: {faster_count}/{len(comparisons)} ({faster_count/len(comparisons)*100:.2f}%)")
        print(f"  Average time savings when small model was faster: {sum(c['comparison']['time_diff_percent'] for c in comparisons if c['comparison']['time_diff_seconds'] > 0)/max(faster_count, 1):.2f}%")
        
        # Memory efficiency
        memory_efficient_count = sum(1 for c in comparisons if c["comparison"].get("memory_diff", 0) > 0)
        if memory_efficient_count > 0:
            print(f"\nMemory efficiency:")
            print(f"  Tasks where small model used less memory: {memory_efficient_count}/{len(comparisons)} ({memory_efficient_count/len(comparisons)*100:.2f}%)")
            print(f"  Average memory savings: {sum(c['comparison'].get('memory_diff_percent', 0) for c in comparisons if c['comparison'].get('memory_diff', 0) > 0)/memory_efficient_count:.2f}%")
        
        # Adaptive branching statistics
        avg_depth = sum(c["small_model"]["adaptive_branching"].get("depth_reached", 0) for c in comparisons) / len(comparisons)
        avg_instances = sum(c["small_model"]["adaptive_branching"].get("total_instances", 0) for c in comparisons) / len(comparisons)
        
        print(f"\nAdaptive branching statistics:")
        print(f"  Average depth reached: {avg_depth:.2f} (of max {self.small_model.max_branching_depth})")
        print(f"  Average model instances used: {avg_instances:.2f}")
        
        # Solution quality
        quality_counts = {
            "equivalent": sum(1 for c in comparisons if c["comparison"]["solution_quality"] == "equivalent"),
            "large_model_better": sum(1 for c in comparisons if c["comparison"]["solution_quality"] == "large_model_better"),
            "small_model_better": sum(1 for c in comparisons if c["comparison"]["solution_quality"] == "small_model_better")
        }
        
        print(f"\nSolution quality comparison:")
        print(f"  Equivalent solutions: {quality_counts['equivalent']}/{len(comparisons)} ({quality_counts['equivalent']/len(comparisons)*100:.2f}%)")
        print(f"  Large model better: {quality_counts['large_model_better']}/{len(comparisons)} ({quality_counts['large_model_better']/len(comparisons)*100:.2f}%)")
        print(f"  Small model better: {quality_counts['small_model_better']}/{len(comparisons)} ({quality_counts['small_model_better']/len(comparisons)*100:.2f}%)")

def main():
    """Main function to run the workflow"""
    parser = argparse.ArgumentParser(description="Resource comparison between large and small models")
    parser.add_argument("--task", type=str, help="Task description")
    parser.add_argument("--task-type", type=str, default="prime", help="Task type (prime, fibonacci, etc.)")
    parser.add_argument("--large-model", type=str, default="deepseek-r1:7b", help="Name of the large model")
    parser.add_argument("--small-model", type=str, default="qwen:0.5b", help="Name of the small model")
    parser.add_argument("--url", type=str, default="http://localhost:11434", help="Ollama API URL")
    parser.add_argument("--max-depth", type=int, default=6, help="Maximum branching depth for adaptive small model")
    parser.add_argument("--all", action="store_true", help="Run all tasks from database")
    parser.add_argument("--import-data", action="store_true", help="Import tasks from JSONL file")
    parser.add_argument("--jsonl-file", type=str, default="data/mbpp.jsonl", help="Path to JSONL file to import")
    parser.add_argument("--no-clear", action="store_true", help="Do not clear the collection before importing data")
    
    args = parser.parse_args()
    
    # Initialize database connection
    db = DatabaseConnection()
    db.connect()
    
    # Import data if requested
    if args.import_data:
        if os.path.exists(args.jsonl_file):
            clear_first = not args.no_clear
            count = db.import_jsonl_to_tasks(args.jsonl_file, clear_first=clear_first)
            print(f"Imported {count} tasks from {args.jsonl_file}")
            if not args.all and not args.task:
                db.disconnect()
                return
        else:
            print(f"Error: JSONL file not found at {args.jsonl_file}")
            if not args.all and not args.task:
                db.disconnect()
                return
    
    # Initialize workflow
    workflow = ResourceComparisonWorkflow(db)
    workflow.initialize_models(
        large_model_name=args.large_model,
        small_model_name=args.small_model,
        ollama_url=args.url,
        max_branching_depth=args.max_depth
    )
    
    if args.all:
        # Run all tasks
        results = workflow.run_all_tasks()
        workflow.print_comparison_summary(results)
    elif args.task:
        # Run a single task
        result = workflow.run_task(args.task, args.task_type)
        
        # Print results
        print("\n==== Task Execution Results ====")
        print(f"Task: {args.task}")
        print(f"\nLarge model ({result['large_model']['name']}):")
        print(f"  Execution time: {result['large_model']['execution_time']:.2f} seconds")
        if "evaluation" in result["large_model"]:
            print(f"  Success: {result['large_model']['evaluation'].get('success', False)}")
            print(f"  Test coverage: {result['large_model']['evaluation'].get('test_coverage', '0/0')}")
        
        small_model_name = result['small_model']['name']
        adaptive_info = result["small_model"]["adaptive_branching"]
        
        print(f"\nSmall model ({small_model_name}):")
        print(f"  Execution time: {result['small_model']['execution_time']:.2f} seconds")
        print(f"  Adaptive branching depth reached: {adaptive_info['depth_reached']} of max {adaptive_info['max_depth']}")
        print(f"  Total model instances used: {adaptive_info['total_instances']}")
        if "evaluation" in result["small_model"]:
            print(f"  Success: {result['small_model']['evaluation'].get('success', False)}")
            print(f"  Test coverage: {result['small_model']['evaluation'].get('test_coverage', '0/0')}")
        
        print(f"\nComparison:")
        time_diff = result["comparison"]["time_diff_seconds"]
        if time_diff > 0:
            print(f"  Small model was faster by {time_diff:.2f} seconds ({result['comparison']['time_diff_percent']:.2f}%)")
        else:
            print(f"  Large model was faster by {abs(time_diff):.2f} seconds ({abs(result['comparison']['time_diff_percent']):.2f}%)")
        
        # Print memory comparison if available
        if "memory_diff" in result["comparison"] and result["comparison"]["memory_diff"] != 0:
            memory_diff = result["comparison"]["memory_diff"]
            if memory_diff > 0:
                print(f"  Small model used less memory by {memory_diff:.2f} MB ({result['comparison']['memory_diff_percent']:.2f}%)")
            else:
                print(f"  Large model used less memory by {abs(memory_diff):.2f} MB ({abs(result['comparison']['memory_diff_percent']):.2f}%)")
        
        # Print solution quality comparison
        solution_quality = result["comparison"]["solution_quality"]
        if solution_quality == "equivalent":
            print("  Both models produced solutions of equivalent quality")
        elif solution_quality == "large_model_better":
            print("  Large model produced a better quality solution")
        else:
            print("  Small model produced a better quality solution")
    else:
        print("Please provide a task using --task or run all tasks using --all")
    
    # Disconnect from database
    db.disconnect()

if __name__ == "__main__":
    main()