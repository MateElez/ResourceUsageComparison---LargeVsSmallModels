from pymongo import MongoClient
from dotenv import load_dotenv
import os
import logging
from typing import Dict, List, Any, Optional

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

class DatabaseConnection:
    """MongoDB database connection manager"""
    
    def __init__(self):
        """Initialize the database connection"""
        self.connection_string = os.getenv('MONGODB_CONNECTION_STRING', 'mongodb://localhost:27017')
        self.client = None
        self.db = None
        self.tasks_collection = None
        self.results_collection = None

    def connect(self):
        """Connect to MongoDB database"""
        try:
            self.client = MongoClient(self.connection_string)
            self.db = self.client['ResourceComparison']
            self.tasks_collection = self.db['tasks']
            self.results_collection = self.db['results']
            logger.info("Successfully connected to MongoDB.")
        except Exception as e:
            logger.error(f"Error connecting to MongoDB: {e}")
            raise

    def disconnect(self):
        """Disconnect from MongoDB database"""
        if self.client:
            self.client.close()
            logger.info("Disconnected from MongoDB.")

    def get_all_tasks(self) -> List[Dict[str, Any]]:
        """
        Retrieve all tasks from the collection
        Returns:
            List of task documents
        """
        try:
            return list(self.tasks_collection.find())
        except Exception as e:
            logger.error(f"Error retrieving tasks: {e}")
            return []
            
    def insert_task(self, task_data: Dict[str, Any]) -> Optional[str]:
        """
        Insert a task document into the collection
        Args:
            task_data: Task document to insert
        Returns:
            ID of the inserted document if successful, None otherwise
        """
        try:
            result = self.tasks_collection.insert_one(task_data)
            logger.info(f"Task inserted with ID: {result.inserted_id}")
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Error inserting task: {e}")
            return None
            
    def get_task_by_id(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a task by its ID
        Args:
            task_id: ID of the task to retrieve
        Returns:
            Task document if found, None otherwise
        """
        try:
            from bson.objectid import ObjectId
            return self.tasks_collection.find_one({"_id": ObjectId(task_id)})
        except Exception as e:
            logger.error(f"Error retrieving task by ID: {e}")
            return None
            
    def update_task(self, task_id: str, update_data: Dict[str, Any]) -> bool:
        """
        Update a task document
        Args:
            task_id: ID of the task to update
            update_data: Data to update
        Returns:
            True if successful, False otherwise
        """
        try:
            from bson.objectid import ObjectId
            result = self.tasks_collection.update_one(
                {"_id": ObjectId(task_id)},
                {"$set": update_data}
            )
            success = result.modified_count > 0
            if success:
                logger.info(f"Task {task_id} updated successfully")
            else:
                logger.warning(f"Task {task_id} not updated, not found or no changes")
            return success
        except Exception as e:
            logger.error(f"Error updating task: {e}")
            return False
            
    def delete_task(self, task_id: str) -> bool:
        """
        Delete a task document
        Args:
            task_id: ID of the task to delete
        Returns:
            True if successful, False otherwise
        """
        try:
            from bson.objectid import ObjectId
            result = self.tasks_collection.delete_one({"_id": ObjectId(task_id)})
            success = result.deleted_count > 0
            if success:
                logger.info(f"Task {task_id} deleted successfully")
            else:
                logger.warning(f"Task {task_id} not deleted, not found")
            return success
        except Exception as e:
            logger.error(f"Error deleting task: {e}")
            return False

    def get_all_results(self) -> List[Dict[str, Any]]:
        """
        Retrieve all results from the results collection
        Returns:
            List of result documents
        """
        try:
            return list(self.results_collection.find())
        except Exception as e:
            logger.error(f"Error retrieving results: {e}")
            return []
            
    def insert_result(self, result_data: Dict[str, Any]) -> Optional[str]:
        """
        Insert a result document into the results collection
        Args:
            result_data: Result document to insert
        Returns:
            ID of the inserted document if successful, None otherwise
        """
        try:
            result = self.results_collection.insert_one(result_data)
            logger.info(f"Result inserted with ID: {result.inserted_id}")
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Error inserting result: {e}")
            return None
            
    def get_result_by_id(self, result_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a result by its ID
        Args:
            result_id: ID of the result to retrieve
        Returns:
            Result document if found, None otherwise
        """
        try:
            from bson.objectid import ObjectId
            return self.results_collection.find_one({"_id": ObjectId(result_id)})
        except Exception as e:
            logger.error(f"Error retrieving result by ID: {e}")
            return None
            
    def update_result(self, result_id: str, update_data: Dict[str, Any]) -> bool:
        """
        Update a result document
        Args:
            result_id: ID of the result to update
            update_data: Data to update
        Returns:
            True if successful, False otherwise
        """
        try:
            from bson.objectid import ObjectId
            result = self.results_collection.update_one(
                {"_id": ObjectId(result_id)},
                {"$set": update_data}
            )
            success = result.modified_count > 0
            if success:
                logger.info(f"Result {result_id} updated successfully")
            else:
                logger.warning(f"Result {result_id} not updated, not found or no changes")
            return success
        except Exception as e:
            logger.error(f"Error updating result: {e}")
            return False
            
    def delete_result(self, result_id: str) -> bool:
        """
        Delete a result document
        Args:
            result_id: ID of the result to delete
        Returns:
            True if successful, False otherwise
        """
        try:
            from bson.objectid import ObjectId
            result = self.results_collection.delete_one({"_id": ObjectId(result_id)})
            success = result.deleted_count > 0
            if success:
                logger.info(f"Result {result_id} deleted successfully")
            else:
                logger.warning(f"Result {result_id} not deleted, not found")
            return success
        except Exception as e:
            logger.error(f"Error deleting result: {e}")
            return False

    def clear_tasks_collection(self) -> bool:
        """
        Remove all documents from the tasks collection
        
        Returns:
            True if operation was successful, False otherwise
        """
        try:
            result = self.tasks_collection.delete_many({})
            deleted_count = result.deleted_count
            logger.info(f"Cleared tasks collection, removed {deleted_count} documents")
            return True
        except Exception as e:
            logger.error(f"Error clearing tasks collection: {e}")
            return False

    def clear_results_collection(self) -> bool:
        """
        Remove all documents from the results collection
        
        Returns:
            True if operation was successful, False otherwise
        """
        try:
            result = self.results_collection.delete_many({})
            deleted_count = result.deleted_count
            logger.info(f"Cleared results collection, removed {deleted_count} documents")
            return True
        except Exception as e:
            logger.error(f"Error clearing results collection: {e}")
            return False

    def import_jsonl_to_tasks(self, jsonl_file_path: str, clear_first: bool = True) -> int:
        """
        Import data from a JSONL file into the tasks collection
        Each line in the JSONL file is parsed as a JSON object and inserted as a document
        
        Args:
            jsonl_file_path: Path to the JSONL file to import
            clear_first: Whether to clear the collection before importing new data (default: True)
            
        Returns:
            Number of documents imported
        """
        import json
        
        # Clear the collection first if requested
        if clear_first:
            self.clear_tasks_collection()
        
        count = 0
        try:
            with open(jsonl_file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    if line.strip():  # Skip empty lines
                        try:
                            task_data = json.loads(line)
                            self.insert_task(task_data)
                            count += 1
                        except json.JSONDecodeError as e:
                            logger.error(f"Error parsing JSON from line: {e}")
                        except Exception as e:
                            logger.error(f"Error inserting task from JSONL: {e}")
            
            logger.info(f"Successfully imported {count} tasks from {jsonl_file_path}")
            return count
        except Exception as e:
            logger.error(f"Error importing JSONL file: {e}")
            return count

    def import_jsonl_to_results(self, jsonl_file_path: str, clear_first: bool = True) -> int:
        """
        Import data from a JSONL file into the results collection
        Each line in the JSONL file is parsed as a JSON object and inserted as a document
        
        Args:
            jsonl_file_path: Path to the JSONL file to import
            clear_first: Whether to clear the collection before importing new data (default: True)
            
        Returns:
            Number of documents imported
        """
        import json
        
        # Clear the collection first if requested
        if clear_first:
            self.clear_results_collection()
        
        count = 0
        try:
            with open(jsonl_file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    if line.strip():  # Skip empty lines
                        try:
                            result_data = json.loads(line)
                            self.insert_result(result_data)
                            count += 1
                        except json.JSONDecodeError as e:
                            logger.error(f"Error parsing JSON from line: {e}")
                        except Exception as e:
                            logger.error(f"Error inserting result from JSONL: {e}")
            
            logger.info(f"Successfully imported {count} results from {jsonl_file_path}")
            return count
        except Exception as e:
            logger.error(f"Error importing JSONL file to results: {e}")
            return count