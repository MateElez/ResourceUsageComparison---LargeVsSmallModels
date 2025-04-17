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

    def connect(self):
        """Connect to MongoDB database"""
        try:
            self.client = MongoClient(self.connection_string)
            self.db = self.client['ResourceComparison']
            self.tasks_collection = self.db['tasks']
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