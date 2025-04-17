# Resource Usage Comparison: Large vs Small Models

## Project Overview
This project implements a system for comparing resource utilization between large, monolithic, models and smaller, branched, models executing the same tasks. The goal is to provide empirical data on resource consumption patterns to help make informed decisions about model architecture choices.

## Architecture

### Components
1. **Task Management**
   - Predefined set of tasks stored in MongoDB
   - Each task represents a specific computation or operation
   - Tasks are executed by both large and small models

2. **Execution Environment**
   - Docker-based sandbox environments for isolated execution
   - Resource monitoring and metrics collection
   - Parallel execution capabilities for small, branched models

3. **Model Types**
   - **Large Model**: Single monolithic model executing the entire task
   - **Small Models**: Multiple specialized models working on subtasks
   - Both types process identical input and produce comparable output

4. **Resource Monitoring**
   - Real-time CPU usage tracking
   - Memory consumption monitoring
   - Execution time measurements
   - Resource limits enforcement

5. **Data Storage**
   - MongoDB database for storing:
     - Task definitions
     - Execution results
     - Resource usage metrics
     - Performance comparisons

## Workflow

1. **Task Definition**
   - Tasks are stored in the MongoDB 'tasks' collection
   - Each task includes input data and expected output
   - Tasks are designed to be divisible for small model processing

2. **Execution Process**
   ```
   Large Model:
   Task → Single Sandbox → Complete Processing → Resource Metrics

   Small Models:
   Task → Attempt → Evaluate → Branch with Feedback → Combine Results → Resource Metrics
   ```

3. **Resource Tracking**
   - For each execution:
     - Peak memory usage
     - Average CPU utilization
     - Total execution time
     - Resource efficiency metrics

4. **Data Collection**
   - Results stored in MongoDB for:
     - Task ID
     - Model type (large/small)
     - Resource consumption metrics
     - Execution success/failure
     - Output accuracy

5. **Analysis**
   - External analysis of collected data
   - Comparison metrics:
     - Resource efficiency
     - Execution speed
     - Scalability
     - Cost-effectiveness

## Expected Outcomes

The project aims to provide:
- Quantitative comparison of resource usage patterns
- Performance metrics for different architectural approaches
- Data-driven insights for model architecture decisions
- Cost-benefit analysis of monolithic vs distributed approaches

## Technical Stack

- **Runtime Environment**: Python 3.9
- **Containerization**: Docker
- **Database**: MongoDB
- **Key Libraries**:
  - pymongo: Database operations
  - docker: Container management
  - psutil: Resource monitoring
  - python-dotenv: Environment configuration

## Data Analysis

The collected data can be analyzed to:
1. Compare total resource consumption
2. Identify efficiency patterns
3. Evaluate scaling characteristics
4. Assess cost implications
5. Determine optimal model architecture choices

## Future Considerations

- Automated scaling based on resource metrics
- Dynamic task distribution
- Real-time performance monitoring
- Cost optimization strategies
- Machine learning for resource prediction