# Resource Usage Comparison: Large vs Small Models

## Specifications Document

### 1. Project Purpose
This project implements a system for comparing resource utilization between large, monolithic models and smaller, adaptive models executing identical programming tasks. The goal is to provide empirical data on resource consumption patterns to help make informed decisions about model architecture choices in AI deployments.

### 2. System Architecture

#### 2.1 Core Components
1. **Task Management**
   - Predefined programming tasks stored in MongoDB
   - Support for importing tasks from JSONL files (e.g., MBPP dataset)
   - Each task includes description, test cases, and expected outputs

2. **Execution Environment**
   - Docker-based sandbox environments for isolated execution
   - Resource monitoring and metrics collection
   - Parallel execution capabilities for small models with adaptive branching

3. **Model Types**
   - **Large Model**: Single monolithic model (e.g., deepseek-r1:7b) executing entire tasks
   - **Small Models**: Smaller models (e.g., qwen:0.5b, tinyllama:1.1b) with adaptive branching capabilities
   - Both models process identical inputs to produce comparable outputs

4. **Resource Monitoring**
   - Real-time CPU usage tracking via Docker metrics
   - Memory consumption monitoring
   - Execution time measurements
   - Resource limits enforcement

5. **Data Storage & Analysis**
   - MongoDB database for storing:
     - Task definitions
     - Execution results
     - Resource usage metrics
     - Performance comparisons

### 3. Adaptive Branching Methodology

The small model approach uses an adaptive branching methodology:

1. Initial task attempt with the small model
2. Evaluation of the solution quality
3. If needed, branch with feedback to improve specific parts
4. Continue branching until success or max depth reached
5. Combine results for final solution

This approach allows small models to tackle complex tasks by dividing them into manageable subtasks with appropriate feedback loops.

### 4. Key Metrics

The system collects and analyzes the following metrics:

1. **Performance Metrics**
   - Execution time (seconds)
   - Success rate (% of tasks completed successfully)
   - Test coverage (ratio of passed tests)
   - Solution quality comparison

2. **Resource Metrics**
   - Peak memory usage (MB)
   - Average CPU utilization (%)
   - Total container count
   - Resource efficiency metrics

3. **Adaptive Branching Statistics**
   - Average branching depth reached
   - Maximum branching depth
   - Total model instances used

### 5. Workflow Process

```
Large Model:
Task → Single Execution → Complete Processing → Resource Metrics

Small Models:
Task → Initial Attempt → Evaluate → Branch with Feedback → Combine Results → Resource Metrics
```

### 6. Implementation Details

#### 6.1 Model Configuration
- Large models: Configured with model name and API endpoint
- Small models: Configured with model name, API endpoint, and maximum branching depth

#### 6.2 Task Execution
- Tasks can be executed individually or in batch mode
- Each task execution collects:
  - Generated solution
  - Resource metrics
  - Execution time
  - Evaluation results

#### 6.3 Result Analysis
- Comparison of large vs small model performance:
  - Time efficiency (% faster)
  - Memory efficiency (% less memory used)
  - Solution quality comparison
  - Success rate comparison

### 7. Technical Requirements

- **Runtime Environment**: Python 3.9+
- **Containerization**: Docker
- **Database**: MongoDB
- **Key Libraries**:
  - pymongo: Database operations
  - docker: Container management
  - psutil: Resource monitoring
  - python-dotenv: Environment configuration

### 8. Command-Line Interface

The system provides a CLI with the following options:
- `--task`: Specify a single task description to execute
- `--task-type`: Specify task type (prime, fibonacci, etc.)
- `--large-model`: Specify the large model name
- `--small-model`: Specify the small model name
- `--url`: Specify the Ollama API URL
- `--max-depth`: Set maximum branching depth for adaptive small model
- `--all`: Run all tasks from database
- `--import-data`: Import tasks from JSONL file
- `--jsonl-file`: Path to JSONL file to import
- `--no-clear`: Do not clear the collection before importing data

### 9. Expected Outcomes

The project aims to provide:
- Quantitative comparison of resource usage patterns
- Performance metrics for different architectural approaches
- Data-driven insights for model architecture decisions
- Cost-benefit analysis of monolithic vs distributed approaches

### 10. Future Considerations

- Automated scaling based on resource metrics
- Dynamic task distribution optimization
- Real-time performance monitoring dashboard
- Cost optimization strategies
- Machine learning for resource prediction
- Support for additional model providers beyond Ollama