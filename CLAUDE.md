# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MPEO (Multi-model collaborative task processing system) is an AI-powered system that decomposes user queries into structured workflows using DAG (Directed Acyclic Graph) approach, then executes them through coordinated AI models with human oversight. The system follows a "model automatic processing + human intervention confirmation" paradigm.

## Core Architecture

The system follows a modular architecture with clear separation of concerns:

### Core Modules (`mpeo/core/`)
1. **Planner Model** (`mpeo/core/planner.py`) - Analyzes user queries and generates task graphs using OpenAI GPT
2. **Human Feedback Interface** (`mpeo/interfaces/cli.py`) - Interactive CLI for task graph visualization and modification
3. **Executor Model** (`mpeo/core/executor.py`) - Executes task graphs with parallel/serial scheduling and MCP service integration
4. **Output Model** (`mpeo/core/output.py`) - Integrates results and resolves conflicts into final output

### Supporting Modules
- **System Coordinator** (`mpeo/core/coordinator.py`) - Orchestrates all components
- **Database Manager** (`mpeo/services/database.py`) - SQLite-based persistence
- **Data Models** (`mpeo/models/`) - Pydantic models for validation
  - `models/task.py` - Task-related models (TaskNode, TaskGraph, ExecutionResult, etc.)
  - `models/session.py` - Session models (TaskSession)
  - `models/config.py` - Configuration models (SystemConfig, MCPServiceConfig)

### Utilities (`mpeo/utils/`)
- **Logging** (`utils/logging.py`) - Centralized logging setup
- **Configuration** (`utils/config.py`) - Configuration loading and management
- **Exceptions** (`utils/exceptions.py`) - Custom exception classes

### Data Organization
- **Databases** (`data/databases/`) - SQLite database files
- **Logs** (`data/logs/`) - System log files
- **Configuration** (`config/`) - Configuration files (MCP services, etc.)

## Development Commands

### Running the System
```bash
# Basic interactive mode
python main.py

# With custom configuration
python main.py --config CONFIG.json --max-parallel 8 --model gpt-4

# Available options:
--config CONFIG       # Configuration file path
--max-parallel MAX    # Maximum parallel tasks (default: 4)
--timeout TIMEOUT     # MCP service timeout (default: 30s)
--retries RETRIES     # Task retry count (default: 3)
--model MODEL         # OpenAI model (default: gpt-3.5-turbo)
--db-path DB_PATH     # Database path (default: data/databases/mpeo.db)
```

### Testing
```bash
# Run system tests
python scripts/test_system.py

# MCP service testing
python scripts/mcp_service_test.py

# MCP fix validation
python scripts/test_mcp_fix.py

# Debug MCP API
python scripts/debug_mcp_api.py
```

### Configuration
- Environment variables in `.env` (OPENAI_API_KEY required)
- MCP services configuration in `config/mcp_services.json`
- Default database: `data/databases/mpeo.db` (SQLite)
- Log files: `data/logs/` (organized by date)

## Key Technical Patterns

### Data Flow
1. User query → Planner (task decomposition)
2. Task graph → Human interface (validation/modification)
3. Approved graph → Executor (parallel/serial execution)
4. Results → Output model (integration/formatting)

### Execution Patterns
- **DAG-based scheduling**: Tasks execute based on dependencies
- **Parallel processing**: Multiple tasks can run simultaneously when independent
- **Retry logic**: Configurable retry attempts for failed tasks
- **MCP integration**: External services via HTTP/SSE

### Database Schema
- `sessions` - Task processing sessions
- `task_graphs` - Versioned task graphs
- `config` - System configuration
- `logs` - System event logs

## Interactive Runtime Commands

During system execution, these commands are available:
- `help/帮助` - Show help information
- `quit/exit/退出` - Exit system
- `status/状态` - Display system status
- `history/历史` - Show session history
- `logs/日志` - Display system logs
- `config set/get <key> <value>` - Configuration management
- `mcp register <service_name> <url>` - Register MCP services

## Development Notes

- Uses Python 3.11+ with async/await patterns throughout
- Strong typing with Pydantic models for data validation
- Rich library for CLI interface with tables and panels
- Comprehensive error handling and logging to files
- Environment-based configuration management
- Database operations use SQLite with transaction management
- Modular architecture with clear separation of concerns
- Centralized logging and configuration utilities
- Custom exception classes for better error handling

## File Structure Best Practices

When working with this codebase:

1. **Core Logic**: Add new core processing components to `mpeo/core/`
2. **Data Models**: Define new models in appropriate `mpeo/models/` submodules
3. **Services**: Add external service integrations to `mpeo/services/`
4. **Utilities**: Use and extend `mpeo/utils/` for common functionality
5. **Configuration**: Place configuration files in `config/`
6. **Scripts**: Add utility scripts to `scripts/`
7. **Tests**: Place tests in `tests/` with appropriate subdirectories

## Import Patterns

```python
# Good - Use specific imports
from mpeo.core import SystemCoordinator
from mpeo.models import TaskGraph, SystemConfig
from mpeo.utils.logging import get_logger

# Avoid - Import from too deep
from mpeo.core.coordinator import SystemCoordinator  # OK but prefer above
```