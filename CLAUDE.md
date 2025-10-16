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
5. **System Coordinator** (`mpeo/core/coordinator.py`) - Orchestrates all components

### Services (`mpeo/services/`)
- **Database Manager** (`mpeo/services/database.py`) - SQLite-based persistence
- **MCP Common** (`mpeo/services/mcp_common.py`) - MCP protocol utilities
- **Configuration Loader** (`mpeo/services/configuration_loader.py`) - Dynamic configuration loading
- **Unified MCP Manager** (`mpeo/services/unified_mcp_manager.py`) - Centralized MCP service management

### Data Models (`mpeo/models/`)
- **Task Models** (`models/task.py`) - TaskNode, TaskGraph, ExecutionResult, etc.
- **Session Models** (`models/session.py`) - TaskSession for workflow tracking
- **Config Models** (`models/config.py`) - SystemConfig, MCPServiceConfig
- **Agent Config** (`models/agent_config.py`) - Per-agent model configurations

### Utilities (`mpeo/utils/`)
- **Logging** (`utils/logging.py`) - Centralized logging setup
- **Configuration** (`utils/config.py`) - Configuration loading and management
- **Exceptions** (`utils/exceptions.py`) - Custom exception classes

### Data Organization
- **Databases** (`data/databases/`) - SQLite database files
- **Logs** (`data/logs/`) - System log files
- **Configuration** (`config/`) - Configuration files (MCP services, etc.)

## Development Commands

### Environment Setup
```bash
# Install dependencies in development mode
pip install -e .

# Or using uv (if available)
uv pip install -e .

# Copy environment configuration
cp .env.example .env
# Edit .env to add your OPENAI_API_KEY and other settings
```

### Running the System
```bash
# Basic interactive mode
python main.py

# With custom configuration
python main.py --config config/custom.json --max-parallel 8 --model gpt-4

# Available options:
--config CONFIG       # Configuration file path
--max-parallel MAX    # Maximum parallel tasks (default: 4)
--timeout TIMEOUT     # MCP service timeout (default: 30s)
--retries RETRIES     # Task retry count (default: 3)
--model MODEL         # OpenAI model (default: gpt-3.5-turbo)
--db-path DB_PATH     # Database path (default: data/databases/mpeo.db)
```

### Database Management
```bash
# Database is automatically created on first run
# Default location: data/databases/mpeo.db
# Log files organized by date: data/logs/YYYY-MM-DD.log

# To reset database (WARNING: deletes all data)
rm data/databases/mpeo.db
```

### Configuration
- **Environment variables**: `.env` file (OPENAI_API_KEY required)
  - Supports per-agent configuration (planner, executor, output models)
  - Global and individual API keys, bases, and organization IDs
- **MCP services**: `config/mcp_services.json` - Pre-configured services (fetch, context7-mcp, Time-MCP)
- **Agent models**: `config/agent_models.json` - Model configurations for different agents
- **Default database**: `data/databases/mpeo.db` (SQLite)
- **Log files**: `data/logs/` (organized by date)

### Package Management
```bash
# The project uses pyproject.toml for dependency management
# Key dependencies: openai>=1.0.0, aiohttp>=3.8.0, pydantic>=2.0.0, rich>=13.0.0, networkx>=3.0

# To add new dependencies, update pyproject.toml and reinstall:
pip install -e .
```

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

- **Python Version**: Requires Python 3.11+ with extensive async/await patterns
- **Type Safety**: Strong typing with Pydantic models for data validation and serialization
- **CLI Interface**: Rich library for beautiful terminal output with tables, panels, and progress indicators
- **Error Handling**: Comprehensive error handling with file-based logging and custom exception classes
- **Configuration Management**: Environment-based configuration with runtime overrides support
- **Database**: SQLite with transaction management, automatic schema creation, and query optimization
- **Architecture**: Modular design with clear separation of concerns and dependency injection
- **Logging**: Centralized logging system with daily rotation and structured log formats
- **MCP Integration**: Full Model Context Protocol support with multiple service types
- **Agent Configuration**: Per-agent model configuration supporting different OpenAI models for different roles

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
# Good - Use specific imports from package level
from mpeo.core import SystemCoordinator
from mpeo.models import TaskGraph, SystemConfig
from mpeo.utils.logging import get_logger
from mpeo.services import DatabaseManager

# Acceptable - Import from submodules when needed
from mpeo.core.coordinator import SystemCoordinator
from mpeo.models.task import TaskNode
from mpeo.services.unified_mcp_manager import UnifiedMCPManager

# Avoid - Deep imports unless necessary
from mpeo.core.coordinator import SomeInternalClass  # Prefer package-level imports
```

## Common Development Patterns

### Adding New MCP Services
1. Update `config/mcp_services.json` with service configuration
2. Use `mcp register <service_name> <url>` command for dynamic registration
3. Services automatically available in executor model for task execution

### Working with Task Graphs
- Task graphs are DAGs (Directed Acyclic Graphs) managed by NetworkX
- Use `TaskNode` for individual tasks with dependency definitions
- Executor handles parallel/serial scheduling based on dependencies
- All task results are stored in SQLite for audit and debugging

### Configuration Management
- Environment variables in `.env` provide base configuration
- Per-agent configuration in `config/agent_models.json` for model customization
- Runtime configuration changes via `config set/get` commands
- Configuration loader supports hot-reloading for most settings

### Error Handling
- Use custom exceptions from `mpeo.utils.exceptions` for consistent error types
- All operations are logged with context and timestamps
- Database operations use transactions with automatic rollback on errors
- MCP service failures trigger automatic retries with exponential backoff