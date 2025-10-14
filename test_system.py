"""
System test script for MPEO
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test if all modules can be imported"""
    print("🔍 Testing imports...")
    
    try:
        from mpeo.models import TaskGraph, TaskNode, TaskEdge, SystemConfig
        print("✓ Models imported successfully")
    except Exception as e:
        print(f"✗ Models import failed: {e}")
        return False
    
    try:
        from mpeo.database import DatabaseManager
        print("✓ Database imported successfully")
    except Exception as e:
        print(f"✗ Database import failed: {e}")
        return False
    
    try:
        from mpeo.planner import PlannerModel
        print("✓ Planner imported successfully")
    except Exception as e:
        print(f"✗ Planner import failed: {e}")
        return False
    
    try:
        from mpeo.executor import TaskExecutor
        print("✓ Executor imported successfully")
    except Exception as e:
        print(f"✗ Executor import failed: {e}")
        return False
    
    try:
        from mpeo.output import OutputModel
        print("✓ Output imported successfully")
    except Exception as e:
        print(f"✗ Output import failed: {e}")
        return False
    
    try:
        from mpeo.interface import HumanFeedbackInterface
        print("✓ Interface imported successfully")
    except Exception as e:
        print(f"✗ Interface import failed: {e}")
        return False
    
    try:
        from mpeo.coordinator import SystemCoordinator, CLIInterface
        print("✓ Coordinator imported successfully")
    except Exception as e:
        print(f"✗ Coordinator import failed: {e}")
        return False
    
    return True

def test_models():
    """Test data models"""
    print("\n🏗️ Testing data models...")
    
    try:
        from mpeo.models import TaskNode, TaskEdge, TaskGraph, TaskType, DependencyType
        
        # Test TaskNode
        task1 = TaskNode(
            task_id="T1",
            task_desc="测试任务",
            task_type=TaskType.LOCAL_COMPUTE,
            expected_output="测试结果",
            priority=3
        )
        task2 = TaskNode(
            task_id="T2",
            task_desc="测试任务2",
            task_type=TaskType.LOCAL_COMPUTE,
            expected_output="测试结果2",
            priority=2
        )
        print("✓ TaskNode creation successful")
        
        # Test TaskEdge
        edge = TaskEdge(
            from_task_id="T1",
            to_task_id="T2",
            dependency_type=DependencyType.RESULT_DEPENDENCY
        )
        print("✓ TaskEdge creation successful")
        
        # Test TaskGraph
        graph = TaskGraph(nodes=[task1, task2], edges=[edge])
        print("✓ TaskGraph creation successful")
        
        # Test cycle detection
        has_cycle = graph.has_cycle()
        print(f"✓ Cycle detection working (has_cycle: {has_cycle})")
        
        return True
        
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        return False

def test_database():
    """Test database functionality"""
    print("\n💾 Testing database...")
    
    try:
        from mpeo.database import DatabaseManager
        
        # Use test database
        test_db_path = "test_mpeo.db"
        db = DatabaseManager(test_db_path)
        print("✓ Database initialization successful")
        
        # Test config operations
        db.save_config("test_key", "test_value")
        value = db.load_config("test_key")
        assert value == "test_value"
        print("✓ Config operations successful")
        
        # Test logging
        db.log_event(None, "test", "test_operation", "test details")
        logs = db.get_logs(limit=1)
        assert len(logs) > 0
        print("✓ Logging operations successful")
        
        # Clean up test database
        if os.path.exists(test_db_path):
            try:
                os.remove(test_db_path)
                print("✓ Database cleanup successful")
            except PermissionError:
                print("⚠️ Database cleanup skipped (file in use)")
        
        return True
        
    except Exception as e:
        print(f"✗ Database test failed: {e}")
        return False

def test_task_graph_validation():
    """Test task graph validation"""
    print("\n🔄 Testing task graph validation...")
    
    try:
        from mpeo.models import TaskNode, TaskEdge, TaskGraph, TaskType, DependencyType
        
        # Test valid graph
        task1 = TaskNode(
            task_id="T1",
            task_desc="Task 1",
            task_type=TaskType.LOCAL_COMPUTE,
            expected_output="Output 1",
            priority=1
        )
        task2 = TaskNode(
            task_id="T2",
            task_desc="Task 2",
            task_type=TaskType.LOCAL_COMPUTE,
            expected_output="Output 2",
            priority=2
        )
        task3 = TaskNode(
            task_id="T3",
            task_desc="Task 3",
            task_type=TaskType.LOCAL_COMPUTE,
            expected_output="Output 3",
            priority=3
        )
        
        edge1 = TaskEdge(
            from_task_id="T1",
            to_task_id="T2",
            dependency_type=DependencyType.RESULT_DEPENDENCY
        )
        edge2 = TaskEdge(
            from_task_id="T2",
            to_task_id="T3",
            dependency_type=DependencyType.RESULT_DEPENDENCY
        )
        
        valid_graph = TaskGraph(nodes=[task1, task2, task3], edges=[edge1, edge2])
        assert not valid_graph.has_cycle()
        print("✓ Valid graph detection successful")
        
        # Test graph with cycle
        edge3 = TaskEdge(
            from_task_id="T3",
            to_task_id="T1",
            dependency_type=DependencyType.RESULT_DEPENDENCY
        )
        cyclic_graph = TaskGraph(nodes=[task1, task2, task3], edges=[edge1, edge2, edge3])
        assert cyclic_graph.has_cycle()
        print("✓ Cycle detection successful")
        
        return True
        
    except Exception as e:
        print(f"✗ Graph validation test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 MPEO System Test Suite")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_models,
        test_database,
        test_task_graph_validation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        else:
            print(f"\n❌ Test {test.__name__} failed!")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready to use.")
        print("\n📋 Next steps:")
        print("1. Copy .env.example to .env")
        print("2. Add your OpenAI API key to .env")
        print("3. Run: python main.py")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)