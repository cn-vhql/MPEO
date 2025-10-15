#!/usr/bin/env python3
"""
批量更新导入路径的脚本
"""

import os
import re
from pathlib import Path

def update_imports_in_file(file_path: str, import_mappings: dict) -> bool:
    """更新文件中的导入路径"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content

        # 应用导入映射
        for old_import, new_import in import_mappings.items():
            # 处理 from ... import ...
            content = re.sub(
                rf'from\s+{re.escape(old_import)}\s+import',
                f'from {new_import} import',
                content
            )

            # 处理 import ...
            content = re.sub(
                rf'import\s+{re.escape(old_import)}',
                f'import {new_import}',
                content)

        # 只有内容发生变化时才写入
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✓ 已更新: {file_path}")
            return True
        else:
            print(f"- 无需更新: {file_path}")
            return False

    except Exception as e:
        print(f"✗ 更新失败 {file_path}: {str(e)}")
        return False

def main():
    """主函数"""
    # 定义导入映射
    import_mappings = {
        # 旧的导入路径 -> 新的导入路径
        'mpeo.models': 'mpeo.models',
        'mpeo.database': 'mpeo.services.database',
        'mpeo.interface': 'mpeo.interfaces.cli',
        '.models': '..models',
        '.database': '..services.database',
        '.interface': '..interfaces.cli',
    }

    # 需要更新的文件
    files_to_update = [
        'mpeo/core/planner.py',
        'mpeo/core/executor.py',
        'mpeo/core/output.py',
        'mpeo/interfaces/cli.py',
        'mpeo/services/database.py',
    ]

    print("🔄 开始批量更新导入路径...")

    updated_count = 0
    for file_path in files_to_update:
        if Path(file_path).exists():
            if update_imports_in_file(file_path, import_mappings):
                updated_count += 1
        else:
            print(f"⚠️  文件不存在: {file_path}")

    print(f"\n✅ 完成！更新了 {updated_count} 个文件")

if __name__ == "__main__":
    main()