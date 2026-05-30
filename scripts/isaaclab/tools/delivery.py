#!/usr/bin/env python3
"""
交付脚本：提取指定的环境配置，创建独立的项目
用法: python delivery_script.py <env1> <env2> ... --project-name <project_name>
例如: python delivery_script.py tps_robot tps_wheel --project-name test_project

文件处理策略（基于.gitignore的智能复制策略）:
1. 智能复制：完全基于.gitignore规则决定复制哪些文件，自动排除.git目录
   - logs/outputs等大目录已在.gitignore中定义，会被自动排除
   - __pycache__、*.pyc、.egg-info等临时文件也会被自动排除
2. 清理tasks目录：只保留tasks/manager_based/locomotion/velocity，删除其他tasks子目录
3. 清理config目录：删除config/internal，只保留指定的env配置
4. 清理assets目录：删除assets/internal，只保留需要的assets文件
5. 清理robots数据：删除所有robot，只保留需要的robot数据
6. 修复导入路径和项目重命名
7. 最终清理：删除额外的不需要目录
"""

import sys
import shutil
import re
from pathlib import Path
from typing import List, Set, Dict, Optional

from InquirerPy import inquirer
from InquirerPy.base.control import Choice
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

# 配置：额外需要删除的目录列表
# 这些目录会在最终清理阶段被删除
ADDITIONAL_CLEANUP_DIRECTORIES = [
    "scripts/cli_trainer",  # CLI训练器目录
    "scripts/tools/delivery.py",
    "scripts/tools/train_batch.py",
    "scripts/reinforcement_learning/rsl_rl/play_cs.py",
    "CONTRIBUTORS.md",
    "sync.sh",
    ".github",
    "docs",
    "README.md",
    # 注意：以下路径会根据项目重命名动态调整
    # "source/{project_name}/{project_name}/tasks/manager_based/locomotion/velocity/mdp/symmetry"
]

# 配置：需要从项目配置文件中删除的代码行
CODE_LINES_TO_REMOVE = [
    """
    # internal
    "InquirerPy",
    "rich",
    """,
    # 可以添加更多代码块
    # """
    # 其他需要删除的代码块
    # """,
]

# 配置：需要排除的文件和目录模式会从 .gitignore 文件中动态读取


class DeliveryScript:
    def __init__(self, root_project_path: Optional[str] = None):
        # 设置根项目路径
        if root_project_path is None:
            script_dir = Path(__file__).parent
            self.root_project_path = script_dir.parent.parent
        else:
            self.root_project_path = Path(root_project_path)

        # 关键路径
        self.source_robot_lab_path = self.root_project_path / "source" / "robot_lab"
        self.config_internal_path = (
            self.source_robot_lab_path / "robot_lab" / "tasks" / "manager_based"
            / "locomotion" / "velocity" / "config" / "internal"
        )
        self.assets_internal_path = self.source_robot_lab_path / "robot_lab" / "assets" / "internal"
        self.robots_data_path = self.source_robot_lab_path / "data" / "Robots"

        # 存储分析结果
        self.env_configs: Dict[str, Path] = {}
        self.required_assets: Set[str] = set()
        self.required_robots: Set[str] = set()

        # 读取 .gitignore 规则
        self.exclude_patterns = self._load_gitignore_patterns()

    def _load_gitignore_patterns(self) -> List[str]:
        """从 .gitignore 文件加载排除模式"""
        patterns = []
        gitignore_path = self.root_project_path / ".gitignore"

        with open(gitignore_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # 跳过空行和注释
                if not line or line.startswith('#'):
                    continue
                # 跳过否定规则 (!)，这些在复制时我们不需要处理
                if line.startswith('!'):
                    continue
                patterns.append(line)

        return patterns

    def should_exclude_item(self, item_path: Path, relative_to: Optional[Path] = None) -> bool:
        """检查文件或目录是否应该被排除"""

        name = item_path.name

        # 如果提供了相对路径基准，计算相对路径
        if relative_to and relative_to in item_path.parents:
            rel_path = str(item_path.relative_to(relative_to))
        else:
            rel_path = name

        # 检查是否匹配排除模式
        for pattern in self.exclude_patterns:
            # 处理不同类型的模式
            if self._matches_pattern(rel_path, name, pattern):
                return True

        return False

    def _matches_pattern(self, rel_path: str, name: str, pattern: str) -> bool:
        """检查路径是否匹配给定模式"""
        import fnmatch

        # 处理特殊的 **/dirname/* 模式（如 **/logs/*）
        if pattern.startswith('**/') and pattern.endswith('/*'):
            # 提取目录名，如从 "**/logs/*" 提取 "logs"
            dir_name = pattern[3:-2]  # 去掉 **/ 和 /*
            # 如果当前项目名就是这个目录名，应该排除整个目录
            if name == dir_name:
                return True
            # 也检查完整路径匹配
            if fnmatch.fnmatch(rel_path, pattern):
                return True

        # 处理 ** 通配符模式
        elif pattern.startswith('**/'):
            sub_pattern = pattern[3:]  # 去掉 **/

            # 如果是目录匹配模式 (以/结尾)
            if sub_pattern.endswith('/'):
                dir_pattern = sub_pattern[:-1]  # 去掉末尾的/
                if fnmatch.fnmatch(name, dir_pattern):
                    return True
            # 普通文件匹配
            elif fnmatch.fnmatch(name, sub_pattern):
                return True

            # 也检查完整路径
            if fnmatch.fnmatch(rel_path, pattern):
                return True

        elif pattern.endswith('/*'):
            # dir/* 匹配目录下的所有内容
            dir_name = pattern[:-2]
            if rel_path.startswith(dir_name + '/') or name == dir_name:
                return True

        elif pattern.endswith('/'):
            # dir/ 匹配目录本身
            dir_name = pattern[:-1]
            if name == dir_name or rel_path.endswith('/' + dir_name):
                return True

        else:
            # 普通模式匹配
            if fnmatch.fnmatch(name, pattern) or fnmatch.fnmatch(rel_path, pattern):
                return True

        return False

    def _create_ignore_function(self, base_path: Path):
        """创建用于 shutil.copytree 的忽略函数"""
        def ignore_func(dir_path: str, names: List[str]) -> List[str]:
            ignored = []
            current_dir = Path(dir_path)

            for name in names:
                item_path = current_dir / name
                if self.should_exclude_item(item_path, base_path):
                    ignored.append(name)

            return ignored

        return ignore_func

    def _create_smart_ignore_function(self, base_path: Path):
        """创建智能忽略函数，基于.gitignore规则 + 自动排除.git目录"""
        def ignore_func(dir_path: str, names: List[str]) -> List[str]:
            ignored = []
            current_dir = Path(dir_path)

            for name in names:
                item_path = current_dir / name
                should_ignore = False

                # 自动排除.git目录（通常不在.gitignore中但确实不需要）
                if name == '.git':
                    should_ignore = True
                    print(f"    🗑️ 自动排除: {name} (.git目录)")

                # 检查.gitignore规则
                elif self.should_exclude_item(item_path, base_path):
                    should_ignore = True
                    # 找出匹配的模式用于日志
                    matched_pattern = self._find_matching_pattern(item_path, base_path)
                    rel_path = item_path.relative_to(base_path) if base_path in item_path.parents else item_path.name
                    print(f"    🗑️ gitignore排除: {rel_path} (匹配规则: {matched_pattern})")

                if should_ignore:
                    ignored.append(name)

            return ignored

        return ignore_func

    def _find_matching_pattern(self, item_path: Path, base_path: Path) -> str:
        """找出匹配的.gitignore模式（用于日志显示）"""
        name = item_path.name

        # 计算相对路径
        if base_path and base_path in item_path.parents:
            rel_path = str(item_path.relative_to(base_path))
        else:
            rel_path = name

        # 检查哪个模式匹配
        for pattern in self.exclude_patterns:
            if self._matches_pattern(rel_path, name, pattern):
                return pattern

        return "未知规则"

    def scan_available_envs(self) -> List[str]:
        """扫描可用的环境配置"""
        env_names = []
        if self.config_internal_path.exists():
            for item in self.config_internal_path.iterdir():
                if item.is_dir() and not item.name.startswith('.'):
                    env_names.append(item.name)
        return sorted(env_names)

    def find_env_configs(self, env_names: List[str]) -> bool:
        """查找指定的环境配置目录"""
        print(f"🔍 查找环境配置: {env_names}")

        missing_envs = []
        for env_name in env_names:
            env_path = self.config_internal_path / env_name
            if env_path.exists():
                self.env_configs[env_name] = env_path
                print(f"  ✅ 找到环境配置: {env_name} -> {env_path}")
            else:
                missing_envs.append(env_name)
                print(f"  ❌ 未找到环境配置: {env_name}")

        if missing_envs:
            print(f"\n❌ 以下环境配置未找到: {missing_envs}")
            print(f"请检查路径: {self.config_internal_path}")
            return False

        return True

    def analyze_dependencies(self) -> bool:
        """分析环境配置文件中使用的assets和robots依赖"""
        print("\n🔍 分析依赖关系...")

        # 1. 分析assets依赖
        import_pattern = re.compile(r'from\s+robot_lab\.assets\.internal\.(\w+)\s+import\s+(\w+)')

        for env_name, env_path in self.env_configs.items():
            print(f"  分析环境: {env_name}")

            # 遍历环境配置目录下的所有Python文件
            for py_file in env_path.rglob("*.py"):
                if py_file.name == "__init__.py":
                    continue

                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()

                    # 查找导入语句
                    matches = import_pattern.findall(content)
                    for asset_module, asset_name in matches:
                        self.required_assets.add(asset_module)
                        print(f"    📦 发现assets依赖: {asset_module}.{asset_name} (在 {py_file.name})")

                except Exception as e:
                    print(f"    ⚠️ 读取文件失败 {py_file}: {e}")

        # 验证assets文件存在
        missing_assets = []
        for asset_name in self.required_assets:
            asset_file = self.assets_internal_path / f"{asset_name}.py"
            if not asset_file.exists():
                missing_assets.append(asset_name)
                print(f"  ❌ 未找到assets文件: {asset_name}.py")
            else:
                print(f"  ✅ 找到assets文件: {asset_name}.py")

        # 2. 分析robots依赖
        print("\n🔍 分析robots依赖...")
        for asset_name in self.required_assets:
            asset_file = self.assets_internal_path / f"{asset_name}.py"
            try:
                with open(asset_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                # 查找asset_path引用，提取robots路径
                asset_path_pattern = re.compile(r'asset_path=f["\'{].*?/Robots/(.*?)["\'}]')
                matches = asset_path_pattern.findall(content)
                for match in matches:
                    # match 格式类似: internal/tps/tps_robot_description/urdf/tps_robot.urdf
                    # 提取第一段作为robot名称: internal/tps
                    parts = match.split('/')
                    if len(parts) >= 2:
                        robot_path = '/'.join(parts[:2])  # 例如: internal/tps
                        self.required_robots.add(robot_path)
                        print(f"    🤖 发现robot依赖: {robot_path} (在 {asset_name}.py)")

                        # 同时记录去掉internal后的路径，用于后续处理
                        if parts[0] == 'internal' and len(parts) >= 2:
                            clean_robot_path = parts[1]  # 例如: tps
                            # 存储映射关系，用于后续路径替换
                            if not hasattr(self, 'robot_path_mapping'):
                                self.robot_path_mapping = {}
                            self.robot_path_mapping[robot_path] = clean_robot_path

            except Exception as e:
                print(f"    ⚠️ 分析assets文件失败 {asset_file}: {e}")

        # 验证robots数据存在
        missing_robots = []
        for robot_path in self.required_robots:
            robot_dir = self.robots_data_path / robot_path
            if not robot_dir.exists():
                missing_robots.append(robot_path)
                print(f"  ❌ 未找到robot数据: {robot_path}")
            else:
                print(f"  ✅ 找到robot数据: {robot_path}")

        if missing_assets or missing_robots:
            if missing_assets:
                print(f"\n❌ 以下assets文件未找到: {missing_assets}")
            if missing_robots:
                print(f"\n❌ 以下robot数据未找到: {missing_robots}")
            return False

        return True

    def create_project_structure(self, output_path: Path) -> bool:
        """先完整复制整个项目，再删除不需要的部分"""
        print(f"\n📋 创建项目结构到: {output_path}")

        try:
            # 如果输出目录存在，询问是否覆盖
            if output_path.exists():
                console.print(f"⚠️ [yellow]输出目录已存在:[/yellow] {output_path}")
                overwrite = inquirer.confirm(
                    message="是否覆盖现有目录?",
                    default=False
                ).execute()
                if overwrite:
                    console.print(f"  🗑️ 删除已存在的输出目录: {output_path}")
                    shutil.rmtree(output_path)
                else:
                    console.print("❌ [red]操作已取消[/red]")
                    return False

            # 步骤1: 智能复制整个项目 (基于.gitignore规则自动排除)
            print("  📁 智能复制项目 (基于.gitignore规则自动排除不需要的文件)...")
            shutil.copytree(
                self.root_project_path,
                output_path,
                ignore=self._create_smart_ignore_function(self.root_project_path)
            )

            return True

        except Exception as e:
            print(f"❌ 创建项目结构失败: {e}")
            return False

    def setup_tasks_and_configs(self, output_path: Path) -> bool:
        """清理tasks目录结构并提取指定的环境配置"""
        print("\n🎯 设置tasks目录和环境配置...")

        try:
            # 1. 清理整个tasks目录结构，只保留manager_based/locomotion/velocity
            target_tasks_path = output_path / "source" / "robot_lab" / "robot_lab" / "tasks"

            if not target_tasks_path.exists():
                print(f"    ⚠️ tasks目录不存在: {target_tasks_path}")
                return False

            print("  🗑️ 清理tasks目录结构，只保留manager_based/locomotion/velocity...")

            # 删除tasks下不需要的目录
            for item in target_tasks_path.iterdir():
                if item.is_dir():
                    if item.name != "manager_based":
                        print(f"    🗑️ 删除tasks子目录: {item.name}")
                        shutil.rmtree(item)
                elif item.is_file() and item.name != "__init__.py":
                    print(f"    🗑️ 删除tasks文件: {item.name}")
                    item.unlink()

            # 清理manager_based目录，只保留locomotion
            manager_based_path = target_tasks_path / "manager_based"
            if manager_based_path.exists():
                for item in manager_based_path.iterdir():
                    if item.is_dir():
                        if item.name != "locomotion":
                            print(f"    🗑️ 删除manager_based子目录: {item.name}")
                            shutil.rmtree(item)
                    elif item.is_file() and item.name != "__init__.py":
                        print(f"    🗑️ 删除manager_based文件: {item.name}")
                        item.unlink()

            # 清理locomotion目录，只保留velocity
            locomotion_path = manager_based_path / "locomotion"
            if locomotion_path.exists():
                for item in locomotion_path.iterdir():
                    if item.is_dir():
                        if item.name != "velocity":
                            print(f"    🗑️ 删除locomotion子目录: {item.name}")
                            shutil.rmtree(item)
                    elif item.is_file() and item.name != "__init__.py":
                        print(f"    🗑️ 删除locomotion文件: {item.name}")
                        item.unlink()

            # 2. 清理config目录
            target_velocity_path = locomotion_path / "velocity"
            target_config_path = target_velocity_path / "config"

            if not target_config_path.exists():
                print(f"    ⚠️ config目录不存在: {target_config_path}")
                return False

            print("  🗑️ 清理config目录...")
            config_internal_path = target_config_path / "internal"
            if config_internal_path.exists():
                print("    🗑️ 删除config/internal目录")
                shutil.rmtree(config_internal_path)

            # 删除其他不需要的config子目录，只保留__init__.py和指定的env配置
            for item in target_config_path.iterdir():
                if item.is_file():
                    if item.name != "__init__.py":
                        print(f"    🗑️ 删除配置文件: {item.name}")
                        item.unlink()
                elif item.is_dir():
                    # 删除所有现有的环境配置目录，稍后重新添加需要的
                    print(f"    🗑️ 删除配置目录: {item.name}")
                    shutil.rmtree(item)

            # 将指定的环境配置从源码internal提取到config根目录
            print("  🎯 提取指定的环境配置到config根目录...")
            for env_name, env_path in self.env_configs.items():
                target_env_path = target_config_path / env_name
                # 应用排除规则到环境配置复制
                shutil.copytree(env_path, target_env_path, ignore=self._create_ignore_function(env_path.parent))
                print(f"    📁 提取环境配置: {env_name}")

            return True

        except Exception as e:
            print(f"❌ 设置tasks目录失败: {e}")
            return False

    def setup_assets_and_robots(self, output_path: Path) -> bool:
        """清理并设置assets和robots数据"""
        print("\n📦 设置assets和robots数据...")

        try:
            # 1. 清理assets目录
            output_robot_lab_path = output_path / "source" / "robot_lab" / "robot_lab"
            target_assets_path = output_robot_lab_path / "assets"

            if not target_assets_path.exists():
                print(f"    ⚠️ assets目录不存在: {target_assets_path}")
                return False

            print("  🗑️ 清理assets目录...")
            # 删除assets/internal目录
            assets_internal_path = target_assets_path / "internal"
            if assets_internal_path.exists():
                print("    🗑️ 删除assets/internal目录")
                shutil.rmtree(assets_internal_path)

            # 删除其他非必需的assets文件，只保留__init__.py和需要的assets
            for item in target_assets_path.iterdir():
                if item.is_file() and item.name != "__init__.py":
                    print(f"    🗑️ 删除assets文件: {item.name}")
                    item.unlink()

            # 2. 提取所需的assets文件到根目录
            print("  📦 提取需要的assets文件...")
            for asset_name in self.required_assets:
                source_file = self.assets_internal_path / f"{asset_name}.py"
                target_file = target_assets_path / f"{asset_name}.py"

                shutil.copy2(source_file, target_file)
                print(f"    📦 提取assets: {asset_name}.py")

                # 修复assets文件中的路径引用
                self._fix_assets_paths(target_file)

                # 更新assets/__init__.py文件
                self._update_assets_init_file(target_assets_path, asset_name)

            # 3. 清理并设置robots数据
            target_data_path = output_path / "source" / "robot_lab" / "data"
            target_robots_path = target_data_path / "Robots"

            if target_robots_path.exists():
                print("  🗑️ 清理robots数据...")
                # 删除所有现有的robot数据，稍后重新添加需要的
                for item in target_robots_path.iterdir():
                    if item.is_dir():
                        print(f"    🗑️ 删除robot目录: {item.name}")
                        shutil.rmtree(item)
                    else:
                        print(f"    🗑️ 删除robot文件: {item.name}")
                        item.unlink()

            # 确保robots目录存在
            target_robots_path.mkdir(parents=True, exist_ok=True)

            print("  🤖 提取需要的robots数据...")
            for robot_path in self.required_robots:
                source_robot_dir = self.robots_data_path / robot_path

                # 如果是internal路径，去掉internal前缀
                if hasattr(self, 'robot_path_mapping') and robot_path in self.robot_path_mapping:
                    clean_robot_path = self.robot_path_mapping[robot_path]
                    target_robot_dir = target_robots_path / clean_robot_path
                    print(f"    🤖 提取robot数据: {robot_path} -> {clean_robot_path}")
                else:
                    target_robot_dir = target_robots_path / robot_path
                    print(f"    🤖 提取robot数据: {robot_path}")

                target_robot_dir.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(source_robot_dir, target_robot_dir)

            # 4. 修复环境配置文件中的导入路径
            self._fix_import_statements(output_path, "robot_lab")

            return True

        except Exception as e:
            print(f"❌ 设置assets和robots失败: {e}")
            return False

    def _update_assets_init_file(self, assets_path: Path, asset_name: str):
        """更新assets/__init__.py文件，添加新的导入"""
        init_file = assets_path / "__init__.py"

        try:
            # 读取现有内容
            content = ""
            if init_file.exists():
                with open(init_file, 'r', encoding='utf-8') as f:
                    content = f.read()

            # 添加新的导入语句（如果不存在）
            new_import = f"from .{asset_name} import *  # noqa: F401,F403"
            if new_import not in content:
                if content and not content.endswith('\n'):
                    content += '\n'
                content += new_import + '\n'

                with open(init_file, 'w', encoding='utf-8') as f:
                    f.write(content)

                print(f"    ✅ 更新assets/__init__.py，添加导入: {asset_name}")

        except Exception as e:
            print(f"    ⚠️ 更新assets/__init__.py失败: {e}")

    def _fix_assets_paths(self, asset_file: Path):
        """修复assets文件中的路径引用，去掉internal路径"""
        try:
            with open(asset_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # 修复asset_path中的internal路径
            # 例如: /Robots/internal/njust/njust_description -> /Robots/njust/njust_description
            pattern = re.compile(r'(/Robots/)internal/([^/"\']+)')
            new_content = pattern.sub(r'\1\2', content)

            if new_content != content:
                with open(asset_file, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                print(f"    🔧 修复assets文件路径: {asset_file.name}")

        except Exception as e:
            print(f"    ⚠️ 修复assets路径失败 {asset_file}: {e}")

    def _fix_import_statements(self, output_path: Path, project_name: str = "robot_lab"):
        """修复环境配置文件中的导入语句"""
        print(f"  🔧 修复导入语句（项目名: {project_name}）...")

        # 修复模式：从 project_name.assets.internal.xxx 改为 project_name.assets.xxx
        pattern = re.compile(rf'from\s+{re.escape(project_name)}\.assets\.internal\.(\w+)\s+import\s+(\w+)')

        # 找到当前的项目目录名
        source_path = output_path / "source"
        project_dirs = [d for d in source_path.iterdir() if d.is_dir()]
        if not project_dirs:
            print("    ⚠️ 未找到项目目录")
            return

        current_project_dir = project_dirs[0].name

        # 现在配置文件在config根目录下，而不是config/internal
        config_path = (
            output_path / "source" / current_project_dir / current_project_dir / "tasks"
            / "manager_based" / "locomotion" / "velocity" / "config"
        )

        if not config_path.exists():
            print(f"    ⚠️ 配置路径不存在: {config_path}")
            return

        for py_file in config_path.rglob("*.py"):
            if py_file.name == "__init__.py":
                continue

            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                # 查找并替换导入语句
                modified = False

                def replace_import(match):
                    nonlocal modified
                    modified = True
                    asset_module = match.group(1)
                    asset_name = match.group(2)
                    rel_path = py_file.relative_to(config_path)
                    print(f"    🔧 修复 {rel_path}: {asset_module}.{asset_name}")
                    return f"from {project_name}.assets.{asset_module} import {asset_name}"

                new_content = pattern.sub(replace_import, content)

                if modified:
                    with open(py_file, 'w', encoding='utf-8') as f:
                        f.write(new_content)

            except Exception as e:
                rel_path = (
                    py_file.relative_to(config_path)
                    if config_path in py_file.parents else py_file.name
                )
                print(f"    ⚠️ 修复文件失败 {rel_path}: {e}")

    def rename_project(self, output_path: Path, new_project_name: str) -> bool:
        """重命名整个项目"""
        print(f"\n🏷️ 重命名项目: robot_lab -> {new_project_name}")

        try:
            # 重命名source/robot_lab目录
            old_main_dir = output_path / "source" / "robot_lab"
            new_main_dir = output_path / "source" / new_project_name

            if old_main_dir.exists():
                print(f"  📁 重命名主目录: source/robot_lab -> source/{new_project_name}")
                shutil.move(str(old_main_dir), str(new_main_dir))

            # 重命名内部的robot_lab目录
            old_inner_dir = new_main_dir / "robot_lab"
            new_inner_dir = new_main_dir / new_project_name

            if old_inner_dir.exists():
                print(f"  📁 重命名内部目录: {new_project_name}/robot_lab -> {new_project_name}/{new_project_name}")
                shutil.move(str(old_inner_dir), str(new_inner_dir))

            # 更新所有Python文件中的导入语句和引用
            self._update_project_references(output_path, new_project_name)

            # 更新项目配置文件
            self._update_project_config_files(output_path, new_project_name)

            # 再次修复导入语句，处理项目重命名后可能遗留的internal导入
            self._fix_import_statements(output_path, new_project_name)

            return True

        except Exception as e:
            print(f"❌ 重命名项目失败: {e}")
            return False

    def _update_project_references(self, output_path: Path, new_project_name: str):
        """更新所有文件中的项目引用"""
        print("  🔧 更新项目引用...")

        # 模式匹配
        patterns = [
            (re.compile(r'\brobot_lab\b'), new_project_name),
        ]

        # 需要处理的文件扩展名（文本文件）
        text_extensions = {
            '.py', '.toml', '.yaml', '.yml', '.json', '.cfg', '.ini',
            '.md', '.rst', '.txt', '.sh', '.bat', '.ps1',
            '.Dockerfile', '.env', '.gitignore', '.gitattributes'
        }

        # 遍历所有文本文件
        for file_path in output_path.rglob("*"):
            if not file_path.is_file():
                continue

            # 检查文件扩展名或特殊文件名
            if (file_path.suffix.lower() in text_extensions
                    or file_path.name in ['Dockerfile', 'Makefile', 'LICENSE']):

                try:
                    # 尝试以文本方式读取
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    modified = False
                    for pattern, replacement in patterns:
                        if pattern.search(content):
                            content = pattern.sub(replacement, content)
                            modified = True

                    if modified:
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(content)

                        rel_path = file_path.relative_to(output_path)
                        print(f"    🔧 更新引用: {rel_path}")

                except (UnicodeDecodeError, PermissionError, OSError):
                    # 跳过二进制文件或无法读取的文件
                    continue
                except Exception as e:
                    rel_path = file_path.relative_to(output_path) if output_path in file_path.parents else file_path.name
                    print(f"    ⚠️ 更新文件失败 {rel_path}: {e}")

    def _update_project_config_files(self, output_path: Path, new_project_name: str):
        """更新项目配置文件"""
        print("  🔧 更新项目配置文件...")

        # 更新setup.py
        setup_file = output_path / "source" / new_project_name / "setup.py"
        if setup_file.exists():
            try:
                with open(setup_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                # 替换项目名
                content = content.replace('robot_lab', new_project_name)
                content = content.replace('Robot Lab', new_project_name.replace('_', ' ').title())

                # 删除指定的代码行
                content = self._remove_specified_code_lines(content)

                with open(setup_file, 'w', encoding='utf-8') as f:
                    f.write(content)

                print("    ✅ 更新setup.py")

            except Exception as e:
                print(f"    ⚠️ 更新setup.py失败: {e}")

        # 更新pyproject.toml
        pyproject_file = output_path / "source" / new_project_name / "pyproject.toml"
        if pyproject_file.exists():
            try:
                with open(pyproject_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                # 替换项目名
                content = content.replace('robot_lab', new_project_name)
                content = content.replace('robot-lab', new_project_name.replace('_', '-'))

                # 删除指定的代码行
                content = self._remove_specified_code_lines(content)

                with open(pyproject_file, 'w', encoding='utf-8') as f:
                    f.write(content)

                print("    ✅ 更新pyproject.toml")

            except Exception as e:
                print(f"    ⚠️ 更新pyproject.toml失败: {e}")

    def _remove_specified_code_lines(self, content: str) -> str:
        """从配置文件内容中删除指定的代码行"""
        lines = content.split('\n')
        filtered_lines = []

        # 将所有配置字符串块分割成行，并过滤空行
        all_code_lines_to_remove = []
        for code_block in CODE_LINES_TO_REMOVE:
            code_lines = [
                line.strip() for line in code_block.strip().split('\n')
                if line.strip()
            ]
            all_code_lines_to_remove.extend(code_lines)

        for line in lines:
            should_remove = False

            # 检查是否需要删除这一行
            for code_to_remove in all_code_lines_to_remove:
                if code_to_remove in line.strip():
                    should_remove = True
                    print(f"    🗑️ 删除代码行: {line.strip()}")
                    break

            if not should_remove:
                filtered_lines.append(line)

        return '\n'.join(filtered_lines)

    def _cleanup_additional_directories(self, output_path: Path, project_name: str):
        """清理配置中指定的额外目录"""
        print("  🧹 清理额外目录...")

        cleanup_dirs = ADDITIONAL_CLEANUP_DIRECTORIES.copy()

        # 添加项目相关的动态路径
        dynamic_cleanup_dirs = [
            f"source/{project_name}/{project_name}/tasks/manager_based/locomotion/velocity/mdp/symmetry"
        ]
        cleanup_dirs.extend(dynamic_cleanup_dirs)

        for dir_path in cleanup_dirs:
            target_dir = output_path / dir_path
            if target_dir.exists():
                if target_dir.is_dir():
                    shutil.rmtree(target_dir)
                    print(f"    🗑️ 删除目录: {dir_path}")
                else:
                    target_dir.unlink()
                    print(f"    🗑️ 删除文件: {dir_path}")
            else:
                print(f"    ⏭️ 跳过不存在的路径: {dir_path}")

        # 额外清理：删除所有 .gitignore 中匹配的文件和目录
        self._cleanup_gitignore_patterns(output_path)

    def _cleanup_gitignore_patterns(self, output_path: Path):
        """根据 .gitignore 模式清理遗留的文件和目录"""
        print("  🧹 清理 .gitignore 匹配的文件...")

        # 特别针对常见的遗留文件进行清理
        patterns_to_clean = [
            "**/__pycache__",
            "**/*.egg-info",
            "**/*.pyc",
            "**/.pytest_cache"
        ]

        cleaned_count = 0

        # 遍历输出目录，查找匹配的文件和目录
        items_to_delete = []
        for item in output_path.rglob("*"):
            if not item.exists():
                continue

            item_name = item.name
            rel_path = str(item.relative_to(output_path))

            should_delete = False
            for pattern in patterns_to_clean:
                if self._matches_gitignore_pattern(rel_path, item_name, pattern):
                    should_delete = True
                    break

            if should_delete:
                items_to_delete.append(item)

        # 删除收集到的项目（按深度排序，先删除深层项目）
        items_to_delete.sort(key=lambda p: len(p.parts), reverse=True)

        for item in items_to_delete:
            if not item.exists():
                continue
            try:
                rel_path = str(item.relative_to(output_path))
                if item.is_dir():
                    shutil.rmtree(item)
                    print(f"    🗑️ 清理目录: {rel_path}")
                else:
                    item.unlink()
                    print(f"    🗑️ 清理文件: {rel_path}")
                cleaned_count += 1
            except Exception as e:
                rel_path = str(item.relative_to(output_path)) if output_path in item.parents else item.name
                print(f"    ⚠️ 清理失败 {rel_path}: {e}")

        if cleaned_count > 0:
            print(f"    ✅ 清理了 {cleaned_count} 个项目")
        else:
            print("    ✅ 没有需要清理的项目")

    def _matches_gitignore_pattern(self, rel_path: str, name: str, pattern: str) -> bool:
        """检查路径是否匹配 gitignore 模式（简化版）"""
        import fnmatch

        if pattern.startswith('**/'):
            # **/*.ext 或 **/__pycache__ 等
            sub_pattern = pattern[3:]
            return fnmatch.fnmatch(name, sub_pattern) or fnmatch.fnmatch(rel_path, pattern)
        else:
            return fnmatch.fnmatch(name, pattern) or fnmatch.fnmatch(rel_path, pattern)

    def run(self, env_names: List[str], project_name: str, output_dir: Optional[str] = None) -> bool:
        """运行完整的交付流程"""
        print("🚀 开始交付流程...")
        print("=" * 60)

        # 设置输出路径
        if output_dir is None:
            output_dir = f"../{project_name}"
        output_path = Path(output_dir).absolute()

        print("📋 交付参数:")
        print(f"  环境配置: {env_names}")
        print(f"  项目名称: {project_name}")
        print(f"  输出路径: {output_path}")
        print(f"  根项目路径: {self.root_project_path}")

        # 步骤1: 查找环境配置
        if not self.find_env_configs(env_names):
            return False

        # 步骤2: 分析依赖关系
        if not self.analyze_dependencies():
            return False

        # 步骤3: 创建项目基础结构
        if not self.create_project_structure(output_path):
            return False

        # 步骤4: 设置tasks目录和环境配置
        if not self.setup_tasks_and_configs(output_path):
            return False

        # 步骤5: 设置assets和robots数据
        if not self.setup_assets_and_robots(output_path):
            return False

        # 步骤6: 重命名项目
        if not self.rename_project(output_path, project_name):
            return False

        # 步骤7: 清理额外目录
        print("\n🧹 最终清理...")
        self._cleanup_additional_directories(output_path, project_name)

        print("\n" + "=" * 60)
        print("🎉 交付完成!")
        print(f"📁 输出目录: {output_path}")
        print(f"📦 包含环境: {list(self.env_configs.keys())}")
        print(f"📦 包含assets: {list(self.required_assets)}")
        print(f"🤖 包含robots: {list(self.required_robots)}")
        print(f"🏷️ 项目名称: {project_name}")

        return True


def interactive_main():
    """交互式主函数"""

    # 显示欢迎信息
    console.print(Panel.fit(
        "[bold blue]🚀 Robot Lab 交付脚本[/bold blue]\n"
        "[dim]提取指定的环境配置，创建独立的项目[/dim]",
        title="欢迎",
        border_style="blue"
    ))

    # 获取根项目路径
    script_dir = Path(__file__).parent
    default_root = script_dir.parent.parent

    root_project = inquirer.text(
        message="根项目路径:",
        default=str(default_root),
        validate=lambda path: Path(path).exists() or "路径不存在"
    ).execute()

    # 创建交付脚本实例
    script = DeliveryScript(root_project)

    # 扫描可用的环境配置
    console.print("\n📡 [cyan]扫描可用的环境配置...[/cyan]")
    available_envs = script.scan_available_envs()

    if not available_envs:
        console.print("❌ [red]未找到任何环境配置[/red]")
        sys.exit(1)

    console.print(f"✅ 找到 {len(available_envs)} 个可用环境")

    # 选择环境配置
    env_choices = [
        Choice(env, name=f"📦 {env}") for env in available_envs
    ]

    selected_envs = inquirer.checkbox(
        message="选择要提取的环境配置:",
        choices=env_choices,
        validate=lambda x: len(x) > 0 or "至少选择一个环境配置",
        instruction="(使用空格选择，回车确认)"
    ).execute()

    # 输入项目名称
    project_name = inquirer.text(
        message="新项目名称:",
        validate=lambda name: (name.replace('_', '').replace('-', '').isalnum() and len(name) > 0) or "项目名称只能包含字母、数字、下划线和连字符"
    ).execute()

    # 输入输出目录
    default_output = f"../{project_name}"
    output_dir = inquirer.text(
        message="输出目录路径:",
        default=default_output,
    ).execute()

    # 显示配置摘要
    table = Table(title="📋 配置摘要", show_header=True, header_style="bold magenta")
    table.add_column("配置项", style="cyan", width=15)
    table.add_column("值", style="white")

    table.add_row("环境配置", ", ".join(selected_envs))
    table.add_row("项目名称", project_name)
    table.add_row("输出目录", output_dir)
    table.add_row("根项目路径", root_project)

    console.print(table)

    # 确认执行
    if not inquirer.confirm(
        message="确认开始交付流程?",
        default=True
    ).execute():
        console.print("❌ [yellow]操作已取消[/yellow]")
        return

    # 运行交付流程
    success = script.run(
        env_names=selected_envs,
        project_name=project_name,
        output_dir=output_dir
    )

    if success:
        console.print("\n🎉 [green bold]交付完成![/green bold]")
        console.print(f"📁 项目已创建到: [cyan]{Path(output_dir).absolute()}[/cyan]")
    else:
        console.print("\n❌ [red bold]交付失败![/red bold]")
        sys.exit(1)


def main():
    """主函数入口"""
    interactive_main()


if __name__ == "__main__":
    main()
