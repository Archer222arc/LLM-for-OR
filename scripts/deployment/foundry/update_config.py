#!/usr/bin/env python3
"""
Azure AI Foundry 配置文件自动更新脚本

功能:
    - 自动更新configs/models/azure_deployments.yaml
    - 将已部署的Foundry模型设置为 enabled: true
    - 备份原配置文件
    - 验证YAML语法正确

Usage:
    python scripts/update_foundry_config.py --enable-deployed

Created: 2026-01-11
"""

import sys
import argparse
import yaml
from pathlib import Path
from datetime import datetime
from typing import List

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# 配置文件路径
CONFIG_FILE = project_root / "configs" / "models" / "azure_deployments.yaml"

# Foundry模型列表
FOUNDRY_MODELS = [
    "claude-opus-4.5",
    "claude-sonnet-4.5",
    "claude-haiku-4.5",
    "deepseek-r1",
    "deepseek-v3.2",
    "qwen3-72b",
]


def backup_config(config_path: Path) -> Path:
    """
    备份配置文件

    Args:
        config_path: 配置文件路径

    Returns:
        Path: 备份文件路径
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = config_path.parent / f"{config_path.stem}.backup.{timestamp}{config_path.suffix}"

    # 复制文件
    import shutil
    shutil.copy2(config_path, backup_path)

    print(f"✓ 配置文件已备份到: {backup_path}")
    return backup_path


def load_yaml_preserving_format(file_path: Path) -> tuple:
    """
    加载YAML文件，保留格式

    Args:
        file_path: YAML文件路径

    Returns:
        tuple: (data, original_content)
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        data = yaml.safe_load(content)

    return data, content


def update_foundry_models_status(models_to_enable: List[str]) -> bool:
    """
    更新Foundry模型的enabled状态

    Args:
        models_to_enable: 要启用的模型键名列表

    Returns:
        bool: 是否成功更新
    """
    if not CONFIG_FILE.exists():
        print(f"❌ 配置文件不存在: {CONFIG_FILE}")
        print(f"请从模板复制: cp {CONFIG_FILE.parent}/azure_deployments_template.yaml {CONFIG_FILE}")
        return False

    # 备份配置文件
    backup_path = backup_config(CONFIG_FILE)

    try:
        # 读取配置文件（使用行级处理以保留格式）
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # 跟踪当前模型
        current_model = None
        updated_models = []
        modified_lines = []
        in_foundry_section = False

        for i, line in enumerate(lines):
            # 检查是否进入foundry配置块
            if 'foundry:' in line and not line.strip().startswith('#'):
                in_foundry_section = True

            # 检查是否退出foundry配置块（通过缩进判断）
            if in_foundry_section and line.strip() and not line.startswith(' ') and 'foundry:' not in line:
                in_foundry_section = False

            # 识别模型定义
            if in_foundry_section:
                for model_key in models_to_enable:
                    if f'{model_key}:' in line and line.strip().startswith(model_key):
                        current_model = model_key
                        break

                # 更新enabled状态
                if current_model in models_to_enable:
                    if 'enabled:' in line:
                        # 替换enabled值
                        indent = len(line) - len(line.lstrip())
                        new_line = ' ' * indent + 'enabled: true\n'

                        if line != new_line:
                            modified_lines.append(line)
                            updated_models.append(current_model)

                        line = new_line

            # 添加处理后的行
            lines[i] = line

        # 写回文件
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            f.writelines(lines)

        # 验证YAML语法
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            yaml.safe_load(f)

        # 输出更新结果
        print(f"\n✓ 配置文件已更新: {CONFIG_FILE}")

        if updated_models:
            print(f"\n已启用以下模型:")
            for model in set(updated_models):
                print(f"  - {model}")
        else:
            print("\n所有模型已经是enabled状态，无需更新")

        return True

    except Exception as e:
        print(f"\n❌ 更新失败: {e}")
        print(f"正在恢复备份...")

        # 恢复备份
        import shutil
        shutil.copy2(backup_path, CONFIG_FILE)
        print(f"✓ 已恢复原配置文件")

        return False


def enable_all_foundry_models():
    """启用所有Foundry模型"""
    print("正在启用所有Foundry模型...")
    print(f"模型列表: {', '.join(FOUNDRY_MODELS)}")

    success = update_foundry_models_status(FOUNDRY_MODELS)

    if success:
        print("\n✓ 所有Foundry模型已启用")
        print("\n下一步:")
        print("  运行验证脚本确认配置正确:")
        print("  python scripts/verify_foundry_deployment.py")


def enable_specific_models(models: List[str]):
    """启用指定模型"""
    # 验证模型键名
    invalid_models = [m for m in models if m not in FOUNDRY_MODELS]

    if invalid_models:
        print(f"❌ 无效的模型键名: {', '.join(invalid_models)}")
        print(f"有效的模型键名: {', '.join(FOUNDRY_MODELS)}")
        return

    print(f"正在启用指定模型: {', '.join(models)}")

    success = update_foundry_models_status(models)

    if success:
        print(f"\n✓ 已启用 {len(models)} 个模型")


def show_current_status():
    """显示当前配置状态"""
    if not CONFIG_FILE.exists():
        print(f"❌ 配置文件不存在: {CONFIG_FILE}")
        return

    try:
        from src.utils.config_loader import AzureConfigLoader

        loader = AzureConfigLoader()
        foundry_config = loader.config.get('foundry', {})
        deployments = foundry_config.get('deployments', {})

        print("Foundry模型配置状态:")
        print(f"{'模型键名':<25} {'启用状态':<10}")
        print("-" * 40)

        for model_key in FOUNDRY_MODELS:
            if model_key in deployments:
                enabled = deployments[model_key].get('enabled', False)
                status = "✓ enabled" if enabled else "✗ disabled"
                print(f"{model_key:<25} {status:<10}")
            else:
                print(f"{model_key:<25} {'❌ 不存在':<10}")

    except Exception as e:
        print(f"❌ 读取配置失败: {e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="更新Azure AI Foundry配置文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 启用所有Foundry模型
  python scripts/update_foundry_config.py --enable-deployed

  # 启用指定模型
  python scripts/update_foundry_config.py --enable claude-opus-4.5 deepseek-r1

  # 查看当前状态
  python scripts/update_foundry_config.py --status
        """
    )

    parser.add_argument(
        '--enable-deployed',
        action='store_true',
        help='启用所有已部署的Foundry模型'
    )

    parser.add_argument(
        '--enable',
        nargs='+',
        metavar='MODEL',
        help='启用指定的模型（模型键名，如 claude-opus-4.5）'
    )

    parser.add_argument(
        '--status',
        action='store_true',
        help='显示当前配置状态'
    )

    args = parser.parse_args()

    # 至少需要一个操作
    if not (args.enable_deployed or args.enable or args.status):
        parser.print_help()
        sys.exit(1)

    # 执行操作
    if args.status:
        show_current_status()

    elif args.enable_deployed:
        enable_all_foundry_models()

    elif args.enable:
        enable_specific_models(args.enable)


if __name__ == "__main__":
    main()
