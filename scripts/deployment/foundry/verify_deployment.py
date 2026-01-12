#!/usr/bin/env python3
"""
验证Azure AI Foundry部署状态

Usage:
    python scripts/verify_foundry_deployment.py

Prerequisites:
    - Azure AI Foundry Hubs已创建
    - 模型已通过Portal部署
    - 环境变量已设置: AZURE_TENANT_ID, AZURE_CLIENT_ID, AZURE_CLIENT_SECRET
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config_loader import AzureConfigLoader
from src.agents.llm_agent import LLMAgent


def verify_deployment(model_key: str) -> bool:
    """
    验证单个Foundry模型部署

    Args:
        model_key: 模型键名（如 'claude-opus-4.5', 'deepseek-r1'）

    Returns:
        bool: 验证是否成功
    """
    print(f"\n{'='*60}")
    print(f"验证模型: {model_key}")
    print(f"{'='*60}")

    try:
        # 1. 检查配置加载
        print("\n[1/3] 检查配置...")
        loader = AzureConfigLoader()
        config = loader.get_foundry_deployment(model_key)

        if config is None:
            print(f"❌ 配置不存在或未启用")
            print(f"   请检查: configs/models/azure_deployments.yaml")
            print(f"   确保模型已配置且 enabled: true")
            return False

        print(f"✓ 配置加载成功")
        print(f"  Hub: {config['hub']}")
        print(f"  Model ID: {config['model_id']}")
        print(f"  Endpoint: {config['endpoint']}")

        # 2. 检查认证
        print("\n[2/3] 检查认证...")
        token = loader.get_foundry_auth_token()
        print(f"✓ 认证成功")
        print(f"  Token前缀: {token[:20]}...")

        # 3. 测试推理调用
        print("\n[3/3] 测试推理...")
        agent = LLMAgent(
            model=model_key,
            provider="azure_foundry",
            temperature=0.0
        )

        test_message = "What is 2+2? Answer with just the number."
        response = agent._call_llm(test_message)

        print(f"✓ 推理成功")
        print(f"  测试问题: {test_message}")
        print(f"  模型回答: {response.strip()}")

        return True

    except FileNotFoundError as e:
        print(f"❌ 配置文件错误: {e}")
        return False

    except ValueError as e:
        print(f"❌ 配置验证失败: {e}")
        return False

    except ImportError as e:
        print(f"❌ 依赖缺失: {e}")
        print(f"   请运行: pip install azure-identity azure-ai-ml requests")
        return False

    except Exception as e:
        print(f"❌ 验证失败: {type(e).__name__}")
        print(f"   错误信息: {e}")
        return False


def main():
    """主函数：验证所有Foundry模型"""
    print("="*60)
    print("Azure AI Foundry 部署验证")
    print("="*60)

    # 检查环境变量
    import os
    required_env_vars = ['AZURE_TENANT_ID', 'AZURE_CLIENT_ID', 'AZURE_CLIENT_SECRET']
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]

    if missing_vars:
        print(f"\n❌ 缺少必需的环境变量: {', '.join(missing_vars)}")
        print(f"\n请设置环境变量:")
        print(f"  export AZURE_TENANT_ID='your-tenant-id'")
        print(f"  export AZURE_CLIENT_ID='your-client-id'")
        print(f"  export AZURE_CLIENT_SECRET='your-client-secret'")
        return 1

    # 要验证的模型列表
    models_to_verify = [
        # Claude系列 (East US 2)
        "claude-opus-4.5",
        "claude-sonnet-4.5",
        "claude-haiku-4.5",
        # DeepSeek系列 (East US)
        "deepseek-r1",
        "deepseek-v3.2",
        # Qwen系列 (Sweden Central)
        "qwen3-72b",
    ]

    # 逐个验证
    results = {}
    for model in models_to_verify:
        results[model] = verify_deployment(model)

    # 汇总报告
    print("\n" + "="*60)
    print("验证汇总")
    print("="*60)

    for model, success in results.items():
        status = "✓ 成功" if success else "❌ 失败"
        print(f"{model:25s} {status}")

    # 统计
    passed = sum(results.values())
    total = len(results)
    pass_rate = (passed / total * 100) if total > 0 else 0

    print(f"\n总计: {passed}/{total} 通过 ({pass_rate:.1f}%)")

    if passed == total:
        print("\n🎉 所有模型验证通过！")
        return 0
    elif passed > 0:
        print(f"\n⚠️  部分模型验证失败，请检查上述错误信息")
        return 1
    else:
        print(f"\n❌ 所有模型验证失败，请检查:")
        print(f"   1. Azure AI Foundry Hubs是否已创建")
        print(f"   2. 模型是否已通过Portal部署")
        print(f"   3. 配置文件中模型是否已启用 (enabled: true)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
