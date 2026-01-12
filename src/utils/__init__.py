"""
Utils 模块 - 工具函数和通用辅助类

提供项目范围内的通用工具函数，包括配置加载、Azure认证等。

Research Direction: All Directions (Infrastructure)
Documentation: docs/modules/utils.md

Key Components:
    - AzureConfigLoader: Azure模型配置统一加载器，支持Azure OpenAI Service
                         和Azure AI Foundry两种部署方式

Example:
    >>> from src.utils import AzureConfigLoader
    >>> loader = AzureConfigLoader()
    >>> config = loader.get_deployment("gpt-4.1")
    >>> print(config['endpoint'])
"""

from .config_loader import AzureConfigLoader

__all__ = ['AzureConfigLoader']
