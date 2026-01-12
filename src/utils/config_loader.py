"""
Azure 模型配置统一加载器

支持 Azure OpenAI Service 和 Azure AI Foundry 两种部署方式的配置管理。
从本地YAML配置文件加载部署信息，支持环境变量覆盖敏感凭证。

Research Direction: All Directions (Infrastructure)
Documentation: docs/modules/config_loader.md

Supported Providers:
    - azure_openai: Azure OpenAI Service (GPT-4.1, GPT-5, o1, o4-mini等)
    - azure_foundry: Azure AI Foundry MaaS (DeepSeek, Claude, Qwen等)

Key Components:
    - AzureConfigLoader: 主配置加载类
    - get_deployment(): 获取Azure OpenAI部署配置
    - get_foundry_deployment(): 获取Azure Foundry部署配置
    - get_foundry_auth_token(): 获取Bearer Token认证

Authentication Priority:
    1. 环境变量 (AZURE_OPENAI_API_KEY, AZURE_TENANT_ID等)
    2. 本地配置文件 (configs/models/azure_deployments.yaml)

Example:
    >>> from src.utils.config_loader import AzureConfigLoader
    >>> loader = AzureConfigLoader()
    >>>
    >>> # Azure OpenAI Service
    >>> config = loader.get_deployment("gpt-4.1")
    >>> # {'endpoint': '...', 'api_key': '...', 'deployment_name': 'gpt-4.1'}
    >>>
    >>> # Azure AI Foundry
    >>> foundry_config = loader.get_foundry_deployment("deepseek-r1")
    >>> # {'endpoint': '...', 'api_key': '...', 'model_id': 'DeepSeek.DeepSeek-R1'}

Available Models (2026-01-11):
    Azure OpenAI: gpt-4.1, gpt-4.1-mini, gpt-5-mini, gpt-5-nano, gpt-5.2-chat,
                  o1, o4-mini, DeepSeek-R1-0528, DeepSeek-V3.2,
                  Kimi-K2-Thinking, Llama-3.3-70B-Instruct
    Azure Foundry: deepseek-r1
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Optional


class AzureConfigLoader:
    """Azure OpenAI 配置加载器"""

    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置加载器

        Args:
            config_path: 配置文件路径，默认为 configs/models/azure_deployments.yaml
        """
        if config_path is None:
            # 默认路径: 项目根目录/configs/models/azure_deployments.yaml
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "configs" / "models" / "azure_deployments.yaml"

        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> Dict:
        """加载YAML配置文件"""
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"配置文件不存在: {self.config_path}\n"
                f"请从 configs/models/azure_deployments_template.yaml 复制并填写实际API key"
            )

        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        if 'azure' not in config:
            raise ValueError(f"配置文件缺少 'azure' 顶级键: {self.config_path}")

        return config['azure']

    def get_endpoint(self, region: Optional[str] = None) -> str:
        """
        获取Azure端点URL

        优先级: 环境变量 > 配置文件 > 默认值

        Args:
            region: 指定区域（可选）

        Returns:
            Azure endpoint URL
        """
        # 1. 环境变量优先
        env_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        if env_endpoint:
            return env_endpoint

        # 2. 配置文件
        return self.config.get('default_endpoint', '')

    def get_api_key(self) -> str:
        """
        获取API密钥

        优先级: 环境变量 > 配置文件

        Returns:
            API key
        """
        # 1. 环境变量优先
        env_key = os.getenv("AZURE_OPENAI_API_KEY")
        if env_key:
            return env_key

        # 2. 配置文件
        key = self.config.get('default_api_key', '')
        if not key or key == "YOUR_API_KEY_HERE":
            raise ValueError(
                "未找到有效的Azure API密钥\n"
                "请设置环境变量 AZURE_OPENAI_API_KEY 或在配置文件中填写 default_api_key"
            )

        return key

    def get_api_version(self) -> str:
        """获取API版本"""
        return self.config.get('api_version', '2024-10-21')

    def get_deployment(self, model_key: str) -> Optional[Dict]:
        """
        获取指定模型的部署配置

        Args:
            model_key: 模型键名（如 'gpt-4o-mini', 'gpt-4-1'）

        Returns:
            部署配置字典，包含:
                - endpoint: Azure endpoint URL
                - api_key: API密钥
                - api_version: API版本
                - deployment_name: 部署名称
                - model_name: 模型名称
                - enabled: 是否启用
                如果模型不存在或未启用，返回 None
        """
        deployments = self.config.get('deployments', {})

        if model_key not in deployments:
            return None

        deployment = deployments[model_key]

        # 检查是否启用
        if not deployment.get('enabled', False):
            return None

        # 组合完整配置
        return {
            'endpoint': self.get_endpoint(deployment.get('region')),
            'api_key': self.get_api_key(),
            'api_version': deployment.get('api_version') or self.get_api_version(),
            'deployment_name': deployment['deployment_name'],
            'model_name': deployment['model_name'],
            'provider_type': deployment.get('provider_type', 'azure_openai'),
            'description': deployment.get('description', ''),
            'no_temperature': deployment.get('no_temperature', False),
            'use_max_completion_tokens': deployment.get('use_max_completion_tokens', False),
        }

    def list_deployments(self, enabled_only: bool = True) -> Dict[str, Dict]:
        """
        列出所有部署配置

        Args:
            enabled_only: 是否只返回启用的部署（默认True）

        Returns:
            {model_key: deployment_config} 字典
        """
        deployments = self.config.get('deployments', {})

        if enabled_only:
            return {
                key: self.get_deployment(key)
                for key, config in deployments.items()
                if config.get('enabled', False)
            }
        else:
            return {
                key: self.get_deployment(key)
                for key in deployments.keys()
            }

    def get_foundry_deployment(self, model_key: str) -> Optional[Dict]:
        """
        获取Azure AI Foundry部署配置

        Args:
            model_key: 模型键名（如 'claude-opus-4.5', 'deepseek-r1'）

        Returns:
            部署配置字典，包含:
                - hub: Hub名称
                - hub_config: Hub配置信息
                - model_id: Azure模型ID
                - endpoint: 推理端点URL
                - deployment_type: serverless或global_standard
                - auth_config: 认证配置
                - provider_type: 'azure_foundry'
                如果模型不存在或未启用，返回 None
        """
        foundry_config = self.config.get('foundry', {})
        deployments = foundry_config.get('deployments', {})

        if model_key not in deployments:
            return None

        deployment = deployments[model_key]

        # 检查是否启用
        if not deployment.get('enabled', False):
            return None

        # 获取Hub配置
        hub_name = deployment['hub']
        hubs = foundry_config.get('hubs', {})

        if hub_name not in hubs:
            raise ValueError(f"Hub '{hub_name}' not found in foundry.hubs configuration")

        hub_config = hubs[hub_name]

        # 构建endpoint URL (如果包含模板变量则替换，否则直接使用)
        endpoint = deployment['endpoint']
        if '{workspace_name}' in endpoint or '{region}' in endpoint:
            endpoint = endpoint.format(
                workspace_name=hub_config['workspace_name'],
                region=hub_config['region']
            )

        result = {
            'hub': hub_name,
            'hub_config': hub_config,
            'model_id': deployment['model_id'],
            'endpoint': endpoint,
            'deployment_type': deployment['deployment_type'],
            'auth_config': foundry_config.get('auth', {}),
            'provider_type': 'azure_foundry',
            'pricing': deployment.get('pricing', {}),
            'quota': deployment.get('quota', {}),
        }

        # 如果配置中有api_key，直接返回
        if deployment.get('api_key'):
            result['api_key'] = deployment['api_key']

        return result

    def get_foundry_auth_token(self) -> str:
        """
        获取Azure AI Foundry的Bearer Token

        认证优先级:
        1. Service Principal (环境变量 AZURE_TENANT_ID, AZURE_CLIENT_ID, AZURE_CLIENT_SECRET)
        2. Azure CLI 凭证 (az login 的当前用户)

        Returns:
            Bearer token字符串

        Raises:
            ImportError: azure-identity包未安装
            Exception: 所有认证方式都失败
        """
        try:
            from azure.identity import ClientSecretCredential, AzureCliCredential
        except ImportError:
            raise ImportError(
                "azure-identity package not installed. Run: pip install azure-identity"
            )

        foundry_config = self.config.get('foundry', {})
        auth_config = foundry_config.get('auth', {})

        # 方式1: 尝试使用Service Principal
        tenant_id = os.getenv('AZURE_TENANT_ID') or auth_config.get('tenant_id')
        client_id = os.getenv('AZURE_CLIENT_ID') or auth_config.get('client_id')
        client_secret = os.getenv('AZURE_CLIENT_SECRET') or auth_config.get('client_secret')

        if all([tenant_id, client_id, client_secret]) and not any(
            v.startswith('${') for v in [str(tenant_id), str(client_id), str(client_secret)]
        ):
            try:
                credential = ClientSecretCredential(
                    tenant_id=tenant_id,
                    client_id=client_id,
                    client_secret=client_secret
                )
                token = credential.get_token("https://cognitiveservices.azure.com/.default")
                return token.token
            except Exception as e:
                print(f"警告: Service Principal认证失败 ({e}), 尝试Azure CLI凭证...")

        # 方式2: 使用Azure CLI凭证 (当前 az login 用户)
        try:
            credential = AzureCliCredential()
            token = credential.get_token("https://cognitiveservices.azure.com/.default")
            return token.token
        except Exception as e:
            raise RuntimeError(
                f"Azure认证失败\n"
                f"请确保已运行 'az login' 登录Azure，或设置Service Principal环境变量\n"
                f"错误: {e}"
            )

    def list_foundry_deployments(self, enabled_only: bool = True) -> Dict[str, Dict]:
        """
        列出所有Azure AI Foundry部署配置

        Args:
            enabled_only: 是否只返回启用的部署（默认True）

        Returns:
            {model_key: deployment_config} 字典
        """
        foundry_config = self.config.get('foundry', {})
        deployments = foundry_config.get('deployments', {})

        if enabled_only:
            return {
                key: self.get_foundry_deployment(key)
                for key, config in deployments.items()
                if config.get('enabled', False)
            }
        else:
            return {
                key: self.get_foundry_deployment(key)
                for key in deployments.keys()
            }
