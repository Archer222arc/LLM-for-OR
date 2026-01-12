# Utils 模块接口文档

## 概述

**模块定位**: 工具函数模块，提供项目范围内的通用工具。

**核心功能**:
- Azure 模型配置统一加载器
- 支持 Azure OpenAI Service 和 Azure AI Foundry 两种部署方式

**模块路径**: `src/utils/`

---

## 模块结构

```
src/utils/
├── __init__.py
└── config_loader.py    # Azure配置加载器
```

---

## AzureConfigLoader

**文件**: `src/utils/config_loader.py`

统一管理 Azure OpenAI Service 和 Azure AI Foundry 的模型配置。

### 初始化

```python
from src.utils.config_loader import AzureConfigLoader

# 使用默认配置文件
loader = AzureConfigLoader()

# 使用自定义配置文件
loader = AzureConfigLoader(config_path="/path/to/config.yaml")
```

### 核心方法

#### get_deployment()

获取 Azure OpenAI Service 部署配置。

```python
config = loader.get_deployment("gpt-4.1")
# 返回:
# {
#     'endpoint': 'https://xxx.cognitiveservices.azure.com/',
#     'api_key': 'xxx',
#     'api_version': '2024-10-21',
#     'deployment_name': 'gpt-4.1',
#     'model_name': 'gpt-4.1',
#     'provider_type': 'azure_openai',
#     'description': 'OpenAI GPT-4.1'
# }
```

#### get_foundry_deployment()

获取 Azure AI Foundry 部署配置。

```python
config = loader.get_foundry_deployment("deepseek-r1")
# 返回:
# {
#     'hub': 'eastus',
#     'hub_config': {...},
#     'model_id': 'DeepSeek.DeepSeek-R1',
#     'endpoint': 'https://xxx.models.ai.azure.com',
#     'api_key': 'xxx',  # 如果配置中有
#     'deployment_type': 'global_standard',
#     'provider_type': 'azure_foundry'
# }
```

#### get_foundry_auth_token()

获取 Azure AI Foundry 的 Bearer Token。

```python
token = loader.get_foundry_auth_token()
# 返回: Bearer token 字符串
```

**认证优先级**:
1. Service Principal (环境变量: `AZURE_TENANT_ID`, `AZURE_CLIENT_ID`, `AZURE_CLIENT_SECRET`)
2. Azure CLI 凭证 (`az login` 的当前用户)

#### list_deployments()

列出所有启用的 Azure OpenAI 部署。

```python
deployments = loader.list_deployments(enabled_only=True)
# 返回: {model_key: deployment_config} 字典
```

#### list_foundry_deployments()

列出所有启用的 Azure Foundry 部署。

```python
foundry_deployments = loader.list_foundry_deployments(enabled_only=True)
```

---

## 配置文件格式

配置文件路径: `configs/models/azure_deployments.yaml`

### Azure OpenAI Service 配置

```yaml
azure:
  default_endpoint: "https://xxx.cognitiveservices.azure.com/"
  default_api_key: "YOUR_API_KEY"
  api_version: "2024-10-21"

  deployments:
    gpt-4.1:
      deployment_name: "gpt-4.1"
      model_name: "gpt-4.1"
      provider_type: "azure_openai"
      region: "eastus2"
      enabled: true
      description: "OpenAI GPT-4.1"

    # O系列需要特殊API版本
    o1:
      deployment_name: "o1"
      model_name: "o1"
      provider_type: "azure_openai"
      region: "eastus2"
      enabled: true
      api_version: "2024-12-01-preview"  # 必须指定
      use_max_completion_tokens: true     # 使用 max_completion_tokens
```

### Azure AI Foundry 配置

```yaml
azure:
  foundry:
    auth:
      method: "service_principal"
      tenant_id: "${AZURE_TENANT_ID}"
      client_id: "${AZURE_CLIENT_ID}"
      client_secret: "${AZURE_CLIENT_SECRET}"

    hubs:
      eastus:
        resource_group: "rg-llm-for-or"
        workspace_name: "foundry-eastus"
        region: "eastus"

    deployments:
      deepseek-r1:
        provider_type: "azure_foundry"
        hub: "eastus"
        model_id: "DeepSeek.DeepSeek-R1"
        deployment_type: "global_standard"
        endpoint: "https://xxx.models.ai.azure.com"
        api_key: "YOUR_API_KEY"  # 可选
        enabled: true
```

---

## 可用模型列表 (2026-01-11)

### Azure AI Services (East US 2)

| 模型 | 部署名称 | 说明 |
|------|----------|------|
| gpt-4.1 | gpt-4.1 | GPT-4.1 最新版 |
| gpt-4.1-mini | gpt-4.1-mini | GPT-4.1 经济型 |
| gpt-5-mini | gpt-5-mini | GPT-5 mini |
| gpt-5-nano | gpt-5-nano | GPT-5 nano 超轻量 |
| gpt-5.2-chat | gpt-5.2-chat | GPT-5.2 最新版 |
| o1 | o1 | O1 推理模型 |
| o4-mini | o4-mini | O4 mini 推理模型 |
| DeepSeek-R1-0528 | DeepSeek-R1-0528 | DeepSeek R1 推理 |
| DeepSeek-V3.2 | DeepSeek-V3.2 | DeepSeek V3.2 通用 |
| Kimi-K2-Thinking | Kimi-K2-Thinking | Moonshot Kimi K2 |
| Llama-3.3-70B-Instruct | Llama-3.3-70B-Instruct | Meta Llama 3.3 |

### Azure AI Foundry (East US)

| 模型 | Endpoint | 说明 |
|------|----------|------|
| deepseek-r1 | llm4or-deepseek-r1 | DeepSeek R1 Foundry |

---

## API版本说明

| 模型类型 | API版本 | max_tokens参数 |
|----------|---------|----------------|
| GPT-4系列 | 2024-10-21 | max_tokens |
| GPT-5系列 | 2024-10-21 | max_completion_tokens |
| O系列 | 2024-12-01-preview | max_completion_tokens |
| DeepSeek | 2024-10-21 | max_tokens |
| Llama/Kimi | 2024-10-21 | max_tokens |

---

## 使用示例

### 与 LLMAgent 集成

```python
from src.agents import LLMAgent

# 使用 Azure OpenAI Service
agent = LLMAgent(
    model="gpt-4.1",
    provider="azure_openai",
    use_local_config=True
)

# 使用 Azure AI Foundry
agent = LLMAgent(
    model="deepseek-r1",
    provider="azure_foundry",
    use_local_config=True
)

# 执行推理
response = agent._call_llm("What is 2+2?")
```

### 直接使用配置

```python
from src.utils.config_loader import AzureConfigLoader

loader = AzureConfigLoader()

# 获取所有启用的模型
for key, config in loader.list_deployments().items():
    print(f"{key}: {config['endpoint']}")
```

---

## 错误处理

```python
# 配置文件不存在
try:
    loader = AzureConfigLoader()
except FileNotFoundError as e:
    print(f"请创建配置文件: {e}")

# 模型未启用
config = loader.get_deployment("disabled-model")
if config is None:
    print("模型未配置或未启用")

# 认证失败
try:
    token = loader.get_foundry_auth_token()
except RuntimeError as e:
    print(f"Azure认证失败: {e}")
```

---

## 扩展指南

添加新模型:

1. 在 `configs/models/azure_deployments.yaml` 中添加部署配置
2. 设置 `enabled: true`
3. 使用 `loader.get_deployment()` 或 `loader.get_foundry_deployment()` 获取配置

---

*最后更新: 2026-01-11*
