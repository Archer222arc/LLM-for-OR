"""
LLM-based debugging agent for OR-Debug-Bench.

基于大语言模型的优化模型调试代理，支持多种LLM提供商和模型。
通过分析求解器状态和反馈，自动选择调试动作修复不可行问题。

Research Direction: Direction A (OR-Debug-Bench)
Documentation: docs/directions/A_OR_Debug_Bench/A2_MDP_Spec.md

Key Components:
    - LLMAgent: 主代理类，使用LLM进行动作选择
    - _call_openai(): OpenAI API调用
    - _call_anthropic(): Anthropic API调用
    - _call_azure_openai(): Azure OpenAI Service调用
    - _call_azure_foundry(): Azure AI Foundry调用
    - _parse_response(): 解析LLM响应为Action

Supported Providers:
    - openai: 直接OpenAI API
    - anthropic: 直接Anthropic API
    - azure_openai: Azure OpenAI Service (推荐用于生产环境)
    - azure_foundry: Azure AI Foundry MaaS

Available Models (2026-01-11):
    Azure OpenAI Service:
        - GPT系列: gpt-4.1, gpt-4.1-mini, gpt-5-mini, gpt-5-nano, gpt-5.2-chat
        - O系列: o1, o4-mini (推理模型)
        - DeepSeek: DeepSeek-R1-0528, DeepSeek-V3.2
        - 其他: Kimi-K2-Thinking, Llama-3.3-70B-Instruct

    Azure AI Foundry:
        - deepseek-r1 (via Serverless Endpoint)

Example:
    >>> from src.agents import LLMAgent
    >>>
    >>> # 使用Azure OpenAI Service (推荐)
    >>> agent = LLMAgent(
    ...     model="gpt-4.1",
    ...     provider="azure_openai",
    ...     use_local_config=True
    ... )
    >>>
    >>> # 使用Azure AI Foundry
    >>> agent = LLMAgent(
    ...     model="deepseek-r1",
    ...     provider="azure_foundry",
    ...     use_local_config=True
    ... )
    >>>
    >>> # 执行动作
    >>> action = agent.act(state)
"""

import json
import os
import re
from datetime import datetime
from typing import Optional, Dict, Any, List

from src.environments import DebugState, Action, ActionType
from src.evaluation.metrics import TokenUsage
from .base_agent import BaseAgent
from .prompts import SYSTEM_PROMPT, format_state, normalize_action_name


class LLMAgent(BaseAgent):
    """
    LLM-based debugging agent.

    Uses a large language model to analyze the optimization model state
    and select debugging actions. Supports multiple LLM providers.

    Supported Providers:
        - openai: Direct OpenAI API (GPT-4, GPT-3.5-turbo, etc.)
        - anthropic: Direct Anthropic API (Claude-3 Sonnet, Opus, Haiku)
        - azure_openai: Azure OpenAI Service (GPT-4, GPT-5, o1, etc.)
        - azure_foundry: Azure AI Foundry (Claude, DeepSeek, Qwen, etc.)

    Attributes:
        model: Model name (e.g., "gpt-4", "claude-3-sonnet", "deepseek-r1")
        provider: LLM provider ("openai", "anthropic", "azure_openai", or "azure_foundry")
        temperature: Sampling temperature (0.0 = deterministic)
        azure_endpoint: Azure endpoint (for azure_openai provider)
        azure_deployment: Azure deployment name (for azure_openai provider)
        foundry_endpoint: Azure Foundry endpoint (for azure_foundry provider)
        foundry_model_id: Azure Foundry model ID (for azure_foundry provider)
    """

    def __init__(
        self,
        model: str = "gpt-4",
        provider: str = "openai",
        temperature: float = 0.0,
        max_retries: int = 3,
        name: Optional[str] = None,
        # Azure OpenAI specific parameters
        azure_endpoint: Optional[str] = None,
        api_version: str = "2024-10-21",
        azure_deployment: Optional[str] = None,
        # Azure Foundry specific parameters
        foundry_endpoint: Optional[str] = None,
        foundry_model_id: Optional[str] = None,
        # Local config support
        use_local_config: bool = True,
        config_path: Optional[str] = None,
    ):
        """
        Initialize LLMAgent.

        Args:
            model: Model name to use
            provider: LLM provider ("openai", "anthropic", "azure_openai", or "azure_foundry")
            temperature: Sampling temperature
            max_retries: Number of retries on parse failure
            name: Agent name
            azure_endpoint: Azure OpenAI endpoint (e.g., https://xxx.openai.azure.com/)
            api_version: Azure API version (default: 2024-10-21)
            azure_deployment: Azure deployment name (defaults to model if not specified)
            foundry_endpoint: Azure Foundry endpoint (e.g., https://xxx.inference.ai.azure.com/)
            foundry_model_id: Azure Foundry model ID (e.g., Anthropic.Claude-Opus-4-5)
            use_local_config: Use local config file for Azure (default: True)
            config_path: Custom config file path (optional)
        """
        super().__init__(name=name or f"LLMAgent-{model}")
        self.model = model
        self.provider = provider.lower()
        self.temperature = temperature
        self.max_retries = max_retries

        # Azure OpenAI configuration
        if self.provider == "azure_openai":
            if use_local_config:
                # Load from local config file
                from src.utils.config_loader import AzureConfigLoader

                loader = AzureConfigLoader(config_path)
                deployment_config = loader.get_deployment(model)

                if deployment_config is None:
                    raise ValueError(
                        f"Model '{model}' not found or not enabled in config file\n"
                        f"Please check configs/models/azure_deployments.yaml"
                    )

                # Use config from file
                self.azure_endpoint = deployment_config['endpoint']
                self.api_version = deployment_config['api_version']
                self.azure_deployment = deployment_config['deployment_name']
                self.azure_api_key = deployment_config.get('api_key')  # Save API key from config
                self.no_temperature = deployment_config.get('no_temperature', False)
                self.use_max_completion_tokens = deployment_config.get('use_max_completion_tokens', False)

                print(f"✓ Loaded Azure config from local file: {self.azure_deployment} ({deployment_config['model_name']})")

            else:
                # Use environment variables or explicit parameters
                self.azure_endpoint = azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
                self.api_version = api_version
                self.azure_deployment = azure_deployment or model
                self.azure_api_key = os.environ.get("AZURE_OPENAI_API_KEY")  # From env var
                self.no_temperature = False
                self.use_max_completion_tokens = False

                if not self.azure_endpoint:
                    raise ValueError(
                        "Azure OpenAI requires AZURE_OPENAI_ENDPOINT\n"
                        "Option 1: Set environment variable\n"
                        "Option 2: Use use_local_config=True (recommended)"
                    )

        # Azure Foundry configuration
        elif self.provider == "azure_foundry":
            if use_local_config:
                # Load from local config file
                from src.utils.config_loader import AzureConfigLoader

                loader = AzureConfigLoader(config_path)
                foundry_config = loader.get_foundry_deployment(model)

                if foundry_config is None:
                    raise ValueError(
                        f"Model '{model}' not found or not enabled in Foundry config\n"
                        f"Please check configs/models/azure_deployments.yaml"
                    )

                # Use config from file
                self.foundry_endpoint = foundry_config['endpoint']
                self.foundry_model_id = foundry_config['model_id']
                self.foundry_hub = foundry_config['hub']
                self.foundry_api_key = foundry_config.get('api_key')  # API key (if available)
                self._foundry_loader = loader  # Save loader for token retrieval

                print(f"✓ Loaded Azure Foundry config: {self.foundry_model_id} from {self.foundry_hub}")

            else:
                # Use explicit parameters
                self.foundry_endpoint = foundry_endpoint
                self.foundry_model_id = foundry_model_id or model
                self.foundry_api_key = None  # Must use Bearer token without config
                self._foundry_loader = None

                if not self.foundry_endpoint:
                    raise ValueError(
                        "Azure Foundry requires foundry_endpoint\n"
                        "Option 1: Provide foundry_endpoint parameter\n"
                        "Option 2: Use use_local_config=True (recommended)"
                    )

            # For Foundry, set Azure OpenAI params to None
            self.azure_endpoint = None
            self.api_version = None
            self.azure_deployment = None
            self.azure_api_key = None
            self.no_temperature = False
            self.use_max_completion_tokens = False

        else:
            # For non-Azure providers, set these to None
            self.azure_endpoint = None
            self.api_version = None
            self.azure_deployment = None
            self.azure_api_key = None
            self.no_temperature = False
            self.use_max_completion_tokens = False
            self.foundry_endpoint = None
            self.foundry_model_id = None
            self._foundry_loader = None

        self._client = None
        self._conversation_history: List[Dict[str, str]] = []

        # Test-Time Compute tracking
        self._last_token_usage: Optional[TokenUsage] = None
        self._total_tokens: int = 0
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0
        self._total_reasoning_tokens: int = 0
        self._api_calls: List[Dict[str, Any]] = []
        self._tokens_per_step: List[int] = []

    def act(self, state: DebugState) -> Action:
        """
        Select an action using the LLM.

        Args:
            state: Current environment state

        Returns:
            Action selected by the LLM
        """
        # Format state as user message
        user_message = format_state(state)

        # Try to get valid action from LLM
        for attempt in range(self.max_retries):
            try:
                response = self._call_llm(user_message)
                action = self._parse_response(response)
                return action
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                if attempt == self.max_retries - 1:
                    # Fallback to GET_IIS on failure
                    return Action(ActionType.GET_IIS)
                # Add error context for retry
                user_message = (
                    f"{format_state(state)}\n\n"
                    f"Previous response was invalid: {e}\n"
                    "Please provide a valid JSON response."
                )

        return Action(ActionType.GET_IIS)

    def reset(self) -> None:
        """Reset agent state."""
        self.clear_history()
        self._conversation_history = []
        self.reset_token_stats()

    def reset_token_stats(self) -> None:
        """Reset token tracking statistics (called at start of each episode)."""
        self._last_token_usage = None
        self._total_tokens = 0
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_reasoning_tokens = 0
        self._api_calls = []
        self._tokens_per_step = []

    def _record_api_call(self, usage: TokenUsage) -> None:
        """
        Record an API call's token usage.

        Args:
            usage: TokenUsage object from the API call
        """
        self._last_token_usage = usage
        self._total_tokens += usage.total_tokens
        self._total_input_tokens += usage.input_tokens
        self._total_output_tokens += usage.output_tokens
        self._total_reasoning_tokens += usage.reasoning_tokens
        self._api_calls.append(usage.to_dict())
        self._tokens_per_step.append(usage.total_tokens)

    def get_token_stats(self) -> Dict[str, Any]:
        """
        Get cumulative token statistics for this episode.

        Returns:
            Dictionary with token usage statistics
        """
        return {
            'total_tokens': self._total_tokens,
            'total_input_tokens': self._total_input_tokens,
            'total_output_tokens': self._total_output_tokens,
            'total_reasoning_tokens': self._total_reasoning_tokens,
            'api_call_count': len(self._api_calls),
            'tokens_per_step': list(self._tokens_per_step),
            'api_calls': list(self._api_calls),
        }

    def _call_llm(self, user_message: str) -> str:
        """
        Call the LLM API.

        Args:
            user_message: User message to send

        Returns:
            LLM response text
        """
        if self.provider == "openai":
            return self._call_openai(user_message)
        elif self.provider == "anthropic":
            return self._call_anthropic(user_message)
        elif self.provider == "azure_openai":
            return self._call_azure_openai(user_message)
        elif self.provider == "azure_foundry":
            return self._call_azure_foundry(user_message)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _call_openai(self, user_message: str) -> str:
        """
        Call OpenAI API.

        Args:
            user_message: User message

        Returns:
            Response text
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "OpenAI package not installed. Run: pip install openai"
            )

        if self._client is None:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            self._client = OpenAI(api_key=api_key)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]
        messages.extend(self._conversation_history)
        messages.append({"role": "user", "content": user_message})

        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
        )

        assistant_message = response.choices[0].message.content

        # Extract and record token usage
        if hasattr(response, 'usage') and response.usage:
            usage = TokenUsage(
                input_tokens=response.usage.prompt_tokens or 0,
                output_tokens=response.usage.completion_tokens or 0,
                total_tokens=response.usage.total_tokens or 0,
                reasoning_tokens=0,  # OpenAI doesn't expose this for non-o1 models
                model=self.model,
                provider='openai',
                timestamp=datetime.now().isoformat(),
            )
            self._record_api_call(usage)

        # Update conversation history
        self._conversation_history.append({"role": "user", "content": user_message})
        self._conversation_history.append({"role": "assistant", "content": assistant_message})

        return assistant_message

    def _call_anthropic(self, user_message: str) -> str:
        """
        Call Anthropic API.

        Args:
            user_message: User message

        Returns:
            Response text
        """
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError(
                "Anthropic package not installed. Run: pip install anthropic"
            )

        if self._client is None:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY environment variable not set")
            self._client = Anthropic(api_key=api_key)

        messages = list(self._conversation_history)
        messages.append({"role": "user", "content": user_message})

        response = self._client.messages.create(
            model=self.model,
            system=SYSTEM_PROMPT,
            messages=messages,
            temperature=self.temperature,
            max_tokens=1024,
        )

        assistant_message = response.content[0].text

        # Extract and record token usage (Anthropic uses different field names)
        if hasattr(response, 'usage') and response.usage:
            usage = TokenUsage(
                input_tokens=response.usage.input_tokens or 0,
                output_tokens=response.usage.output_tokens or 0,
                total_tokens=(response.usage.input_tokens or 0) + (response.usage.output_tokens or 0),
                reasoning_tokens=0,
                model=self.model,
                provider='anthropic',
                timestamp=datetime.now().isoformat(),
            )
            self._record_api_call(usage)

        # Update conversation history
        self._conversation_history.append({"role": "user", "content": user_message})
        self._conversation_history.append({"role": "assistant", "content": assistant_message})

        return assistant_message

    def _call_azure_openai(self, user_message: str) -> str:
        """
        Call Azure OpenAI API.

        Azure OpenAI uses the same SDK as OpenAI but with different configuration:
        - Custom endpoint (azure_endpoint)
        - API version parameter
        - Deployment name instead of model name

        Args:
            user_message: User message

        Returns:
            Response text
        """
        try:
            from openai import AzureOpenAI
        except ImportError:
            raise ImportError(
                "OpenAI package not installed. Run: pip install openai"
            )

        if self._client is None:
            # Use API key from config (loaded in __init__) or env var
            api_key = getattr(self, 'azure_api_key', None) or os.environ.get("AZURE_OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "Azure OpenAI API key not found.\n"
                    "Option 1: Set AZURE_OPENAI_API_KEY environment variable\n"
                    "Option 2: Configure api_key in configs/models/azure_deployments.yaml"
                )

            # Create HTTP client without any proxy to avoid network issues
            # Azure endpoints should be accessed directly
            import httpx

            # Backup and temporarily remove all proxy settings
            proxy_vars = ['ALL_PROXY', 'all_proxy', 'HTTP_PROXY', 'http_proxy', 'HTTPS_PROXY', 'https_proxy']
            proxy_backup = {var: os.environ.pop(var, None) for var in proxy_vars}
            try:
                http_client = httpx.Client(timeout=60.0)  # 60 second timeout
            finally:
                # Restore proxy settings
                for var, value in proxy_backup.items():
                    if value is not None:
                        os.environ[var] = value

            self._client = AzureOpenAI(
                azure_endpoint=self.azure_endpoint,
                api_key=api_key,
                api_version=self.api_version,
                http_client=http_client,
            )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]
        messages.extend(self._conversation_history)
        messages.append({"role": "user", "content": user_message})

        try:
            # Build request parameters
            request_params = {
                "model": self.azure_deployment,  # Use deployment name
                "messages": messages,
            }

            # Add temperature if model supports it
            if not getattr(self, 'no_temperature', False):
                request_params["temperature"] = self.temperature

            # Use max_completion_tokens for newer models
            if getattr(self, 'use_max_completion_tokens', False):
                request_params["max_completion_tokens"] = 2048
            else:
                request_params["max_tokens"] = 2048

            response = self._client.chat.completions.create(**request_params)
        except Exception as e:
            raise RuntimeError(f"Azure OpenAI API call failed: {e}")

        assistant_message = response.choices[0].message.content

        # Extract and record token usage
        if hasattr(response, 'usage') and response.usage:
            # Handle reasoning tokens for o1/o4-mini models
            # These models report completion_tokens_details.reasoning_tokens
            reasoning_tokens = 0
            if hasattr(response.usage, 'completion_tokens_details'):
                details = response.usage.completion_tokens_details
                if details and hasattr(details, 'reasoning_tokens'):
                    reasoning_tokens = details.reasoning_tokens or 0

            usage = TokenUsage(
                input_tokens=response.usage.prompt_tokens or 0,
                output_tokens=response.usage.completion_tokens or 0,
                total_tokens=response.usage.total_tokens or 0,
                reasoning_tokens=reasoning_tokens,
                model=self.model,
                provider='azure_openai',
                timestamp=datetime.now().isoformat(),
            )
            self._record_api_call(usage)

        # Update conversation history
        self._conversation_history.append({"role": "user", "content": user_message})
        self._conversation_history.append({"role": "assistant", "content": assistant_message})

        return assistant_message

    def _call_azure_foundry(self, user_message: str) -> str:
        """
        Call Azure AI Foundry API.

        Azure AI Foundry supports two authentication methods:
        1. API Key (preferred if available in config)
        2. Bearer Token (Azure Entra ID / Azure CLI)

        Args:
            user_message: User message

        Returns:
            Response text
        """
        try:
            import requests
        except ImportError:
            raise ImportError(
                "requests package not installed. Run: pip install requests"
            )

        # Build request headers based on available authentication
        if hasattr(self, 'foundry_api_key') and self.foundry_api_key:
            # Use API Key authentication
            headers = {
                "Authorization": f"Bearer {self.foundry_api_key}",
                "Content-Type": "application/json",
            }
        else:
            # Use Bearer Token authentication
            if self._foundry_loader:
                token = self._foundry_loader.get_foundry_auth_token()
            else:
                # If no loader, try to get token from environment
                from src.utils.config_loader import AzureConfigLoader
                loader = AzureConfigLoader()
                token = loader.get_foundry_auth_token()

            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            }

        # Build messages
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]
        messages.extend(self._conversation_history)
        messages.append({"role": "user", "content": user_message})

        # Build payload
        payload = {
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": 2048,  # Azure Foundry recommended setting
        }

        # Call API
        url = f"{self.foundry_endpoint}/v1/chat/completions"

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(
                f"Azure Foundry API call failed: {e}\n"
                f"Endpoint: {url}\n"
                f"Model: {self.foundry_model_id}"
            )

        # Parse response (OpenAI-compatible format)
        data = response.json()
        assistant_message = data['choices'][0]['message']['content']

        # Extract and record token usage from response
        if 'usage' in data:
            usage_data = data['usage']
            usage = TokenUsage(
                input_tokens=usage_data.get('prompt_tokens', 0),
                output_tokens=usage_data.get('completion_tokens', 0),
                total_tokens=usage_data.get('total_tokens', 0),
                reasoning_tokens=0,  # Foundry may not expose this
                model=self.foundry_model_id or self.model,
                provider='azure_foundry',
                timestamp=datetime.now().isoformat(),
            )
            self._record_api_call(usage)

        # Update conversation history
        self._conversation_history.append({"role": "user", "content": user_message})
        self._conversation_history.append({"role": "assistant", "content": assistant_message})

        return assistant_message

    def _parse_response(self, response: str) -> Action:
        """
        Parse LLM response into an Action.

        Args:
            response: LLM response text

        Returns:
            Parsed Action object

        Raises:
            ValueError: If response cannot be parsed
        """
        # Try to extract JSON from response
        json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if not json_match:
            raise ValueError("No JSON object found in response")

        data = json.loads(json_match.group())

        if "action" not in data:
            raise ValueError("Response missing 'action' field")

        action_name = normalize_action_name(data["action"])

        # Map to ActionType
        action_map = {
            "get_iis": ActionType.GET_IIS,
            "check_slack": ActionType.CHECK_SLACK,
            "drop_constraint": ActionType.DROP_CONSTRAINT,
            "relax_constraint": ActionType.RELAX_CONSTRAINT,
            "update_rhs": ActionType.UPDATE_RHS,
            "update_bounds": ActionType.UPDATE_BOUNDS,
            "reset": ActionType.RESET,
            "submit": ActionType.SUBMIT,
        }

        if action_name not in action_map:
            raise ValueError(f"Unknown action: {action_name}")

        action_type = action_map[action_name]

        # Extract parameters
        target = data.get("target")
        value = data.get("value")
        value2 = data.get("value2")

        # Validate required parameters
        if action_type.requires_target and not target:
            raise ValueError(f"Action {action_name} requires 'target' parameter")

        if action_type.requires_value and value is None:
            raise ValueError(f"Action {action_name} requires 'value' parameter")

        return Action(
            action_type=action_type,
            target=target,
            value=float(value) if value is not None else None,
            value2=float(value2) if value2 is not None else None,
        )


class MockLLMAgent(BaseAgent):
    """
    Mock LLM agent for testing without API calls.

    Uses predefined responses or follows a simple heuristic.
    """

    def __init__(
        self,
        responses: Optional[List[Dict[str, Any]]] = None,
        name: str = "MockLLMAgent",
    ):
        """
        Initialize MockLLMAgent.

        Args:
            responses: Predefined list of response dicts
            name: Agent name
        """
        super().__init__(name=name)
        self._responses = responses or []
        self._response_index = 0

    def act(self, state: DebugState) -> Action:
        """
        Return next predefined action or use heuristic.

        Args:
            state: Current environment state

        Returns:
            Action
        """
        if self._response_index < len(self._responses):
            data = self._responses[self._response_index]
            self._response_index += 1

            action_map = {
                "get_iis": ActionType.GET_IIS,
                "drop_constraint": ActionType.DROP_CONSTRAINT,
                "submit": ActionType.SUBMIT,
            }

            action_name = data.get("action", "get_iis").lower()
            action_type = action_map.get(action_name, ActionType.GET_IIS)

            return Action(
                action_type=action_type,
                target=data.get("target"),
                value=data.get("value"),
            )

        # Fallback heuristic
        if state.is_optimal():
            return Action(ActionType.SUBMIT)
        elif state.iis_constraints:
            return Action(ActionType.DROP_CONSTRAINT, target=state.iis_constraints[0])
        else:
            return Action(ActionType.GET_IIS)

    def reset(self) -> None:
        """Reset agent state."""
        self.clear_history()
        self._response_index = 0
