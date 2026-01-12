#!/usr/bin/env python3
"""
Test Azure OpenAI connection

Usage:
    python scripts/test_azure_connection.py
"""

import os
import sys

# Add project root to path
from pathlib import Path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

print("="*60)
print("Testing Azure OpenAI Connection")
print("="*60)
print()

# Check environment variables
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_API_KEY")

print("1. Environment Variables Check:")
if endpoint:
    print(f"   ✓ AZURE_OPENAI_ENDPOINT: {endpoint}")
else:
    print("   ❌ AZURE_OPENAI_ENDPOINT not set")

if api_key:
    print(f"   ✓ AZURE_OPENAI_API_KEY: ****{api_key[-4:]}")
else:
    print("   ❌ AZURE_OPENAI_API_KEY not set")

if not endpoint or not api_key:
    print("\n❌ Missing environment variables. Set them with:")
    print('   export AZURE_OPENAI_ENDPOINT="https://llm-for-or-openai.openai.azure.com/"')
    print('   export AZURE_OPENAI_API_KEY="your-api-key"')
    sys.exit(1)

print("\n2. Testing Azure OpenAI SDK:")
try:
    from openai import AzureOpenAI
    print("   ✓ OpenAI SDK imported successfully")
except ImportError:
    print("   ❌ OpenAI SDK not installed. Run: pip install openai")
    sys.exit(1)

print("\n3. Creating Azure OpenAI client:")
try:
    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version="2024-10-21"
    )
    print("   ✓ Client created successfully")
except Exception as e:
    print(f"   ❌ Failed to create client: {e}")
    sys.exit(1)

print("\n4. Testing simple API call:")
try:
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # Deployment name
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say 'Azure OpenAI is working!' in one line."}
        ],
        max_tokens=50,
        temperature=0.0
    )

    result = response.choices[0].message.content
    print(f"   ✓ API call successful!")
    print(f"   Response: {result}")

except Exception as e:
    print(f"   ❌ API call failed: {e}")
    print(f"\n   Debug info:")
    print(f"   - Endpoint: {endpoint}")
    print(f"   - Deployment: gpt-4o-mini")
    print(f"   - API Version: 2024-10-21")
    sys.exit(1)

print("\n5. Testing LLMAgent integration:")
try:
    from src.agents import LLMAgent

    agent = LLMAgent(
        model="gpt-4o-mini",
        provider="azure_openai",
        azure_deployment="gpt-4o-mini",
        name="test-agent"
    )
    print("   ✓ LLMAgent created successfully")
    print(f"   Agent name: {agent.name}")
    print(f"   Provider: {agent.provider}")
    print(f"   Deployment: {agent.azure_deployment}")

except Exception as e:
    print(f"   ❌ LLMAgent creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("✅ All tests passed! Azure OpenAI is ready.")
print("="*60)
print("\nNext steps:")
print("1. Run quick experiment:")
print("   python scripts/run_llm_experiment.py --config configs/experiments/llm_eval_azure.yaml --limit 2")
print("\n2. Run full experiment:")
print("   python scripts/run_llm_experiment.py --config configs/experiments/llm_eval_azure.yaml")
