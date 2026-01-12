#!/bin/bash
# Azure OpenAI Environment Setup Script
# Created: 2026-01-11

echo "=========================================="
echo "Azure OpenAI Environment Setup"
echo "=========================================="
echo ""

# Check if Azure CLI is installed
if command -v az &> /dev/null; then
    echo "✓ Azure CLI installed: $(az version --query '"azure-cli"' -o tsv)"

    # Check login status
    if az account show &> /dev/null; then
        ACCOUNT=$(az account show --query "user.name" -o tsv)
        SUBSCRIPTION=$(az account show --query "name" -o tsv)
        echo "✓ Logged in as: $ACCOUNT"
        echo "✓ Subscription: $SUBSCRIPTION"
    else
        echo "⚠ Not logged in to Azure"
        echo "  Run: az login"
    fi
else
    echo "❌ Azure CLI not installed"
    echo "  Install: https://docs.microsoft.com/en-us/cli/azure/install-azure-cli"
fi

echo ""
echo "==========================================

"
echo "Azure OpenAI Configuration"
echo "=========================================="
echo ""
echo "You need to:"
echo "1. Create Azure OpenAI resource (if not exists)"
echo "2. Deploy models (gpt-4o-mini, gpt-4.1, etc.)"
echo "3. Get endpoint and API key"
echo ""
echo "Then set environment variables:"
echo ""

# Prompt for Azure OpenAI credentials
read -p "Enter Azure OpenAI Endpoint (or press Enter to skip): " AZURE_ENDPOINT
if [ -n "$AZURE_ENDPOINT" ]; then
    export AZURE_OPENAI_ENDPOINT="$AZURE_ENDPOINT"
    echo "✓ AZURE_OPENAI_ENDPOINT set"

    # Optionally save to .zshrc/.bashrc
    read -p "Save to ~/.zshrc for permanent use? (y/n): " SAVE_PROFILE
    if [ "$SAVE_PROFILE" = "y" ] || [ "$SAVE_PROFILE" = "Y" ]; then
        echo "export AZURE_OPENAI_ENDPOINT=\"$AZURE_ENDPOINT\"" >> ~/.zshrc
        echo "✓ Saved to ~/.zshrc"
    fi
fi

read -p "Enter Azure OpenAI API Key (or press Enter to skip): " AZURE_KEY
if [ -n "$AZURE_KEY" ]; then
    export AZURE_OPENAI_API_KEY="$AZURE_KEY"
    echo "✓ AZURE_OPENAI_API_KEY set"

    # Optionally save to .zshrc/.bashrc
    read -p "Save to ~/.zshrc for permanent use? (y/n): " SAVE_PROFILE
    if [ "$SAVE_PROFILE" = "y" ] || [ "$SAVE_PROFILE" = "Y" ]; then
        echo "export AZURE_OPENAI_API_KEY=\"$AZURE_KEY\"" >> ~/.zshrc
        echo "✓ Saved to ~/.zshrc"
    fi
fi

echo ""
echo "=========================================="
echo "Current Environment Variables"
echo "=========================================="
echo ""

if [ -n "$AZURE_OPENAI_ENDPOINT" ]; then
    echo "✓ AZURE_OPENAI_ENDPOINT: $AZURE_OPENAI_ENDPOINT"
else
    echo "❌ AZURE_OPENAI_ENDPOINT not set"
fi

if [ -n "$AZURE_OPENAI_API_KEY" ]; then
    echo "✓ AZURE_OPENAI_API_KEY: ****${AZURE_OPENAI_API_KEY: -4}"
else
    echo "❌ AZURE_OPENAI_API_KEY not set"
fi

echo ""
echo "=========================================="
echo "Next Steps"
echo "=========================================="
echo ""
echo "1. If credentials are set, run a quick test:"
echo "   python -c 'from src.agents import LLMAgent; agent = LLMAgent(model=\"gpt-4o-mini\", provider=\"azure_openai\"); print(\"✓ Azure OpenAI configured!\")'"
echo ""
echo "2. Run the experiment:"
echo "   python scripts/run_llm_experiment.py --config configs/experiments/llm_eval_azure.yaml --limit 2"
echo ""
echo "3. Check results:"
echo "   cat outputs/experiments/llm_eval_azure_*/report.md"
echo ""
