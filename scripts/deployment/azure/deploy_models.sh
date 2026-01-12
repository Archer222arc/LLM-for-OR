#!/bin/bash
# Azure Models Batch Deployment Script
# Created: 2026-01-11

RESOURCE_NAME="llm-for-or-openai"
RESOURCE_GROUP="rg-llm-research"

echo "========================================"
echo "Azure Models Batch Deployment"
echo "========================================"
echo ""

# 检查已部署模型
echo "[1/3] 检查已部署模型..."
EXISTING=$(az cognitiveservices account deployment list \
    --name $RESOURCE_NAME \
    --resource-group $RESOURCE_GROUP \
    --query "[].name" -o tsv)

echo "已部署: $(echo $EXISTING | tr '\n' ', ' | sed 's/, $//')"
echo ""

# 批量部署
echo "[2/3] 开始批量部署..."
SUCCESS_COUNT=0
SKIP_COUNT=0
FAIL_COUNT=0

# 函数: 部署单个模型
deploy_model() {
    local DEPLOYMENT_NAME="$1"
    local MODEL_NAME="$2"
    local VERSION="$3"
    local KIND="$4"

    # 检查是否已部署
    if echo "$EXISTING" | grep -q "^${DEPLOYMENT_NAME}$"; then
        echo "  ⏭️  跳过 $DEPLOYMENT_NAME (已部署)"
        SKIP_COUNT=$((SKIP_COUNT + 1))
        return 0
    fi

    echo "  🚀 部署 $DEPLOYMENT_NAME ($MODEL_NAME v$VERSION)..."

    if az cognitiveservices account deployment create \
        --name $RESOURCE_NAME \
        --resource-group $RESOURCE_GROUP \
        --deployment-name "$DEPLOYMENT_NAME" \
        --model-name "$MODEL_NAME" \
        --model-version "$VERSION" \
        --model-format "$KIND" \
        --sku-capacity 10 \
        --sku-name "Standard" 2>&1 | grep -q "succeeded"; then
        echo "  ✅ 成功部署 $DEPLOYMENT_NAME"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        echo "  ❌ 部署失败 $DEPLOYMENT_NAME"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
}

# 部署各个模型
deploy_model "gpt-4o" "gpt-4o" "2024-11-20" "OpenAI"
deploy_model "gpt-4-1" "gpt-4.1" "2025-04-14" "OpenAI"
deploy_model "o1" "o1" "2024-12-17" "OpenAI"
deploy_model "deepseek-r1" "DeepSeek-R1" "1" "OpenAI"
deploy_model "deepseek-v3" "DeepSeek-V3" "1" "OpenAI"
deploy_model "qwen3-32b" "qwen3-32b" "1" "OpenAI"
deploy_model "llama-3-2-11b" "Llama-3.2-11B-Vision-Instruct" "2" "OpenAI"
deploy_model "mistral-large" "Mistral-Large-2411" "2" "OpenAI"
deploy_model "cohere-command-r-plus" "Cohere-command-r-plus" "1" "OpenAI"

echo ""
echo "[3/3] 部署完成统计:"
echo "  ✅ 成功: $SUCCESS_COUNT"
echo "  ⏭️  跳过: $SKIP_COUNT"
echo "  ❌ 失败: $FAIL_COUNT"
echo ""

# 列出所有部署
echo "========================================"
echo "当前部署列表:"
echo "========================================"
az cognitiveservices account deployment list \
    --name $RESOURCE_NAME \
    --resource-group $RESOURCE_GROUP \
    --query "[].{Name:name, Model:properties.model.name, Version:properties.model.version}" \
    -o table

echo ""
echo "✅ 批量部署完成！"
