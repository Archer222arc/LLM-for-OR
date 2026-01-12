#!/bin/bash
# ============================================================================
# Azure AI Foundry 基础设施自动化部署脚本
# ============================================================================
# 功能:
#   - 创建Service Principal用于Foundry认证
#   - 创建3个Azure AI Foundry Hubs（eastus2, eastus, swedencentral）
#   - 配置RBAC权限
#   - 生成环境变量文件
#
# Usage:
#   1. 确保已登录Azure: az login
#   2. 设置订阅: az account set --subscription "your-subscription-id"
#   3. 运行脚本: bash scripts/deploy_foundry_infrastructure.sh
#   4. 加载环境变量: source ~/.azure_foundry_credentials
#
# Created: 2026-01-11
# ============================================================================

set -e  # 遇到错误立即退出

# ============== 颜色输出 ==============
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============== 配置变量 ==============
RESOURCE_GROUP="rg-llm-for-or"
SP_NAME="sp-llm-for-or-foundry"
CREDENTIALS_FILE="$HOME/.azure_foundry_credentials"

# Hubs配置
declare -A HUBS
HUBS["foundry-eastus2"]="eastus2"
HUBS["foundry-eastus"]="eastus"
HUBS["foundry-sweden"]="swedencentral"

# 统计变量
TOTAL_STEPS=6
CURRENT_STEP=0
ERRORS=0

# ============== 辅助函数 ==============
print_header() {
    echo -e "${BLUE}============================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}============================================================${NC}"
}

print_step() {
    CURRENT_STEP=$((CURRENT_STEP + 1))
    echo -e "\n${YELLOW}[步骤 $CURRENT_STEP/$TOTAL_STEPS] $1${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
    ERRORS=$((ERRORS + 1))
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# ============== 步骤1: 检查前置条件 ==============
check_prerequisites() {
    print_step "检查前置条件"

    # 检查Azure CLI
    if ! command -v az &> /dev/null; then
        print_error "Azure CLI未安装"
        echo "请安装Azure CLI: https://docs.microsoft.com/en-us/cli/azure/install-azure-cli"
        exit 1
    fi
    print_success "Azure CLI已安装"

    # 检查登录状态
    if ! az account show &> /dev/null; then
        print_error "未登录Azure"
        echo "请运行: az login"
        exit 1
    fi
    print_success "已登录Azure"

    # 获取当前订阅信息
    SUBSCRIPTION_ID=$(az account show --query id -o tsv)
    SUBSCRIPTION_NAME=$(az account show --query name -o tsv)
    TENANT_ID=$(az account show --query tenantId -o tsv)

    print_info "订阅: $SUBSCRIPTION_NAME"
    print_info "订阅ID: $SUBSCRIPTION_ID"
    print_info "租户ID: $TENANT_ID"

    # 检查资源组是否存在
    if ! az group show --name "$RESOURCE_GROUP" &> /dev/null; then
        print_error "资源组 '$RESOURCE_GROUP' 不存在"
        echo "请先创建资源组: az group create --name $RESOURCE_GROUP --location eastus"
        exit 1
    fi
    print_success "资源组 '$RESOURCE_GROUP' 已存在"
}

# ============== 步骤2: 创建Service Principal ==============
create_service_principal() {
    print_step "创建Service Principal"

    # 检查SP是否已存在
    if az ad sp list --display-name "$SP_NAME" --query "[0].appId" -o tsv 2>/dev/null | grep -q .; then
        print_info "Service Principal '$SP_NAME' 已存在，跳过创建"
        SP_APP_ID=$(az ad sp list --display-name "$SP_NAME" --query "[0].appId" -o tsv)
        print_info "使用现有的App ID: $SP_APP_ID"

        # 重置凭证以获取新的secret
        echo "正在重置凭证..."
        SP_CREDENTIALS=$(az ad sp credential reset --id "$SP_APP_ID" --output json)
        SP_PASSWORD=$(echo "$SP_CREDENTIALS" | jq -r '.password')
    else
        # 创建新的Service Principal
        print_info "创建新的Service Principal: $SP_NAME"

        SP_CREDENTIALS=$(az ad sp create-for-rbac \
            --name "$SP_NAME" \
            --role "Cognitive Services User" \
            --scopes "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP" \
            --output json)

        SP_APP_ID=$(echo "$SP_CREDENTIALS" | jq -r '.appId')
        SP_PASSWORD=$(echo "$SP_CREDENTIALS" | jq -r '.password')

        print_success "Service Principal创建成功"
    fi

    print_info "App ID: $SP_APP_ID"
    print_info "Secret: ${SP_PASSWORD:0:8}... (已隐藏)"

    # 等待Service Principal传播
    echo "等待Service Principal传播..."
    sleep 10
}

# ============== 步骤3: 创建Azure AI Foundry Hubs ==============
create_foundry_hubs() {
    print_step "创建Azure AI Foundry Hubs"

    # 检查azure-ai-ml扩展
    if ! az extension list --query "[?name=='ml'].name" -o tsv | grep -q "ml"; then
        print_info "安装Azure ML扩展..."
        az extension add --name ml --yes
    fi

    # 创建Hubs（串行创建以避免冲突）
    for hub_name in "${!HUBS[@]}"; do
        region="${HUBS[$hub_name]}"

        # 检查Hub是否已存在
        if az ml workspace show --name "$hub_name" --resource-group "$RESOURCE_GROUP" &> /dev/null; then
            print_info "Hub '$hub_name' 已存在，跳过创建"
            continue
        fi

        print_info "创建Hub: $hub_name (region: $region)"

        # 创建Hub
        if az ml workspace create \
            --resource-group "$RESOURCE_GROUP" \
            --name "$hub_name" \
            --location "$region" \
            --kind project \
            --output none; then
            print_success "Hub '$hub_name' 创建成功"
        else
            print_error "Hub '$hub_name' 创建失败"
        fi
    done
}

# ============== 步骤4: 配置RBAC权限 ==============
configure_rbac() {
    print_step "配置RBAC权限"

    # 为Service Principal分配"Azure AI Developer"角色
    ROLE="Azure AI Developer"
    SCOPE="/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP"

    # 检查角色分配是否已存在
    if az role assignment list \
        --assignee "$SP_APP_ID" \
        --role "$ROLE" \
        --scope "$SCOPE" \
        --query "[0].id" -o tsv | grep -q .; then
        print_info "角色分配已存在，跳过"
    else
        print_info "分配角色: $ROLE"

        if az role assignment create \
            --assignee "$SP_APP_ID" \
            --role "$ROLE" \
            --scope "$SCOPE" \
            --output none; then
            print_success "RBAC权限配置成功"
        else
            # 如果"Azure AI Developer"不存在，尝试使用"Contributor"
            print_info "角色'$ROLE'不存在，尝试使用'Contributor'角色"

            if az role assignment create \
                --assignee "$SP_APP_ID" \
                --role "Contributor" \
                --scope "$SCOPE" \
                --output none; then
                print_success "RBAC权限配置成功（使用Contributor角色）"
            else
                print_error "RBAC权限配置失败"
            fi
        fi
    fi

    # 等待权限传播
    echo "等待权限传播..."
    sleep 5
}

# ============== 步骤5: 生成环境变量文件 ==============
generate_credentials_file() {
    print_step "生成环境变量文件"

    # 备份现有文件
    if [ -f "$CREDENTIALS_FILE" ]; then
        backup_file="${CREDENTIALS_FILE}.backup.$(date +%Y%m%d_%H%M%S)"
        cp "$CREDENTIALS_FILE" "$backup_file"
        print_info "已备份现有文件到: $backup_file"
    fi

    # 创建新的凭证文件
    cat > "$CREDENTIALS_FILE" << EOF
# ============================================================================
# Azure AI Foundry 环境变量
# 自动生成时间: $(date)
# ============================================================================

# Azure认证信息
export AZURE_TENANT_ID="$TENANT_ID"
export AZURE_CLIENT_ID="$SP_APP_ID"
export AZURE_CLIENT_SECRET="$SP_PASSWORD"
export AZURE_SUBSCRIPTION_ID="$SUBSCRIPTION_ID"

# 资源组信息
export AZURE_RESOURCE_GROUP="$RESOURCE_GROUP"

# Foundry Hubs
export FOUNDRY_HUB_EASTUS2="foundry-eastus2"
export FOUNDRY_HUB_EASTUS="foundry-eastus"
export FOUNDRY_HUB_SWEDEN="foundry-sweden"

# ============================================================================
# 使用说明:
#   1. 加载环境变量: source ~/.azure_foundry_credentials
#   2. 验证部署: python scripts/verify_foundry_deployment.py
# ============================================================================
EOF

    # 设置文件权限（仅用户可读写）
    chmod 600 "$CREDENTIALS_FILE"

    print_success "环境变量文件已生成: $CREDENTIALS_FILE"
    print_info "运行以下命令加载环境变量:"
    echo -e "${GREEN}    source $CREDENTIALS_FILE${NC}"
}

# ============== 步骤6: 验证部署 ==============
verify_deployment() {
    print_step "验证部署"

    # 验证Service Principal
    if az ad sp show --id "$SP_APP_ID" &> /dev/null; then
        print_success "Service Principal验证成功"
    else
        print_error "Service Principal验证失败"
    fi

    # 验证Hubs
    for hub_name in "${!HUBS[@]}"; do
        if az ml workspace show --name "$hub_name" --resource-group "$RESOURCE_GROUP" &> /dev/null; then
            print_success "Hub '$hub_name' 验证成功"
        else
            print_error "Hub '$hub_name' 验证失败"
        fi
    done

    # 验证RBAC权限
    if az role assignment list --assignee "$SP_APP_ID" --scope "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP" --query "[0].id" -o tsv | grep -q .; then
        print_success "RBAC权限验证成功"
    else
        print_error "RBAC权限验证失败"
    fi
}

# ============== 主函数 ==============
main() {
    print_header "Azure AI Foundry 基础设施自动化部署"

    echo "此脚本将创建以下资源:"
    echo "  - Service Principal: $SP_NAME"
    echo "  - Azure AI Foundry Hubs: ${!HUBS[*]}"
    echo "  - RBAC权限配置"
    echo ""
    read -p "是否继续? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "部署已取消"
        exit 0
    fi

    # 执行部署步骤
    check_prerequisites
    create_service_principal
    create_foundry_hubs
    configure_rbac
    generate_credentials_file
    verify_deployment

    # 输出汇总报告
    print_header "部署汇总"

    if [ $ERRORS -eq 0 ]; then
        echo -e "${GREEN}✓ 所有步骤成功完成！${NC}\n"

        echo "下一步操作:"
        echo "  1. 加载环境变量:"
        echo -e "     ${GREEN}source $CREDENTIALS_FILE${NC}"
        echo ""
        echo "  2. 创建本地配置文件:"
        echo -e "     ${GREEN}cp configs/models/azure_deployments_template.yaml configs/models/azure_deployments.yaml${NC}"
        echo ""
        echo "  3. 部署Foundry模型（交互式）:"
        echo -e "     ${GREEN}python scripts/guide_foundry_model_deployment.py${NC}"
        echo ""
        echo "  4. 验证部署:"
        echo -e "     ${GREEN}python scripts/verify_foundry_deployment.py${NC}"

    else
        echo -e "${RED}✗ 部署过程中遇到 $ERRORS 个错误${NC}\n"
        echo "请检查上述错误信息并重新运行脚本"
        exit 1
    fi
}

# ============== 执行主函数 ==============
main
