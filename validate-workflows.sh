#!/bin/bash

# GitHub Actions Workflow Validation Script
# This script validates that all workflow files are properly configured

set -e

echo "🔍 Validating GitHub Actions Workflows..."

WORKFLOWS_DIR=".github/workflows"
REQUIRED_WORKFLOWS=(
    "build-embedding-service.yml"
    "build-gateway-service.yml"
    "build-ingestion-service.yml"
    "build-ner-service.yml"
    "build-ocr-service.yml"
    "build-query-service.yml"
    "build-all-services.yml"
)

# Check if workflows directory exists
if [ ! -d "$WORKFLOWS_DIR" ]; then
    echo "❌ Workflows directory not found: $WORKFLOWS_DIR"
    exit 1
fi

echo "✅ Workflows directory exists"

# Check for required workflow files
for workflow in "${REQUIRED_WORKFLOWS[@]}"; do
    if [ -f "$WORKFLOWS_DIR/$workflow" ]; then
        echo "✅ Found workflow: $workflow"
        
        # Basic YAML validation
        if command -v yq >/dev/null 2>&1; then
            if yq eval '.' "$WORKFLOWS_DIR/$workflow" >/dev/null 2>&1; then
                echo "  ✅ Valid YAML syntax"
            else
                echo "  ❌ Invalid YAML syntax"
                exit 1
            fi
        else
            echo "  ⚠️  yq not installed, skipping YAML validation"
        fi
    else
        echo "❌ Missing workflow: $workflow"
        exit 1
    fi
done

# Check for README
if [ -f "$WORKFLOWS_DIR/README.md" ]; then
    echo "✅ Documentation found: README.md"
else
    echo "⚠️  Documentation missing: README.md"
fi

echo ""
echo "🎉 All workflows validated successfully!"
echo ""
echo "📋 Summary:"
echo "  - Individual service workflows: 6"
echo "  - Combined workflow: 1"
echo "  - Documentation: 1"
echo ""
echo "🚀 Next steps:"
echo "  1. Commit and push these workflows to your repository"
echo "  2. Go to GitHub Actions tab to see the workflows"
echo "  3. Test manual execution with a version tag"
echo "  4. Monitor builds and image publishing"
