# GitHub Actions CI/CD Workflows

This directory contains GitHub Actions workflows for building and publishing Docker images for all microservices in the Abysalto project.

## 🚀 Available Workflows

### Individual Service Workflows
- **`build-embedding-service.yml`** - Builds and publishes the embedding service Docker image
- **`build-gateway-service.yml`** - Builds and publishes the gateway service Docker image  
- **`build-ingestion-service.yml`** - Builds and publishes the ingestion service Docker image
- **`build-ner-service.yml`** - Builds and publishes the NER service Docker image
- **`build-ocr-service.yml`** - Builds and publishes the OCR service Docker image
- **`build-query-service.yml`** - Builds and publishes the query service Docker image

### Combined Workflow
- **`build-all-services.yml`** - Builds and publishes all services in parallel

## 🔄 Trigger Conditions

Each workflow is triggered in the following scenarios:

1. **Automatic Triggers:**
   - When code is pushed to the `main` branch
   - When pull requests are opened/updated targeting the `main` branch
   - Only triggers if files in the respective service directory are modified

2. **Manual Triggers:**
   - Can be triggered manually via GitHub Actions UI
   - Allows specifying custom version tags
   - Useful for releases and hotfixes

## 🏷️ Image Tagging Strategy

Images are tagged with multiple tags for flexibility:

- **`latest`** - Always points to the latest version from main branch
- **`main`** - Latest version from main branch
- **`v1.0.0`** - Semantic version tags (when manually triggered)
- **`1.0`** - Major.minor version tags
- **`pr-123`** - Pull request tags (for testing)

## 📦 Container Registry

All images are published to **GitHub Container Registry (ghcr.io)**:

```
ghcr.io/{owner}/{repository}/embedding-service
ghcr.io/{owner}/{repository}/gateway-service
ghcr.io/{owner}/{repository}/ingestion-service
ghcr.io/{owner}/{repository}/ner-service
ghcr.io/{owner}/{repository}/ocr-service
ghcr.io/{owner}/{repository}/query-service
```

## 🛠️ Usage Examples

### Pull Latest Images
```bash
# Pull all latest images
docker pull ghcr.io/{owner}/{repository}/embedding-service:latest
docker pull ghcr.io/{owner}/{repository}/gateway-service:latest
docker pull ghcr.io/{owner}/{repository}/ingestion-service:latest
docker pull ghcr.io/{owner}/{repository}/ner-service:latest
docker pull ghcr.io/{owner}/{repository}/ocr-service:latest
docker pull ghcr.io/{owner}/{repository}/query-service:latest
```

### Pull Specific Version
```bash
# Pull specific version
docker pull ghcr.io/{owner}/{repository}/embedding-service:v1.2.0
docker pull ghcr.io/{owner}/{repository}/gateway-service:v1.2.0
```

### Use in Docker Compose
```yaml
version: '3.8'
services:
  embedding-service:
    image: ghcr.io/{owner}/{repository}/embedding-service:latest
    ports:
      - "8002:8002"
  
  gateway-service:
    image: ghcr.io/{owner}/{repository}/gateway-service:latest
    ports:
      - "8005:8005"
```

## 🔧 Manual Workflow Execution

To manually trigger a workflow:

1. Go to the **Actions** tab in your GitHub repository
2. Select the workflow you want to run
3. Click **"Run workflow"**
4. Choose the branch and enter a version tag (e.g., `v1.0.0`)
5. Click **"Run workflow"**

## 🏗️ Build Features

- **Multi-platform builds** - Images built for both `linux/amd64` and `linux/arm64`
- **Build caching** - Uses GitHub Actions cache for faster builds
- **Security** - Images run as non-root user
- **Health checks** - Built-in health check endpoints
- **Optimized layers** - Efficient Docker layer caching

## 📋 Prerequisites

- GitHub repository with Actions enabled
- Proper permissions for GitHub Container Registry
- Dockerfiles present in each service directory

## 🔍 Monitoring

- Check the **Actions** tab for build status
- View build logs for debugging
- Monitor image sizes and build times
- Set up notifications for failed builds

## 🚨 Troubleshooting

### Common Issues

1. **Permission Denied**
   - Ensure `GITHUB_TOKEN` has `packages: write` permission
   - Check repository settings for Actions permissions

2. **Build Failures**
   - Check Dockerfile syntax
   - Verify all dependencies are available
   - Review build logs for specific errors

3. **Image Not Found**
   - Ensure workflow completed successfully
   - Check if image was pushed to correct registry
   - Verify image name and tag

### Getting Help

- Check GitHub Actions documentation
- Review Docker build logs
- Verify service-specific requirements in each service's README
