# Digital Ocean Pricing Analysis

This document provides a comprehensive cost analysis for running the document processing system on Digital Ocean, based on the production Docker Compose configuration.

## Infrastructure Requirements

Based on the [Docker Compose Production](backend/docker-compose.production.yml) file, the system requires:

### Resource Summary
- **Total CPU Cores**: ~38 cores
- **Total Memory**: ~95GB RAM
- **Storage**: 2TB+ (PostgreSQL, MinIO, Qdrant, monitoring data)
- **Target Throughput**: 50-100 documents/minute (10-20 MB files)

### Service Breakdown
| Service | Replicas | CPU/Replica | RAM/Replica | Total CPU | Total RAM |
|---------|----------|-------------|-------------|-----------|-----------|
| OCR Service | 3 | 2.0 | 4GB | 6.0 | 12GB |
| NER Service | 2 | 1.5 | 3GB | 3.0 | 6GB |
| Embedding Service | 2 | 1.0 | 4GB | 2.0 | 8GB |
| Query Service | 2 | 1.0 | 2GB | 2.0 | 4GB |
| API Gateway | 2 | 0.5 | 1GB | 1.0 | 2GB |
| Qdrant | 1 | 2.0 | 8GB | 2.0 | 8GB |
| PostgreSQL | 1 | 2.0 | 8GB | 2.0 | 8GB |
| MinIO | 1 | 1.0 | 4GB | 1.0 | 4GB |
| Redis | 1 | 1.0 | 2GB | 1.0 | 2GB |
| OCR Workers | 5 | 2.0 | 4GB | 10.0 | 20GB |
| NER Workers | 3 | 1.5 | 3GB | 4.5 | 9GB |
| Embedding Workers | 3 | 1.0 | 4GB | 3.0 | 12GB |
| Completion Workers | 1 | 0.5 | 1GB | 0.5 | 1GB |
| Monitoring (Prometheus/Grafana/Flower) | 3 | 0.6 | 2.2GB | 1.8 | 6.6GB |
| **TOTAL** | | | | **38.8** | **95.6GB** |

## Digital Ocean Pricing

### Infrastructure Costs (Monthly)

**Option 1: Multiple Medium Droplets**
- 4x General Purpose Droplets (8 vCPU, 16GB RAM each) = 32 vCPU, 64GB RAM
- 2x General Purpose Droplets (4 vCPU, 8GB RAM each) = 8 vCPU, 16GB RAM
- **Cost**: ~$1,200/month

**Option 2: Fewer Large Droplets**
- 2x General Purpose Droplets (16 vCPU, 32GB RAM each) = 32 vCPU, 64GB RAM
- 1x General Purpose Droplet (8 vCPU, 16GB RAM) = 8 vCPU, 16GB RAM
- **Cost**: ~$1,100/month

### Additional Costs
- **Block Storage** (2TB+): ~$30/month
- **Load Balancer**: $12/month
- **Data Transfer**: ~$50/month (moderate usage)
- **Monitoring**: Included in droplet costs

### Total Monthly Infrastructure Cost: ~$1,200

## Document Processing Capacity

### Throughput Analysis
- **Target**: 50-100 documents/minute (as configured)
- **File Size**: 10-20 MB per document
- **Processing Time**: 20-45 seconds per document (OCR bottleneck)
- **Monthly Capacity**: 
  - Conservative: 50 docs/min × 60 × 24 × 30 = **2.16M documents/month**
  - Optimistic: 100 docs/min × 60 × 24 × 30 = **4.32M documents/month**
  - Average: **3.24M documents/month**

## Cost Per Thousand Documents

| Scenario | Monthly Documents | Cost per 1,000 Documents |
|----------|------------------|---------------------------|
| Conservative (50 docs/min) | 2.16M | **$0.56** |
| Average (75 docs/min) | 3.24M | **$0.37** |
| Optimistic (100 docs/min) | 4.32M | **$0.28** |

## Cost Breakdown Summary

- **Infrastructure**: $1,200/month
- **Storage**: $30/month
- **Load Balancer**: $12/month
- **Data Transfer**: $50/month
- **Total**: **$1,292/month**

**Average Cost per 1,000 Documents: $0.37**

## Notes

- Costs are based on 2024 Digital Ocean pricing
- Processing capacity assumes 10-20 MB file sizes as per production requirements
- OCR processing is the primary bottleneck and scales with file size
- Costs include high availability with multiple replicas
- Monitoring and observability included in infrastructure costs
- Pricing assumes continuous operation (24/7)

## Scaling Considerations

- **CPU-bound scaling**: OCR workers can be scaled horizontally
- **Memory-bound scaling**: Embedding workers limited by transformer model memory requirements
- **Storage scaling**: Costs increase linearly with document volume
- **Geographic distribution**: Additional costs for multi-region deployment