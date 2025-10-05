# Ingestion Service Monitoring Setup

This document describes the monitoring setup for the Ingestion Service using Prometheus and Grafana.

## Overview

The monitoring setup includes:
- **Prometheus**: Metrics collection and storage
- **Grafana**: Metrics visualization and dashboards
- **Ingestion Service**: Enhanced with Prometheus metrics

## Quick Start

### 1. Start the Services

```bash
# Start all services including monitoring
cd backend
docker-compose up -d

# Check that all services are running
docker-compose ps
```

### 2. Access the Monitoring Tools

- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090
- **Ingestion Service**: http://localhost:8003
- **Flower (Celery)**: http://localhost:5555 (admin/admin)
- 
### 2. Tools Display

#### Flower
<img width="3411" height="603" alt="Image" src="https://github.com/user-attachments/assets/57f64e28-d7f9-4e4a-8a3d-43f46cb520fe" />

#### Grafana

<img width="3037" height="1174" alt="Image" src="https://github.com/user-attachments/assets/f8cd39cb-a2b1-4189-a142-ca24c60c8eec" />

<img width="3042" height="1197" alt="Image" src="https://github.com/user-attachments/assets/d908abb2-475e-456e-b8b7-1059d94fabf4" />


### 4. Test Metrics Collection

```bash
# Run the test script
./test-metrics.sh
```

## Metrics Collected

### HTTP Request Metrics
- `ingestion_http_requests_total`: Total HTTP requests by method, endpoint, status code, and tenant
- `ingestion_http_request_duration_seconds`: Request duration histogram
- `ingestion_http_request_size_bytes`: Request size histogram
- `ingestion_http_response_size_bytes`: Response size histogram

### Upload Metrics
- `ingestion_uploads_total`: Total file uploads by tenant, file type, and status
- `ingestion_upload_duration_seconds`: Upload duration histogram
- `ingestion_upload_size_bytes`: Upload file size histogram

### Service Health Metrics
- `ingestion_service_health_status`: Health status of components (database, minio, redis)
- `ingestion_service_uptime_seconds`: Service uptime

## Grafana Dashboard

The dashboard includes:
1. **Request Rate**: HTTP requests per second
2. **Response Time**: 95th and 50th percentile response times
3. **Upload Rate**: File uploads per second by tenant and type
4. **Upload Duration**: Upload processing times
5. **Service Health**: Component health status
6. **Service Uptime**: Service uptime tracking

## Configuration Files

### Prometheus Configuration
- `prometheus/prometheus.yml`: Prometheus scraping configuration

### Grafana Configuration
- `grafana-dashboards/datasources.yml`: Prometheus datasource configuration
- `grafana-dashboards/dashboard.yml`: Dashboard provisioning configuration
- `grafana-dashboards/ingestion-service-overview.json`: Main dashboard definition

## Adding New Metrics

To add new metrics to the ingestion service:

1. **Define the metric** in `src/main.py`:
```python
NEW_METRIC = Counter('ingestion_new_metric_total', 'Description', ['label1', 'label2'])
```

2. **Use the metric** in your code:
```python
NEW_METRIC.labels(label1='value1', label2='value2').inc()
```

3. **Update the Grafana dashboard** to visualize the new metric

## Troubleshooting

### Metrics Not Appearing
1. Check that the ingestion service is running: `docker-compose ps`
2. Verify metrics endpoint: `curl http://localhost:8003/metrics`
3. Check Prometheus targets: http://localhost:9090/targets
4. Verify Grafana datasource: http://localhost:3000/datasources

### Dashboard Not Loading
1. Check Grafana logs: `docker-compose logs grafana`
2. Verify dashboard JSON syntax
3. Check datasource configuration

### Performance Impact
- Metrics collection adds <1% CPU overhead
- Memory usage increases by ~5-10MB
- Network overhead is negligible

## Next Steps

1. **Add Celery Task Metrics**: Monitor processing pipeline performance
2. **Add Database Metrics**: Track database operations and performance
3. **Add Storage Metrics**: Monitor MinIO operations
4. **Set Up Alerting**: Configure alerts for critical metrics
5. **Add Custom Business Metrics**: Track business-specific KPIs

## Monitoring Best Practices

1. **Start Small**: Begin with basic metrics and expand gradually
2. **Monitor Performance**: Watch for performance impact of metrics collection
3. **Set Up Alerting**: Configure alerts for critical metrics
4. **Regular Review**: Review and update dashboards regularly
5. **Documentation**: Keep monitoring documentation up to date
