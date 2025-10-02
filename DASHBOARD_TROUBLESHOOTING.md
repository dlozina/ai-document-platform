# Grafana Dashboard Troubleshooting Guide

## Issue: Dashboard Not Showing Data

### Root Cause
The dashboard was using a datasource variable `${DS_PROMETHEUS}` that wasn't defined in the template variables, causing Grafana to fail to resolve the datasource.

### Solution Applied
Changed all datasource references from:
```json
"datasource": {
  "type": "prometheus",
  "uid": "${DS_PROMETHEUS}"
}
```

To:
```json
"datasource": "Prometheus"
```

This uses Grafana's name-based datasource resolution, which is more reliable for provisioned datasources.

## Verification Steps

### 1. Check Services are Running
```bash
cd backend && docker compose ps
```

All services (especially `grafana`, `prometheus`, and `ingestion-service`) should be `Up`.

### 2. Verify Metrics Endpoint
```bash
curl http://localhost:8003/metrics
```

Should return Prometheus-formatted metrics.

### 3. Check Prometheus is Scraping
```bash
# Check targets
curl -s "http://localhost:9090/api/v1/targets" | jq '.data.activeTargets[] | select(.labels.job=="ingestion-service")'

# Query metrics
curl -s "http://localhost:9090/api/v1/query?query=ingestion_http_requests_total" | jq '.data.result | length'
```

Should show `"health": "up"` and return results.

### 4. Verify Grafana Datasource
```bash
curl -s -u admin:admin "http://localhost:3000/api/datasources" | jq '.'
```

Should show Prometheus datasource with name "Prometheus".

### 5. Check Grafana Datasource Health
```bash
curl -s -u admin:admin "http://localhost:3000/api/datasources/1/health" | jq '.'
```

Should return `"status": "OK"`.

### 6. Verify Dashboard Configuration
```bash
curl -s -u admin:admin "http://localhost:3000/api/dashboards/uid/ingestion-overview" | jq '.dashboard.panels[0].datasource'
```

Should return `"Prometheus"` (not a variable).

### 7. Generate Test Traffic
```bash
./test-metrics-simple.sh
```

This will create traffic and populate metrics.

## Common Issues

### Issue: Dashboard shows "No data"

**Possible causes:**
1. Not enough data points (need at least 2 data points for rate calculations)
2. Time range is outside of data availability
3. Prometheus hasn't scraped yet (wait 15-30 seconds)

**Solution:**
- Generate traffic using `./test-metrics-simple.sh`
- Wait 30 seconds for Prometheus to scrape
- Refresh the Grafana dashboard
- Check time range is set to "Last 1 hour"

### Issue: Dashboard shows "Data source not found"

**Possible causes:**
1. Datasource UID is hardcoded and doesn't match
2. Datasource name mismatch
3. Provisioning didn't work correctly

**Solution:**
- Use name-based reference: `"datasource": "Prometheus"`
- Check datasource exists: `curl -s -u admin:admin "http://localhost:3000/api/datasources"`
- Restart Grafana: `docker compose restart grafana`

### Issue: Metrics endpoint returns 404

**Possible causes:**
1. Ingestion service isn't running
2. Wrong port

**Solution:**
- Check service: `docker compose ps ingestion-service`
- Verify port: should be 8003
- Check logs: `docker compose logs ingestion-service`

### Issue: Prometheus shows target as "down"

**Possible causes:**
1. Network connectivity issue
2. Service restarted recently
3. Wrong target address

**Solution:**
- Check Prometheus config: `curl -s "http://localhost:9090/api/v1/status/config"`
- Verify target can be reached from Prometheus container
- Check target address matches service name in docker-compose

## Dashboard URLs

- **Grafana Dashboard**: http://localhost:3000/d/ingestion-overview/ingestion-service-overview
- **Grafana Login**: admin / admin
- **Prometheus**: http://localhost:9090
- **Prometheus Targets**: http://localhost:9090/targets
- **Ingestion Service Health**: http://localhost:8003/health
- **Ingestion Service Metrics**: http://localhost:8003/metrics

## Quick Fix Commands

```bash
# Restart all monitoring services
cd backend && docker compose restart prometheus grafana ingestion-service

# Generate test traffic
cd .. && ./test-metrics-simple.sh

# Wait for scrape
sleep 30

# Verify data
curl -s "http://localhost:9090/api/v1/query?query=ingestion_http_requests_total" | jq '.data.result | length'
```

## What Should You See in the Dashboard?

After generating traffic with the test script, you should see:

1. **Service Uptime**: Shows service uptime in seconds (e.g., 415 seconds = ~7 minutes)
2. **Request Rate**: Shows requests per second (should be > 0)
3. **Response Times**: Shows average response times
4. **Upload Statistics**: Shows upload counts and sizes
5. **Health Status**: Shows 1.0 for healthy components (database, minio, redis)

## Still Not Working?

1. Check Grafana logs: `docker compose logs grafana | tail -50`
2. Check Prometheus logs: `docker compose logs prometheus | tail -50`
3. Check ingestion service logs: `docker compose logs ingestion-service | tail -50`
4. Run the verification script: `./verify-monitoring.sh`
5. Ensure you're viewing the dashboard in your browser with the correct URL
