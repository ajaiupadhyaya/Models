# DEPLOYMENT.md

## Production Deployment Guide

This guide covers deploying the Trading ML API to production environments.

---

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Pre-Deployment Checklist](#pre-deployment-checklist)
3. [Deployment Options](#deployment-options)
4. [Configuration](#configuration)
5. [Security Hardening](#security-hardening)
6. [Monitoring & Logging](#monitoring--logging)
7. [Scaling](#scaling)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements
- Python 3.11+ (3.12 recommended)
- Docker & Docker Compose (for containerized deployment)
- 4GB RAM minimum, 8GB recommended
- 20GB+ disk space for data/models
- Stable internet connection for market data

### Required Credentials
```
Alpaca Trading:
- ALPACA_API_KEY
- ALPACA_API_SECRET
- ALPACA_API_BASE (paper-api.alpaca.markets or api.alpaca.markets)

Market Data (optional):
- IEX Cloud API key (if using paid market data)
- Alternative: Use yfinance (free but rate-limited)
```

### Network Requirements
- **Production API Port**: 8000 (configurable)
- **Redis Cache**: 6379 (internal only)
- **PostgreSQL**: 5432 (internal only)
- **Prometheus Metrics**: 9090 (internal only)

---

## Pre-Deployment Checklist

### Code Quality
- [ ] All tests passing locally
- [ ] No syntax errors or import issues
- [ ] All dependencies pinned in requirements.txt
- [ ] Environment variables documented
- [ ] API endpoints tested with curl/Postman
- [ ] Database migrations applied
- [ ] Logs review shows no errors

### Security
- [ ] No hardcoded secrets in code
- [ ] API key length verified (Alpaca requires specific format)
- [ ] CORS origins whitelist configured
- [ ] SSL/TLS certificates ready (for HTTPS)
- [ ] Firewall rules configured
- [ ] Rate limiting enabled on endpoints
- [ ] Input validation on all endpoints
- [ ] No debugging mode enabled

### Performance
- [ ] Database indexes created
- [ ] Cache strategy defined
- [ ] Model files optimized
- [ ] Response times < 500ms
- [ ] Memory usage profiled
- [ ] CPU usage under 80%

### Infrastructure
- [ ] Load balancer configured (if multi-instance)
- [ ] Health checks defined
- [ ] Auto-restart enabled
- [ ] Log rotation configured
- [ ] Backup strategy defined
- [ ] Disaster recovery plan documented

---

## Deployment Options

### Option 1: Docker Compose (Recommended)

**Best for**: Single server, development, small production

```bash
# 1. Build images
docker-compose build

# 2. Start all services
docker-compose up -d

# 3. Verify health
curl http://localhost:8000/health

# 4. Check logs
docker-compose logs -f api
```

**Services Started**:
- API (FastAPI) on :8000
- Redis on :6379
- PostgreSQL on :5432
- Prometheus on :9090

### Option 2: Docker Swarm

**Best for**: Multiple servers, high availability

```bash
# Initialize swarm
docker swarm init

# Deploy stack
docker stack deploy -c docker-compose.yml trading-api

# Monitor services
docker service ls
docker service logs trading-api_api

# Scale API service
docker service scale trading-api_api=3
```

### Option 3: Kubernetes

**Best for**: Large scale, complex requirements

```bash
# Create namespace
kubectl create namespace trading-ml

# Apply manifests (see kubernetes/ directory)
kubectl apply -f kubernetes/ -n trading-ml

# Check deployment
kubectl get pods -n trading-ml
kubectl logs deployment/trading-api -n trading-ml

# Scale replicas
kubectl scale deployment/trading-api --replicas=3 -n trading-ml
```

### Option 4: Traditional VM/Bare Metal

**Best for**: Existing infrastructure, maximum control

```bash
# 1. SSH to server
ssh user@production-server

# 2. Clone repository
git clone <repo-url>
cd Models

# 3. Create virtual environment
python3.12 -m venv venv
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt
pip install -r requirements-api.txt

# 5. Configure environment
cp .env.example .env
# Edit .env with production values

# 6. Start API
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4

# 7. (Optional) Use systemd service
sudo cp start-api.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable start-api
sudo systemctl start start-api
```

---

## Configuration

### Environment Variables

Create `.env` file in project root:

```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_LOG_LEVEL=info
API_WORKERS=4

# Alpaca (Paper Trading)
ALPACA_API_KEY=your_api_key_here
ALPACA_API_SECRET=your_api_secret_here
ALPACA_API_BASE=https://paper-api.alpaca.markets

# Redis (Caching)
REDIS_URL=redis://localhost:6379/0

# PostgreSQL (Metrics Storage)
POSTGRES_URL=postgresql://trader:trading123@localhost:5432/trading_metrics

# Data Settings
DATA_DIR=./data
MODELS_DIR=./data/models
CACHE_TTL=3600

# Security
CORS_ORIGINS=["http://localhost:3000", "https://yourdomain.com"]
API_KEY_REQUIRED=false
API_KEY=your_api_key_here

# Monitoring
SENTRY_DSN=https://your-sentry-key@sentry.io/project-id
DATADOG_API_KEY=your_datadog_key
ENABLE_METRICS=true

# Feature Flags
ENABLE_PAPER_TRADING=true
ENABLE_WEBSOCKETS=true
ENABLE_BACKTESTING=true
```

### Production Config

```ini
# deployment/production.ini

[api]
workers = 8
worker_class = uvicorn.workers.UvicornWorker
timeout = 120
keepalive = 5
max_requests = 1000
max_requests_jitter = 100

[logging]
level = info
format = json
output = /var/log/trading-api/api.log

[database]
pool_size = 20
pool_recycle = 3600
echo = false

[redis]
max_connections = 50
decode_responses = true
```

---

## Security Hardening

### API Security

```python
# 1. Enable authentication
from fastapi.security import HTTPBearer, HTTPAuthCredential

security = HTTPBearer()

@app.get("/api/v1/protected")
async def protected_endpoint(credentials: HTTPAuthCredential = Depends(security)):
    # Validate JWT token
    pass
```

### Network Security

```bash
# 1. Configure firewall
ufw allow 22/tcp
ufw allow 8000/tcp
ufw enable

# 2. Setup reverse proxy (Nginx)
# /etc/nginx/sites-available/trading-api
upstream trading_api {
    server localhost:8000;
}

server {
    listen 80;
    server_name yourdomain.com;
    
    # Redirect to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name yourdomain.com;
    
    ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;
    
    location / {
        proxy_pass http://trading_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

### Secrets Management

```bash
# Option 1: AWS Secrets Manager
aws secretsmanager create-secret \
    --name trading-api/alpaca \
    --secret-string '{"api_key":"...","api_secret":"..."}'

# Option 2: HashiCorp Vault
vault kv put secret/trading-api alpaca_api_key=xxx alpaca_api_secret=yyy

# Option 3: Environment variables (with .env.local)
# Add to .gitignore:
.env
.env.*.local
```

---

## Monitoring & Logging

### Application Monitoring

```python
# Enable Prometheus metrics
from fastapi_prometheus_instrumentator import Instrumentator

Instrumentator().instrument(app).expose(app)

# Metrics available at /metrics
```

### Log Aggregation

```bash
# Setup ELK Stack (Elasticsearch, Logstash, Kibana)
docker-compose -f elk-docker-compose.yml up -d

# Configure app to send logs to Logstash
# In api/main.py:
import logging.handlers
handler = logging.handlers.SocketHandler('logstash.example.com', 5000)
logger.addHandler(handler)
```

### Alerting

```yaml
# Prometheus alerts (monitoring/alerts.yml)
groups:
- name: trading_api
  rules:
  - alert: APIDown
    expr: up{job="trading-api"} == 0
    for: 5m
    
  - alert: HighErrorRate
    expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.05
    for: 5m
    
  - alert: HighLatency
    expr: histogram_quantile(0.95, request_duration_seconds) > 1.0
    for: 5m
```

---

## Scaling

### Horizontal Scaling

```bash
# Docker Compose
docker-compose up -d --scale api=4

# Kubernetes
kubectl scale deployment/trading-api --replicas=5

# Manual (Load Balancer)
# Deploy to 3+ instances behind HAProxy/Nginx
```

### Performance Optimization

```python
# 1. Enable caching
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend

@app.get("/api/v1/models")
@cached(namespace="models", expire=3600)
async def get_models():
    pass

# 2. Use connection pooling
# In database configuration:
pool_size = 20
max_overflow = 10

# 3. Optimize model loading
import joblib
model = joblib.load('models/ensemble.pkl')  # Cached in memory
```

### Database Optimization

```sql
-- Create indexes for frequently queried fields
CREATE INDEX idx_trades_symbol ON trades(symbol);
CREATE INDEX idx_trades_timestamp ON trades(timestamp);
CREATE INDEX idx_predictions_model ON predictions(model_name);

-- Partition large tables
ALTER TABLE trades PARTITION BY RANGE (YEAR(timestamp)) (
    PARTITION p2023 VALUES LESS THAN (2024),
    PARTITION p2024 VALUES LESS THAN (2025)
);
```

---

## Troubleshooting

### Common Issues

**Issue**: API won't start
```bash
# Check for port conflicts
lsof -i :8000

# Check Python version
python --version  # Should be 3.11+

# Check imports
python -c "import api.main"
```

**Issue**: Out of memory
```bash
# Monitor memory usage
docker stats

# Reduce worker count
# In api/main.py or uvicorn config
workers = 2  # Reduce from 4

# Enable garbage collection
import gc
gc.collect()
```

**Issue**: Database connection errors
```bash
# Test database connection
python -c "import psycopg2; psycopg2.connect(os.getenv('POSTGRES_URL'))"

# Check PostgreSQL is running
docker ps | grep postgres
```

**Issue**: WebSocket connection fails
```bash
# Check WebSocket support
curl -i -N -H "Connection: Upgrade" -H "Upgrade: websocket" \
  http://localhost:8000/api/v1/ws/prices/AAPL

# Verify firewall allows WebSocket
sudo ufw allow 8000/tcp
```

### Debug Mode

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Run with debug mode
uvicorn api.main:app --reload --log-level debug

# Check API response
curl -v http://localhost:8000/health
```

### Health Checks

```bash
# Full health check
curl -X GET http://localhost:8000/health

# Component health
curl http://localhost:8000/api/v1/monitoring/system
curl http://localhost:8000/api/v1/paper-trading/health
```

---

## Maintenance

### Regular Tasks

```bash
# Daily
- Monitor logs for errors
- Check API response times
- Verify data integrity

# Weekly
- Review database size
- Check cache hit rates
- Update market data

# Monthly
- Backup database
- Update dependencies
- Review security logs
- Analyze performance metrics
```

### Backup Strategy

```bash
# Automated daily backup (cron job)
0 2 * * * /opt/trading-api/backup.sh

# Backup script
#!/bin/bash
BACKUP_DIR=/backups/trading-api
DATE=$(date +%Y%m%d_%H%M%S)

# Backup database
pg_dump trading_metrics > $BACKUP_DIR/db_$DATE.sql

# Backup models
tar -czf $BACKUP_DIR/models_$DATE.tar.gz data/models/

# Upload to S3
aws s3 cp $BACKUP_DIR/ s3://trading-api-backups/
```

### Zero-Downtime Deployments

```bash
# Using load balancer
1. Deploy new version to secondary instance
2. Run smoke tests on secondary
3. Update load balancer to include secondary
4. Remove primary from load balancer
5. Update primary with new version
6. Restore primary to load balancer
7. Complete
```

---

## Support & Monitoring

### Key Metrics to Watch

```
- API Response Time (p50, p95, p99)
- Error Rate (5xx, 4xx)
- Trade Execution Latency
- Model Prediction Confidence
- Database Connection Pool Usage
- Memory Usage
- CPU Usage
- Request Volume
```

### Alerting Thresholds

```
- Response Time p95 > 1000ms  ⚠️
- Error Rate > 5%  🚨
- Memory Usage > 80%  🚨
- CPU Usage > 90%  🚨
- Database Connections > 15/20  ⚠️
```

### Runbook

See [RUNBOOK.md](./RUNBOOK.md) for incident response procedures.

---

## Version Control

```bash
# Tag releases
git tag -a v1.0.0 -m "Production release v1.0.0"
git push origin v1.0.0

# Rollback if needed
git checkout v0.9.9
docker-compose rebuild api
docker-compose up -d api
```

---

For questions or issues, contact the DevOps team or create an issue in the repository.
