# Docker Deployment Guide

## Quick Start

### Prerequisites
- Docker 20.10+
- Docker Compose 2.0+ (optional, for multi-service setup)
- 4GB RAM, 20GB disk space

### Environment (required for API keys)

The API service loads environment variables from a `.env` file. `.env` is not committed; create it from the template before running Compose:

```bash
cp .env.example .env
# Edit .env and set at least FRED_API_KEY, ALPHA_VANTAGE_API_KEY; optionally OPENAI_API_KEY, Alpaca keys.
```

When you run `docker-compose up`, the API container uses `env_file: .env`, so all keys and settings from `.env` are available inside the container. Without a valid `.env`, data and AI endpoints may return errors or empty results.

### Option 1: Run with Docker Compose (Recommended)

```bash
# Clone/navigate to project
cd /path/to/Models

# Start all services
docker-compose up -d

# Verify services
docker-compose ps

# Check API logs
docker-compose logs -f api

# Test API
curl http://localhost:8000/health
```

**Services Started:**
- **API**: FastAPI server on `http://localhost:8000`
- **Redis**: Cache on `localhost:6379`
- **PostgreSQL**: Metrics DB on `localhost:5432`
- **Prometheus**: Metrics on `http://localhost:9090`

### Option 2: Run API Only with Docker

```bash
# Build image
docker build -t trading-ml-api:latest .

# Run container
docker run -d \
  --name trading-api \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  trading-ml-api:latest

# View logs
docker logs -f trading-api

# Test API
curl http://localhost:8000/health

# Stop container
docker stop trading-api
docker rm trading-api
```

### Option 3: Run with Environment Variables

```bash
# Create .env file
cat > .env << EOF
ALPACA_API_KEY=your_key
ALPACA_API_SECRET=your_secret
ALPACA_API_BASE=https://paper-api.alpaca.markets
API_LOG_LEVEL=info
ENABLE_PAPER_TRADING=true
EOF

# Run with environment file
docker run -d \
  --name trading-api \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  --env-file .env \
  trading-ml-api:latest
```

---

## Docker Compose Services

### Configuration

The API service in `docker-compose.yml` uses `env_file: .env`, so create `.env` from `.env.example` and fill in your keys. You can override or add non-secret variables via `environment:` in the compose file:

```yaml
services:
  api:
    env_file: .env
    environment:
      - API_LOG_LEVEL=info  # debug, info, warning, error
      - ENABLE_PAPER_TRADING=true
      - ENABLE_WEBSOCKETS=true
    ports:
      - "8000:8000"  # Change port if needed
```

### Scaling the API

```bash
# Scale to 3 instances (requires load balancer)
docker-compose up -d --scale api=3

# View scaled services
docker-compose ps
```

### Stopping Services

```bash
# Stop all services
docker-compose down

# Stop and remove volumes (data loss!)
docker-compose down -v
```

---

## Environment Variables

Create `.env` file in project root:

```bash
# API Settings
API_HOST=0.0.0.0
API_PORT=8000
API_LOG_LEVEL=info
API_WORKERS=4

# Alpaca Paper Trading
ALPACA_API_KEY=your_api_key
ALPACA_API_SECRET=your_api_secret
ALPACA_API_BASE=https://paper-api.alpaca.markets

# Data
DATA_DIR=/app/data
MODELS_DIR=/app/data/models
CACHE_TTL=3600

# Features
ENABLE_PAPER_TRADING=true
ENABLE_WEBSOCKETS=true
ENABLE_BACKTESTING=true

# Security (production only)
CORS_ORIGINS=["http://localhost:3000", "https://yourdomain.com"]
API_KEY_REQUIRED=false
```

---

## Monitoring & Logs

### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f api
docker-compose logs -f redis

# Last 100 lines
docker-compose logs --tail=100 api

# Follow errors only
docker-compose logs -f api 2>&1 | grep -i error
```

### Access Monitoring

```bash
# Prometheus metrics (if enabled)
curl http://localhost:9090

# API health
curl http://localhost:8000/health

# System metrics
curl http://localhost:8000/api/v1/monitoring/system

# Recent predictions
curl http://localhost:8000/api/v1/monitoring/predictions/recent?limit=10
```

### Check Container Status

```bash
# Detailed info
docker-compose ps

# Resource usage
docker stats

# Container logs with timestamps
docker-compose logs --timestamps api
```

---

## Data Persistence

### Volume Mounting

```bash
# Data is persisted to ./data directory
# Models: ./data/models/
# Cache: ./data/cache/
# Metrics: ./data/metrics/

# Backup data
tar -czf backup-$(date +%Y%m%d).tar.gz data/

# Restore data
tar -xzf backup-20240113.tar.gz
```

### Database Backup

```bash
# Export PostgreSQL data
docker-compose exec postgres pg_dump -U trader trading_metrics > backup.sql

# Restore PostgreSQL
docker-compose exec -T postgres psql -U trader trading_metrics < backup.sql

# Backup Redis
docker-compose exec redis redis-cli BGSAVE
docker cp trading-ml-redis:/data/dump.rdb ./redis-backup.rdb
```

---

## Production Deployment

### Multi-Stage Build

```bash
# Build only production layer
docker build --target production -t trading-ml-api:prod .
```

### Health Checks

The container includes a health check:

```bash
# Manual health check
docker exec trading-api curl http://localhost:8000/health

# Check last 10 health check results
docker inspect trading-api | grep -A 10 '"Health"'
```

### Security

```bash
# Run with read-only filesystem (except /app/data)
docker run -d \
  --name trading-api \
  --read-only \
  --tmpfs /tmp \
  -v $(pwd)/data:/app/data \
  -p 8000:8000 \
  trading-ml-api:latest

# Run with resource limits
docker run -d \
  --name trading-api \
  --memory=2g \
  --cpus=2 \
  -p 8000:8000 \
  trading-ml-api:latest

# Run with user (no root)
# Check Dockerfile - app runs as 'appuser'
```

---

## Networking

### Container-to-Container Communication

```yaml
# In docker-compose.yml - services on same network can use service names
redis:
  container_name: trading-ml-redis
  
api:
  environment:
    - REDIS_URL=redis://redis:6379/0  # Use service name!
```

### Expose to Host

```bash
# Map port to specific IP
docker run -d \
  -p 127.0.0.1:8000:8000 \  # Only localhost
  trading-ml-api:latest

# Map multiple ports
docker run -d \
  -p 8000:8000 \
  -p 8001:8000 \
  trading-ml-api:latest
```

---

## Troubleshooting

### Container won't start

```bash
# Check logs
docker-compose logs api

# Check image exists
docker images | grep trading-ml-api

# Try with verbose output
docker-compose up api  # Don't use -d, see errors
```

### Port already in use

```bash
# Find process using port 8000
lsof -i :8000

# Kill it
kill -9 <PID>

# Or use different port
docker run -d -p 8001:8000 trading-ml-api:latest
```

### Out of memory

```bash
# Check container memory usage
docker stats

# Limit memory in docker-compose.yml
services:
  api:
    deploy:
      resources:
        limits:
          memory: 2G
        reservations:
          memory: 1G
```

### Database connection errors

```bash
# Check PostgreSQL is running
docker-compose ps postgres

# Test connection
docker-compose exec postgres psql -U trader -d trading_metrics -c "SELECT 1"

# Reset database
docker-compose down -v
docker-compose up -d
```

### API not responding

```bash
# Test connectivity
docker-compose exec api curl http://localhost:8000/health

# Check if API process is running
docker-compose exec api ps aux | grep uvicorn

# Restart API
docker-compose restart api

# Check API logs for errors
docker-compose logs api | grep -i error
```

---

## Performance Tuning

### Increase Workers

```yaml
# docker-compose.yml
services:
  api:
    environment:
      - API_WORKERS=8  # Increase from 4
```

### Enable Caching

```yaml
services:
  redis:
    # Redis is already included
    # API will use it for caching if available
```

### Database Connection Pooling

```yaml
services:
  postgres:
    environment:
      - shared_buffers=256MB
      - max_connections=200
```

---

## Cleanup

```bash
# Stop all containers
docker-compose stop

# Remove stopped containers
docker-compose rm

# Remove images
docker rmi trading-ml-api:latest

# Remove unused volumes
docker volume prune

# Remove unused networks
docker network prune

# Full cleanup (WARNING: loses all data!)
docker-compose down -v && docker system prune -a
```

---

## Advanced Usage

### Custom Docker Network

```bash
# Create network
docker network create trading-network

# Run services on custom network
docker run -d \
  --network trading-network \
  --name api \
  -p 8000:8000 \
  trading-ml-api:latest
```

### Using Docker Secrets (Swarm)

```bash
# Create secret
echo "your-api-key" | docker secret create alpaca_key -

# Use in compose
services:
  api:
    secrets:
      - alpaca_key
    environment:
      - ALPACA_API_KEY_FILE=/run/secrets/alpaca_key
```

### Log Drivers

```yaml
# Send logs to syslog
services:
  api:
    logging:
      driver: syslog
      options:
        syslog-address: "udp://127.0.0.1:514"
        tag: "trading-api"
```

---

## Testing

```bash
# Run smoke tests
curl -v http://localhost:8000/
curl -v http://localhost:8000/health
curl -v http://localhost:8000/api/v1/monitoring/system

# Test paper trading initialization
curl -X POST http://localhost:8000/api/v1/paper-trading/initialize

# Test model training
curl -X POST http://localhost:8000/api/v1/models/train \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "AAPL",
    "model_type": "simple",
    "start_date": "2023-01-01",
    "end_date": "2024-01-01"
  }'
```

---

For production deployment, see [LAUNCH_GUIDE.md](LAUNCH_GUIDE.md) (sections 4–5).
