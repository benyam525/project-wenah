# Wenah Deployment Guide

This guide covers deploying the Wenah Civil Rights Compliance API to various environments.

## Prerequisites

- Docker 20.10+ and Docker Compose 2.0+
- Anthropic API key
- (Optional) Kubernetes cluster for production deployments

## Quick Start

### Local Development

1. **Clone and configure:**
   ```bash
   git clone <repository-url>
   cd project-wenah
   cp .env.example .env
   ```

2. **Set your API key:**
   ```bash
   # Edit .env and set ANTHROPIC_API_KEY
   ```

3. **Run with Docker Compose:**
   ```bash
   # Development mode with hot reload
   docker compose --profile dev up wenah-dev

   # Production mode
   docker compose up wenah-api
   ```

4. **Access the API:**
   - API: http://localhost:8000
   - Docs: http://localhost:8000/docs
   - Health: http://localhost:8000/health

## Docker Deployment

### Building the Image

```bash
# Build production image
docker build -t wenah-api:latest .

# Build with specific tag
docker build -t wenah-api:v0.1.0 .
```

### Running the Container

```bash
docker run -d \
  --name wenah-api \
  -p 8000:8000 \
  -e ANTHROPIC_API_KEY=your_key_here \
  -e DEBUG=false \
  -v wenah-data:/app/data/vector_db \
  wenah-api:latest
```

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | Yes | - | Anthropic API key |
| `CLAUDE_MODEL` | No | `claude-sonnet-4-20250514` | Claude model to use |
| `API_HOST` | No | `0.0.0.0` | API host binding |
| `API_PORT` | No | `8000` | API port |
| `DEBUG` | No | `false` | Enable debug mode |
| `CHROMA_PERSIST_DIRECTORY` | No | `/app/data/vector_db` | Vector DB storage |
| `EMBEDDING_MODEL` | No | `all-MiniLM-L6-v2` | Embedding model |
| `RULE_ENGINE_BASE_WEIGHT` | No | `0.6` | Rule engine weight |
| `LLM_ANALYSIS_BASE_WEIGHT` | No | `0.4` | LLM analysis weight |
| `ENABLE_HALLUCINATION_GUARDRAILS` | No | `true` | Enable guardrails |

## Kubernetes Deployment

### Sample Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: wenah-api
  labels:
    app: wenah-api
spec:
  replicas: 2
  selector:
    matchLabels:
      app: wenah-api
  template:
    metadata:
      labels:
        app: wenah-api
    spec:
      containers:
      - name: wenah-api
        image: wenah-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: ANTHROPIC_API_KEY
          valueFrom:
            secretKeyRef:
              name: wenah-secrets
              key: anthropic-api-key
        - name: DEBUG
          value: "false"
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health/live
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 10
        volumeMounts:
        - name: vector-db
          mountPath: /app/data/vector_db
      volumes:
      - name: vector-db
        persistentVolumeClaim:
          claimName: wenah-vector-db
---
apiVersion: v1
kind: Service
metadata:
  name: wenah-api
spec:
  selector:
    app: wenah-api
  ports:
  - port: 80
    targetPort: 8000
  type: ClusterIP
```

### Secrets

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: wenah-secrets
type: Opaque
stringData:
  anthropic-api-key: your_api_key_here
```

### Persistent Volume Claim

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: wenah-vector-db
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 1Gi
```

## Health Endpoints

The API provides several health endpoints for monitoring:

| Endpoint | Purpose | Use Case |
|----------|---------|----------|
| `GET /health` | Basic health check | Load balancers |
| `GET /health/live` | Liveness probe | Kubernetes liveness |
| `GET /health/ready` | Readiness probe | Kubernetes readiness |
| `GET /health/detailed` | Detailed status | Debugging/monitoring |
| `GET /metrics` | Application metrics | Monitoring systems |

### Health Check Responses

**Basic Health (`/health`):**
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "timestamp": "2024-01-15T12:00:00Z"
}
```

**Detailed Health (`/health/detailed`):**
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "timestamp": "2024-01-15T12:00:00Z",
  "uptime_seconds": 3600.5,
  "components": [
    {"name": "rule_engine", "status": "healthy", "latency_ms": 5.2},
    {"name": "scoring_engine", "status": "healthy", "latency_ms": 3.1},
    {"name": "design_guidance", "status": "healthy", "latency_ms": 2.8}
  ],
  "checks_passed": 3,
  "checks_failed": 0
}
```

## Production Checklist

Before deploying to production:

- [ ] Set `DEBUG=false`
- [ ] Configure proper CORS origins
- [ ] Set up HTTPS/TLS termination
- [ ] Configure rate limiting
- [ ] Set up log aggregation
- [ ] Configure alerting on health endpoints
- [ ] Enable request logging
- [ ] Set up persistent storage for vector DB
- [ ] Configure backup strategy for data
- [ ] Review and set appropriate resource limits

## Scaling Considerations

### Horizontal Scaling

The API is stateless except for the vector database. For horizontal scaling:

1. Use a shared persistent volume for the vector DB
2. Or deploy ChromaDB as a separate service
3. Configure sticky sessions if needed

### Resource Requirements

| Component | Min Memory | Recommended Memory | CPU |
|-----------|------------|-------------------|-----|
| API Server | 256MB | 512MB | 0.25 cores |
| With Embeddings | 512MB | 1GB | 0.5 cores |
| Full Stack | 1GB | 2GB | 1 core |

## Troubleshooting

### Container Won't Start

1. Check logs: `docker logs wenah-api`
2. Verify environment variables are set
3. Check port availability

### Health Check Failures

1. Check `/health/detailed` for component status
2. Verify vector DB directory permissions
3. Check API key validity

### Performance Issues

1. Check `/metrics` for response times
2. Review resource usage
3. Consider scaling replicas

## Security Recommendations

1. **API Key Management:**
   - Use secrets management (Vault, AWS Secrets Manager)
   - Rotate keys regularly
   - Never commit keys to source control

2. **Network Security:**
   - Deploy behind a reverse proxy
   - Enable TLS/HTTPS
   - Configure firewall rules

3. **Container Security:**
   - Run as non-root user (already configured)
   - Use read-only file systems where possible
   - Scan images for vulnerabilities

## Monitoring Integration

### Prometheus Metrics

The `/metrics` endpoint provides basic metrics compatible with monitoring systems.

### Log Format

Set `LOG_FORMAT=json` for structured logging compatible with log aggregators:

```json
{
  "timestamp": "2024-01-15T12:00:00Z",
  "level": "INFO",
  "message": "Request processed",
  "path": "/api/v1/assess/quick",
  "method": "POST",
  "duration_ms": 150
}
```

## Support

For issues and feature requests, please open an issue on the project repository.
