# Production deployment

The production stack targets a CPU-only Ubuntu VPS with Docker Compose. Only Caddy is exposed publicly. Caddy obtains HTTPS certificates, forwards traffic to the frontend, and the frontend proxies `/api` to the private backend network.

## Recommended VPS

- Public demo: 4 vCPU, 8 GB RAM, 40 GB SSD.
- Comfortable deployment: 4-8 vCPU, 16 GB RAM, 60 GB SSD.
- DNS: point an `A` record for the chosen domain to the VPS before the first start.

The default `.env.production.example` memory limits target the 16 GB profile. For an 8 GB demo VPS, start with:

```dotenv
MGN_MEMORY_LIMIT=2g
FNO_MEMORY_LIMIT=1g
PINN_MEMORY_LIMIT=1536m
TRANSFORMER_MEMORY_LIMIT=1536m
BACKEND_MEMORY_LIMIT=512m
```

## First deploy

Install Docker Engine and the Compose plugin on the VPS, then clone the repository into `/opt/thermoelastic`.

Some MGN runtime artifacts are intentionally not stored in Git. Sync them from the development machine:

```bash
rsync -av --progress \
  mgn-service/datasets/sandstone_comsol_real/processed/ \
  user@server:/opt/thermoelastic/mgn-service/datasets/sandstone_comsol_real/processed/
```

If the server clone does not contain model checkpoints, sync these directories as well:

```bash
rsync -avR --progress \
  ./fno-service/artifacts/checkpoints/baseline \
  ./fno-service/artifacts/datasets/sandstone_fno \
  ./pinn-service/artifacts/checkpoints/baseline_cpu \
  ./transformer-service/artifacts/checkpoints/smoke \
  ./mgn-service/outputs/checkpoints \
  user@server:/opt/thermoelastic/
```

Then configure and start:

```bash
cd /opt/thermoelastic
cp .env.production.example .env.production
nano .env.production
./scripts/check-prod-assets.sh
./scripts/deploy-prod.sh up
```

Set at least:

```dotenv
DEPLOY_HOST=thermoelastic.example.com
ACME_EMAIL=admin@example.com
CORS_ORIGINS=https://thermoelastic.example.com
```

## Operations

```bash
./scripts/deploy-prod.sh status
./scripts/deploy-prod.sh logs
./scripts/deploy-prod.sh logs backend
./scripts/deploy-prod.sh down
```

Update an existing deployment:

```bash
git pull
./scripts/deploy-prod.sh up
```

The production compose file is `docker-compose.prod.yml`. It uses runtime Dockerfiles with CPU-only PyTorch, log rotation, restart policies, memory limits, and a private Docker network for backend and model services.
