=====================

# CMD (pwd)

## Run Container
docker compose -f docker-compose.yml up -d --scale spark-worker=2

## Remove Container
docker compose -f docker-compose.yml down

## Data visualization
python -m projects.dataviz

# CMD (any dir)

## Run jobs
docker exec spark-master spark-submit --deploy-mode client ./jobs/train.py
docker exec spark-master spark-submit --deploy-mode client ./jobs/deploy.py

=====================

# Spark Master
http://localhost:9090

# History Server
http://localhost:18080