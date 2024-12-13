#!/usr/bin/bash

CCACHE_NAME=ccache-cache
REDIS_CONTAINER_STATUS=$(docker ps --filter="name=$CCACHE_NAME" --filter="status=exited" --format "{{.ID}}")
REDIS_CONTAINER_ID=$(docker ps --filter="name=$CCACHE_NAME" --format "{{.ID}}")
if [ -n "$REDIS_CONTAINER_STATUS" ]
then
    echo "starting previous redis container"
    docker start "$CCACHE_NAME"
elif [ ! -n "$REDIS_CONTAINER_ID" ]
then
    echo "no previous ccache container found, starting a new one"
    docker run -p 6379:6379 --name "$CCACHE_NAME" -d redis
fi
docker build --network=host -t anofel-snark .
