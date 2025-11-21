#!/usr/bin/env bash
# Before running, assume IAM role via aws-sso and set .env variables including AWS_DEFAULT_REGION=eu-central-1
docker run -v ~/.aws:/root/.aws -p 8501:8501 --env-file .env hsemining