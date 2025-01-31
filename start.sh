#!/bin/bash
gunicorn main:app --workers 50 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8080 --timeout 120