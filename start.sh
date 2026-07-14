#!/bin/bash
set -e

# 1. Запуск Backend в фоне
uvicorn backend.app_backend:app --host 0.0.0.0 --port 8000 > /tmp/fastapi.log 2>&1 &

# 2. Ожидание запуска Backend
for i in {1..30}; do
    if curl -s http://localhost:8000/health > /dev/null; then
        break
    fi
    if [ $i -eq 30 ]; then
        echo "Backend failed to start"
        cat /tmp/fastapi.log
        exit 1
    fi
    sleep 1
done

# 3. Запуск Frontend (exec заменяет текущий процесс)
exec streamlit run frontend/app.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true