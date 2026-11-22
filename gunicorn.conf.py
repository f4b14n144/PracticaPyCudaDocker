import os

bind = f"0.0.0.0:{os.environ.get('PORT', '5000')}"
workers = 1
worker_class = "gthread"
threads = int(os.environ.get("HILOS_HTTP", 8))
timeout = int(os.environ.get("TIMEOUT_S", 120)) + 30
graceful_timeout = 30
accesslog = "-"
errorlog = "-"
loglevel = os.environ.get("LOG_LEVEL", "info").lower()
reload = False
preload_app = False
