"""Root WSGI entry-point used by gunicorn / Render / Railway / Heroku / Docker.

PaaS platforms look for an `app` object at the repository root.
This file imports the Flask `app` instance from Backend/app.py.
"""
import os
import sys

# Make sure the repo root is on the path so 'Backend' is importable
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from Backend.app import app  # noqa: E402  (import not at top of file)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
