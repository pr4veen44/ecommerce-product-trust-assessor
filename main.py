#!/usr/bin/env python
"""Launch the E-Commerce Product Trust Assessor API:  python main.py"""

import uvicorn

from trust_assessor.api.app import app
from trust_assessor.config import get_settings

if __name__ in {"__main__", "__mp_main__"}:
    settings = get_settings()
    uvicorn.run(app, host=settings.ui_host, port=settings.ui_port, reload=False)
