import io
import os
import sys
import threading
import time
import webbrowser

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for p in (REPO, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("IRONPULSE_PROJECT_ROOT", HERE)


def main():
    cfg = yaml.safe_load(io.open(os.path.join(HERE, "config.yaml"), encoding="utf-8"))
    host = cfg["web"]["host"]
    port = int(cfg["web"]["port"])
    url = "http://%s:%d/" % (host, port)

    import uvicorn
    from web.app import app

    if cfg["web"]["open_browser"]:
        def opener():
            time.sleep(1.2)
            webbrowser.open(url)
        threading.Thread(target=opener, daemon=True).start()

    print("IronPulse Zero")
    print("  ตรวจจับความไม่สมดุลของมวลหมุนเท่านั้น")
    print("  เปิดที่ %s" % url)
    print("  กด Ctrl+C เพื่อหยุด")
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
