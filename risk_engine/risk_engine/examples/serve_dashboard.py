"""
Serve the CCR / XVA / SA-CCR dashboard with a production WSGI server (waitress)
instead of Flask's dev server. Same 5 tabs, same data.

    python risk_engine/examples/serve_dashboard.py               # LAN, port 8050
    python risk_engine/examples/serve_dashboard.py --port 9000
    DASH_PORT=9000 python risk_engine/examples/serve_dashboard.py

Binds 0.0.0.0 -> reachable from other machines on the same network as
    http://<this-machine-LAN-IP>:<port>
No authentication (per project decision -- LAN only).

If colleagues on the LAN cannot connect, add a one-time inbound firewall rule
(PowerShell **as Administrator**):

    New-NetFirewallRule -DisplayName "Dash CCR Dashboard 8050" `
        -Direction Inbound -Protocol TCP -LocalPort 8050 -Action Allow -Profile Private
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from dashboard import server, _lan_ip  # noqa: E402  (Dash WSGI app + helper)

from waitress import serve  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Serve the dashboard (waitress, LAN)")
    ap.add_argument("--host", default=os.environ.get("DASH_HOST", "0.0.0.0"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("DASH_PORT", "8050")))
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    if args.host in ("0.0.0.0", "::"):
        ip = _lan_ip()
        print("Production server (waitress). Reachable on this LAN at:")
        print(f"    http://{ip}:{args.port}      <- share with colleagues on the same network")
        print(f"    http://127.0.0.1:{args.port}   (this machine)")
        print("No authentication -- anyone on the network with the URL can view the data.")
    else:
        print(f"Serving at http://{args.host}:{args.port}")
    print("Ctrl+C to stop.\n")
    serve(server, host=args.host, port=args.port, threads=args.threads)


if __name__ == "__main__":
    main()
