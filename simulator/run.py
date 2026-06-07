#!/usr/bin/env python3
"""Smart Agriculture Simulator — entry point.

Usage:
    python run.py [--time-scale N] [--port PORT] [--no-browser]

Options:
    --time-scale N   Simulation speed multiplier (1=realtime, 3600=1h/s). Default: 1
    --port PORT      HTTP port. Default: auto-find 5000-5010
    --no-browser     Don't open browser automatically
"""

import argparse
import os
import sys
import socket
import signal
import webbrowser

# ── Ensure project root is on sys.path ──
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def find_available_port(start: int = 5000, end: int = 5020) -> int:
    """Find the first available port in [start, end)."""
    for port in range(start, end):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"No available port in range {start}-{end}")


def main():
    parser = argparse.ArgumentParser(description="Smart Agriculture Simulator")
    parser.add_argument("--time-scale", type=float, default=1.0,
                        help="Simulation speed (1=realtime, 60=1min/s, 3600=1h/s)")
    parser.add_argument("--port", type=int, default=0,
                        help="HTTP port (0=auto)")
    parser.add_argument("--no-browser", action="store_true",
                        help="Don't open browser")
    args = parser.parse_args()

    # ── Resolve model path ──
    model_path = os.path.join(ROOT, "data", "plant_disease_model.tflite")
    data_dir = os.path.join(ROOT, "data")

    # ── Create Flask app ──
    from flask import Flask
    from flask_cors import CORS
    from api.routes import api_bp, init_routes
    from simulator.system import AgriSystem

    app = Flask(__name__, static_folder=None)
    CORS(app)
    app.register_blueprint(api_bp)

    # ── Initialize system ──
    system = AgriSystem(model_path=model_path, data_dir=data_dir)
    system.begin(time_scale=args.time_scale)
    init_routes(system)

    # ── Start simulation ──
    system.start()

    # ── Find port ──
    port = args.port if args.port > 0 else find_available_port()

    # ── Graceful shutdown ──
    def shutdown(signum, frame):
        print("\n[Main] shutting down...")
        system.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # ── Open browser ──
    url = f"http://localhost:{port}/"
    if not args.no_browser:
        print(f"[Main] opening browser: {url}")
        webbrowser.open(url)
    else:
        print(f"[Main] dashboard at: {url}")

    print(f"[Main] Smart Agriculture Simulator running (time_scale={args.time_scale}x)")
    print(f"[Main] Press Ctrl+C to stop")

    # ── Run Flask ──
    try:
        app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
    except KeyboardInterrupt:
        pass
    finally:
        system.stop()


if __name__ == "__main__":
    main()
