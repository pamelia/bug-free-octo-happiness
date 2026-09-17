"""Wait for the worker-cluster rendezvous DNS record before torchrun starts."""

import argparse
import socket
import time


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("host")
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()

    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        try:
            socket.getaddrinfo(args.host, 29500)
            return
        except socket.gaierror:
            time.sleep(2)

    raise SystemExit(f"rendezvous DNS did not become ready: {args.host}")


if __name__ == "__main__":
    main()
