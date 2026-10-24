"""Create one disposable Daytona sandbox, execute echo, and confirm deletion.

Run from the repository root with DAYTONA_API_KEY already exported:
    uv run --no-project --with daytona==0.225.0 python \
        torchtitan/experiments/rl/tests/daytona_lifecycle_smoke.py

The network must permit Daytona. Proxy configuration does not grant access.
"""

import os
import time
import uuid
import warnings
from unittest.mock import patch

from daytona import CreateSandboxFromImageParams, Daytona, DaytonaConfig, Resources
from daytona_api_client import Configuration


def main() -> None:
    if not os.environ.get("DAYTONA_API_KEY"):
        raise RuntimeError("Export DAYTONA_API_KEY before running this smoke test")

    proxy = os.environ.get("https_proxy") or os.environ.get("HTTPS_PROXY")
    original_init = Configuration.__init__

    def configure_proxy(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if proxy:
            self.proxy = proxy

    # The synchronous SDK does not read HTTPS_PROXY for generated API clients.
    # Set it while constructing the client; its toolbox inherits this config.
    # Polling avoids a separate websocket transport for this short smoke test.
    with patch.object(Configuration, "__init__", configure_proxy):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Polling-only mode.*")
            client = Daytona(DaytonaConfig(use_deprecated_polling=True))

    name = "andy-daytona-smoke-" + uuid.uuid4().hex[:12]
    sandbox = None
    started = time.monotonic()
    print(f"creating: {name}", flush=True)
    try:
        sandbox = client.create(
            CreateSandboxFromImageParams(
                name=name,
                image="debian:bookworm-slim",
                resources=Resources(cpu=1, memory=1, disk=1),
                labels={"owner": "andy", "purpose": "local-daytona-smoke"},
                ephemeral=True,
                auto_stop_interval=5,
                ttl_minutes=15,
            ),
            timeout=120,
        )
        print(f"created: {sandbox.id}", flush=True)
        print(f"create_seconds: {time.monotonic() - started:.2f}", flush=True)
        result = sandbox.process.exec("echo hello-from-daytona", timeout=30)
        print(f"exit_code: {result.exit_code}\n{result.result}", flush=True)
        if result.exit_code != 0 or result.result.strip() != "hello-from-daytona":
            raise RuntimeError("Unexpected echo response")
    finally:
        if sandbox is not None:
            print(f"deleting: {sandbox.id}", flush=True)
            client.delete(sandbox, timeout=60, wait=True)
            print("deleted: true", flush=True)


if __name__ == "__main__":
    main()
