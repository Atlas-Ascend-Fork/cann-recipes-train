#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Short-lived callback relay for Vime-managed AgentENV rollouts.

The bearer value is the per-trial Vime adapter session id.  The relay only
checks that a non-trivial bearer is present; the NPU adapter remains the
authority that accepts or rejects the session id.
"""

from __future__ import annotations

import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


ALLOWED_PATHS = {"/v1/models", "/v1/responses", "/healthz"}
MAX_BODY_BYTES = 2 * 1024 * 1024


class RelayHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _relay(self) -> None:
        if self.path not in ALLOWED_PATHS:
            self.send_error(404)
            return
        authorization = self.headers.get("Authorization", "")
        if not authorization.startswith("Bearer ") or len(authorization) < 24:
            self.send_error(401)
            return
        try:
            body_size = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.send_error(400)
            return
        if body_size > MAX_BODY_BYTES:
            self.send_error(413)
            return

        body = self.rfile.read(body_size) if body_size else None
        headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower() not in {"host", "connection", "content-length", "transfer-encoding"}
        }
        upstream = http.client.HTTPConnection("127.0.0.1", 18002, timeout=900)
        try:
            upstream.request(self.command, self.path, body=body, headers=headers)
            response = upstream.getresponse()
            response_body = response.read()
            self.send_response(response.status)
            for key, value in response.getheaders():
                if key.lower() not in {"connection", "content-length", "transfer-encoding"}:
                    self.send_header(key, value)
            self.send_header("Content-Length", str(len(response_body)))
            self.end_headers()
            self.wfile.write(response_body)
        finally:
            upstream.close()

    do_GET = _relay
    do_POST = _relay

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"relay {self.client_address[0]} {self.command} {self.path} {args[1]}", flush=True)


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", 18080), RelayHandler).serve_forever()
