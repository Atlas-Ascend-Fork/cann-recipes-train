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

"""Small async v1 HTTP client. No queues, reward policy or global sessions."""

from __future__ import annotations

import asyncio
import json
import logging
from urllib.parse import quote

import aiohttp

from .protocol import EvaluatorConfig, ProtocolError

logger = logging.getLogger(__name__)


class ServiceUnavailable(RuntimeError):
    pass


class SubmissionUnknown(ServiceUnavailable):
    """Submission attempts exhausted without a job id; remote jobs may exist."""

    def __init__(self, message, *, attempt_count=1):
        self.attempt_count = attempt_count
        super().__init__(message)


class HTTPError(ProtocolError):
    def __init__(self, status, message):
        self.status = status
        super().__init__(message)


class EvaluatorClient:
    def __init__(self, config: EvaluatorConfig):
        self.config = config
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.config.request_timeout))
        return self

    async def __aexit__(self, *exc):
        await self.session.close()

    async def submit(self, *, op: str, submission: str, alias: str, filename: str | None = None):
        filename = filename or f"{op}.txt"
        if not op or any(c in op for c in "/\\\r\n"):
            raise ValueError("op must be an explicit operator name")
        if not filename.endswith(".txt") or any(c in filename for c in "/\\\r\n"):
            raise ValueError("filename must be a plain .txt basename")

        async def upload(_):
            # aiohttp consumes FormData; retries must rebuild the full upload.
            form = aiohttp.FormData()
            form.add_field("alias", alias)
            form.add_field("device_id", "-1")
            form.add_field("files", submission, filename=filename, content_type="text/plain")
            return await self._request("POST", "submit", data=form)

        try:
            return await self._retry(
                upload, self.config.request_retries, label=f"submit {op} (remote duplicates possible)"
            )
        except ServiceUnavailable as exc:
            count = self.config.request_retries + 1
            raise SubmissionUnknown(f"submission failed after {count} attempts: {exc}", attempt_count=count) from exc

    async def status(self, job_id: str):
        return await self._get(f"jobs/{quote(job_id, safe='')}")

    async def result(self, job_id: str):
        return await self._get(f"jobs/{quote(job_id, safe='')}/result")

    async def log(self, job_id: str, stage: str) -> str:
        if stage not in {"compile", "evaluate"}:
            raise ValueError("log stage must be compile or evaluate")
        # Use the configured service, not an arbitrary URL from the submission/result.
        path = f"logs/{stage}.log"
        return await self._get(f"/download/{quote(job_id, safe='')}/{path}", text=True)

    async def delete(self, job_id: str) -> dict:
        outcome = {"delete_attempted": True, "delete_succeeded": False}

        async def cancel(attempt):
            outcome["delete_attempt_count"] = attempt
            try:
                payload = await self._request("DELETE", f"jobs/{quote(job_id, safe='')}")
                if payload.get("deleted") != job_id:
                    raise ProtocolError("DELETE did not acknowledge the requested job")
                return {**outcome, "delete_succeeded": True}
            except HTTPError as exc:
                if exc.status != 404:
                    raise
                if await self._job_is_absent(job_id):
                    return {**outcome, "delete_succeeded": True, "already_absent": True}
                raise ServiceUnavailable(str(exc)) from exc

        try:
            return await self._retry(
                cancel,
                self.config.delete_retries,
                label=f"DELETE {job_id}",
                errors=(ServiceUnavailable, ProtocolError),
            )
        except (ServiceUnavailable, ProtocolError) as exc:
            logger.warning("CANN DELETE failed: job_id=%s, error=%s", job_id, exc)
            return {**outcome, "delete_error": str(exc)}

    async def _job_is_absent(self, job_id: str) -> bool:
        """A missing DELETE route is not proof the job was removed."""
        try:
            await self.status(job_id)
        except HTTPError as exc:
            if exc.status == 404:
                return True
            raise
        except ServiceUnavailable:
            pass
        return False

    def _url(self, suffix: str) -> str:
        path = suffix if suffix.startswith("/") else f"/api/v1/{suffix}"
        return f"{self.config.url.rstrip('/')}{path}"

    async def _retry(self, operation, retries, *, label, errors=(ServiceUnavailable,)):
        for attempt in range(retries + 1):
            try:
                return await operation(attempt + 1)
            except HTTPError:
                # Authentication/configuration errors are never transient.
                raise
            except errors as exc:
                if attempt == retries:
                    raise
                logger.warning("CANN %s attempt %s failed; retrying: %s", label, attempt + 1, exc)
                await asyncio.sleep(min(2**attempt, 8))

    async def _request(self, method: str, suffix: str, *, text: bool = False, **kwargs):
        try:
            async with self.session.request(method, self._url(suffix), allow_redirects=False, **kwargs) as response:
                body = await response.text()
                if response.status >= 500 or response.status in {408, 429}:
                    raise ServiceUnavailable(f"HTTP {response.status}: {body[:2000]}")
                if not 200 <= response.status < 300:
                    raise HTTPError(response.status, f"{method} {suffix}: HTTP {response.status}: {body[:2000]}")
                if text:
                    return body
                try:
                    payload = json.loads(body)
                except json.JSONDecodeError as exc:
                    raise ProtocolError(f"{method} {suffix} returned non-JSON") from exc
                if not isinstance(payload, dict):
                    raise ProtocolError("v1 responses must be JSON objects")
                return payload
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            raise ServiceUnavailable(str(exc)) from exc

    async def _get(self, suffix, **kwargs):
        return await self._retry(
            lambda _: self._request("GET", suffix, **kwargs), self.config.request_retries, label=f"GET {suffix}"
        )
