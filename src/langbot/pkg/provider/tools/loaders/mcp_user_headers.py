from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from typing import Mapping
from urllib.parse import urlparse

import httpx


USER_HEADERS_CONFIG_KEY = 'user_headers'
USER_HEADERS_DEFAULT_MAX_SESSIONS = 128
USER_HEADERS_MAX_SESSIONS_LIMIT = 1024

_HEADER_NAME_PATTERN = re.compile(r"^[!#$%&'*+.^_`|~0-9A-Za-z-]+$")
_ENV_NAME_PATTERN = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')
_MCP_SERVER_ID_PATTERN = re.compile(r'^[A-Za-z0-9._:-]+$')
_CORDYS_REQUIRED_HEADERS = frozenset({'X-Access-Key', 'X-Secret-Key'})
_CORDYS_DEFAULT_TIMEOUT_SECONDS = 5.0
_CORDYS_MAX_TIMEOUT_SECONDS = 30.0
_CORDYS_MINIMUM_SERVICE_TOKEN_LENGTH = 32


class MCPUserHeadersError(ValueError):
    """A user-scoped MCP header configuration cannot be resolved safely."""


@dataclass(frozen=True)
class ResolvedMCPUserHeaders:
    actor_key: str
    credential_fingerprint: str
    headers: dict[str, str]


def has_user_headers(server_config: Mapping[str, object]) -> bool:
    return USER_HEADERS_CONFIG_KEY in server_config


def user_header_session_limit(server_config: Mapping[str, object]) -> int:
    config = server_config.get(USER_HEADERS_CONFIG_KEY)
    if not isinstance(config, Mapping):
        raise MCPUserHeadersError('MCP user header configuration must be an object')

    value = config.get('max_sessions', USER_HEADERS_DEFAULT_MAX_SESSIONS)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise MCPUserHeadersError('MCP user header max_sessions must be a positive integer')
    return min(value, USER_HEADERS_MAX_SESSIONS_LIMIT)


def validate_user_headers_config(server_config: Mapping[str, object]) -> None:
    if not has_user_headers(server_config):
        return

    config = server_config.get(USER_HEADERS_CONFIG_KEY)
    if not isinstance(config, Mapping):
        raise MCPUserHeadersError('MCP user header configuration must be an object')
    if config.get('actor_source', 'sender_id') != 'sender_id':
        raise MCPUserHeadersError('MCP user headers must use sender_id as the actor source')
    user_header_session_limit(server_config)

    resolver_type = config.get('type', 'env')
    if resolver_type == 'env':
        _validate_env_config(config)
        return
    if resolver_type == 'cordys':
        _validate_cordys_config(config)
        return
    raise MCPUserHeadersError('Unsupported MCP user header resolver type')


def _validate_env_config(config: Mapping[str, object]) -> None:
    actors = config.get('actors')
    if not isinstance(actors, Mapping) or not actors:
        raise MCPUserHeadersError('MCP user header actors must be a non-empty object')
    for raw_actor_id, actor_config in actors.items():
        if not str(raw_actor_id or '').strip():
            raise MCPUserHeadersError('MCP user header actor identity is invalid')
        if not isinstance(actor_config, Mapping):
            raise MCPUserHeadersError('MCP user header actor configuration must be an object')
        header_refs = actor_config.get('headers')
        if not isinstance(header_refs, Mapping) or not header_refs:
            raise MCPUserHeadersError('MCP user header mapping must be a non-empty object')
        if len(header_refs) > 32:
            raise MCPUserHeadersError('MCP user header mapping exceeds the maximum of 32 headers')
        for raw_header_name, raw_env_name in header_refs.items():
            header_name = str(raw_header_name or '').strip()
            env_name = str(raw_env_name or '').strip()
            if not _HEADER_NAME_PATTERN.fullmatch(header_name):
                raise MCPUserHeadersError('MCP user header name is invalid')
            if not _ENV_NAME_PATTERN.fullmatch(env_name):
                raise MCPUserHeadersError('MCP user header environment reference is invalid')


def _validate_cordys_config(config: Mapping[str, object]) -> None:
    resolver_url = str(config.get('resolver_url', '') or '').strip()
    parsed_url = urlparse(resolver_url)
    if parsed_url.scheme not in ('http', 'https') or not parsed_url.hostname:
        raise MCPUserHeadersError('Cordys credential resolver URL must use HTTP or HTTPS')
    if parsed_url.username or parsed_url.password or parsed_url.fragment:
        raise MCPUserHeadersError('Cordys credential resolver URL contains unsupported components')

    service_token_env = str(config.get('service_token_env', '') or '').strip()
    if not _ENV_NAME_PATTERN.fullmatch(service_token_env):
        raise MCPUserHeadersError('Cordys credential resolver service token reference is invalid')

    mcp_server_id = str(config.get('mcp_server_id', '') or '').strip()
    if not _MCP_SERVER_ID_PATTERN.fullmatch(mcp_server_id):
        raise MCPUserHeadersError('Cordys credential resolver MCP server ID is invalid')

    platform = str(config.get('platform', 'LARK') or '').strip()
    if not platform or len(platform) > 32:
        raise MCPUserHeadersError('Cordys credential resolver platform is invalid')

    timeout = config.get('resolver_timeout_seconds', _CORDYS_DEFAULT_TIMEOUT_SECONDS)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
        raise MCPUserHeadersError('Cordys credential resolver timeout must be positive')
    if timeout > _CORDYS_MAX_TIMEOUT_SECONDS:
        raise MCPUserHeadersError('Cordys credential resolver timeout exceeds the maximum')


async def resolve_user_headers(
    server_config: Mapping[str, object],
    query: object,
    *,
    environ: Mapping[str, str] | None = None,
    http_client: httpx.AsyncClient | None = None,
) -> ResolvedMCPUserHeaders | None:
    """Resolve one user's MCP headers from the configured credential source.

    Secret values are returned only to the MCP connection boundary and are never
    written into the MCP database.
    """

    if not has_user_headers(server_config):
        return None

    validate_user_headers_config(server_config)
    config = server_config.get(USER_HEADERS_CONFIG_KEY)
    assert isinstance(config, Mapping)

    actor_id = str(getattr(query, 'sender_id', '') or '').strip()
    if not actor_id:
        raise MCPUserHeadersError('MCP user identity is unavailable')

    if config.get('type', 'env') == 'cordys':
        return await _resolve_cordys_user_headers(
            config,
            actor_id,
            environ=os.environ if environ is None else environ,
            http_client=http_client,
        )

    return _resolve_env_user_headers(config, actor_id, environ=os.environ if environ is None else environ)


def _resolve_env_user_headers(
    config: Mapping[str, object],
    actor_id: str,
    *,
    environ: Mapping[str, str],
) -> ResolvedMCPUserHeaders:
    actors = config.get('actors')
    assert isinstance(actors, Mapping)
    actor_config = actors.get(actor_id)
    if not isinstance(actor_config, Mapping):
        raise MCPUserHeadersError('No MCP credentials are configured for the current user')

    header_refs = actor_config.get('headers')
    assert isinstance(header_refs, Mapping)

    resolved: dict[str, str] = {}
    for raw_header_name, raw_env_name in header_refs.items():
        header_name = str(raw_header_name or '').strip()
        env_name = str(raw_env_name or '').strip()
        value = environ.get(env_name, '')
        if not value:
            raise MCPUserHeadersError('A required MCP user credential is unavailable')
        if '\r' in value or '\n' in value:
            raise MCPUserHeadersError('MCP user credential contains invalid header characters')
        resolved[header_name] = value

    return _build_resolved_headers(actor_id, resolved)


async def _resolve_cordys_user_headers(
    config: Mapping[str, object],
    external_user_id: str,
    *,
    environ: Mapping[str, str],
    http_client: httpx.AsyncClient | None,
) -> ResolvedMCPUserHeaders:
    service_token_env = str(config['service_token_env'])
    service_token = environ.get(service_token_env, '')
    if len(service_token) < _CORDYS_MINIMUM_SERVICE_TOKEN_LENGTH:
        raise MCPUserHeadersError('Cordys credential resolver service credential is unavailable')
    if '\r' in service_token or '\n' in service_token:
        raise MCPUserHeadersError('Cordys credential resolver service credential is invalid')

    request_body = {
        'platform': str(config.get('platform', 'LARK')),
        'externalUserId': external_user_id,
        'mcpServerId': str(config['mcp_server_id']),
    }
    request_headers = {'Authorization': f'Bearer {service_token}'}
    timeout = float(config.get('resolver_timeout_seconds', _CORDYS_DEFAULT_TIMEOUT_SECONDS))

    try:
        if http_client is None:
            async with httpx.AsyncClient(
                timeout=timeout,
                follow_redirects=False,
                trust_env=False,
            ) as client:
                response = await client.post(
                    str(config['resolver_url']),
                    json=request_body,
                    headers=request_headers,
                )
        else:
            response = await http_client.post(
                str(config['resolver_url']),
                json=request_body,
                headers=request_headers,
                timeout=timeout,
                follow_redirects=False,
            )
    except httpx.HTTPError as exc:
        raise MCPUserHeadersError('Cordys credential resolver request failed') from exc

    if response.status_code != httpx.codes.OK:
        raise MCPUserHeadersError(f'Cordys credential resolver rejected the request ({response.status_code})')

    try:
        payload = response.json()
    except ValueError as exc:
        raise MCPUserHeadersError('Cordys credential resolver returned invalid JSON') from exc
    if not isinstance(payload, Mapping):
        raise MCPUserHeadersError('Cordys credential resolver returned an invalid response')
    data = payload.get('data', payload)
    if not isinstance(data, Mapping):
        raise MCPUserHeadersError('Cordys credential resolver returned an invalid response')

    actor_id = str(data.get('actorId', '') or '').strip()
    raw_headers = data.get('headers')
    if not actor_id or not isinstance(raw_headers, Mapping):
        raise MCPUserHeadersError('Cordys credential resolver returned incomplete credentials')

    resolved: dict[str, str] = {}
    for raw_header_name, raw_value in raw_headers.items():
        header_name = str(raw_header_name or '').strip()
        value = str(raw_value or '')
        if not _HEADER_NAME_PATTERN.fullmatch(header_name) or not value:
            raise MCPUserHeadersError('Cordys credential resolver returned invalid headers')
        if '\r' in value or '\n' in value:
            raise MCPUserHeadersError('Cordys credential resolver returned invalid headers')
        resolved[header_name] = value
    if set(resolved) != _CORDYS_REQUIRED_HEADERS:
        raise MCPUserHeadersError('Cordys credential resolver returned unexpected headers')

    return _build_resolved_headers(actor_id, resolved)


def _build_resolved_headers(actor_id: str, headers: dict[str, str]) -> ResolvedMCPUserHeaders:
    actor_key = hashlib.sha256(actor_id.encode('utf-8')).hexdigest()[:24]
    fingerprint_payload = json.dumps(headers, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    credential_fingerprint = hashlib.sha256(fingerprint_payload.encode('utf-8')).hexdigest()[:24]
    return ResolvedMCPUserHeaders(
        actor_key=actor_key,
        credential_fingerprint=credential_fingerprint,
        headers=headers,
    )
