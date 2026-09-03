from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from typing import Mapping


USER_HEADERS_CONFIG_KEY = 'user_headers'
USER_HEADERS_DEFAULT_MAX_SESSIONS = 128
USER_HEADERS_MAX_SESSIONS_LIMIT = 1024

_HEADER_NAME_PATTERN = re.compile(r"^[!#$%&'*+.^_`|~0-9A-Za-z-]+$")
_ENV_NAME_PATTERN = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')


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
    if config.get('type', 'env') != 'env':
        raise MCPUserHeadersError('Only env MCP user headers are supported')
    if config.get('actor_source', 'sender_id') != 'sender_id':
        raise MCPUserHeadersError('MCP user headers must use sender_id as the actor source')
    user_header_session_limit(server_config)

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


def resolve_user_headers(
    server_config: Mapping[str, object],
    query: object,
    *,
    environ: Mapping[str, str] | None = None,
) -> ResolvedMCPUserHeaders | None:
    """Resolve one user's MCP headers from environment-variable references.

    Configuration stores only environment variable names. Secret values remain
    in the process environment and are never written into the MCP database.
    """

    if not has_user_headers(server_config):
        return None

    validate_user_headers_config(server_config)
    config = server_config.get(USER_HEADERS_CONFIG_KEY)
    assert isinstance(config, Mapping)

    actor_id = str(getattr(query, 'sender_id', '') or '').strip()
    if not actor_id:
        raise MCPUserHeadersError('MCP user identity is unavailable')

    actors = config.get('actors')
    assert isinstance(actors, Mapping)
    actor_config = actors.get(actor_id)
    if not isinstance(actor_config, Mapping):
        raise MCPUserHeadersError('No MCP credentials are configured for the current user')

    header_refs = actor_config.get('headers')
    assert isinstance(header_refs, Mapping)

    source = os.environ if environ is None else environ
    resolved: dict[str, str] = {}
    for raw_header_name, raw_env_name in header_refs.items():
        header_name = str(raw_header_name or '').strip()
        env_name = str(raw_env_name or '').strip()
        value = source.get(env_name, '')
        if not value:
            raise MCPUserHeadersError('A required MCP user credential is unavailable')
        if '\r' in value or '\n' in value:
            raise MCPUserHeadersError('MCP user credential contains invalid header characters')
        resolved[header_name] = value

    actor_key = hashlib.sha256(actor_id.encode('utf-8')).hexdigest()[:24]
    fingerprint_payload = json.dumps(resolved, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    credential_fingerprint = hashlib.sha256(fingerprint_payload.encode('utf-8')).hexdigest()[:24]
    return ResolvedMCPUserHeaders(
        actor_key=actor_key,
        credential_fingerprint=credential_fingerprint,
        headers=resolved,
    )
