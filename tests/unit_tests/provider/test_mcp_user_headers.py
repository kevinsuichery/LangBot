from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from langbot.pkg.api.http.context import ExecutionContext
from langbot.pkg.provider.tools.loaders import mcp as mcp_module
from langbot.pkg.provider.tools.loaders.mcp import MCPLoader, MCPSessionStatus
from langbot.pkg.provider.tools.loaders.mcp_user_headers import (
    MCPUserHeadersError,
    resolve_user_headers,
    user_header_session_limit,
)
from langbot_plugin.api.entities.builtin.resource.tool import LLMTool


TEST_CONTEXT = ExecutionContext(
    instance_uuid='instance-a',
    workspace_uuid='workspace-a',
    placement_generation=1,
)
TEST_RESOLVER_TOKEN = 'resolver-service-token-with-at-least-32-chars'


def _app() -> SimpleNamespace:
    return SimpleNamespace(
        logger=Mock(),
        workspace_service=SimpleNamespace(
            get_execution_binding=AsyncMock(
                return_value=SimpleNamespace(
                    instance_uuid=TEST_CONTEXT.instance_uuid,
                    workspace_uuid=TEST_CONTEXT.workspace_uuid,
                    placement_generation=TEST_CONTEXT.placement_generation,
                )
            )
        ),
    )


def _tool() -> LLMTool:
    return LLMTool(
        name='whoami',
        human_desc='Return the authenticated user',
        description='Return the authenticated user',
        parameters={'type': 'object'},
        func=lambda: None,
    )


def _server_config(*, max_sessions: int = 128) -> dict:
    return {
        'uuid': 'server-a',
        'mode': 'http',
        'url': 'https://mcp.example.test/mcp',
        'headers': {'X-Static': 'static-value'},
        'user_headers': {
            'type': 'env',
            'actor_source': 'sender_id',
            'max_sessions': max_sessions,
            'actors': {
                'user-a': {
                    'headers': {
                        'X-Access-Key': 'TEST_MCP_USER_A_ACCESS_KEY',
                        'X-Secret-Key': 'TEST_MCP_USER_A_SECRET_KEY',
                    }
                },
                'user-b': {
                    'headers': {
                        'X-Access-Key': 'TEST_MCP_USER_B_ACCESS_KEY',
                        'X-Secret-Key': 'TEST_MCP_USER_B_SECRET_KEY',
                    }
                },
            },
        },
    }


def _query(sender_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        instance_uuid=TEST_CONTEXT.instance_uuid,
        workspace_uuid=TEST_CONTEXT.workspace_uuid,
        placement_generation=TEST_CONTEXT.placement_generation,
        sender_id=sender_id,
        variables={},
        bot_uuid=None,
        pipeline_uuid=None,
        query_uuid=f'query-{sender_id}',
    )


class _FakeSession:
    def __init__(
        self,
        server_name: str,
        server_config: dict,
        enable: bool,
        ap: object,
        execution_context: ExecutionContext,
    ) -> None:
        self.server_name = server_name
        self.server_uuid = server_config['uuid']
        self.server_config = server_config
        self.enable = enable
        self.ap = ap
        self.execution_context = execution_context
        self.status = MCPSessionStatus.CONNECTING
        self.session = None
        self.functions = [_tool()]
        self.invoke_count = 0
        self.shutdown_count = 0

    async def start(self) -> None:
        self.status = MCPSessionStatus.CONNECTED
        self.session = object()

    async def shutdown(self) -> None:
        self.shutdown_count += 1
        self.status = MCPSessionStatus.ERROR
        self.session = None

    def get_tools(self) -> list[LLMTool]:
        return self.functions

    async def invoke_mcp_tool(self, _name: str, _parameters: dict, *, query: object) -> dict:
        self.invoke_count += 1
        return {
            'sender_id': query.sender_id,
            'headers': dict(self.server_config.get('headers', {})),
        }


def _discovery_session(config: dict) -> _FakeSession:
    session = _FakeSession('cordys', config, True, _app(), TEST_CONTEXT)
    session.status = MCPSessionStatus.CONNECTED
    session.session = object()
    return session


def _set_test_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('TEST_MCP_USER_A_ACCESS_KEY', 'access-a')
    monkeypatch.setenv('TEST_MCP_USER_A_SECRET_KEY', 'secret-a')
    monkeypatch.setenv('TEST_MCP_USER_B_ACCESS_KEY', 'access-b')
    monkeypatch.setenv('TEST_MCP_USER_B_SECRET_KEY', 'secret-b')


@pytest.mark.asyncio
async def test_resolve_user_headers_uses_exact_sender_mapping_without_exposing_actor_id():
    resolved = await resolve_user_headers(
        _server_config(),
        _query('user-a'),
        environ={
            'TEST_MCP_USER_A_ACCESS_KEY': 'access-a',
            'TEST_MCP_USER_A_SECRET_KEY': 'secret-a',
        },
    )

    assert resolved is not None
    assert resolved.headers == {'X-Access-Key': 'access-a', 'X-Secret-Key': 'secret-a'}
    assert resolved.actor_key != 'user-a'
    assert 'access-a' not in resolved.credential_fingerprint
    assert 'secret-a' not in resolved.credential_fingerprint


@pytest.mark.parametrize(
    ('config_change', 'expected_message'),
    [
        ({'actor_source': 'launcher_id'}, 'must use sender_id'),
        ({'actors': {}}, 'actors must be a non-empty object'),
        ({'max_sessions': 0}, 'positive integer'),
    ],
)
@pytest.mark.asyncio
async def test_invalid_or_unmapped_user_header_configuration_fails_closed(config_change, expected_message):
    config = _server_config()
    config['user_headers'].update(config_change)

    with pytest.raises(MCPUserHeadersError, match=expected_message):
        if 'max_sessions' in config_change:
            user_header_session_limit(config)
        else:
            await resolve_user_headers(config, _query('user-a'), environ={})


def _cordys_server_config() -> dict:
    config = _server_config()
    config['user_headers'] = {
        'type': 'cordys',
        'actor_source': 'sender_id',
        'resolver_url': 'https://cordys.example.test/internal/agent/credential/resolve',
        'service_token_env': 'TEST_CORDYS_RESOLVER_TOKEN',
        'platform': 'LARK',
        'mcp_server_id': 'cordys-crm',
        'resolver_timeout_seconds': 3,
        'max_sessions': 32,
    }
    return config


@pytest.mark.asyncio
async def test_cordys_resolver_uses_sender_id_and_returns_official_headers():
    client = SimpleNamespace(
        post=AsyncMock(
            return_value=httpx.Response(
                200,
                json={
                    'code': 100200,
                    'data': {
                        'actorId': 'cordys-user-a',
                        'credentialId': 'key-a',
                        'headers': {
                            'X-Access-Key': 'access-a',
                            'X-Secret-Key': 'secret-a',
                        },
                    },
                },
            )
        )
    )

    resolved = await resolve_user_headers(
        _cordys_server_config(),
        _query('ou-user-a'),
        environ={'TEST_CORDYS_RESOLVER_TOKEN': TEST_RESOLVER_TOKEN},
        http_client=client,
    )

    assert resolved is not None
    assert resolved.headers == {'X-Access-Key': 'access-a', 'X-Secret-Key': 'secret-a'}
    assert resolved.actor_key != 'cordys-user-a'
    assert 'access-a' not in resolved.credential_fingerprint
    client.post.assert_awaited_once_with(
        'https://cordys.example.test/internal/agent/credential/resolve',
        json={
            'platform': 'LARK',
            'externalUserId': 'ou-user-a',
            'mcpServerId': 'cordys-crm',
        },
        headers={'Authorization': f'Bearer {TEST_RESOLVER_TOKEN}'},
        timeout=3.0,
        follow_redirects=False,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize('service_token', ['', 'too-short'])
async def test_cordys_resolver_rejects_invalid_service_token_without_calling_endpoint(service_token):
    client = SimpleNamespace(post=AsyncMock())
    environ = {'TEST_CORDYS_RESOLVER_TOKEN': service_token} if service_token else {}

    with pytest.raises(MCPUserHeadersError, match='service credential is unavailable'):
        await resolve_user_headers(
            _cordys_server_config(),
            _query('ou-user-a'),
            environ=environ,
            http_client=client,
        )

    client.post.assert_not_awaited()


@pytest.mark.asyncio
async def test_cordys_resolver_rejects_unexpected_response_headers():
    client = SimpleNamespace(
        post=AsyncMock(
            return_value=httpx.Response(
                200,
                json={
                    'data': {
                        'actorId': 'cordys-user-a',
                        'headers': {'Authorization': 'unexpected'},
                    }
                },
            )
        )
    )

    with pytest.raises(MCPUserHeadersError, match='unexpected headers'):
        await resolve_user_headers(
            _cordys_server_config(),
            _query('ou-user-a'),
            environ={'TEST_CORDYS_RESOLVER_TOKEN': TEST_RESOLVER_TOKEN},
            http_client=client,
        )


@pytest.mark.asyncio
async def test_loader_uses_distinct_cached_mcp_sessions_for_each_sender(monkeypatch: pytest.MonkeyPatch):
    _set_test_credentials(monkeypatch)
    monkeypatch.setattr(mcp_module, 'RuntimeMCPSession', _FakeSession)
    loader = MCPLoader(_app())
    discovery = _discovery_session(_server_config())
    loader._register_session(TEST_CONTEXT, discovery.server_name, discovery)

    result_a = await loader.invoke_tool('whoami', {}, _query('user-a'))
    result_b = await loader.invoke_tool('whoami', {}, _query('user-b'))
    result_a_again = await loader.invoke_tool('whoami', {}, _query('user-a'))

    assert result_a['headers']['X-Access-Key'] == 'access-a'
    assert result_b['headers']['X-Access-Key'] == 'access-b'
    assert result_a_again['headers']['X-Access-Key'] == 'access-a'
    assert len(loader._actor_sessions) == 2
    assert sorted(session.invoke_count for session in loader._actor_sessions.values()) == [1, 2]
    assert all('user-a' not in key and 'user-b' not in key for key in loader._actor_sessions)
    assert discovery.invoke_count == 0


@pytest.mark.asyncio
async def test_loader_rejects_unmapped_sender_without_using_discovery_session(monkeypatch: pytest.MonkeyPatch):
    _set_test_credentials(monkeypatch)
    monkeypatch.setattr(mcp_module, 'RuntimeMCPSession', _FakeSession)
    loader = MCPLoader(_app())
    discovery = _discovery_session(_server_config())
    loader._register_session(TEST_CONTEXT, discovery.server_name, discovery)

    with pytest.raises(MCPUserHeadersError, match='No MCP credentials'):
        await loader.invoke_tool('whoami', {}, _query('user-c'))

    assert loader._actor_sessions == {}
    assert discovery.invoke_count == 0


@pytest.mark.asyncio
async def test_credential_rotation_replaces_and_closes_the_previous_actor_session(monkeypatch: pytest.MonkeyPatch):
    _set_test_credentials(monkeypatch)
    monkeypatch.setattr(mcp_module, 'RuntimeMCPSession', _FakeSession)
    loader = MCPLoader(_app())
    discovery = _discovery_session(_server_config())
    loader._register_session(TEST_CONTEXT, discovery.server_name, discovery)

    await loader.invoke_tool('whoami', {}, _query('user-a'))
    previous = next(iter(loader._actor_sessions.values()))
    monkeypatch.setenv('TEST_MCP_USER_A_ACCESS_KEY', 'rotated-access-a')

    result = await loader.invoke_tool('whoami', {}, _query('user-a'))

    assert result['headers']['X-Access-Key'] == 'rotated-access-a'
    assert previous.shutdown_count == 1
    assert len(loader._actor_sessions) == 1


@pytest.mark.asyncio
async def test_actor_session_limit_evicts_the_least_recently_used_session(monkeypatch: pytest.MonkeyPatch):
    _set_test_credentials(monkeypatch)
    monkeypatch.setattr(mcp_module, 'RuntimeMCPSession', _FakeSession)
    loader = MCPLoader(_app())
    discovery = _discovery_session(_server_config(max_sessions=1))
    loader._register_session(TEST_CONTEXT, discovery.server_name, discovery)

    await loader.invoke_tool('whoami', {}, _query('user-a'))
    evicted = next(iter(loader._actor_sessions.values()))
    await loader.invoke_tool('whoami', {}, _query('user-b'))

    assert evicted.shutdown_count == 1
    assert len(loader._actor_sessions) == 1


@pytest.mark.asyncio
async def test_static_mcp_configuration_keeps_using_the_discovery_session():
    loader = MCPLoader(_app())
    config = _server_config()
    config.pop('user_headers')
    discovery = _discovery_session(config)
    loader._register_session(TEST_CONTEXT, discovery.server_name, discovery)

    result = await loader.invoke_tool('whoami', {}, _query('unmapped-user'))

    assert result['sender_id'] == 'unmapped-user'
    assert result['headers'] == {'X-Static': 'static-value'}
    assert discovery.invoke_count == 1
    assert loader._actor_sessions == {}


@pytest.mark.asyncio
async def test_removing_server_closes_discovery_and_actor_sessions(monkeypatch: pytest.MonkeyPatch):
    _set_test_credentials(monkeypatch)
    monkeypatch.setattr(mcp_module, 'RuntimeMCPSession', _FakeSession)
    loader = MCPLoader(_app())
    discovery = _discovery_session(_server_config())
    loader._register_session(TEST_CONTEXT, discovery.server_name, discovery)
    await loader.invoke_tool('whoami', {}, _query('user-a'))
    actor = next(iter(loader._actor_sessions.values()))

    await loader.remove_mcp_server(TEST_CONTEXT, discovery.server_name)

    assert discovery.shutdown_count == 1
    assert actor.shutdown_count == 1
    assert loader.sessions == {}
    assert loader._actor_sessions == {}
