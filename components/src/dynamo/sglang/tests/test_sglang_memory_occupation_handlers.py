# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from dynamo.sglang.request_handlers.handler_base import (
    BaseWorkerHandler,
    SGLangEnginePauseController,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


@pytest.fixture(autouse=True)
def _stub_sglang_io_struct(monkeypatch):
    """Keep unit tests independent from CUDA-only sglang imports."""

    io_struct = types.ModuleType("sglang.srt.managers.io_struct")

    class _Req:
        def __init__(self, tags=None):
            self.tags = tags

    io_struct.PauseGenerationReqInput = _Req
    io_struct.ReleaseMemoryOccupationReqInput = _Req
    io_struct.ResumeMemoryOccupationReqInput = _Req
    io_struct.ContinueGenerationReqInput = _Req

    monkeypatch.setitem(sys.modules, "sglang.srt.managers.io_struct", io_struct)


class _TestWorkerHandler(BaseWorkerHandler):
    async def generate(self, request, context):
        yield {}


@pytest.fixture
def handler():
    handler = _TestWorkerHandler.__new__(_TestWorkerHandler)
    handler.engine = SimpleNamespace(
        tokenizer_manager=SimpleNamespace(
            pause_generation=AsyncMock(),
            release_memory_occupation=AsyncMock(),
            resume_memory_occupation=AsyncMock(),
            continue_generation=AsyncMock(),
            auto_create_handle_loop=MagicMock(),
            rid_to_state={},
            flush_cache=AsyncMock(return_value=SimpleNamespace(success=True)),
            clear_hicache_storage=AsyncMock(return_value=SimpleNamespace(success=True)),
            server_args=SimpleNamespace(hicache_storage_backend=None),
        )
    )
    handler.generate_endpoint = SimpleNamespace(
        unregister_endpoint_instance=AsyncMock(),
        register_endpoint_instance=AsyncMock(),
        connection_id=MagicMock(return_value=0xA1),
    )
    handler._pause_controller = SGLangEnginePauseController(handler.engine)
    handler._pause_lock = asyncio.Lock()
    handler._serving_enabled = True
    handler._endpoint_registered = True
    return handler


@pytest.mark.asyncio
async def test_release_and_resume_are_idempotent(handler):
    first_release = await handler.release_memory_occupation({})
    second_release = await handler.release_memory_occupation({})

    first_resume = await handler.resume_memory_occupation({})
    second_resume = await handler.resume_memory_occupation({})

    assert first_release["status"] == "ok"
    assert second_release["status"] == "ok"
    assert first_resume["status"] == "ok"
    assert second_resume["status"] == "ok"
    assert second_release["message"] == "Memory already released"
    assert second_resume["message"] == "Memory already resumed"

    release_req = (
        handler.engine.tokenizer_manager.release_memory_occupation.await_args.args[0]
    )
    resume_req = (
        handler.engine.tokenizer_manager.resume_memory_occupation.await_args.args[0]
    )
    assert release_req.tags is None
    assert resume_req.tags is None

    handler.engine.tokenizer_manager.pause_generation.assert_awaited_once()
    handler.engine.tokenizer_manager.release_memory_occupation.assert_awaited_once()
    handler.generate_endpoint.unregister_endpoint_instance.assert_awaited_once()

    handler.engine.tokenizer_manager.resume_memory_occupation.assert_awaited_once()
    handler.engine.tokenizer_manager.continue_generation.assert_awaited_once()
    handler.generate_endpoint.register_endpoint_instance.assert_awaited_once()


@pytest.mark.asyncio
async def test_serving_membership_controls_are_idempotent(handler):
    handler._serving_enabled = False
    handler._endpoint_registered = False

    first_enable = await handler.enable_serving({})
    second_enable = await handler.enable_serving({})
    first_disable = await handler.disable_serving({})
    second_disable = await handler.disable_serving({})

    assert first_enable == {
        "status": "ok",
        "message": "Serving enabled",
        "changed": True,
        "serving_enabled": True,
        "endpoint_registered": True,
        "connection_id": 0xA1,
    }
    assert second_enable["changed"] is False
    assert first_disable["changed"] is True
    assert second_disable["changed"] is False
    assert second_disable["serving_enabled"] is False
    assert second_disable["endpoint_registered"] is False
    handler.generate_endpoint.register_endpoint_instance.assert_awaited_once()
    handler.generate_endpoint.unregister_endpoint_instance.assert_awaited_once()


@pytest.mark.asyncio
async def test_enable_serving_restores_intent_when_registration_fails(handler):
    handler._serving_enabled = False
    handler._endpoint_registered = False
    handler.generate_endpoint.register_endpoint_instance = AsyncMock(
        side_effect=RuntimeError("discovery write timeout")
    )

    result = await handler.enable_serving({})

    assert result["status"] == "error"
    assert result["changed"] is False
    assert result["serving_enabled"] is False
    assert result["endpoint_registered"] is False


@pytest.mark.asyncio
async def test_disable_serving_keeps_fail_closed_intent_and_can_retry(handler):
    handler.generate_endpoint.unregister_endpoint_instance = AsyncMock(
        side_effect=[RuntimeError("discovery write timeout"), None]
    )

    first_result = await handler.disable_serving({})
    second_result = await handler.disable_serving({})

    assert first_result["status"] == "error"
    assert first_result["serving_enabled"] is False
    assert first_result["endpoint_registered"] is True
    assert second_result["status"] == "ok"
    assert second_result["changed"] is True
    assert second_result["endpoint_registered"] is False
    assert handler.generate_endpoint.unregister_endpoint_instance.await_count == 2


@pytest.mark.asyncio
async def test_deferred_memory_cycle_stays_unregistered(handler):
    handler._serving_enabled = False
    handler._endpoint_registered = False

    release_result = await handler.release_memory_occupation({})
    resume_result = await handler.resume_memory_occupation({})

    assert release_result["status"] == "ok"
    assert resume_result["status"] == "ok"
    assert handler._serving_enabled is False
    assert handler._endpoint_registered is False
    handler.generate_endpoint.unregister_endpoint_instance.assert_not_awaited()
    handler.generate_endpoint.register_endpoint_instance.assert_not_awaited()


@pytest.mark.asyncio
async def test_deferred_worker_cannot_enable_serving_while_paused(handler):
    handler._serving_enabled = False
    handler._endpoint_registered = False
    await handler.release_memory_occupation({})

    paused_result = await handler.enable_serving({})

    assert paused_result["status"] == "error"
    assert paused_result["changed"] is False
    assert handler._serving_enabled is False
    assert handler._endpoint_registered is False

    await handler.resume_memory_occupation({})
    handler.generate_endpoint.register_endpoint_instance.assert_not_awaited()

    enabled_result = await handler.enable_serving({})

    assert enabled_result["status"] == "ok"
    assert enabled_result["changed"] is True
    handler.generate_endpoint.register_endpoint_instance.assert_awaited_once()


@pytest.mark.asyncio
async def test_memory_occupation_handlers_forward_tags_exactly(handler):
    await handler.release_memory_occupation({"tags": []})
    resume_result = await handler.resume_memory_occupation({"tags": []})

    assert resume_result["status"] == "ok"
    release_req = (
        handler.engine.tokenizer_manager.release_memory_occupation.await_args.args[0]
    )
    resume_req = (
        handler.engine.tokenizer_manager.resume_memory_occupation.await_args.args[0]
    )
    assert release_req.tags == []
    assert resume_req.tags == []

    handler.engine.tokenizer_manager.pause_generation.reset_mock()
    handler.engine.tokenizer_manager.release_memory_occupation.reset_mock()
    handler.engine.tokenizer_manager.resume_memory_occupation.reset_mock()
    handler.engine.tokenizer_manager.continue_generation.reset_mock()
    handler.generate_endpoint.unregister_endpoint_instance.reset_mock()
    handler.generate_endpoint.register_endpoint_instance.reset_mock()

    await handler.release_memory_occupation({"tags": ["weights"]})
    resume_result = await handler.resume_memory_occupation({})

    assert resume_result["status"] == "ok"
    release_req = (
        handler.engine.tokenizer_manager.release_memory_occupation.await_args.args[0]
    )
    resume_req = (
        handler.engine.tokenizer_manager.resume_memory_occupation.await_args.args[0]
    )
    assert release_req.tags == ["weights"]
    assert resume_req.tags is None
    handler.engine.tokenizer_manager.continue_generation.assert_awaited_once()
    handler.generate_endpoint.register_endpoint_instance.assert_awaited_once()


@pytest.mark.asyncio
async def test_resume_recovers_generation_pause_after_failed_release_rollback(handler):
    manager = handler.engine.tokenizer_manager
    manager.release_memory_occupation = AsyncMock(
        side_effect=RuntimeError("release failed")
    )
    failed_continue = AsyncMock(side_effect=RuntimeError("continue failed"))
    manager.continue_generation = failed_continue

    release_result = await handler.release_memory_occupation({})

    assert release_result["status"] == "error"
    assert handler._pause_controller.is_paused is False
    assert handler._pause_controller.needs_resume_recovery is True
    failed_continue.assert_awaited_once()
    handler.generate_endpoint.unregister_endpoint_instance.assert_awaited_once()

    manager.continue_generation = AsyncMock()
    resume_result = await handler.resume_memory_occupation({})

    assert resume_result["status"] == "ok"
    manager.resume_memory_occupation.assert_not_awaited()
    manager.continue_generation.assert_awaited_once()
    handler.generate_endpoint.register_endpoint_instance.assert_awaited_once()
    assert handler._pause_controller.needs_resume_recovery is False


@pytest.mark.asyncio
async def test_release_reregisters_after_clean_pause_rollback(handler):
    manager = handler.engine.tokenizer_manager
    manager.release_memory_occupation = AsyncMock(
        side_effect=RuntimeError("release failed")
    )

    release_result = await handler.release_memory_occupation({})

    # Rollback continued generation cleanly, so the engine is serving-safe again.
    assert release_result["status"] == "error"
    assert handler._pause_controller.is_paused is False
    assert handler._pause_controller.needs_resume_recovery is False
    manager.continue_generation.assert_awaited_once()
    handler.generate_endpoint.unregister_endpoint_instance.assert_awaited_once()
    # The endpoint must rejoin the routing pool since resume will early-return.
    handler.generate_endpoint.register_endpoint_instance.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method_name", "endpoint_method"),
    [
        ("release_memory_occupation", "unregister_endpoint_instance"),
        ("resume_memory_occupation", "register_endpoint_instance"),
    ],
)
async def test_memory_control_returns_error_without_pause_controller(
    handler,
    method_name,
    endpoint_method,
):
    handler.engine = None
    handler._pause_controller = None

    result = await getattr(handler, method_name)({})

    assert result == {
        "status": "error",
        "message": "memory control not supported on this worker",
    }
    getattr(handler.generate_endpoint, endpoint_method).assert_not_awaited()


@pytest.mark.asyncio
async def test_resume_keeps_paused_state_when_register_fails(handler):
    await handler.release_memory_occupation({})
    handler.generate_endpoint.register_endpoint_instance = AsyncMock(
        side_effect=RuntimeError("discovery write timeout")
    )

    result = await handler.resume_memory_occupation({})

    assert result["status"] == "error"
    assert handler._pause_controller is not None
    assert handler._pause_controller.is_paused is True


@pytest.mark.asyncio
async def test_clear_kv_blocks_flushes_sglang_cache(handler):
    handler.engine.tokenizer_manager.server_args = SimpleNamespace(
        hicache_storage_backend="none"
    )

    chunks = [chunk async for chunk in handler.clear_kv_blocks({})]

    assert chunks == [{"status": "success", "message": "KV cache cleared"}]
    handler.engine.tokenizer_manager.auto_create_handle_loop.assert_called_once_with()
    handler.engine.tokenizer_manager.flush_cache.assert_awaited_once_with()
    handler.engine.tokenizer_manager.clear_hicache_storage.assert_not_awaited()


@pytest.mark.asyncio
async def test_clear_kv_blocks_clears_configured_sglang_external_cache(handler):
    handler.engine.tokenizer_manager.server_args = SimpleNamespace(
        hicache_storage_backend="nixl"
    )

    chunks = [chunk async for chunk in handler.clear_kv_blocks({})]

    assert chunks == [{"status": "success", "message": "KV cache cleared"}]
    handler.engine.tokenizer_manager.auto_create_handle_loop.assert_called_once_with()
    handler.engine.tokenizer_manager.flush_cache.assert_awaited_once_with()
    handler.engine.tokenizer_manager.clear_hicache_storage.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_clear_kv_blocks_rejects_active_sglang_requests(handler):
    handler.engine.tokenizer_manager.rid_to_state = {"request-1": object()}

    chunks = [chunk async for chunk in handler.clear_kv_blocks({})]

    assert chunks == [
        {
            "status": "error",
            "message": "Cannot clear KV cache while requests are active",
        }
    ]
    handler.engine.tokenizer_manager.auto_create_handle_loop.assert_not_called()
    handler.engine.tokenizer_manager.flush_cache.assert_not_awaited()
    handler.engine.tokenizer_manager.clear_hicache_storage.assert_not_awaited()


@pytest.mark.asyncio
async def test_clear_kv_blocks_returns_error_without_engine(handler):
    handler.engine = None

    chunks = [chunk async for chunk in handler.clear_kv_blocks({})]

    assert chunks == [
        {
            "status": "error",
            "message": "KV cache clear not supported on this worker",
        }
    ]


@pytest.mark.asyncio
async def test_clear_kv_blocks_reports_flush_failure(handler):
    handler.engine.tokenizer_manager.flush_cache = AsyncMock(
        return_value=SimpleNamespace(success=False, message="cache busy")
    )

    chunks = [chunk async for chunk in handler.clear_kv_blocks({})]

    assert chunks == [{"status": "error", "message": "cache busy"}]
    handler.engine.tokenizer_manager.flush_cache.assert_awaited_once_with()
    handler.engine.tokenizer_manager.clear_hicache_storage.assert_not_awaited()


@pytest.mark.asyncio
async def test_clear_kv_blocks_reports_sglang_external_cache_failure(handler):
    handler.engine.tokenizer_manager.server_args = SimpleNamespace(
        hicache_storage_backend="nixl"
    )
    handler.engine.tokenizer_manager.clear_hicache_storage = AsyncMock(
        return_value=SimpleNamespace(success=False, message="storage busy")
    )

    chunks = [chunk async for chunk in handler.clear_kv_blocks({})]

    assert chunks == [{"status": "error", "message": "storage busy"}]
    handler.engine.tokenizer_manager.flush_cache.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_clear_kv_blocks_reports_flush_exception(handler):
    handler.engine.tokenizer_manager.flush_cache = AsyncMock(
        side_effect=RuntimeError("flush crashed")
    )

    chunks = [chunk async for chunk in handler.clear_kv_blocks({})]

    assert chunks == [{"status": "error", "message": "flush crashed"}]
