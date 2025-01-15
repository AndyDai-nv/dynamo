# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import logging
import sys
from multiprocessing import Manager, Process

import cupy
import numpy
import pytest
import ucp
from cupy_backends.cuda.api.runtime import CUDARuntimeError
from triton_distributed.icp.nats_request_plane import NatsRequestPlane
from triton_distributed.icp.ucp_data_plane import UcpDataPlane
from triton_distributed.worker.log_formatter import LOGGER_NAME
from triton_distributed.worker.operator import OperatorConfig
from triton_distributed.worker.remote_operator import RemoteOperator
from triton_distributed.worker.triton_core_operator import TritonCoreOperator
from triton_distributed.worker.worker import WorkerConfig

NATS_PORT = 4223
MODEL_REPOSITORY = "/workspace/worker/tests/python/integration/operators/models"
WORKFLOW_REPOSITORY = "/workspace/worker/tests/python/integration/operators"
TRITON_LOG_LEVEL = 6

logger = logging.getLogger(LOGGER_NAME)

# Run cupy's cuda.is_available once to
# avoid the exception hitting runtime code.
try:
    if cupy.cuda.is_available():
        pass
    else:
        print("CUDA not available.")
except CUDARuntimeError:
    print("CUDA not available")

# TODO
# Decide if this should be
# pre merge, nightly, or weekly
pytestmark = pytest.mark.pre_merge


@pytest.fixture
def workers(worker_manager, request):
    worker_config = WorkerConfig(
        request_plane=NatsRequestPlane,
        data_plane=UcpDataPlane,
        request_plane_args=([], {"request_plane_uri": f"nats://localhost:{NATS_PORT}"}),
        log_level=TRITON_LOG_LEVEL,
    )
    store_outputs_in_response = request.getfixturevalue("store_outputs_in_response")

    add_model = OperatorConfig(
        name="add",
        implementation=TritonCoreOperator,
        version=1,
        max_inflight_requests=10,
        parameters={"store_outputs_in_response": store_outputs_in_response},
        repository=MODEL_REPOSITORY,
    )
    multiply_model = OperatorConfig(
        name="multiply",
        implementation=TritonCoreOperator,
        version=1,
        max_inflight_requests=10,
        parameters={"store_outputs_in_response": store_outputs_in_response},
        repository=MODEL_REPOSITORY,
    )
    divide_model = OperatorConfig(
        name="divide",
        implementation=TritonCoreOperator,
        version=1,
        max_inflight_requests=10,
        parameters={"store_outputs_in_response": store_outputs_in_response},
        repository=MODEL_REPOSITORY,
    )
    workflow = OperatorConfig(
        name="add_multiply_divide",
        implementation="add_multiply_divide:AddMultiplyDivide",
        version=1,
        max_inflight_requests=10,
        parameters={"store_outputs_in_response": store_outputs_in_response},
        repository=WORKFLOW_REPOSITORY,
    )

    with Manager() as manager:
        workers = []
        queues = []

        queues.append(manager.Queue(maxsize=1))
        workers.append(
            worker_manager.setup_worker_process(
                [add_model], "add", queues[-1], worker_config
            )
        )

        queues.append(manager.Queue(maxsize=1))
        workers.append(
            worker_manager.setup_worker_process(
                [multiply_model], "multiply", queues[-1], worker_config
            )
        )

        queues.append(manager.Queue(maxsize=1))
        workers.append(
            worker_manager.setup_worker_process(
                [divide_model], "divide", queues[-1], worker_config
            )
        )

        queues.append(manager.Queue(maxsize=1))
        workers.append(
            worker_manager.setup_worker_process(
                [workflow], "add_multiply_divide", queues[-1], worker_config
            )
        )

        workers_failed = False
        status_list = []
        for queue, worker in zip(queues, workers):
            status = queue.get()
            status_list.append(status)
            if status != "READY":
                workers_failed = True

        if workers_failed:
            worker_manager.cleanup_workers(workers, check_status=False)
            raise Exception(f"Failed to start worker processes: {status_list}")
        yield workers
        worker_manager.cleanup_workers(workers)


def _create_inputs(number, size):
    inputs = []
    outputs = []

    for index in range(number):
        input_ = numpy.random.randint(low=1, high=100, size=[2, size])

        expected_ = {}

        expected_["add_int64_output_total"] = numpy.array([[input_.sum()]])

        expected_["add_int64_output_partial"] = numpy.array([[x.sum() for x in input_]])

        expected_["multiply_int64_output_total"] = numpy.array(
            [[x.prod() for x in expected_["add_int64_output_partial"]]]
        )

        divisor = expected_["add_int64_output_total"][0][0]

        dividends = expected_["add_int64_output_partial"]

        expected_["divide_fp64_output_partial"] = numpy.array(
            [numpy.divide(dividends, divisor)]
        )
        inputs.append(input_)
        outputs.append(expected_)
    return inputs, outputs


async def post_requests(num_requests, store_inputs_in_request):
    ucp.reset()
    timeout = 5

    data_plane = UcpDataPlane()
    data_plane.connect()

    request_plane = NatsRequestPlane(f"nats://localhost:{NATS_PORT}")
    await request_plane.connect()

    add_multiply_divide_model = RemoteOperator(
        "add_multiply_divide", 1, request_plane, data_plane
    )

    results = []
    expected_results = {}

    inputs, outputs = _create_inputs(num_requests, 40)

    for i, input_ in enumerate(inputs):
        request_id = str(i)
        request = add_multiply_divide_model.create_request(
            inputs={"int64_input": input_}, request_id=request_id
        )
        if store_inputs_in_request:
            request.store_inputs_in_request.add("int64_input")
        print(request)
        results.append(add_multiply_divide_model.async_infer(request))
        expected_results[request_id] = outputs[i]

    for result in asyncio.as_completed(results):
        responses = await result
        async for response in responses:
            print(response)

            for output_name, expected_value in expected_results[
                response.request_id
            ].items():
                output = response.outputs[output_name]
                output_value = numpy.from_dlpack(output.to_host())
                numpy.testing.assert_equal(output_value, expected_value)
                del output

            print(expected_results[response.request_id])

            del response

    timeout = 5
    data_plane.close(timeout)
    await request_plane.close()


def run(num_requests, store_inputs_in_request=False):
    sys.exit(
        asyncio.run(
            post_requests(
                num_requests=num_requests,
                store_inputs_in_request=store_inputs_in_request,
            )
        )
    )


@pytest.mark.skipif(
    "(not os.path.exists('/usr/local/bin/nats-server'))",
    reason="NATS.io not present",
)
@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    ["store_inputs_in_request", "store_outputs_in_response"],
    [(False, False), (True, True)],
)
def test_add_multiply_divide(
    request, nats_server, workers, store_inputs_in_request, store_outputs_in_response
):
    # Using a separate process to use data plane across multiple tests.
    p = Process(target=run, args=(2, store_inputs_in_request))
    p.start()
    p.join()
    assert p.exitcode == 0
