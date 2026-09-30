// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Engine-observed membership. Engines publish facts; only the worker writes discovery.

use std::sync::Arc;
use std::time::Duration;

use dynamo_runtime::component::Endpoint;
use dynamo_runtime::config::HealthStatus;
use dynamo_runtime::system_health::ReadinessHold;
use dynamo_runtime::traits::DistributedRuntimeProvider;
use tokio::sync::{Mutex, watch};
use tokio_util::sync::CancellationToken;

/// A full observation of one engine incarnation, not a controller enable command.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EngineServingState {
    pub instance_id: u64,
    pub revision: u64,
    pub ready: bool,
}

/// `None` means unknown/disconnected and withdraws membership. Closing the sender
/// also withdraws membership. Engines must never mutate discovery themselves.
pub type EngineServingStates = watch::Receiver<Option<EngineServingState>>;

/// Runs only after the request handler and model card exist. A deferred endpoint
/// remains absent until the first eligible observation. The mutation lock is also
/// used by shutdown, so the final unregister cannot be followed by a late register.
pub(crate) async fn follow_engine_state(
    endpoint: Endpoint,
    mut states: EngineServingStates,
    mutation: Arc<Mutex<()>>,
    shutdown: CancellationToken,
    readiness_hold: ReadinessHold,
) {
    let mut registered_instance = None;
    let mut readiness_hold = Some(readiness_hold);
    let mut closed = false;
    loop {
        let observation = if closed {
            None
        } else {
            states.borrow_and_update().clone()
        };
        let result = {
            let _guard = mutation.lock().await;
            if shutdown.is_cancelled() {
                return;
            }
            reconcile(
                &endpoint,
                &mut registered_instance,
                &mut readiness_hold,
                observation.as_ref(),
            )
            .await
        };
        let retry = match result {
            Ok(()) => false,
            Err(error) => {
                // A failed RPC may have taken effect remotely. Forget the local
                // cache and force an unregister before attempting registration again.
                registered_instance = None;
                super::worker::set_worker_health(&endpoint, HealthStatus::NotReady);
                tracing::warn!(%error, "engine membership reconciliation failed; retrying");
                true
            }
        };
        tokio::select! {
            biased;
            _ = shutdown.cancelled() => return,
            changed = states.changed(), if !closed => {
                closed = changed.is_err();
            }
            _ = tokio::time::sleep(Duration::from_millis(250)), if retry => {}
        }
    }
}

async fn reconcile(
    endpoint: &Endpoint,
    registered_instance: &mut Option<u64>,
    readiness_hold: &mut Option<ReadinessHold>,
    observation: Option<&EngineServingState>,
) -> anyhow::Result<()> {
    let eligible = observation.filter(|state| state.ready);
    if eligible.is_some_and(|state| Some(state.instance_id) == *registered_instance) {
        return Ok(());
    }

    // Also withdraw on initial/unknown state and after an ambiguous register
    // failure. Idempotence is provided by the deferred-endpoint runtime primitive.
    readiness_hold.get_or_insert_with(|| {
        ReadinessHold::take(endpoint.drt().system_health(), endpoint.name())
    });
    super::worker::set_worker_health(endpoint, HealthStatus::NotReady);
    endpoint.unregister_endpoint_instance().await?;
    *registered_instance = None;
    if let Some(state) = eligible {
        endpoint.register_endpoint_instance().await?;
        *registered_instance = Some(state.instance_id);
        drop(readiness_hold.take());
        super::worker::set_worker_health(endpoint, HealthStatus::Ready);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_runtime::{
        DistributedRuntime, Runtime,
        discovery::DiscoveryQuery,
        distributed::DistributedConfig,
        pipeline::{ManyOut, SingleIn, network::Ingress},
        protocols::annotated::Annotated,
        traits::DistributedRuntimeProvider,
    };

    async fn count(endpoint: &Endpoint) -> usize {
        let id = endpoint.id();
        endpoint
            .drt()
            .discovery()
            .list(DiscoveryQuery::Endpoint {
                namespace: id.namespace,
                component: id.component,
                endpoint: id.name,
            })
            .await
            .unwrap()
            .len()
    }

    async fn wait_count(endpoint: &Endpoint, expected: usize) {
        tokio::time::timeout(Duration::from_secs(5), async {
            while count(endpoint).await != expected {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn observed_lifecycle_defers_withdraws_and_stops_before_shutdown() {
        let runtime = Runtime::from_current().unwrap();
        let drt = DistributedRuntime::new(runtime.clone(), DistributedConfig::process_local())
            .await
            .unwrap();
        let endpoint = drt
            .namespace("observed_lifecycle_test")
            .unwrap()
            .component("backend")
            .unwrap()
            .endpoint("generate");
        let started = endpoint
            .endpoint_builder()
            .handler(Ingress::<SingleIn<String>, ManyOut<Annotated<String>>>::new())
            .initially_registered(false)
            .start_with_registration()
            .await
            .unwrap();
        let (tx, rx) = watch::channel(None);
        let shutdown = CancellationToken::new();
        let mutation = Arc::new(Mutex::new(()));
        let task = tokio::spawn(follow_engine_state(
            endpoint.clone(),
            rx,
            mutation.clone(),
            shutdown.clone(),
            ReadinessHold::take(endpoint.drt().system_health(), endpoint.name()),
        ));
        assert_eq!(count(&endpoint).await, 0);

        tx.send_replace(Some(EngineServingState {
            instance_id: 1,
            revision: 1,
            ready: false,
        }));
        assert_eq!(count(&endpoint).await, 0);
        tx.send_replace(Some(EngineServingState {
            instance_id: 1,
            revision: 2,
            ready: true,
        }));
        wait_count(&endpoint, 1).await;
        tx.send_replace(None);
        wait_count(&endpoint, 0).await;
        // A late successful transport/canary signal cannot make a withdrawn
        // observed worker ready while the membership owner retains its hold.
        {
            let health = endpoint.drt().system_health();
            let mut health = health.lock();
            health.set_endpoint_health_status(endpoint.name(), HealthStatus::Ready);
            health.set_health_status(HealthStatus::Ready);
            assert!(!health.get_health_status().0);
        }
        tx.send_replace(Some(EngineServingState {
            instance_id: 2,
            revision: 1,
            ready: true,
        }));
        wait_count(&endpoint, 1).await;
        tx.send_replace(Some(EngineServingState {
            instance_id: 2,
            revision: 2,
            ready: false,
        }));
        wait_count(&endpoint, 0).await;
        tx.send_replace(Some(EngineServingState {
            instance_id: 2,
            revision: 3,
            ready: true,
        }));
        wait_count(&endpoint, 1).await;

        shutdown.cancel();
        let guard = mutation.lock().await;
        endpoint.unregister_endpoint_instance().await.unwrap();
        drop(guard);
        tx.send_replace(Some(EngineServingState {
            instance_id: 3,
            revision: 1,
            ready: true,
        }));
        task.await.unwrap();
        assert_eq!(count(&endpoint).await, 0);
        started.shutdown().await.unwrap();
        runtime.shutdown();
    }

    #[tokio::test]
    async fn sender_exit_withdraws_a_registered_worker() {
        let runtime = Runtime::from_current().unwrap();
        let drt = DistributedRuntime::new(runtime.clone(), DistributedConfig::process_local())
            .await
            .unwrap();
        let endpoint = drt
            .namespace("observed_sender_exit")
            .unwrap()
            .component("backend")
            .unwrap()
            .endpoint("generate");
        let started = endpoint
            .endpoint_builder()
            .handler(Ingress::<SingleIn<String>, ManyOut<Annotated<String>>>::new())
            .initially_registered(false)
            .start_with_registration()
            .await
            .unwrap();
        let (tx, rx) = watch::channel(Some(EngineServingState {
            instance_id: 1,
            revision: 1,
            ready: true,
        }));
        let shutdown = CancellationToken::new();
        let task = tokio::spawn(follow_engine_state(
            endpoint.clone(),
            rx,
            Arc::new(Mutex::new(())),
            shutdown.clone(),
            ReadinessHold::take(endpoint.drt().system_health(), endpoint.name()),
        ));
        wait_count(&endpoint, 1).await;
        drop(tx);
        wait_count(&endpoint, 0).await;
        shutdown.cancel();
        task.await.unwrap();
        started.shutdown().await.unwrap();
        runtime.shutdown();
    }
}
