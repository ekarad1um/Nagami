//! Compile + smoke gate for the `tests/common` WS scaffolding
//! introduced in refactor plan . Ensures the helper module
//! compiles even before its first real consumer ( / )
//! lands.

#[path = "stream_io_helpers/mod.rs"]
mod stream_io_helpers;

use bytes::Bytes;
use stream_io_helpers::{connect_tcp_ws, recv_binary, spawn_tcp_router};

/// Round-trip one byte payload through the audio WS path using the
/// shared helper.  Smaller than `ws_e2e::tcp_audio_ws_round_trip`,
/// pinned at this layer purely to gate the helper's compile.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn helper_round_trip_audio() {
    let harness = spawn_tcp_router().await;
    let mut ws = connect_tcp_ws(&harness, "/stream/audio").await;

    let payload = Bytes::from_static(b" helper smoke");
    harness.audio_tx.send(payload.clone()).expect("send");

    let received = recv_binary(&mut ws).await;
    assert_eq!(received.as_ref(), payload.as_ref());
}
