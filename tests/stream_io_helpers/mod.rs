//! Shared WS test scaffolding (refactor plan  deliverable).
//!
//! `tests/stream_io_ws_e2e.rs` already shipped a working WS-over-TCP test
//! pattern; this module factors out the connect + ready-handshake
//! dance so future tests ( admission control, envelope
//! framing) can call one helper instead of copy-pasting ~30 lines
//! of router-spawn + connect + subscriber-wait setup.
//!
//! Concretely it provides:
//!
//!   * [`spawn_tcp_router`] -- start a `StreamRouter` on an
//!     ephemeral 127.0.0.1 port, return the [`RouterHarness`] guard
//!     (cancels the server task on drop).
//!   * [`connect_tcp_ws`] -- open a tokio_tungstenite WS connection
//!     to a given path (`/stream/audio` or `/stream/infer`) and
//!     wait until the corresponding subscriber count is observed
//!     positive on the server side, so callers can immediately
//!     `tx.send(...)` and expect the receiver to see it.
//!
//! UDS-side scaffolding is more bespoke (the WS handshake on a
//! UnixStream is hand-rolled because tokio-tungstenite's
//! `connect_async` doesn't speak `unix:` URIs); see `ws_e2e.rs`
//! for that pattern.

#![allow(dead_code)] // tests pull individual items as needed

use acoustics_lab::stream_io::{StreamRouter, TransportPolicy, serve_tcp};
use bytes::Bytes;
use futures_util::{SinkExt, StreamExt};
use std::time::Duration;
use tokio::net::TcpListener;
use tokio::sync::watch;
use tokio::task::JoinHandle;
use tokio::time::timeout;
use tokio_tungstenite::WebSocketStream;
use tokio_tungstenite::tungstenite::Message as TtMessage;
use tokio_util::sync::CancellationToken;

/// Default per-await timeout.  WS round-trips on a loopback socket
/// settle in ~1 ms; this generous budget catches CI hangs without
/// flaking on a busy runner.
pub const TEST_TIMEOUT: Duration = Duration::from_secs(5);

/// Owns a live `StreamRouter` + `tokio::spawn`'d server task.  On
/// drop the server is cancelled and joined; tests that hold a
/// `RouterHarness` for their full duration get clean teardown.
pub struct RouterHarness {
    pub router: StreamRouter,
    pub local_addr: std::net::SocketAddr,
    pub audio_tx: tokio::sync::broadcast::Sender<Bytes>,
    pub infer_tx: tokio::sync::broadcast::Sender<Bytes>,
    pub audio_subs: watch::Receiver<usize>,
    pub infer_subs: watch::Receiver<usize>,
    cancel: CancellationToken,
    server: Option<JoinHandle<()>>,
}

impl Drop for RouterHarness {
    fn drop(&mut self) {
        self.cancel.cancel();
        // Best-effort: don't block_on inside Drop; the test runtime
        // will reap the task.  Aborting is enough to release the
        // listener.
        if let Some(h) = self.server.take() {
            h.abort();
        }
    }
}

/// Spawn a fresh router + TCP listener on `127.0.0.1:0`, return
/// the harness.  Caller can then `connect_tcp_ws` to either
/// `/stream/audio` or `/stream/infer`.
///
/// The harness uses a relaxed [`TransportPolicy`]
/// (`require_subprotocol = false`) so callers can use any
/// `tokio_tungstenite::connect_async` URL without hand-rolling
/// the `Sec-WebSocket-Protocol` header.  Tests that specifically
/// want to exercise the strict default should construct their
/// own router via [`StreamRouter::new`].
pub async fn spawn_tcp_router() -> RouterHarness {
    let policy = TransportPolicy {
        require_subprotocol: false,
        ..TransportPolicy::default()
    };
    let router = StreamRouter::with_capacities_and_policy(64, 64, policy);
    let audio_tx = router.audio_tx();
    let infer_tx = router.infer_tx();
    let audio_subs = router.audio_subscribers();
    let infer_subs = router.infer_subscribers();

    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let local_addr = listener.local_addr().expect("local_addr");
    let cancel = CancellationToken::new();
    let token_srv = cancel.clone();
    let app = router.router();
    let server = tokio::spawn(async move {
        let _ = serve_tcp(listener, app, token_srv).await;
    });

    RouterHarness {
        router,
        local_addr,
        audio_tx,
        infer_tx,
        audio_subs,
        infer_subs,
        cancel,
        server: Some(server),
    }
}

/// Connect a tungstenite WS client to `ws://<harness>/stream/...`,
/// wait for the corresponding subscriber count to reach 1, return
/// the open stream.  After this returns, broadcasting on the matching
/// `*_tx` will land at the receiver.
pub async fn connect_tcp_ws(
    harness: &RouterHarness,
    path: &str,
) -> WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>> {
    assert!(
        path == "/stream/audio" || path == "/stream/infer",
        "connect_tcp_ws: unknown path `{path}` (use /stream/audio or /stream/infer)"
    );
    let url = format!("ws://{}{}", harness.local_addr, path);
    // `Spawn_tcp_router` runs a relaxed
    // `TransportPolicy`, so a bare `connect_async` (no
    // `Sec-WebSocket-Protocol` header) is admitted.  Tests that
    // need to exercise the strict default build their own
    // router + request explicitly.
    let (ws, _resp) = tokio_tungstenite::connect_async(&url)
        .await
        .expect("ws connect");

    // Wait until subscriber count goes positive on the server side.
    let mut subs_rx = if path == "/stream/audio" {
        harness.audio_subs.clone()
    } else {
        harness.infer_subs.clone()
    };
    let waited = timeout(TEST_TIMEOUT, async {
        while *subs_rx.borrow_and_update() == 0 {
            subs_rx.changed().await.expect("subs watch closed");
        }
    })
    .await;
    waited.expect("subscriber count should reach 1 within timeout");
    ws
}

/// Read one binary message from a WS stream with the test timeout.
/// Panics on timeout, clean close, or text frame.
pub async fn recv_binary<S>(ws: &mut WebSocketStream<S>) -> Bytes
where
    S: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin,
{
    let next = timeout(TEST_TIMEOUT, ws.next())
        .await
        .expect("ws recv timeout")
        .expect("ws stream closed");
    match next.expect("ws msg") {
        TtMessage::Binary(b) => Bytes::from(b.to_vec()),
        other => panic!("expected Binary frame, got {other:?}"),
    }
}

/// Send a binary message on a WS stream (rare for the daemon --
/// the protocol is server-initiated -- but useful for negative
/// tests around envelope-version negotiation in ).
pub async fn send_binary<S>(ws: &mut WebSocketStream<S>, bytes: Bytes)
where
    S: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin,
{
    timeout(TEST_TIMEOUT, ws.send(TtMessage::Binary(bytes)))
        .await
        .expect("ws send timeout")
        .expect("ws send error");
}
