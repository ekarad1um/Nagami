//! End-to-end tests for the WS fan-out:
//!
//!   * TCP bind + WS upgrade + binary forward
//!   * UDS bind + WS upgrade + binary forward
//!   * Subscriber count toggles via watch::Receiver
//!   * RecvError::Lagged closes the WS with code 1011
//!
//! Each test brings up a fresh router + listener pair on an ephemeral
//! port / tempdir socket; cancellation token tears it down at the
//! end so the test process exits cleanly.

use acoustics_lab::stream_io::{
    StreamRouter, TransportPolicy, bind_uds, serve_tcp, serve_uds, set_uds_permissions,
};
use bytes::Bytes;
use futures_util::StreamExt;
use std::time::Duration;
use tokio::net::TcpListener;
use tokio::time::timeout;
use tokio_tungstenite::tungstenite::Message as TtMessage;
use tokio_util::sync::CancellationToken;

/// Most tests use a relaxed `TransportPolicy` so a bare
/// `tokio_tungstenite::connect_async(&url)` works without
/// hand-rolling `Sec-WebSocket-Protocol`.  The
/// `tcp_ws_refuses_missing_subprotocol` test below keeps the
/// strict default so the production gate stays covered.
fn relaxed_router(audio_cap: usize, infer_cap: usize) -> StreamRouter {
    let policy = TransportPolicy {
        require_subprotocol: false,
        ..TransportPolicy::default()
    };
    StreamRouter::with_capacities_and_policy(audio_cap, infer_cap, policy)
}

const TEST_TIMEOUT: Duration = Duration::from_secs(5);

/// WS handlers refuse upgrades whose
/// `Sec-WebSocket-Protocol` header doesn't request
/// `acoustics.v1` under the default strict policy
/// (`require_subprotocol = true`).  Production daemons
/// reject silently-old clients that would otherwise
/// stream pre-Envelope payloads they can't decode.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn tcp_ws_refuses_missing_subprotocol() {
    let router = StreamRouter::new(); // strict default policy
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("local_addr");
    let token = CancellationToken::new();
    let token_srv = token.clone();
    let app = router.router();
    let server = tokio::spawn(async move { serve_tcp(listener, app, token_srv).await });

    let url = format!("ws://{addr}/stream/audio");
    // `connect_async` with the bare URL omits the
    // subprotocol header -- handler returns 400.
    let err = tokio_tungstenite::connect_async(&url)
        .await
        .expect_err("missing subprotocol must reject");
    let msg = format!("{err:?}");
    assert!(
        msg.contains("400"),
        "expected 400 Bad Request on missing subprotocol; got {msg}",
    );

    token.cancel();
    let _ = server.await;
}

/// Counterpart: under the relaxed policy
/// (`require_subprotocol = false`) the same bare-URL upgrade
/// succeeds.  Covers the new operator escape hatch for
/// UDS-only / dev-CLI deployments.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn tcp_ws_relaxed_admits_missing_subprotocol() {
    let router = relaxed_router(64, 64);
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("local_addr");
    let token = CancellationToken::new();
    let token_srv = token.clone();
    let app = router.router();
    let server = tokio::spawn(async move { serve_tcp(listener, app, token_srv).await });

    let url = format!("ws://{addr}/stream/audio");
    let (_ws, _resp) = tokio_tungstenite::connect_async(&url)
        .await
        .expect("relaxed policy must admit bare WS upgrade");

    token.cancel();
    let _ = server.await;
}

/// TCP bind, WS upgrade on /stream/audio, send -> receive bytes.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn tcp_audio_ws_round_trip() {
    let router = relaxed_router(64, 64);
    let audio_tx = router.audio_tx();
    let audio_subs = router.audio_subscribers();

    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("local_addr");
    let token = CancellationToken::new();
    let token_srv = token.clone();
    let app = router.router();
    let server = tokio::spawn(async move { serve_tcp(listener, app, token_srv).await });

    // Connect a WS client to /stream/audio.  Bare URL: relaxed
    // policy admits without the Sec-WebSocket-Protocol header.
    let url = format!("ws://{addr}/stream/audio");
    let (mut ws, _resp) = tokio_tungstenite::connect_async(&url)
        .await
        .expect("ws connect");

    // Wait until subscriber count goes positive (a quick poll loop --
    // the connect_async returns before the server-side handler has
    // necessarily incremented the guard).
    let mut subs_rx = audio_subs.clone();
    let waited = timeout(TEST_TIMEOUT, async {
        while *subs_rx.borrow_and_update() == 0 {
            subs_rx.changed().await.expect("watch");
        }
    })
    .await;
    waited.expect("subscriber count should reach 1 within timeout");

    // Broadcast a payload and read it from the client side.
    let payload = Bytes::from_static(b"hello opus");
    audio_tx.send(payload.clone()).expect("send");

    let recv = timeout(TEST_TIMEOUT, ws.next())
        .await
        .expect("recv timeout")
        .expect("ws stream closed");
    let msg = recv.expect("ws msg");
    match msg {
        TtMessage::Binary(b) => assert_eq!(&b[..], payload.as_ref()),
        other => panic!("expected Binary, got {other:?}"),
    }

    // Disconnect and verify subscriber count drops.
    drop(ws);
    let waited = timeout(TEST_TIMEOUT, async {
        while *subs_rx.borrow_and_update() > 0 {
            subs_rx.changed().await.expect("watch");
        }
    })
    .await;
    waited.expect("subscriber count should return to 0 within timeout");

    token.cancel();
    let _ = server.await;
}

/// UDS bind, WS upgrade on /stream/infer, send -> receive bytes.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn uds_infer_ws_round_trip() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::UnixStream as TokioUnixStream;

    let router = relaxed_router(64, 64);
    let infer_tx = router.infer_tx();
    let infer_subs = router.infer_subscribers();

    let dir = tempfile::tempdir().expect("tempdir");
    let sock = dir.path().join("test.sock");
    let listener = bind_uds(&sock).await.expect("bind_uds");
    set_uds_permissions(&sock, 0o666).expect("chmod");
    let token = CancellationToken::new();
    let token_srv = token.clone();
    let app = router.router();
    let server = tokio::spawn(async move { serve_uds(listener, app, token_srv).await });

    // tokio-tungstenite's `connect_async` doesn't speak UDS directly;
    // we hand-roll the WS handshake on a UnixStream.  The handshake
    // is HTTP/1.1 GET with Upgrade headers -- easy enough to script.
    //
    // For an actual product, a downstream CLI would use
    // tokio-tungstenite's `client_async` with a `tokio::net::UnixStream`,
    // but the API requires a Request-with-Uri, and `unix:` URIs are
    // not valid HTTP.  We stick with raw bytes here; this still
    // exercises the entire server path through the upgrade.
    let mut sock_conn = TokioUnixStream::connect(&sock).await.expect("uds connect");

    // Minimal WS handshake.  Sec-WebSocket-Key is the canonical empty
    // 16-byte ASCII pad ("dGhlIHNhbXBsZSBub25jZQ==" in tungstenite's
    // examples).
    let req = "GET /stream/infer HTTP/1.1\r\n\
        Host: localhost\r\n\
        Upgrade: websocket\r\n\
        Connection: Upgrade\r\n\
        Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n\
        Sec-WebSocket-Version: 13\r\n\
        Sec-WebSocket-Protocol: acoustics.v1\r\n\r\n";
    sock_conn
        .write_all(req.as_bytes())
        .await
        .expect("write req");

    // Read until "\r\n\r\n" to consume the response headers.
    let mut buf = [0u8; 4096];
    let mut total = 0usize;
    let resp_end = loop {
        let n = sock_conn.read(&mut buf[total..]).await.expect("read resp");
        assert!(n > 0, "uds connection closed before response");
        total += n;
        if let Some(idx) = find_subseq(&buf[..total], b"\r\n\r\n") {
            break idx + 4;
        }
        if total == buf.len() {
            panic!("response too large for buffer");
        }
    };
    let head = std::str::from_utf8(&buf[..resp_end]).expect("utf-8 resp");
    assert!(
        head.starts_with("HTTP/1.1 101"),
        "expected 101 Switching Protocols; got: {head:?}",
    );

    // Wait for subscriber to register.
    let mut subs_rx = infer_subs.clone();
    let waited = timeout(TEST_TIMEOUT, async {
        while *subs_rx.borrow_and_update() == 0 {
            subs_rx.changed().await.expect("watch");
        }
    })
    .await;
    waited.expect("subscriber count should reach 1");

    // Broadcast a payload.  The server writes a binary WS frame; we
    // parse the minimal frame format to recover the payload bytes.
    // (RFC 6455 server-to-client frames are unmasked, so the
    // header is just FIN/opcode + length, then payload.)
    let payload = Bytes::from_static(b"hello inference");
    infer_tx.send(payload.clone()).expect("send");

    // Read a single frame.
    let mut frame_hdr = [0u8; 2];
    timeout(TEST_TIMEOUT, sock_conn.read_exact(&mut frame_hdr))
        .await
        .expect("read frame header timeout")
        .expect("read frame header");
    let opcode = frame_hdr[0] & 0x0f;
    let masked = (frame_hdr[1] & 0x80) != 0;
    let payload_len = (frame_hdr[1] & 0x7f) as usize;
    assert_eq!(
        opcode, 0x2,
        "expected binary frame opcode 0x2; got 0x{opcode:x}"
    );
    assert!(!masked, "server frames should not be masked");
    assert_eq!(payload_len, payload.len(), "frame length mismatch");
    let mut frame_body = vec![0u8; payload_len];
    timeout(TEST_TIMEOUT, sock_conn.read_exact(&mut frame_body))
        .await
        .expect("read frame body timeout")
        .expect("read frame body");
    assert_eq!(frame_body.as_slice(), payload.as_ref());

    // Drop the connection; verify subscriber count returns to 0.
    drop(sock_conn);
    let waited = timeout(TEST_TIMEOUT, async {
        while *subs_rx.borrow_and_update() > 0 {
            subs_rx.changed().await.expect("watch");
        }
    })
    .await;
    waited.expect("subscriber count should return to 0");

    token.cancel();
    let _ = server.await;
}

fn find_subseq(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || haystack.len() < needle.len() {
        return None;
    }
    haystack.windows(needle.len()).position(|w| w == needle)
}

/// Two clients on the same stream both receive every message.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fan_out_to_two_subscribers() {
    let router = relaxed_router(64, 64);
    let audio_tx = router.audio_tx();

    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("local_addr");
    let token = CancellationToken::new();
    let token_srv = token.clone();
    let app = router.router();
    let server = tokio::spawn(async move { serve_tcp(listener, app, token_srv).await });

    let url = format!("ws://{addr}/stream/audio");
    let (mut ws_a, _) = tokio_tungstenite::connect_async(&url).await.expect("ws a");
    let (mut ws_b, _) = tokio_tungstenite::connect_async(&url).await.expect("ws b");

    // Wait for both subscribers to register.
    let mut subs_rx = router.audio_subscribers();
    timeout(TEST_TIMEOUT, async {
        while *subs_rx.borrow_and_update() < 2 {
            subs_rx.changed().await.expect("watch");
        }
    })
    .await
    .expect("two subs within timeout");

    // Send 5 messages; both clients receive in order.
    for i in 0..5u32 {
        let payload = Bytes::copy_from_slice(&i.to_le_bytes());
        audio_tx.send(payload.clone()).expect("send");
    }

    // Drain both clients.
    for _ in 0..5 {
        for ws in [&mut ws_a, &mut ws_b] {
            let m = timeout(TEST_TIMEOUT, ws.next())
                .await
                .expect("recv timeout")
                .expect("stream closed")
                .expect("ws err");
            assert!(matches!(m, TtMessage::Binary(_)));
        }
    }

    drop(ws_a);
    drop(ws_b);
    timeout(TEST_TIMEOUT, async {
        while *subs_rx.borrow_and_update() > 0 {
            subs_rx.changed().await.expect("watch");
        }
    })
    .await
    .expect("subs return to 0");

    token.cancel();
    let _ = server.await;
}

/// Slow consumer triggers `RecvError::Lagged` -> server closes the WS.
/// We saturate the broadcast channel without the client reading; the
/// next `recv()` after capacity overflow yields Lagged, and the
/// handler sends Close{1011}.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn lagged_consumer_gets_closed_with_1011() {
    // Tiny capacity to lag fast.
    let router = relaxed_router(2, 2);
    let audio_tx = router.audio_tx();
    let lag_counters = router.lag_counters();

    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("local_addr");
    let token = CancellationToken::new();
    let token_srv = token.clone();
    let app = router.router();
    let server = tokio::spawn(async move { serve_tcp(listener, app, token_srv).await });

    let url = format!("ws://{addr}/stream/audio");
    let (mut ws, _) = tokio_tungstenite::connect_async(&url).await.expect("ws");

    // Wait for subscriber registration.
    let mut subs_rx = router.audio_subscribers();
    timeout(TEST_TIMEOUT, async {
        while *subs_rx.borrow_and_update() == 0 {
            subs_rx.changed().await.expect("watch");
        }
    })
    .await
    .expect("subs reach 1");

    // Send 100 messages without reading. broadcast cap=2 -> 98 lagged.
    for i in 0..100u32 {
        let _ = audio_tx.send(Bytes::copy_from_slice(&i.to_le_bytes()));
    }

    // Read until we see a Close frame. axum forwards prior unread
    // messages first up to the cap, then the Lagged-induced Close.
    let mut saw_close = false;
    let collect_loop_deadline = std::time::Instant::now() + TEST_TIMEOUT;
    while std::time::Instant::now() < collect_loop_deadline {
        let m = match timeout(Duration::from_millis(500), ws.next()).await {
            Ok(Some(Ok(m))) => m,
            Ok(Some(Err(_))) => break,
            Ok(None) => break,
            Err(_) => continue,
        };
        match m {
            TtMessage::Close(Some(frame)) => {
                assert_eq!(u16::from(frame.code), 1011, "wrong close code: {:?}", frame);
                assert!(
                    frame.reason.starts_with("lagged"),
                    "wrong reason: {:?}",
                    frame.reason,
                );
                saw_close = true;
                break;
            }
            TtMessage::Close(None) => {
                saw_close = true;
                break;
            }
            _ => continue,
        }
    }
    assert!(saw_close, "did not observe Close frame within timeout");

    // The Lagged path should have ticked the audio counter (and only
    // the audio counter -- the inference channel was untouched).  This
    // is the same number `/api/v1/status` will surface to operators,
    // so locking it down end-to-end avoids the counter rotting if a
    // future refactor moves the increment.
    assert!(
        lag_counters.audio_messages_dropped() > 0,
        "audio_messages_dropped counter did not advance; got {}",
        lag_counters.audio_messages_dropped(),
    );
    assert_eq!(lag_counters.inference_messages_dropped(), 0);

    token.cancel();
    let _ = server.await;
}
