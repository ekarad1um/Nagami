//! WebSocket fan-out for the audio + inference broadcast streams.
//!
//! ## Endpoints
//!
//! * `GET /stream/audio` -- Opus-encoded mic audio.  WS upgrade; binary
//!   frames carry the bytes emitted by `opus_stream::run`.
//! * `GET /stream/infer` -- `crate::proto::InferenceFrame` protobuf.
//!   WS upgrade; binary frames carry the bytes emitted by
//!   `inference::InferenceEngine::run_blocking`.
//!
//! Same `Router` is mounted on both a TCP listener and a UDS listener
//! (so a local CLI tool can speak WS over `unix:///run/...`).
//! `axum::serve` accepts `TcpListener` natively; for UDS we hand-roll
//! the accept loop using `hyper-util`.
//!
//! ## Subscriber counting
//!
//! Each WS connection holds a `SubscriberGuard` that increments the
//! relevant `watch::Sender<usize>` on construction and decrements on
//! drop.  `opus_stream::run` watches that count to auto-pause the
//! encoder when no client is connected (saving ~3 % CPU on a Pi 5).
//!
//! ## Backpressure
//!
//! Each WS handler holds its own `broadcast::Receiver<Bytes>`; if the
//! client falls behind by more than the channel's capacity (64 by
//! default), `broadcast::Receiver::recv()` returns
//! `RecvError::Lagged(n)`.  We close the WS with status 1011 and a
//! `"lagged {n}"` reason; the client is expected to reconnect.
//! (Cleaner than silently skipping frames -- the user observes a
//! glitch and we don't end up with a torn protobuf decode at the
//! receiver.)

#![warn(missing_debug_implementations)]

// Envelope-wrap helpers + UDS length-prefix framing
// helpers + the `acoustics.v1` WS subprotocol token.  Producers
// (`opus_stream`, `inference::engine`) and the WS handlers
// below import from here.
pub mod framing;
pub use framing::{
    FramingEncodeError, FramingError, MAX_UDS_FRAME_BYTES, ProtoDecodeError, WS_SUBPROTOCOL,
    decode_envelope, decode_length_prefixed, try_encode_length_prefixed, wrap_audio,
    wrap_inference,
};

use std::sync::Arc;

use axum::Router;
use axum::extract::State;
use axum::extract::ws::{CloseFrame, Message, WebSocket, WebSocketUpgrade};
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::IntoResponse;
use axum::routing::get;
use bytes::Bytes;
use thiserror::Error;
use tokio::net::{TcpListener, UnixListener};
use tokio::sync::{broadcast, watch};
use tokio_util::sync::CancellationToken;

/// Failures from listener bind / serve and from
/// envelope-encoding the broadcast payload.  Most variants
/// are daemon-internal; [`Self::Serve`] is the only
/// operator-visible shape (logged by the API layer when a
/// listener task exits unexpectedly).
#[derive(Debug, Error)]
pub enum StreamError {
    #[error("uds bind {path}: {source}")]
    UdsBind {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("tcp bind {addr}: {source}")]
    TcpBind {
        addr: String,
        #[source]
        source: std::io::Error,
    },
    #[error("uds permissions {path}: {source}")]
    UdsPerms {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("uds remove stale {path}: {source}")]
    UdsRemove {
        path: String,
        #[source]
        source: std::io::Error,
    },
    /// `bind_uds` failed to stat the existing path entry while
    /// deciding whether it is a stale socket safe to unlink.
    /// Distinct from [`Self::UdsRemove`] because no `unlink` was
    /// attempted -- typical causes are `EACCES` on the path or
    /// its parent and surface as operator-actionable diagnostics
    /// (permission misconfiguration, not stale-socket cleanup).
    #[error("uds stat {path}: {source}")]
    UdsStat {
        path: String,
        #[source]
        source: std::io::Error,
    },
    /// `bind_uds` refused to touch the existing entry at `path`
    /// because it is not a Unix socket.  Surfaced before any
    /// `unlink` so a misconfigured path that points at operator
    /// data (regular file, symlink, FIFO, ...) is rejected
    /// instead of silently destroyed.  `kind` is a short
    /// human-readable label (`"regular file"`, `"symlink"`,
    /// `"directory"`, ...) so the operator log shows what was
    /// actually at the path.
    #[error(
        "uds path {path} is a {kind}, not a unix socket; refusing to bind (would unlink operator data)"
    )]
    UdsPathNotSocket { path: String, kind: &'static str },
    /// `bind_uds` refused to bind because the parent directory
    /// is not safely confined.  The path-based `chmod` after
    /// `bind` is a TOCTOU vector iff a non-trusted user can
    /// enter the parent directory (replace the just-bound
    /// socket with a different file between bind and chmod).
    /// We require the parent dir to be writable only by the
    /// daemon's user (no `o+w`; `g+w` only when the group is
    /// the daemon's primary group) -- the well-known
    /// systemd-tmpfiles `/run/<service>` shape.  Sticky bit
    /// (`/tmp`-shape) is acceptable as a relaxation because
    /// the sticky-bit semantic prevents non-owner unlink.
    #[error("uds parent dir {parent} for {path} is not safely confined: {detail}")]
    UdsParentInsecure {
        path: String,
        parent: String,
        detail: String,
    },
    /// Listener serve-loop failure.  `transport` is `"tcp"` or
    /// `"uds"` so a downstream `Categorized` impl (and operator
    /// log triage) can distinguish which listener died.
    ///
    /// Surfaced from `axum::serve(...).await` returning Err on
    /// the TCP path (`serve_tcp`) and a fatal (non-transient per
    /// `classify_uds_accept_error`) accept failure on the UDS
    /// path (`serve_uds`).  Both indicate the listener fd is
    /// gone; the daemon's external supervisor restarts.
    #[error("{transport} serve loop: {source}")]
    Serve {
        transport: &'static str,
        #[source]
        source: std::io::Error,
    },
}

/// Every public error in the workspace
/// implements `Categorized` so it can map to an HTTP status if
/// crossed into the api boundary.  `StreamError` doesn't reach the
/// REST surface today (the daemon launches listeners during boot
/// and any failure aborts the process via `?` in main_body), but
/// the impl future-proofs against a path that surfaces a serve
/// failure as a status snapshot field.
///
/// Every variant maps to `Internal` -- these are server-side
/// failures (bind / permissions / serve-loop), never operator
/// input errors.  A future variant that DOES describe operator
/// input (e.g., a malformed connection upgrade) should classify
/// as `UserInput` -- currently no such case exists.
impl crate::common::error::Categorized for StreamError {
    fn kind(&self) -> crate::common::error::ErrorKind {
        crate::common::error::ErrorKind::Internal
    }
}

/// Per-stream broadcast lag counters.  Increments each time a WS
/// receiver lags the broadcast channel and is closed with 1011.
/// Exposed by the `StreamRouter` so the daemon can surface the
/// numbers in `/api/v1/status` -- without this, the only signal is
/// log lines, which operators won't see in production.
///
/// Cheap to clone (two `Arc<AtomicU64>`).  Loads use `Relaxed` --
/// these are pure counters, no other state depends on their
/// ordering with respect to the broadcast itself.
#[derive(Clone, Debug, Default)]
pub struct BroadcastLagCounters {
    audio: Arc<std::sync::atomic::AtomicU64>,
    inference: Arc<std::sync::atomic::AtomicU64>,
}

impl BroadcastLagCounters {
    /// Cumulative messages dropped on the audio broadcast
    /// channel.  See `BroadcastLagSnapshot` doc for the
    /// dropped-messages-not-events semantic.
    pub fn audio_messages_dropped(&self) -> u64 {
        self.audio.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Cumulative messages dropped on the inference broadcast
    /// channel.
    pub fn inference_messages_dropped(&self) -> u64 {
        self.inference.load(std::sync::atomic::Ordering::Relaxed)
    }
}

/// Read-side abstraction over the per-stream counters.
/// `api::AppState` carries `Arc<dyn LagSource>` so the API crate
/// doesn't depend on `stream_io`; production wiring uses this
/// impl, tests substitute a mock.
impl crate::common::traits::lag_source::LagSource for BroadcastLagCounters {
    fn snapshot(&self) -> crate::common::traits::lag_source::BroadcastLagSnapshot {
        crate::common::traits::lag_source::BroadcastLagSnapshot {
            audio_messages_dropped: self.audio_messages_dropped(),
            inference_messages_dropped: self.inference_messages_dropped(),
        }
    }
}

/// Transport-level admission policy for the WS endpoints.
/// Defaults are open-but-capped: any peer can connect, but at
/// most `max_connections_per_stream` concurrent connections per
/// stream.  The daemon does not terminate auth -- production
/// deployments front it with a reverse proxy that handles TLS
/// termination and authentication.
///
/// All three fields are operator-tunable via TOML
/// (`[stream.tcp_policy]` / `[stream.uds_policy]`).  The policy
/// is enforced in the WS upgrade handlers BEFORE the upgrade
/// completes, so a rejected client never reaches the broadcast
/// subscribe step.
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TransportPolicy {
    /// `0` disables the cap entirely.  Otherwise the WS handler
    /// rejects new connections with 429 once `subscribers >= max`.
    pub max_connections_per_stream: u32,
    /// When `true` (the safe default) every WS upgrade MUST list
    /// `acoustics.v1` in `Sec-WebSocket-Protocol` or it is
    /// rejected with 400.  Set to `false` for deployments where
    /// listener admission is already gated by something else
    /// (UDS filesystem permissions, a localhost-only TCP bind)
    /// and the strict header is just ergonomic friction.
    pub require_subprotocol: bool,
    /// `None` = any Origin accepted (default).
    /// `Some([...])` = only the listed values pass; comparison
    /// is case-sensitive byte-equality.
    ///
    /// ```toml
    /// [stream.tcp_policy]
    /// allowed_origins = ["https://app.example.com"]
    /// ```
    ///
    /// `None` also admits requests that omit the Origin header
    /// (typical of non-browser clients like `curl`).
    pub allowed_origins: Option<Vec<String>>,
}

impl Default for TransportPolicy {
    fn default() -> Self {
        Self {
            max_connections_per_stream: 0,
            allowed_origins: None,
            require_subprotocol: true,
        }
    }
}

impl TransportPolicy {
    /// Conservative production defaults: 32 connections per
    /// stream, no origin restriction, strict subprotocol check.
    pub fn capped() -> Self {
        Self {
            max_connections_per_stream: 32,
            allowed_origins: None,
            require_subprotocol: true,
        }
    }

    /// Origin check.  `None` policy (any origin) always passes.
    /// `Some(allowlist)` requires the request's Origin header to
    /// match one entry exactly.  A request with no Origin header
    /// is admitted under either policy -- non-browser clients
    /// (curl, internal services) typically don't send Origin.
    fn origin_ok(&self, supplied: Option<&str>) -> bool {
        let Some(allowed) = self.allowed_origins.as_deref() else {
            return true;
        };
        let Some(origin) = supplied else {
            return true;
        };
        allowed.iter().any(|a| a == origin)
    }
}

/// State threaded into every WS upgrade handler.  Cheap to clone (one
/// Arc bump for each watch::Sender).  Constructed once by `StreamRouter`.
#[derive(Clone)]
struct AppState {
    audio_tx: broadcast::Sender<Bytes>,
    infer_tx: broadcast::Sender<Bytes>,
    audio_subs: Arc<watch::Sender<usize>>,
    infer_subs: Arc<watch::Sender<usize>>,
    lag_counters: BroadcastLagCounters,
    policy: Arc<TransportPolicy>,
}

impl std::fmt::Debug for AppState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AppState")
            .field("audio_subs", &*self.audio_subs.borrow())
            .field("infer_subs", &*self.infer_subs.borrow())
            .finish_non_exhaustive()
    }
}

/// Owned WS fan-out state.  Construct once in `daemon::main`; spawn
/// one `serve_tcp` and one `serve_uds` task; pipe the broadcast/watch
/// handles to the rest of the daemon.
#[derive(Debug)]
pub struct StreamRouter {
    audio_tx: broadcast::Sender<Bytes>,
    infer_tx: broadcast::Sender<Bytes>,
    audio_subs_tx: Arc<watch::Sender<usize>>,
    audio_subs_rx: watch::Receiver<usize>,
    infer_subs_tx: Arc<watch::Sender<usize>>,
    infer_subs_rx: watch::Receiver<usize>,
    lag_counters: BroadcastLagCounters,
    policy: Arc<TransportPolicy>,
}

impl StreamRouter {
    /// Build a new router with the default broadcast capacities (64
    /// slots per stream).
    ///
    /// At the daemon's nominal cadences this gives ~1.3 s of buffered
    /// audio (50 Hz / 20 ms Opus packets) and ~16 s of buffered
    /// inference frames (4 Hz / 250 ms hop) before a slow client
    /// trips `RecvError::Lagged` and gets closed with WS status 1011.
    pub fn new() -> Self {
        Self::with_capacities(64, 64)
    }

    pub fn with_capacities(audio_cap: usize, infer_cap: usize) -> Self {
        Self::with_capacities_and_policy(audio_cap, infer_cap, TransportPolicy::default())
    }

    /// Full constructor with explicit broadcast capacities
    /// AND a [`TransportPolicy`].  Use [`TransportPolicy::default`]
    /// for the legacy un-policied behaviour (no cap, no auth, no
    /// origin check) or [`TransportPolicy::capped`] for production
    /// defaults (32 connections per stream, no auth).
    pub fn with_capacities_and_policy(
        audio_cap: usize,
        infer_cap: usize,
        policy: TransportPolicy,
    ) -> Self {
        let (audio_tx, _) = broadcast::channel::<Bytes>(audio_cap);
        let (infer_tx, _) = broadcast::channel::<Bytes>(infer_cap);
        let (audio_subs_tx, audio_subs_rx) = watch::channel(0usize);
        let (infer_subs_tx, infer_subs_rx) = watch::channel(0usize);
        Self {
            audio_tx,
            infer_tx,
            audio_subs_tx: Arc::new(audio_subs_tx),
            audio_subs_rx,
            infer_subs_tx: Arc::new(infer_subs_tx),
            infer_subs_rx,
            lag_counters: BroadcastLagCounters::default(),
            policy: Arc::new(policy),
        }
    }

    /// Cheap-clone handle on the per-stream lag counters.  Daemon
    /// passes this into the API layer so `/api/v1/status` can
    /// report broadcast lags in addition to audio-buffer lags.
    pub fn lag_counters(&self) -> BroadcastLagCounters {
        self.lag_counters.clone()
    }

    /// Producer end for audio packets (used by `opus_stream::run`).
    pub fn audio_tx(&self) -> broadcast::Sender<Bytes> {
        self.audio_tx.clone()
    }

    /// Producer end for inference frames (used by
    /// `inference::InferenceEngine::run_blocking`).
    pub fn infer_tx(&self) -> broadcast::Sender<Bytes> {
        self.infer_tx.clone()
    }

    /// Watch receiver for the audio-subscriber count.  Used by
    /// `opus_stream::run` to auto-pause when 0.
    pub fn audio_subscribers(&self) -> watch::Receiver<usize> {
        self.audio_subs_rx.clone()
    }

    /// Watch receiver for the inference-subscriber count.  Currently
    /// not consumed (inference always runs); kept symmetric with
    /// audio for future use (e.g. selectively pausing inference if
    /// the daemon is in a low-power mode).
    pub fn infer_subscribers(&self) -> watch::Receiver<usize> {
        self.infer_subs_rx.clone()
    }

    /// Build the axum router with the router-level [`TransportPolicy`]
    /// passed to the constructor.  Call once and clone the result for
    /// every listener that should share that policy.
    ///
    /// For per-listener policies (Different policy on TCP
    /// vs UDS) use [`StreamRouter::router_with_policy`] instead, so
    /// the daemon can build one Router per listener while sharing
    /// the broadcast + subscriber-counter state.
    pub fn router(&self) -> Router {
        self.router_with_policy(self.policy.as_ref().clone())
    }

    /// Build the axum router with an explicit
    /// [`TransportPolicy`], overriding the router-level default
    /// captured at construction time.  Use one per listener so the
    /// daemon can apply different admission policies on TCP vs UDS
    /// while still sharing the underlying broadcast channels and
    /// subscriber counters.
    pub fn router_with_policy(&self, policy: TransportPolicy) -> Router {
        let state = AppState {
            audio_tx: self.audio_tx.clone(),
            infer_tx: self.infer_tx.clone(),
            audio_subs: self.audio_subs_tx.clone(),
            infer_subs: self.infer_subs_tx.clone(),
            lag_counters: self.lag_counters.clone(),
            policy: Arc::new(policy),
        };
        Router::new()
            .route("/stream/audio", get(audio_ws_handler))
            .route("/stream/infer", get(infer_ws_handler))
            .with_state(state)
    }
}

impl Default for StreamRouter {
    fn default() -> Self {
        Self::new()
    }
}

/// Common policy + cap enforcement.  Returns `Ok(guard)` on
/// admit (caller passes the guard into `ws.on_upgrade`); returns
/// `Err(status)` on rejection (caller returns the response
/// directly, short-circuiting the upgrade so a rejected client
/// never subscribes to the broadcast).
fn enforce_admission(
    headers: &HeaderMap,
    subs_tx: Arc<watch::Sender<usize>>,
    policy: &TransportPolicy,
) -> Result<SubscriberGuard, StatusCode> {
    let supplied_origin = headers.get(header::ORIGIN).and_then(|v| v.to_str().ok());
    if !policy.origin_ok(supplied_origin) {
        return Err(StatusCode::FORBIDDEN);
    }
    // Connection cap: atomic acquire-or-reject.  The guard
    // lifetime begins here; if the upgrade closure runs, the
    // guard moves into it.  If the caller drops the guard
    // before on_upgrade, the slot releases.
    SubscriberGuard::try_acquire(subs_tx, policy.max_connections_per_stream)
        .ok_or(StatusCode::TOO_MANY_REQUESTS)
}

async fn audio_ws_handler(
    ws: WebSocketUpgrade,
    headers: HeaderMap,
    State(state): State<AppState>,
) -> axum::response::Response {
    if let Err(s) = enforce_subprotocol(&headers, &state.policy) {
        return s.into_response();
    }
    let guard = match enforce_admission(&headers, state.audio_subs.clone(), &state.policy) {
        Ok(g) => g,
        Err(s) => return s.into_response(),
    };
    let rx = state.audio_tx.subscribe();
    let lag = state.lag_counters.audio.clone();
    ws.protocols([WS_SUBPROTOCOL])
        .on_upgrade(move |socket| handle_ws(socket, rx, guard, lag, "audio"))
        .into_response()
}

async fn infer_ws_handler(
    ws: WebSocketUpgrade,
    headers: HeaderMap,
    State(state): State<AppState>,
) -> axum::response::Response {
    if let Err(s) = enforce_subprotocol(&headers, &state.policy) {
        return s.into_response();
    }
    let guard = match enforce_admission(&headers, state.infer_subs.clone(), &state.policy) {
        Ok(g) => g,
        Err(s) => return s.into_response(),
    };
    let rx = state.infer_tx.subscribe();
    let lag = state.lag_counters.inference.clone();
    ws.protocols([WS_SUBPROTOCOL])
        .on_upgrade(move |socket| handle_ws(socket, rx, guard, lag, "infer"))
        .into_response()
}

/// Require `Sec-WebSocket-Protocol: acoustics.v1` on every
/// WS upgrade when [`TransportPolicy::require_subprotocol`]
/// is `true` (the default).  axum's
/// `WebSocketUpgrade::protocols([...])` echoes the matched
/// token if the client lists it but does NOT reject
/// clients that omit the header (RFC 6455 lets a server
/// accept without echoing).  Production browser
/// deployments want the stricter contract: an outdated
/// client that doesn't request `acoustics.v1` cannot
/// silently connect and stream pre-envelope payloads it
/// can't decode.
///
/// When the policy is set to `false`, the check is
/// skipped: clients that omit the header are admitted.
/// This is the right knob for
/// UDS-only / localhost-only deployments where the listener itself
/// is the auth boundary and forcing every CLI smoke check to
/// hand-craft the header is just friction.  The handler still echoes
/// the subprotocol when the client sends it, so a real browser
/// client behaves identically under either policy.
///
/// `Sec-WebSocket-Protocol` is a comma-separated list of
/// client-preferred subprotocols.  The header value MAY have
/// per-token whitespace; trim before matching.
fn enforce_subprotocol(headers: &HeaderMap, policy: &TransportPolicy) -> Result<(), StatusCode> {
    if !policy.require_subprotocol {
        return Ok(());
    }
    let raw = match headers.get(header::SEC_WEBSOCKET_PROTOCOL) {
        Some(v) => v,
        None => return Err(StatusCode::BAD_REQUEST),
    };
    let s = raw.to_str().map_err(|_| StatusCode::BAD_REQUEST)?;
    let listed = s.split(',').any(|tok| tok.trim() == WS_SUBPROTOCOL);
    if listed {
        Ok(())
    } else {
        Err(StatusCode::BAD_REQUEST)
    }
}

/// RAII guard for the subscriber count.  Increments on construction,
/// decrements on drop (panic-safe, async-cancellation-safe).
struct SubscriberGuard {
    tx: Arc<watch::Sender<usize>>,
}

impl SubscriberGuard {
    /// Atomic acquire-or-reject.  `send_if_modified`'s
    /// closure runs under the watch's internal lock so the
    /// observe-and-bump is race-free against concurrent acquires:
    /// at the moment we read `*c`, no other handler can interleave.
    /// Returns `None` when the subscriber count is already at
    /// `max_per_stream` -- handler responds 429 without upgrading.
    /// `max_per_stream = 0` is interpreted as "no cap" and skips
    /// the check entirely.
    fn try_acquire(tx: Arc<watch::Sender<usize>>, max_per_stream: u32) -> Option<Self> {
        if max_per_stream == 0 {
            tx.send_modify(|c| *c = c.saturating_add(1));
            return Some(Self { tx });
        }
        let admitted = tx.send_if_modified(|c| {
            if (*c as u32) < max_per_stream {
                *c = c.saturating_add(1);
                true
            } else {
                false
            }
        });
        if admitted { Some(Self { tx }) } else { None }
    }
}

impl Drop for SubscriberGuard {
    fn drop(&mut self) {
        self.tx.send_modify(|c| *c = c.saturating_sub(1));
    }
}

async fn handle_ws(
    mut socket: WebSocket,
    mut rx: broadcast::Receiver<Bytes>,
    guard: SubscriberGuard,
    lag_counter: Arc<std::sync::atomic::AtomicU64>,
    stream_name: &'static str,
) {
    // Guard moved in from `*_ws_handler`; the cap was honoured *before*
    // the upgrade so a rejected client returns 429 without occupying a
    // slot.  Drop on this function's exit (clean close, lagged-receiver
    // close, or panic) decrements the subscriber count.
    let _guard = guard;
    tracing::debug!(target: "stream_io", stream = stream_name, "ws subscribed");

    loop {
        // CANCEL-SAFE branches:
        //   * broadcast `Receiver::recv` -- documented cancel-safe.
        //   * axum `WebSocket::recv` -- cancel-safe at message
        //     granularity (the underlying reader buffers the
        //     in-flight frame).
        // Match-arm `socket.send(...)` may be cancelled mid-send;
        // safe because the only durable invariant tied to this
        // future's lifetime is the subscriber-count decrement,
        // which runs from `_guard`'s Drop.
        tokio::select! {
            biased;
            recv = rx.recv() => match recv {
                Ok(payload) => {
                    // axum 0.8's WebSocket binary takes Bytes directly,
                    // so the broadcast Bytes flows through without a
                    // deep copy.
                    if socket.send(Message::Binary(payload)).await.is_err() {
                        // Client disconnected.
                        break;
                    }
                }
                Err(broadcast::error::RecvError::Lagged(n)) => {
                    // Bump the per-stream counter so `/api/v1/status`
                    // can surface this.  `fetch_add` wraps on overflow
                    // (not saturating), but the wrap point on u64
                    // (~5.8e11 years at 1 kHz lag events) makes that
                    // a non-event in practice.  `Relaxed` is the right
                    // ordering -- the counter has no happens-before
                    // relationship with anything else.
                    lag_counter.fetch_add(n, std::sync::atomic::Ordering::Relaxed);
                    tracing::warn!(
                        target: "stream_io",
                        stream = stream_name,
                        skipped = n,
                        "ws receiver lagged; closing 1011",
                    );
                    let _ = socket
                        .send(Message::Close(Some(CloseFrame {
                            // 1011 = "internal error" (RFC 6455 sec.7.4.1)
                            // -- closest match for "we can't keep up".
                            code: 1011,
                            reason: format!("lagged {n}").into(),
                        })))
                        .await;
                    break;
                }
                Err(broadcast::error::RecvError::Closed) => {
                    let _ = socket
                        .send(Message::Close(Some(CloseFrame {
                            code: 1001, // going away
                            reason: "stream closed".into(),
                        })))
                        .await;
                    break;
                }
            },
            client_msg = socket.recv() => match client_msg {
                None => break,
                Some(Ok(Message::Close(_))) => break,
                Some(Ok(Message::Ping(payload))) => {
                    // axum auto-replies to Pings, but we explicitly
                    // accept them so the select-arm doesn't churn.
                    let _ = socket.send(Message::Pong(payload)).await;
                }
                Some(Ok(_)) => {
                    // Ignore client text/binary; we are a producer.
                }
                Some(Err(e)) => {
                    tracing::debug!(target: "stream_io", err = %e, "ws recv error");
                    break;
                }
            }
        }
    }

    tracing::debug!(target: "stream_io", stream = stream_name, "ws unsubscribed");
}

/// Drive a TCP listener serving the given router.  Returns when
/// `shutdown` fires (graceful) or when the listener errors.
pub async fn serve_tcp(
    listener: TcpListener,
    router: Router,
    shutdown: CancellationToken,
) -> Result<(), StreamError> {
    axum::serve(listener, router)
        .with_graceful_shutdown(async move {
            shutdown.cancelled().await;
        })
        .await
        .map_err(|source| StreamError::Serve {
            transport: "tcp",
            source,
        })
}

/// Classify a `UnixListener::accept` error as transient
/// (FD pressure / momentary client churn) vs fatal.  Transient errors
/// must not tear down the listener: under EMFILE the daemon is
/// briefly out of file descriptors but will recover within
/// milliseconds as in-flight WS connections close; ECONNABORTED is
/// a connect that vanished between the kernel queueing it and us
/// dequeuing it -- never a listener-level problem.
///
/// `EAGAIN` shouldn't strictly happen for a blocking-from-our-PoV
/// `accept().await` (tokio re-arms readiness internally), but we
/// classify it transient defensively -- same treatment as the
/// upstream `tokio::io` examples for accept-loop hardening.
fn is_transient_accept_error(e: &std::io::Error) -> bool {
    matches!(
        e.raw_os_error(),
        Some(libc::EMFILE) | Some(libc::ENFILE) | Some(libc::EAGAIN) | Some(libc::ECONNABORTED)
    )
}

/// Back-off duration on transient accept failure.
/// 100 ms is short enough that a one-off FD-pressure spike costs
/// at most a few accept iterations and long enough that a
/// sustained EMFILE storm doesn't burn a CPU spinning on the
/// failing syscall.
const UDS_ACCEPT_BACKOFF: std::time::Duration = std::time::Duration::from_millis(100);

/// Drive a UDS listener serving the given router.  `axum::serve` does
/// not accept `UnixListener`, so we hand-roll the accept loop using
/// `hyper`'s HTTP/1.1 connection builder (the daemon serves
/// HTTP/1.1 only -- the WebSocket upgrade in axum requires it, and
/// no client we ship speaks HTTP/2 over UDS).
///
/// Each accepted connection is spawned as its own task so a slow
/// client doesn't block the accept loop.
///
/// Accept-loop is resilient against transient FD-pressure
/// (EMFILE / ENFILE / EAGAIN / ECONNABORTED): on those errors we
/// log + sleep 100 ms + continue rather than propagating, so a
/// brief FD spike doesn't kill the listener.  Non-transient errors
/// still propagate as `StreamError::Serve`.
pub async fn serve_uds(
    listener: UnixListener,
    router: Router,
    shutdown: CancellationToken,
) -> Result<(), StreamError> {
    use hyper::server::conn::http1;
    use hyper_util::rt::TokioIo;
    use hyper_util::service::TowerToHyperService;

    let mut conn_id = 0u64;
    loop {
        tokio::select! {
            biased;
            _ = shutdown.cancelled() => {
                tracing::debug!(target: "stream_io", "uds accept loop: shutdown");
                return Ok(());
            }
            accept = listener.accept() => {
                let (stream, _addr) = match accept {
                    Ok(s) => s,
                    Err(e) => {
                        if is_transient_accept_error(&e) {
                            tracing::warn!(
                                target: "stream_io",
                                err = %e,
                                errno = ?e.raw_os_error(),
                                "uds accept transient failure; backing off",
                            );
                            tokio::time::sleep(UDS_ACCEPT_BACKOFF).await;
                            continue;
                        }
                        return Err(StreamError::Serve {
                            transport: "uds",
                            source: e,
                        });
                    }
                };
                conn_id = conn_id.wrapping_add(1);
                let app = router.clone();
                let svc = TowerToHyperService::new(app);
                let shutdown_cloned = shutdown.clone();
                tokio::spawn(async move {
                    // `serve_connection().with_upgrades()` returns a
                    // future whose contained Builder is a temporary;
                    // bind the Builder to a local first so the future
                    // can borrow from it for the duration of the select.
                    let builder = http1::Builder::new();
                    let conn = builder
                        .serve_connection(TokioIo::new(stream), svc)
                        .with_upgrades();
                    tokio::pin!(conn);
                    tokio::select! {
                        biased;
                        _ = shutdown_cloned.cancelled() => {
                            // Best-effort graceful: hyper has no
                            // direct close on the HTTP/1 connection
                            // future.  Dropping the future closes
                            // the connection cleanly enough.
                            tracing::trace!(target: "stream_io", conn_id, "uds conn cancelled");
                        }
                        res = &mut conn => {
                            if let Err(e) = res {
                                tracing::debug!(target: "stream_io", conn_id, err = %e, "uds conn ended");
                            }
                        }
                    }
                });
            }
        }
    }
}

/// Bind a UDS, safely removing any stale socket file at the path.
/// Returns the listener; caller is responsible for chmod via
/// [`set_uds_permissions`].
///
/// ## Safety contract
///
/// 1. **Refuses to unlink non-socket files.**  The pre-fix
///    behaviour was `std::fs::remove_file(path)` ignoring only
///    `NotFound`, which would happily delete a regular file or
///    follow + delete a symlink target if an operator typo or
///    hostile peer staged one at the configured path.  The
///    rewritten flow `symlink_metadata`s the path first and
///    rejects every entry that is not a Unix socket
///    (`StreamError::UdsPathNotSocket`).  Symlinks are rejected
///    even when their target is itself a socket -- following a
///    symlink at unlink time is the classic TOCTOU vector.
/// 2. **Validates parent-directory confinement.**  The
///    path-based `chmod` in [`set_uds_permissions`] is only
///    safe when the parent directory cannot be entered by an
///    untrusted user; otherwise an attacker can swap the
///    just-bound socket for a different file between bind and
///    chmod.  We refuse to bind into a parent directory that
///    is world-writable without the sticky bit
///    (`StreamError::UdsParentInsecure`).  The well-known
///    `systemd-tmpfiles` `/run/<service>` shape (mode `0o755`
///    or `0o750`, daemon-owned) and the `/tmp`-shape sticky
///    parent (`0o1777`) both pass; an unconfined `chmod 0o777
///    /var/run/foo/` does not.
/// 3. **Stale-socket cleanup is best-effort.**  When the
///    pre-existing entry IS a Unix socket -- the legitimate
///    "previous daemon crashed mid-shutdown" case -- we
///    `remove_file` it before binding, so a single-instance
///    daemon restart succeeds without operator intervention.
///    Multi-instance deployments must use distinct paths.
///
/// `config::StreamCfg::validate_uds_path` is the *config-time*
/// partner: it catches the same footguns at TOML-load time
/// before the daemon ever attempts a bind.  Both layers fail
/// closed; this function is the bind-time backstop.
pub async fn bind_uds(path: &std::path::Path) -> Result<UnixListener, StreamError> {
    // Reject obviously-unsafe parent dirs first, so an operator
    // who points `uds_path` at e.g. `/tmp/acoustics_lab.sock`
    // (sticky -- OK) versus a misconfigured `/srv/public-rw/...`
    // (`o+w` no sticky -- rejected) gets the safer diagnostic
    // before we touch the filesystem.  `parent()` returning
    // None (`"/"`) or empty (`"foo.sock"`) is itself a typo we
    // reject -- the daemon never legitimately binds at the
    // filesystem root or at CWD.
    let parent = path
        .parent()
        .ok_or_else(|| StreamError::UdsParentInsecure {
            path: path.display().to_string(),
            parent: String::new(),
            detail: "no parent directory; pick a full path (e.g. /run/acoustics_lab.sock)".into(),
        })?;
    if parent.as_os_str().is_empty() {
        return Err(StreamError::UdsParentInsecure {
            path: path.display().to_string(),
            parent: String::new(),
            detail: "empty parent directory; pick a full path (e.g. /run/acoustics_lab.sock)"
                .into(),
        });
    }
    validate_parent_dir_confinement(path, parent)?;

    // `symlink_metadata` (NOT `metadata`) so a symlink at the
    // path is observable as a symlink instead of being silently
    // followed.  The branch order matters: NotFound is the
    // common fresh-boot case; the existence branches handle
    // either a stale daemon socket (legitimate cleanup) or a
    // hostile / typo entry (refuse).
    match std::fs::symlink_metadata(path) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            // Fresh path -- bind will create it.
        }
        Err(e) => {
            return Err(StreamError::UdsStat {
                path: path.display().to_string(),
                source: e,
            });
        }
        Ok(md) => {
            let ft = md.file_type();
            // Use `unix::fs::FileTypeExt::is_socket` -- the only
            // shape for which `remove_file + bind` is legitimate.
            // Everything else (symlink even to a socket, regular
            // file, directory, FIFO, block / char device) is a
            // hard reject.  The `&'static str` `kind` label
            // surfaces in the `Display` impl so the operator log
            // line shows what was actually at the path.
            #[cfg(unix)]
            {
                use std::os::unix::fs::FileTypeExt;
                let kind: Option<&'static str> = if ft.is_symlink() {
                    Some("symlink")
                } else if ft.is_file() {
                    Some("regular file")
                } else if ft.is_dir() {
                    Some("directory")
                } else if ft.is_fifo() {
                    Some("fifo")
                } else if ft.is_block_device() {
                    Some("block device")
                } else if ft.is_char_device() {
                    Some("char device")
                } else if ft.is_socket() {
                    None // OK to remove
                } else {
                    Some("unknown file type")
                };
                if let Some(kind) = kind {
                    return Err(StreamError::UdsPathNotSocket {
                        path: path.display().to_string(),
                        kind,
                    });
                }
            }
            // Confirmed socket -- safe to remove.
            std::fs::remove_file(path).map_err(|e| StreamError::UdsRemove {
                path: path.display().to_string(),
                source: e,
            })?;
        }
    }
    UnixListener::bind(path).map_err(|e| StreamError::UdsBind {
        path: path.display().to_string(),
        source: e,
    })
}

/// Reject parent directories that an untrusted user could
/// enter.  The path-based `chmod` in [`set_uds_permissions`]
/// must run after `bind` (the inode doesn't exist before
/// `bind` creates it), and `chmod(path, mode)` follows the
/// path -- so an attacker who can replace the just-bound
/// socket with a different file between `bind` and `chmod`
/// can cause `chmod` to apply the operator's mode to the
/// attacker's file.  The textbook fix is `fchmod(fd, mode)`,
/// but Unix-domain socket fds reject `fchmod` (EINVAL on
/// Linux + macOS): `fchmod` operates on the inode the fd
/// refers to, and the listener fd refers to the in-kernel
/// socket object, not the bound path's filesystem inode.
///
/// Parent-dir confinement closes the race: if no untrusted
/// user can `lookup` / `unlink` / `create` inside the parent
/// dir, the path-based `chmod` after `bind` is safe by
/// construction.  Acceptance criteria:
///
/// * `o+w` clear, OR
/// * `o+w` set AND sticky (`/tmp`-shape, `0o1777`) -- the
///   sticky bit prevents non-owner unlink, which is what
///   makes the swap impossible.
///
/// Group-writable (`g+w`) is accepted: production deployments
/// run the daemon as a service user with a service group, and
/// operators legitimately want to grant trusted CLI tools
/// (debug shells, monitoring agents) into the group.
fn validate_parent_dir_confinement(
    path: &std::path::Path,
    parent: &std::path::Path,
) -> Result<(), StreamError> {
    // The parent must already exist; we don't auto-create it
    // (the absence is almost certainly a typo, and creating it
    // would mask the typo and silently install a socket in an
    // unexpected location).  `config::StreamCfg::validate_uds_path`
    // enforces the same at config-load time; this is the
    // bind-time backstop.
    let md = match std::fs::metadata(parent) {
        Ok(md) => md,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Err(StreamError::UdsParentInsecure {
                path: path.display().to_string(),
                parent: parent.display().to_string(),
                detail: format!(
                    "parent directory does not exist (create it with systemd-tmpfiles or pick an existing path); \
                     stat error: {e}"
                ),
            });
        }
        Err(e) => {
            return Err(StreamError::UdsParentInsecure {
                path: path.display().to_string(),
                parent: parent.display().to_string(),
                detail: format!("stat failed: {e}"),
            });
        }
    };
    if !md.is_dir() {
        return Err(StreamError::UdsParentInsecure {
            path: path.display().to_string(),
            parent: parent.display().to_string(),
            detail: "parent path is not a directory".into(),
        });
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = md.permissions().mode();
        let world_writable = (mode & 0o002) != 0;
        let sticky = (mode & 0o1000) != 0;
        if world_writable && !sticky {
            return Err(StreamError::UdsParentInsecure {
                path: path.display().to_string(),
                parent: parent.display().to_string(),
                detail: format!(
                    "parent dir is world-writable without sticky bit (mode {mode:#o}); \
                     would allow a TOCTOU swap between bind and chmod -- restrict to \
                     daemon user (mode 0o755 or 0o750) or set the sticky bit (/tmp-shape, 0o1777)"
                ),
            });
        }
    }
    Ok(())
}

/// Apply Unix mode bits to the bound socket file.  Defaults at bind
/// time are 0660 (umask-dependent), too tight for non-root tools to
/// connect; daemons that want broader access should call this with
/// `0o666` (world R/W) immediately after `bind_uds`.
///
/// ## Safety contract
///
/// `set_uds_permissions` is `chmod(path, mode)` -- a path-based
/// syscall that follows the filesystem path at call time.  It
/// is safe iff [`bind_uds`]'s parent-dir confinement check
/// passed: an attacker who cannot enter the parent directory
/// cannot swap the just-bound socket for a different file
/// before this `chmod` runs.  We DO NOT use `fchmod(fd, mode)`
/// on the listener fd: Unix-domain socket fds reject `fchmod`
/// with EINVAL on both Linux and macOS (the fd refers to the
/// in-kernel socket object, not the bound path's filesystem
/// inode).  See the doc comment on the private
/// `validate_parent_dir_confinement` helper above for the
/// full TOCTOU argument.
///
/// Defence-in-depth: this function re-stats the path with
/// `symlink_metadata` first and refuses to chmod anything
/// that is not a socket, so a swap to a non-socket file
/// (e.g. `/etc/passwd`) is detected before the chmod fires.
/// A swap from one socket to another would still chmod the
/// attacker's socket -- only parent-dir confinement closes
/// that gap.
pub fn set_uds_permissions(path: &std::path::Path, mode: u32) -> Result<(), StreamError> {
    use std::os::unix::fs::PermissionsExt;
    // Refuse to chmod a non-socket file: if `bind_uds` was
    // called against a confined parent dir, this stat is
    // race-free; if it wasn't, this is best-effort defence in
    // depth (better to refuse than to chmod `/etc/passwd`).
    match std::fs::symlink_metadata(path) {
        Ok(md) => {
            let ft = md.file_type();
            #[cfg(unix)]
            {
                use std::os::unix::fs::FileTypeExt;
                if !ft.is_socket() {
                    let kind = if ft.is_symlink() {
                        "symlink"
                    } else if ft.is_file() {
                        "regular file"
                    } else if ft.is_dir() {
                        "directory"
                    } else if ft.is_fifo() {
                        "fifo"
                    } else if ft.is_block_device() {
                        "block device"
                    } else if ft.is_char_device() {
                        "char device"
                    } else {
                        "unknown file type"
                    };
                    return Err(StreamError::UdsPathNotSocket {
                        path: path.display().to_string(),
                        kind,
                    });
                }
            }
        }
        Err(e) => {
            return Err(StreamError::UdsPerms {
                path: path.display().to_string(),
                source: e,
            });
        }
    }
    let perms = std::fs::Permissions::from_mode(mode);
    std::fs::set_permissions(path, perms).map_err(|e| StreamError::UdsPerms {
        path: path.display().to_string(),
        source: e,
    })
}

#[cfg(test)]
mod tests {
    // Test code: stages a tempfile via `std::fs::write` to exercise
    // the UDS-cleanup path on a non-socket file.  The production
    // constraint in `clippy.toml` does not apply here.
    #![allow(clippy::disallowed_methods)]
    use super::*;

    /// `StreamRouter::new` constructs without panicking and exposes
    /// the expected handles.
    #[test]
    fn router_construction_smoke() {
        let r = StreamRouter::new();
        let _audio_tx = r.audio_tx();
        let _infer_tx = r.infer_tx();
        assert_eq!(*r.audio_subscribers().borrow(), 0);
        assert_eq!(*r.infer_subscribers().borrow(), 0);
        // `router()` returns a Router; we just check it builds.
        let _ = r.router();
    }

    /// Subscriber guard increments + decrements correctly.
    #[test]
    fn subscriber_guard_round_trip() {
        let (tx, rx) = watch::channel(0usize);
        let tx = Arc::new(tx);
        {
            let _g1 = SubscriberGuard::try_acquire(tx.clone(), 0).expect("uncapped");
            assert_eq!(*rx.borrow(), 1);
            {
                let _g2 = SubscriberGuard::try_acquire(tx.clone(), 0).expect("uncapped");
                assert_eq!(*rx.borrow(), 2);
            }
            assert_eq!(*rx.borrow(), 1);
        }
        assert_eq!(*rx.borrow(), 0);
    }

    /// `try_acquire` enforces the cap atomically: the Nth+1
    /// concurrent acquire fails without bumping the counter, and a
    /// later acquire after one slot frees succeeds again.
    #[test]
    fn subscriber_guard_caps_concurrent() {
        let (tx, rx) = watch::channel(0usize);
        let tx = Arc::new(tx);
        let g1 = SubscriberGuard::try_acquire(tx.clone(), 2).expect("first slot");
        let g2 = SubscriberGuard::try_acquire(tx.clone(), 2).expect("second slot");
        assert_eq!(*rx.borrow(), 2);
        // Cap is 2; third acquire must reject.
        assert!(
            SubscriberGuard::try_acquire(tx.clone(), 2).is_none(),
            "third acquire must reject at cap"
        );
        // Counter unchanged (no bump on rejected acquire).
        assert_eq!(*rx.borrow(), 2);
        // Free a slot; next acquire succeeds.
        drop(g1);
        assert_eq!(*rx.borrow(), 1);
        let _g3 = SubscriberGuard::try_acquire(tx.clone(), 2).expect("after free");
        assert_eq!(*rx.borrow(), 2);
        drop(g2);
    }

    /// `TransportPolicy::origin_ok` only enforces when an
    /// allowlist is configured.  Missing Origin is admitted under
    /// either policy (non-browser clients typically don't send it).
    #[test]
    fn transport_policy_origin_allowlist() {
        let open = TransportPolicy::default();
        assert!(open.origin_ok(None));
        assert!(open.origin_ok(Some("https://example.com")));

        let restricted = TransportPolicy {
            allowed_origins: Some(vec![
                "https://app.example.com".into(),
                "https://localhost:5173".into(),
            ]),
            ..TransportPolicy::default()
        };
        assert!(restricted.origin_ok(None), "missing Origin admitted");
        assert!(restricted.origin_ok(Some("https://app.example.com")));
        assert!(restricted.origin_ok(Some("https://localhost:5173")));
        assert!(!restricted.origin_ok(Some("https://evil.example.com")));
        assert!(
            !restricted.origin_ok(Some("https://app.example.com:8443")),
            "exact match required, port-aware"
        );
    }

    /// `Enforce_subprotocol` is gated by
    /// `TransportPolicy::require_subprotocol`.  Strict default rejects
    /// requests that omit the header; the relaxed policy accepts
    /// them; both policies still accept a request that DOES list
    /// the v1 token (so production browsers behave identically).
    #[test]
    fn enforce_subprotocol_policy_controlled() {
        let strict = TransportPolicy::default();
        assert!(strict.require_subprotocol, "default must be strict");
        let relaxed = TransportPolicy {
            require_subprotocol: false,
            ..TransportPolicy::default()
        };

        // Empty header map -> strict rejects, relaxed accepts.
        let empty = HeaderMap::new();
        assert_eq!(
            enforce_subprotocol(&empty, &strict),
            Err(StatusCode::BAD_REQUEST),
            "strict policy must reject when no Sec-WebSocket-Protocol",
        );
        assert_eq!(
            enforce_subprotocol(&empty, &relaxed),
            Ok(()),
            "relaxed policy must admit when no Sec-WebSocket-Protocol",
        );

        // Header lists the right token -> both policies accept.
        let mut listed = HeaderMap::new();
        listed.insert(
            header::SEC_WEBSOCKET_PROTOCOL,
            WS_SUBPROTOCOL.parse().expect("hv"),
        );
        assert_eq!(enforce_subprotocol(&listed, &strict), Ok(()));
        assert_eq!(enforce_subprotocol(&listed, &relaxed), Ok(()));

        // Header lists OTHER tokens only -> strict rejects, relaxed
        // still admits (it doesn't enforce membership at all).
        let mut other = HeaderMap::new();
        other.insert(
            header::SEC_WEBSOCKET_PROTOCOL,
            "acoustics.v0, soap".parse().expect("hv"),
        );
        assert_eq!(
            enforce_subprotocol(&other, &strict),
            Err(StatusCode::BAD_REQUEST),
        );
        assert_eq!(enforce_subprotocol(&other, &relaxed), Ok(()));
    }

    /// `Is_transient_accept_error` correctly classifies
    /// the four transient errno values that the `serve_uds`
    /// accept-loop now retries on (EMFILE, ENFILE, EAGAIN,
    /// ECONNABORTED) AND rejects everything else (so a real
    /// listener-fatal error like EBADF still propagates as a
    /// `StreamError::Serve`).  The runtime-side test that injects a
    /// fake transient into a live `serve_uds` would need a mock
    /// `UnixListener` impl -- `tokio::net::UnixListener` does not
    /// expose its trait surface for shimming, so we test the
    /// classifier directly.  Operationally this is the part that
    /// matters: the loop body is a one-liner around it.
    #[test]
    fn is_transient_accept_error_classifies_correctly() {
        for &errno in &[libc::EMFILE, libc::ENFILE, libc::EAGAIN, libc::ECONNABORTED] {
            let e = std::io::Error::from_raw_os_error(errno);
            assert!(
                is_transient_accept_error(&e),
                "errno {errno} ({e}) must classify as transient",
            );
        }
        // Sample of fatal errnos the daemon should NOT silently
        // retry on -- bad descriptor, invalid arg, etc.
        for &errno in &[libc::EBADF, libc::EINVAL, libc::ENOTSOCK, libc::EFAULT] {
            let e = std::io::Error::from_raw_os_error(errno);
            assert!(
                !is_transient_accept_error(&e),
                "errno {errno} ({e}) must NOT classify as transient",
            );
        }
        // Non-OS errors (no `raw_os_error()`) are also non-transient.
        let other = std::io::Error::other("synthetic");
        assert!(
            !is_transient_accept_error(&other),
            "non-OS error must not classify as transient",
        );
    }

    /// `BroadcastLagCounters` starts at zero, is observably mutated
    /// via the inner Arcs, and survives clone (the daemon clones
    /// the counters into the api state and into the WS handler;
    /// both views must see the same numbers).
    #[test]
    fn broadcast_lag_counters_start_zero_and_share_state() {
        let r = StreamRouter::new();
        let view_a = r.lag_counters();
        let view_b = r.lag_counters();
        assert_eq!(view_a.audio_messages_dropped(), 0);
        assert_eq!(view_a.inference_messages_dropped(), 0);

        // Simulate the WS handler bumping the inference counter
        // (this is the same path `handle_ws` takes on Lagged).
        view_a
            .inference
            .fetch_add(7, std::sync::atomic::Ordering::Relaxed);
        // Other clones must see the update.
        assert_eq!(view_b.inference_messages_dropped(), 7);
        assert_eq!(view_b.audio_messages_dropped(), 0);
    }

    /// `set_uds_permissions` actually applies the requested mode
    /// to a real Unix socket file.  The pre-fix version of this
    /// test staged a regular file via `std::fs::write` and called
    /// `set_uds_permissions` on it; the rewritten function refuses
    /// to chmod non-socket files (defence in depth against a swap
    /// to e.g. `/etc/passwd`), so we now stage a real socket via
    /// `bind_uds`.
    #[tokio::test(flavor = "current_thread")]
    async fn set_uds_permissions_applies_mode() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("test.sock");
        let _listener = bind_uds(&path).await.expect("bind");
        set_uds_permissions(&path, 0o666).expect("chmod");
        let mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o666, "got mode 0o{mode:o}");
    }

    /// `bind_uds` rejects a regular file at the configured path
    /// instead of unlinking it.  Pre-fix behaviour was an
    /// unconditional `remove_file` that silently destroyed
    /// operator data; this test stages a regular file with a
    /// distinctive payload, asserts `bind_uds` returns
    /// `UdsPathNotSocket`, and asserts the file is still
    /// present + has its original contents.
    #[tokio::test(flavor = "current_thread")]
    async fn bind_uds_refuses_to_unlink_regular_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("not-a-socket");
        let payload = b"do not delete me";
        std::fs::write(&path, payload).expect("stage regular file");
        let err = bind_uds(&path)
            .await
            .expect_err("bind_uds must refuse a regular file");
        match err {
            StreamError::UdsPathNotSocket { kind, path: p } => {
                assert_eq!(kind, "regular file");
                assert!(p.contains("not-a-socket"), "path in err: {p}");
            }
            other => panic!("wrong error variant: {other:?}"),
        }
        // File MUST still exist and be byte-identical -- the
        // load-bearing property of the fix.
        let still_there = std::fs::read(&path).expect("file still exists");
        assert_eq!(still_there, payload, "file contents were modified");
    }

    /// `bind_uds` rejects a symlink at the configured path even
    /// when the symlink target is itself a socket.  Following the
    /// symlink at unlink time is the classic TOCTOU vector; the
    /// fix uses `symlink_metadata` so the entry is observable as
    /// a symlink and rejected.
    #[tokio::test(flavor = "current_thread")]
    async fn bind_uds_refuses_symlink_at_path() {
        let dir = tempfile::tempdir().expect("tempdir");
        // Stage a real socket at `target.sock`, then symlink
        // `link.sock -> target.sock`.  The symlink branch must
        // reject regardless of what the target is.
        let target = dir.path().join("target.sock");
        let _real = std::os::unix::net::UnixListener::bind(&target).expect("stage socket");
        let link = dir.path().join("link.sock");
        std::os::unix::fs::symlink(&target, &link).expect("stage symlink");
        let err = bind_uds(&link)
            .await
            .expect_err("bind_uds must refuse a symlink");
        match err {
            StreamError::UdsPathNotSocket { kind, .. } => {
                assert_eq!(kind, "symlink");
            }
            other => panic!("wrong error variant: {other:?}"),
        }
        // The symlink must still point at the original target;
        // the original socket file must still exist.
        let read_link = std::fs::read_link(&link).expect("symlink still present");
        assert_eq!(read_link, target);
        assert!(
            std::fs::symlink_metadata(&target).is_ok(),
            "target file should still exist",
        );
    }

    /// `bind_uds` removes a stale socket file (the legitimate
    /// "previous daemon crashed mid-shutdown" case) and binds a
    /// new listener.  This is the operationally-required success
    /// path the fix must preserve while tightening the safety
    /// gate around it.
    #[tokio::test(flavor = "current_thread")]
    async fn bind_uds_removes_stale_socket_and_binds() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("stale.sock");
        // Stage a real socket file (bound + dropped, so the
        // path persists until removed).
        {
            let _stale = std::os::unix::net::UnixListener::bind(&path).expect("stage stale");
        }
        assert!(
            std::fs::symlink_metadata(&path).is_ok(),
            "stale socket file should exist before bind",
        );
        let listener = bind_uds(&path)
            .await
            .expect("bind_uds must remove stale socket and rebind");
        // The new listener owns a freshly-created socket at the
        // same path.
        assert!(std::fs::symlink_metadata(&path).is_ok());
        drop(listener);
    }

    /// `bind_uds` rejects a parent dir that is world-writable
    /// without the sticky bit (the unconfined `chmod 0o777` shape).
    /// Sticky-bit (`/tmp`-shape) parents pass.
    #[tokio::test(flavor = "current_thread")]
    async fn bind_uds_refuses_unconfined_parent_dir() {
        use std::os::unix::fs::PermissionsExt;
        let outer = tempfile::tempdir().expect("tempdir");
        // Sub-dir we will set 0o777 (no sticky) on -- the
        // reject-shape.  Doing this on the temp root would also
        // chmod the tempdir's cleanup behaviour.
        let unsafe_parent = outer.path().join("public-rw");
        std::fs::create_dir(&unsafe_parent).expect("mkdir unsafe");
        std::fs::set_permissions(&unsafe_parent, std::fs::Permissions::from_mode(0o777))
            .expect("chmod unsafe");
        let path = unsafe_parent.join("test.sock");
        let err = bind_uds(&path)
            .await
            .expect_err("bind_uds must refuse an unconfined parent dir");
        match err {
            StreamError::UdsParentInsecure { detail, .. } => {
                assert!(
                    detail.contains("world-writable") && detail.contains("sticky"),
                    "wrong detail: {detail}",
                );
            }
            other => panic!("wrong error variant: {other:?}"),
        }
        // Now flip the sticky bit on -- /tmp-shape -- and
        // confirm bind_uds accepts.
        std::fs::set_permissions(&unsafe_parent, std::fs::Permissions::from_mode(0o1777))
            .expect("chmod sticky");
        let listener = bind_uds(&path)
            .await
            .expect("bind_uds must accept sticky-bit parent");
        drop(listener);
    }

    /// `set_uds_permissions` refuses to chmod a non-socket file
    /// (defence-in-depth against a hostile swap that beat the
    /// parent-dir confinement check).  Pre-fix the function
    /// would happily chmod whatever was at the path.
    #[tokio::test(flavor = "current_thread")]
    async fn set_uds_permissions_refuses_non_socket() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("not-a-socket");
        std::fs::write(&path, b"hello").expect("stage regular file");
        let err = set_uds_permissions(&path, 0o600)
            .expect_err("set_uds_permissions must refuse a regular file");
        match err {
            StreamError::UdsPathNotSocket { kind, .. } => {
                assert_eq!(kind, "regular file");
            }
            other => panic!("wrong error variant: {other:?}"),
        }
    }

    /// `[stream.tcp_policy]` round-trips through TOML --
    /// serialize+reparse must equal the original, populated
    /// values must differ from `default()` (so a regression
    /// that drops a field can't silently still pass), and the
    /// empty TOML path falls back to `default()` via
    /// `serde(default)`.  Uses `require_subprotocol = false`
    /// (a non-default value) so deser-skip of the field would
    /// fail the round-trip.
    #[test]
    fn transport_policy_toml_round_trips() {
        let toml_input = r#"
max_connections_per_stream = 16
require_subprotocol = false
allowed_origins = ["https://app.example.com", "https://localhost:5173"]
"#;
        let parsed: TransportPolicy = toml::from_str(toml_input).expect("toml load");
        assert_eq!(parsed.max_connections_per_stream, 16);
        assert!(!parsed.require_subprotocol);
        assert_eq!(
            parsed.allowed_origins.as_deref().unwrap_or(&[]),
            &[
                "https://app.example.com".to_string(),
                "https://localhost:5173".to_string(),
            ],
        );
        assert_ne!(
            parsed,
            TransportPolicy::default(),
            "populated case must not equal default; otherwise a deser regression \
             could silently pass",
        );

        // Real round-trip: serialize -> reparse -> compare.
        let serialized = toml::to_string(&parsed).expect("serialize");
        let reparsed: TransportPolicy = toml::from_str(&serialized).expect("re-parse");
        assert_eq!(parsed, reparsed, "TOML round-trip must preserve all fields");

        let empty: TransportPolicy = toml::from_str("").expect("empty toml");
        assert_eq!(empty, TransportPolicy::default());
        assert!(empty.allowed_origins.is_none());
    }
}
