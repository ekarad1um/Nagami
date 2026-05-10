# `acoustics` wire-format contract

The streaming protocol shared by the WebSocket and raw-UDS
surfaces.  Schema files live at
[`modules/proto/`](../modules/proto/).

## Versioning

The protocol is **single-version**: there is no `acoustics.v1`
/ `acoustics.v2` fork, no `proto/v1/` namespace, and no
`Envelope.schema_version` field.  The daemon's clients are
limited to its own consumers (no third-party peers in the
wild), so re-versioning at any future point is a fresh-start
replacement rather than an incremental upgrade.  Re-introducing
a versioning surface is justified only if a real cross-
deployment compatibility need appears (the daemon shipping to
externally-owned consumers); see the rationale block in
`modules/proto.rs`.

## WebSocket transport

- **Subprotocol negotiation.** Clients MUST connect with
  `Sec-WebSocket-Protocol: acoustics`.  Daemons reject
  connections whose header is absent or does not list this
  exact token with HTTP 400 Bad Request.  The header value MAY
  be a comma-separated list (per RFC 6455); the daemon accepts
  as long as one trimmed token equals `acoustics`.
- **Frame format.** Each WebSocket binary message is exactly one
  prost-encoded `Envelope`.  The WS layer's own length-prefix bounds
  the message; no per-Envelope length prefix is added.  No
  fragmentation across WS frames -- if an Envelope ever exceeds the
  WS implementation's max frame size the producer MUST close the
  connection rather than fragment.
- **Per-stream URL routing.** `/stream/audio` carries `Envelope {
  payload: Audio(AudioFrame) }`; `/stream/infer` carries
  `Envelope { payload: Inference(InferenceFrame) }`.  Cross-payload
  variants on the wrong URL are a daemon bug.

## Raw UDS transport (deferred wiring; contract documented)

Raw UDS clients that bypass the WebSocket upgrade ceremony follow
this contract:

- Each Envelope on the stream is prefixed with a 4-byte little-endian
  unsigned length, followed by exactly that many bytes of prost-
  encoded `Envelope`:

  ```text
  [u32 LE: payload_len] [payload_len bytes: Envelope-encoded]
  ```

- The maximum accepted prefix is 64 KiB
  (`proto::framing::MAX_UDS_FRAME_BYTES`; re-exported as
  `stream_io::framing::MAX_UDS_FRAME_BYTES`).  Readers MUST close
  the connection on any prefix beyond this -- the prefix IS the
  synchronization point and resync is undefined.
- Helpers ship in `proto::framing::try_encode_length_prefixed`
  (sync, producer-side; rejects payloads over `MAX_UDS_FRAME_BYTES`
  rather than silently truncating the length prefix) and
  `stream_io::framing::decode_length_prefixed` (async, server-side
  reader); the server-side wiring is not yet in tree.
- Receivers that decode envelope bytes MUST go through
  `proto::framing::decode_envelope`, which validates prost decode
  + payload presence in one place; routing each receiver through
  its own ad-hoc check is how decode policies drift.
- Readers MUST close on any framing error (oversized prefix,
  truncated payload, I/O error, malformed envelope).

## Per-message overhead

Wrapping a payload in `Envelope` adds ~2 bytes:

- 1 byte for the `payload` oneof discriminator (`audio = 10` /
  `inference = 11`, both 1-byte tags).
- 1-2 bytes for the inner length varint.

At production cadence (4 Hz inference + 50 Hz audio) the envelope
tax is ~108 B/s -- single-digit ppm of broadcast bandwidth.

## Field numbering convention

See the per-message comments in
[`audio_stream.proto`](../modules/proto/audio_stream.proto),
[`inference_stream.proto`](../modules/proto/inference_stream.proto),
and [`envelope.proto`](../modules/proto/envelope.proto):

- Numbers 1-15 carry 1-byte tags; the daemon reserves these for
  hot fields that appear on every message.
- Numbers 16+ carry 2-byte tags; new additions go here without
  renumbering existing slots.
- Renumbering or removing an existing field is a wire-breaking
  change; the project's no-third-party-peers stance accepts it
  without a `reserved` block.

## Forward-compatibility

- `optional` scalars are absent in the proto3 wire when the producer
  omits them; receivers MUST treat absent as "no value", not as
  "value = 0".  This is the load-bearing distinction for `head_id`,
  `head_version`, `t_us_*`, `sample_rate`, `frame_duration_ms`.
- Unknown field tags in an Envelope's `payload` oneof MAY be dropped
  by the receiver (proto3 unknown-field semantics).  A future
  control-plane variant (e.g. `heartbeat = 12`) is additive --
  older clients skip the unrecognized tag without breaking.
- The `head_id` + `head_version` pair on `InferenceFrame` is atomic
  (engine snapshots both via `HotHead::snapshot_with_version`).
  Receivers can correlate `(head_id, head_version)` to disambiguate
  weights generations under a swap.

## Capture-time semantics (`t_us_capture_monotonic`)

Both `AudioFrame.t_us_capture_monotonic` and
`InferenceFrame.t_us_capture_monotonic` carry the
**capture monotonic time of the audio the frame covers**, not
the encode/publish moment:

- `AudioFrame`: the time the FIRST 44.1 kHz sample of the
  encoded packet's input chunk reached the producer (per the
  daemon's `BufferTimingAnchor` machinery).
- `InferenceFrame`: the time the FIRST 44.1 kHz sample of the
  inference window reached the producer (window-start
  convention: "this prediction is about audio that started at
  t").

The two stamps are wall-aligned within ~1 ms (one resampler
chunk + one mic-arbitrator push period), letting clients pair
audio + inference frames whose `t_us_capture_monotonic` values
agree.  Per-message proto comments document the precision and
the `BufferTimingAnchor` projection math.
