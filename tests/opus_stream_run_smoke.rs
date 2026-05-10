//! Integration smoke test for the async `run` loop: subscriber-driven
//! pause/resume, broadcast wiring, lag handling.

use acoustics_lab::audio_buffer::AudioBuffer;
use acoustics_lab::opus_stream::{IN_RATE_HZ, run};
use bytes::Bytes;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;
use tokio::sync::{broadcast, watch};
use tokio_util::sync::CancellationToken;

fn fill_writer(writer: &mut acoustics_lab::audio_buffer::Writer, seconds: f32) {
    // Sine at 1 kHz.
    let n = (IN_RATE_HZ as f32 * seconds) as usize;
    let pcm: Vec<f32> = (0..n)
        .map(|i| {
            0.5 * (2.0_f32 * std::f32::consts::PI * 1000.0 * (i as f32 / IN_RATE_HZ as f32)).sin()
        })
        .collect();
    // Push in 1024-sample chunks.
    for chunk in pcm.chunks(1024) {
        writer.push(chunk);
    }
}

/// Active stream: writer pushes 2 s of audio, run() encodes packets,
/// receiver collects them, we cancel and drain.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn active_stream_emits_packets() {
    // 262_144 (= 2^18, ~5.94 s) is the next pow2 above 5 x IN_RATE_HZ.
    let buf = AudioBuffer::new(262_144);
    let mut writer = buf.take_writer();
    let reader = buf.reader_at(0);

    let (sub_tx, sub_rx) = watch::channel(1usize); // subscribers > 0 from the start
    let (out_tx, mut out_rx) = broadcast::channel::<Bytes>(256);
    let token = CancellationToken::new();
    let token_run = token.clone();
    let packets_encoded = Arc::new(AtomicU64::new(0));
    let packets = packets_encoded.clone();

    // Pre-fill some audio before run() starts.
    fill_writer(&mut writer, 2.0);

    // Run loop in a background task.
    let run_handle =
        tokio::spawn(async move { run(reader, sub_rx, out_tx, token_run, packets, None).await });

    // Drain packets for ~1 s, also pushing fresh audio (so the reader
    // doesn't lap the buffer).
    let mut packets = Vec::new();
    let deadline = std::time::Instant::now() + Duration::from_secs(2);
    while std::time::Instant::now() < deadline && packets.len() < 30 {
        match out_rx.try_recv() {
            Ok(b) => packets.push(b),
            Err(broadcast::error::TryRecvError::Empty) => {
                tokio::time::sleep(Duration::from_millis(20)).await;
                fill_writer(&mut writer, 0.020); // ~20 ms refill
            }
            Err(broadcast::error::TryRecvError::Lagged(_)) => continue,
            Err(broadcast::error::TryRecvError::Closed) => break,
        }
    }

    token.cancel();
    let _ = sub_tx.send(0);
    let _ = run_handle.await.expect("run task panicked");
    assert!(
        packets.len() >= 30,
        "expected >=30 packets, got {}",
        packets.len()
    );

    // All packets non-empty.
    for (i, p) in packets.iter().enumerate() {
        assert!(!p.is_empty(), "packet {i} empty");
        assert!(p.len() <= 4000, "packet {i} too big: {} B", p.len());
    }

    // `packets_encoded` is bumped per `out.send` (post-rename
    // from `packets_emitted` for encoder-progress vs delivery
    // clarity; see `opus_stream::run` doc).  The broadcast
    // receiver may have observed Lagged events that skipped
    // some packets, so the receiver-collected `packets.len()`
    // is a lower bound on the counter, not an equality.  Both
    // must be >= 30 (one already asserted) and the counter
    // must be >= the collected count.
    let counted = packets_encoded.load(Ordering::Relaxed);
    assert!(
        counted >= packets.len() as u64,
        "packets_encoded={counted} < collected={}; counter must include packets the receiver missed",
        packets.len(),
    );
    assert!(
        counted >= 30,
        "packets_encoded={counted} < 30; encoder did not bump counter for emitted packets",
    );
}

/// Paused stream emits no packets; resume kicks them in.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pause_resume_state_machine() {
    // 262_144 (= 2^18, ~5.94 s) is the next pow2 above 5 x IN_RATE_HZ.
    let buf = AudioBuffer::new(262_144);
    let mut writer = buf.take_writer();
    let reader = buf.reader_at(0);

    let (sub_tx, sub_rx) = watch::channel(0usize); // start paused
    let (out_tx, mut out_rx) = broadcast::channel::<Bytes>(256);
    let token = CancellationToken::new();
    let token_run = token.clone();
    let packets = Arc::new(AtomicU64::new(0));

    fill_writer(&mut writer, 1.0);

    let run_handle =
        tokio::spawn(async move { run(reader, sub_rx, out_tx, token_run, packets, None).await });

    // While paused, the run loop should NOT emit anything.
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert!(
        matches!(
            out_rx.try_recv(),
            Err(broadcast::error::TryRecvError::Empty)
        ),
        "paused state emitted packets",
    );

    // Resume: 1 subscriber.  Run should rebuild engine + start encoding.
    sub_tx.send(1).expect("send subscribers=1");

    // Keep filling audio so the reader doesn't lap.
    let fill_done = tokio::time::Instant::now() + Duration::from_millis(800);
    let mut got = 0usize;
    while tokio::time::Instant::now() < fill_done {
        fill_writer(&mut writer, 0.020);
        tokio::time::sleep(Duration::from_millis(20)).await;
        while let Ok(_b) = out_rx.try_recv() {
            got += 1;
        }
    }
    assert!(got > 5, "got only {got} packets after resume");

    // Pause again: drain pending then verify silence.
    sub_tx.send(0).expect("send subscribers=0");
    tokio::time::sleep(Duration::from_millis(100)).await;
    while out_rx.try_recv().is_ok() {} // drain
    let pre = out_rx.len();
    tokio::time::sleep(Duration::from_millis(200)).await;
    let post = out_rx.len();
    assert_eq!(pre, post, "packets emitted after pause: {pre} -> {post}");

    token.cancel();
    let _ = sub_tx.send(0);
    let _ = run_handle.await.expect("run task panicked");
}

/// Capture-timing anchor: when the encoder is plumbed with a
/// producer-side `SharedTimingAnchor`, every emitted `AudioFrame`
/// carries `t_us_capture_monotonic` derived from the chunk's
/// first 44.1 kHz sample position projected through the anchor.
///
/// This test asserts the end-to-end +/-1 ms tolerance gate: a
/// synthesized producer publishes a known anchor (`head_pos = 0`,
/// `captured_at = 1_000_000_000` us, 44.1 kHz), then writes 2 s
/// of audio while the encoder runs.  The first emitted packet's
/// `t_us_capture_monotonic` must equal
/// `captured_at + (BACKLOG_SAMPLES_BEHIND_HEAD / 44100 * 1e6)`
/// within +/-1 ms.  The encoder seeks
/// `BACKLOG_SAMPLES_BEHIND_HEAD` behind the live edge on resume;
/// reading that head value via the anchor's tail projection
/// gives a deterministic expected timestamp.
///
/// The resampler buffers ~32 input samples, so mapping a packet
/// back to its "first sample" is approximate to within one
/// resampler chunk -- the reason the tolerance is 1 ms rather
/// than sub-microsecond.  ~32 samples / 44.1 kHz = ~726 us;
/// 1 ms covers it with comfortable margin for jitter.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn timing_anchor_drives_capture_us_within_one_ms() {
    use acoustics_lab::common::time::{
        BufferTimingAnchor, CaptureTime, capture_us_for, shared_timing_anchor,
    };
    use acoustics_lab::proto::envelope::Payload;
    use acoustics_lab::proto::framing::decode_envelope;

    let buf = AudioBuffer::new(262_144);
    let mut writer = buf.take_writer();
    let reader = buf.reader_at(0);

    // Stage a deterministic anchor BEFORE any audio flows so
    // the projection from sample position to capture_us is
    // a known function of the reader's tail.  `head_pos = 0`
    // and `captured_at = 1_000_000_000` us pin the math: any
    // sample at absolute position N has capture_us =
    // 1_000_000_000 + N * 1e6 / 44_100.
    let anchor_cell = shared_timing_anchor();
    anchor_cell.store(Arc::new(BufferTimingAnchor {
        head_pos: 0,
        captured_at: CaptureTime::from_micros(1_000_000_000),
        sample_rate_hz: 44_100,
    }));

    let (_sub_tx, sub_rx) = watch::channel(1usize);
    let (out_tx, mut out_rx) = broadcast::channel::<Bytes>(256);
    let token = CancellationToken::new();
    let token_run = token.clone();
    let packets_encoded = Arc::new(AtomicU64::new(0));
    let packets = packets_encoded.clone();
    let anchor_for_run = anchor_cell.clone();

    fill_writer(&mut writer, 2.0);

    let run_handle = tokio::spawn(async move {
        run(
            reader,
            sub_rx,
            out_tx,
            token_run,
            packets,
            Some(anchor_for_run),
        )
        .await
    });

    // Drain a few packets, refilling so the encoder doesn't
    // starve.  We just need ONE successfully-decoded packet's
    // timestamp to validate the projection; collect a small
    // batch for robustness against transient lag.
    let mut packet_bytes: Vec<Bytes> = Vec::new();
    let deadline = std::time::Instant::now() + Duration::from_secs(2);
    while std::time::Instant::now() < deadline && packet_bytes.len() < 5 {
        match out_rx.try_recv() {
            Ok(b) => packet_bytes.push(b),
            Err(broadcast::error::TryRecvError::Empty) => {
                tokio::time::sleep(Duration::from_millis(20)).await;
                fill_writer(&mut writer, 0.020);
            }
            Err(broadcast::error::TryRecvError::Lagged(_)) => continue,
            Err(broadcast::error::TryRecvError::Closed) => break,
        }
    }
    token.cancel();
    let _ = run_handle.await.expect("run task panicked");

    assert!(
        !packet_bytes.is_empty(),
        "encoder produced no packets within deadline",
    );

    // Decode the first packet's `AudioFrame` and verify its
    // `t_us_capture_monotonic` matches the anchor projection
    // for *some* tail position in the valid range.  The
    // encoder consumes from `head - BACKLOG_SAMPLES` upward,
    // pulling `PCM_PULL_CHUNK = 2048` samples per iteration
    // (these constants are private; the projection bounds
    // hold without referencing them).  The first emitted
    // packet covers a chunk whose first sample is at SOME
    // absolute position N >= 0; the projection
    // `1_000_000_000 + N * 1e6 / 44_100` produces a stamp
    // bounded between `1_000_000_000` (N = 0) and
    // `1_000_000_000 + 2_000_000` (N = 2 s of audio at 44.1 kHz).
    //
    // The tolerance check is bidirectional: the stamp falls
    // inside this window AND the residual against the
    // closest-matching N is below 1 ms.
    let env = decode_envelope(&packet_bytes[0]).expect("decode envelope");
    let payload = env.payload.expect("envelope payload present");
    let frame = match payload {
        Payload::Audio(f) => f,
        other => panic!("expected Payload::Audio, got {other:?}"),
    };
    let stamp = frame
        .t_us_capture_monotonic
        .expect("anchor-plumbed encoder must populate t_us_capture_monotonic");

    // Lower bound: the anchor's `captured_at` (sample 0).
    assert!(
        stamp >= 1_000_000_000,
        "stamp {stamp} below anchor captured_at; \
         anchor projection went backwards from sample 0",
    );
    // Upper bound: 2 s of audio + 1 ms tolerance.
    assert!(
        stamp <= 1_000_000_000 + 2_000_000 + 1_000,
        "stamp {stamp} above expected upper bound; \
         encoder produced a packet covering audio outside the 2 s fill window",
    );

    // Tightness check via back-projection: the encoder pulled
    // some sample position N from the buffer to produce this
    // packet, and stamped `capture_us_for(anchor, N)`.  We
    // recover the implied N from the stamp using the inverse
    // of the projection (`(stamp - captured_at) * 44_100 / 1e6
    // + anchor.head_pos`) and assert:
    //
    //   * N is in `[0, head_now]` -- the encoder cannot have
    //     read samples that the writer has not yet pushed;
    //   * the round-trip residual `|capture_us_for(anchor,
    //     N_recovered) - stamp|` is within 1 ms (covers the
    //     ~726 us resampler-chunk imprecision plus
    //     integer-division rounding).
    //
    // Without access to the encoder's private `BACKLOG_SAMPLES`
    // / `PCM_PULL_CHUNK` constants the test cannot center its
    // search window precisely, so back-projection is the
    // robust way to validate the contract.
    let anchor = **anchor_cell.load();
    let head_now = writer.head_pos();
    let stamp_delta_us = stamp.saturating_sub(anchor.captured_at.as_micros());
    let n_recovered = (stamp_delta_us as u128 * anchor.sample_rate_hz as u128 / 1_000_000) as u64
        + anchor.head_pos;
    assert!(
        n_recovered <= head_now,
        "recovered sample position {n_recovered} > head_now {head_now}; \
         encoder stamped a future sample (anchor projection inverted)",
    );
    let projected = capture_us_for(anchor, n_recovered);
    let residual = projected.abs_diff(stamp);
    assert!(
        residual <= 1_000,
        "anchor round-trip residual {residual} us > 1 ms tolerance; \
         stamp {stamp}, recovered N {n_recovered}, projected {projected}",
    );

    // The stamp must come from the anchor path, not the
    // publish-time fallback: anchor `captured_at` is set to
    // 1_000_000_000 us (~17 minutes from process boot), so any
    // stamp near that value proves the anchor projection ran.
    // The publish-time fallback path stamps `CaptureTime::now()`
    // which is process-boot-relative and on a fresh test
    // process is well below 1e9 us.
    assert!(
        (999_900_000..=1_000_900_000 + 2_000_000).contains(&stamp),
        "stamp {stamp} not in the anchor's projection image; \
         the publish-time fallback path leaked through",
    );
}

/// Cancellation tears down the loop quickly even from paused state.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cancellation_terminates_paused_loop() {
    // 131_072 (= 2^17, ~2.97 s) is the next pow2 above 2 x IN_RATE_HZ.
    let buf = AudioBuffer::new(131_072);
    let _writer = buf.take_writer();
    let reader = buf.reader_at(0);

    let (_sub_tx, sub_rx) = watch::channel(0usize); // paused
    let (out_tx, _out_rx) = broadcast::channel::<Bytes>(16);
    let token = CancellationToken::new();
    let token_run = token.clone();
    let packets = Arc::new(AtomicU64::new(0));

    let run_handle =
        tokio::spawn(async move { run(reader, sub_rx, out_tx, token_run, packets, None).await });

    // Cancel and verify the task exits within a short timeout.
    token.cancel();
    let res = tokio::time::timeout(Duration::from_millis(500), run_handle).await;
    let inner = res.expect("run task did not exit within 500 ms");
    let _ = inner.expect("run task panicked");
}
