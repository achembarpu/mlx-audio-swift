import Foundation
import MLX

// Incremental (online) streaming for IBM Granite Speech — growing-window re-decode.
//
// Granite Speech is NOT a streaming architecture. Its front end, encoder, and prompt
// layout all assume the full utterance is available up front:
//
//   1. Mel normalization is utterance-global: `extractFeatures` floors the log-mel at
//      `maxVal - 8.0` where `maxVal` is the whole-utterance max, so no mel frame is
//      frozen until the loudest frame has arrived.
//   2. The conformer's block-wise attention (context_size = 200) is self-contained per
//      block, but the depthwise conv (kernel 15) crosses the block boundary in BOTH
//      directions (±7 frames), so a block's boundary frames depend on the next block's
//      frames — there is no causal right-edge.
//   3. The prompt places every `<|audio|>` placeholder BEFORE the text prompt. As audio
//      grows, the text prompt's absolute positions shift, so a downstream LLM KV-cache
//      (prompt text + generated tokens) is invalidated by every new audio window — the
//      LLM cannot be resumed where it left off.
//
// Unlike Nemotron (chunked_limited attention + causal conv + RNN-T decoder, all designed
// for online decoding), there is no per-chunk state that reproduces the offline encoder
// bit-for-bit. This session therefore implements the best correct BOUNDED approximation:
// it re-runs the offline pipeline on the growing buffer every time a new audio-token
// window completes (window_size = 15 encoder frames → 3 audio tokens ≈ 300 ms of audio),
// and de-duplicates against the previously emitted transcript by longest-common-prefix.
//
// Bit-identical vs approximate:
//   * BIT-IDENTICAL: `finish()` runs the exact offline pipeline on the full buffer (via
//     `GraniteSpeechModel.transcribeOffline`), so the final transcript equals
//     `generate(wholeAudio)`.
//   * APPROXIMATE (intermediate steps only) — the partial-buffer decode can differ from
//     the final transcript for three reasons, none of which survive `finish()`:
//       1. encoder right-context: mid-utterance, the conformer conv sees a reflect-padded
//          right edge instead of future frames, so frames near the (moving) right edge
//          shift;
//       2. QFormer / audio-token alignment: the trailing partial 15-frame window is
//          zero-padded now but fills with real frames later, so its 3 audio tokens (and
//          the LLM decode after them) change;
//       3. mel normalization: the log-mel global max is the utterance max SO FAR, which
//          can still grow (rare once a loud frame has arrived).
//     When any of these alter the transcript PREFIX, the prefix de-dup falls back to
//     re-emitting the whole text (no backspaces in the insertion path — the server's
//     TranscriptDeltaEmitter applies the same forward-only rule on `session.text`).
//
// Latency / WER tradeoff: re-decoding at window granularity adds ~300 ms of latency over
// the offline result, at identical final WER (finish() is exact). The cost is O(buffer²)
// encode over a session — each decode re-encodes the whole buffer (mel ≈ 1% of encode, so
// the encoder dominates). For a bounded approximation that is the accepted tradeoff this
// pass; the follow-up below removes it.
//
// FOLLOW-UP DESIGN (cache-aware conformer, O(buffer)): keep a per-layer conv cache of the
// last 7 GLU frames so 200-frame blocks can be encoded independently, and hold back the
// last 7 frames of each block until the next block arrives (recompute them then) — the
// attention is already block-local, so only the conv right-edge is approximate. Feed the
// projector window-by-window (15 encoder frames → 3 audio tokens) rather than re-projecting
// the whole sequence. The LLM prefill remains the hard part: because audio tokens precede
// the text prompt, the text-prompt KV must be re-prefilled each step (M tokens, cheap) and
// the audio-token KV appended incrementally — but the decode KV is still invalidated by the
// growing prefix, so re-decode (or a speculative-carry scheme) is required regardless. That
// is a larger change with its own WER/bit-exactness risks, so it is left out of this pass.

/// Controls whether a Granite session exposes provisional transcript snapshots.
///
/// Granite is an offline architecture: growing-window re-decodes can revise text
/// already returned by an earlier window. `finalOnly` is therefore the safe default
/// for append-only consumers. `growingWindow` is retained for callers that can
/// handle replacement semantics themselves.
public enum GraniteSpeechStreamingMode: Sendable, Equatable {
    case finalOnly
    case growingWindow
}

public final class GraniteSpeechStreamSession {
    /// Text + token ids decoded by a single `step` / `finish` call.
    public struct Delta {
        public let text: String
        public let tokenIds: [Int]
    }

    private let model: GraniteSpeechModel
    private let userPrompt: String?
    private let maxTokens: Int
    private let temperature: Float
    private let mode: GraniteSpeechStreamingMode

    private var rawBuffer: [Float] = []
    private var emittedText = ""
    private var tokenIds: [Int] = []
    private var lastNumAudioTokens = 0
    private var done = false

    init(
        model: GraniteSpeechModel,
        userPrompt: String?,
        maxTokens: Int,
        temperature: Float,
        mode: GraniteSpeechStreamingMode
    ) {
        self.model = model
        self.userPrompt = userPrompt
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.mode = mode
    }

    /// Full transcript decoded so far (the server's append-only emitter consumes this
    /// snapshot, not the raw `Delta`, exactly as with the Voxtral/Nemotron sessions).
    public var text: String { emittedText }
    /// Token ids decoded so far.
    public var tokens: [Int] { tokenIds }
    /// Whether `finish()` has been called.
    public var isFinished: Bool { done }

    /// Whether a non-final audio chunk may trigger model inference.
    static func shouldDecodeIntermediate(mode: GraniteSpeechStreamingMode) -> Bool {
        mode == .growingWindow
    }

    /// Returns a suffix only when the latest decode preserves the already-emitted
    /// transcript. A growing-window re-decode can temporarily shorten or rewrite
    /// its result; the insertion contract is append-only, so those snapshots must
    /// be held back rather than emitted as a duplicate or rewrite.
    static func appendOnlyTextDelta(previous: String, latest: String) -> String {
        guard latest.hasPrefix(previous) else { return "" }
        return String(latest.dropFirst(previous.count))
    }

    /// Returns token IDs beyond the previously emitted prefix. Re-decodes are
    /// allowed to return fewer IDs than an earlier pass, so slicing is guarded.
    static func appendOnlyTokenDelta(previousCount: Int, latest: [Int]) -> [Int] {
        guard previousCount >= 0, latest.count >= previousCount else { return [] }
        return Array(latest.dropFirst(previousCount))
    }

    /// Ingest a chunk of 16 kHz mono samples; returns the text decoded by this call
    /// (usually empty until a new audio-token window completes).
    @discardableResult
    public func step(_ samples: [Float]) -> Delta {
        rawBuffer.append(contentsOf: samples)
        return advance(final: false)
    }

    @discardableResult
    public func step(_ samples: MLXArray) -> Delta {
        let mono = samples.ndim > 1 ? samples.mean(axis: -1) : samples
        return step(mono.asType(.float32).asArray(Float.self))
    }

    /// Flush the trailing partial window so the final transcript equals
    /// `generate(wholeAudio)`. Call once after the last `step`.
    @discardableResult
    public func finish() -> Delta {
        advance(final: true)
    }

    private func advance(final: Bool) -> Delta {
        guard !done else { return Delta(text: "", tokenIds: []) }
        guard !rawBuffer.isEmpty else {
            if final { done = true }
            return Delta(text: "", tokenIds: [])
        }

        // No intermediate result can be made append-only-safe for Granite: the
        // global mel normalization, right-edge convolution, and audio-before-text
        // prompt can all revise an earlier decode. In finalOnly mode, defer all
        // inference until finish(), preserving both correctness and O(1) interim
        // CPU/GPU work. `growingWindow` remains an explicit opt-in for replacement-
        // capable consumers.
        guard final || Self.shouldDecodeIntermediate(mode: mode) else {
            return Delta(text: "", tokenIds: [])
        }

        let firstNew = tokenIds.count
        let audio = MLXArray(rawBuffer)

        // Decode only when a new audio-token window has completed (or on finish). The
        // window gate is what bounds the re-decode cadence to ~300 ms; extractFeatures is
        // the cheap mel front end, so running it just for the token count is negligible.
        let (_, numAudioTokens) = model.extractFeatures(audio)
        guard final || numAudioTokens > lastNumAudioTokens else {
            return Delta(text: "", tokenIds: [])
        }
        lastNumAudioTokens = numAudioTokens

        let result = model.transcribeOffline(
            audio: audio,
            maxTokens: maxTokens,
            temperature: temperature,
            userPrompt: userPrompt
        )

        let fullText = result.text
        let deltaText = Self.appendOnlyTextDelta(previous: emittedText, latest: fullText)
        if !deltaText.isEmpty || fullText == emittedText {
            emittedText = fullText
        }
        tokenIds = result.tokenIds
        let deltaIds = Self.appendOnlyTokenDelta(
            previousCount: firstNew,
            latest: result.tokenIds
        )

        if final { done = true }
        Memory.clearCache()
        return Delta(text: deltaText, tokenIds: deltaIds)
    }
}

public extension GraniteSpeechModel {
    /// Create an online streaming session. Feed audio with `step(_:)`, then `finish()`.
    ///
    /// - Parameters:
    ///   - prompt: explicit transcription/translation instruction; `nil` uses the default.
    ///   - language: language code/name (e.g. "fr") for speech translation; `nil` for
    ///     transcription. Same resolution as `generate`.
    ///   - maxTokens / temperature: forwarded to the greedy decode, matching `generate`.
    func makeStreamSession(
        prompt: String? = nil,
        language: String? = nil,
        maxTokens: Int = 4096,
        temperature: Float = 0.0,
        mode: GraniteSpeechStreamingMode = .finalOnly
    ) -> GraniteSpeechStreamSession {
        GraniteSpeechStreamSession(
            model: self,
            userPrompt: resolveUserPrompt(prompt: prompt, language: language),
            maxTokens: maxTokens,
            temperature: temperature,
            mode: mode
        )
    }

    /// Transcribe a whole audio buffer through the online streaming session, feeding
    /// fixed `chunkMs`-sized chunks as a live caller would — instead of the whole-buffer
    /// `generateStream`. `onDelta` receives each newly decoded fragment as it is produced
    /// (use it to render live output); the returned `STTOutput` is the full transcript
    /// (bit-identical to `generate(audio:)` because the session's `finish()` is exact).
    func transcribeStreaming(
        audio: MLXArray,
        generationParameters: STTGenerateParameters = STTGenerateParameters(),
        chunkMs: Int = 480,
        onDelta: ((String) -> Void)? = nil,
        mode: GraniteSpeechStreamingMode = .finalOnly
    ) -> STTOutput {
        let mono = audio.ndim > 1 ? audio.mean(axis: -1) : audio
        let samples = mono.asType(.float32).asArray(Float.self)
        let chunk = max(1, 16000 * chunkMs / 1000)

        let session = makeStreamSession(
            language: generationParameters.language,
            maxTokens: generationParameters.maxTokens,
            temperature: generationParameters.temperature,
            mode: mode
        )
        let start = CFAbsoluteTimeGetCurrent()

        var idx = 0
        while idx < samples.count {
            let end = min(idx + chunk, samples.count)
            let delta = session.step(Array(samples[idx..<end]))
            if !delta.text.isEmpty { onDelta?(delta.text) }
            idx = end
        }
        let tail = session.finish()
        if !tail.text.isEmpty { onDelta?(tail.text) }

        let totalTime = CFAbsoluteTimeGetCurrent() - start
        let tokenCount = session.tokens.count
        return STTOutput(
            text: session.text.trimmingCharacters(in: .whitespacesAndNewlines),
            language: generationParameters.language,
            generationTokens: tokenCount,
            totalTokens: tokenCount,
            generationTps: totalTime > 0 ? Double(tokenCount) / totalTime : 0,
            totalTime: totalTime
        )
    }
}
