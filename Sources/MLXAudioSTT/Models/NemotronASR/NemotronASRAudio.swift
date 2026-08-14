import Foundation
import MLX
import MLXAudioCore

enum NemotronASRAudio {
    static func logMelSpectrogram(
        _ audio: MLXArray,
        config: NemotronASRPreprocessConfig
    ) -> MLXArray {
        let originalDType = audio.dtype
        var x = audio

        if config.padTo > 0 && x.shape[0] < config.padTo {
            let padLength = config.padTo - x.shape[0]
            let paddedTail = MLXArray(Array(repeating: config.padValue, count: padLength))
            x = MLX.concatenated([x, paddedTail], axis: 0)
        }

        if config.preemph > 0 && x.shape[0] > 1 {
            let first = x[0..<1]
            let rest = x[1...] - Float(config.preemph) * x[..<(x.shape[0] - 1)]
            x = MLX.concatenated([first, rest], axis: 0)
        }

        let window = makeWindow(name: config.window, winLength: config.winLength, fftLength: config.nFft)
        let stftOutput = stft(
            audio: x,
            window: window,
            nFft: config.nFft,
            hopLength: config.hopLength,
            padMode: .constant
        )

        let power = MLX.abs(stftOutput).square().asType(originalDType)
        let filters = melFilters(
            sampleRate: config.sampleRate,
            nFft: config.nFft,
            nMels: config.features,
            norm: "slaney",
            melScale: .slaney
        )

        var mel = MLX.matmul(power, filters.asType(power.dtype))
        mel = MLX.log(mel + MLXArray(config.logZeroGuardValue, dtype: mel.dtype))

        switch config.normalize.lowercased() {
        case "na", "none":
            return mel.expandedDimensions(axis: 0).asType(originalDType)
        case "per_feature":
            let mean = MLX.mean(mel, axis: 0, keepDims: true)
            let denominator = max(mel.dim(0) - 1, 1)
            let variance = MLX.sum((mel - mean).square(), axis: 0, keepDims: true) / Float(denominator)
            let std = MLX.sqrt(variance)
            mel = (mel - mean) / (std + MLXArray(1e-5, dtype: mel.dtype))
        default:
            let mean = MLX.mean(mel)
            let std = MLX.std(mel)
            mel = (mel - mean) / (std + MLXArray(1e-5, dtype: mel.dtype))
        }

        return mel.expandedDimensions(axis: 0).asType(originalDType)
    }

    static func makeWindow(name: String, winLength: Int, fftLength: Int) -> MLXArray {
        let base: MLXArray
        switch name.lowercased() {
        case "hann", "hanning":
            base = hanningWindow(size: winLength)
        case "hamming":
            base = hammingWindow(size: winLength)
        case "blackman":
            base = blackmanWindow(size: winLength)
        case "bartlett":
            base = bartlettWindow(size: winLength)
        default:
            base = hanningWindow(size: winLength)
        }

        if winLength >= fftLength {
            return base[0..<fftLength]
        }

        let left = (fftLength - winLength) / 2
        let right = fftLength - winLength - left
        return MLX.concatenated([
            MLXArray.zeros([left]),
            base,
            MLXArray.zeros([right])
        ], axis: 0)
    }

    private static func hammingWindow(size: Int) -> MLXArray {
        if size <= 1 {
            return MLXArray(Array(repeating: Float(1), count: max(size, 1)))
        }
        let denom = Float(size - 1)
        let values = (0..<size).map { n in
            Float(0.54) - Float(0.46) * cos(2 * Float.pi * Float(n) / denom)
        }
        return MLXArray(values)
    }

    private static func blackmanWindow(size: Int) -> MLXArray {
        if size <= 1 {
            return MLXArray(Array(repeating: Float(1), count: max(size, 1)))
        }
        let denom = Float(size - 1)
        let values = (0..<size).map { n in
            let k = 2 * Float.pi * Float(n) / denom
            return Float(0.42) - Float(0.5) * cos(k) + Float(0.08) * cos(2 * k)
        }
        return MLXArray(values)
    }

    private static func bartlettWindow(size: Int) -> MLXArray {
        if size <= 1 {
            return MLXArray(Array(repeating: Float(1), count: max(size, 1)))
        }
        let mid = Float(size - 1) / 2
        let values = (0..<size).map { n in
            Float(1) - abs((Float(n) - mid) / mid)
        }
        return MLXArray(values)
    }
}

// MARK: - Incremental (online) mel

extension NemotronASRAudio {
    /// Static window / mel-filterbank for a preprocess config, reused across the
    /// per-frame incremental computation below (computed once per session).
    static func melWindow(_ config: NemotronASRPreprocessConfig) -> MLXArray {
        makeWindow(name: config.window, winLength: config.winLength, fftLength: config.nFft)
    }

    static func melFilterbank(_ config: NemotronASRPreprocessConfig) -> MLXArray {
        melFilters(
            sampleRate: config.sampleRate,
            nFft: config.nFft,
            nMels: config.features,
            norm: "slaney",
            melScale: .slaney
        )
    }

    /// Compute the mel row for one absolute frame index of a growing
    /// pre-emphasized signal, bit-identical to `logMelSpectrogram(...)[0, frame]`.
    ///
    /// `logMelSpectrogram` zero-pads the signal with `nFft/2` on both ends and
    /// frames it at `hop` (strided view), so frame `f` is a function of the
    /// samples `[f·hop − nFft/2, f·hop + nFft/2)` (zero outside the signal). Once
    /// `f·hop + nFft/2 <= samples.count`, the frame is FROZEN — unaffected by any
    /// future audio — so it can be computed here and cached. This is what lets
    /// `NemotronASRStreamSession` avoid recomputing the whole mel on every step
    /// (O(buffer²) total) and instead pay O(buffer) overall.
    static func melFrame(
        _ preemphSignal: [Float],
        frame: Int,
        window: MLXArray,
        filters: MLXArray,
        config: NemotronASRPreprocessConfig
    ) -> MLXArray {
        let nFft = config.nFft
        let hop = config.hopLength
        let half = nFft / 2
        let start = frame * hop - half
        let end = start + nFft

        // padded[f·hop ...] == signal window with nFft/2 zero-pad at both edges.
        var samples: [Float] = []
        samples.reserveCapacity(nFft)
        if start < 0 {
            samples.append(contentsOf: [Float](repeating: 0, count: -start))
        }
        let lo = max(0, start)
        let hi = min(preemphSignal.count, end)
        if lo < hi {
            samples.append(contentsOf: preemphSignal[lo..<hi])
        }
        if end > preemphSignal.count {
            samples.append(contentsOf: [Float](repeating: 0, count: end - preemphSignal.count))
        }

        let x = MLXArray(samples).asType(.float32)
        let windowed = x * window.asType(.float32)
        let fft = MLXFFT.rfft(windowed.expandedDimensions(axis: 0), axis: 1)
        let power = MLX.abs(fft).square()
        let mel = MLX.matmul(power, filters.asType(power.dtype))
        // Keep the leading batch axis: the caller concatenates rows along axis 0
        // into a (T, F) mel, so each row must be [1, F], not a 1-D [F].
        return MLX.log(mel + MLXArray(config.logZeroGuardValue, dtype: mel.dtype))
    }
}
