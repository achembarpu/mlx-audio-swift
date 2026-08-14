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

    /// Compute the mel rows for a contiguous range of absolute frame indices of a
    /// growing pre-emphasized signal, matching `logMelSpectrogram(...)[0, range]`
    /// within the last ulp (see the numerics note in `NemotronASRStreamSession`).
    ///
    /// `logMelSpectrogram` zero-pads the signal with `nFft/2` on both ends and
    /// frames it at `hop` via a strided view (`asStrided`), so frame `f` is a
    /// function of the samples `[f·hop − nFft/2, f·hop + nFft/2)` (zero outside
    /// the signal). Once `f·hop + nFft/2 <= samples.count`, the frame is FROZEN —
    /// unaffected by any future audio — so it can be computed here and cached.
    /// This is what lets `NemotronASRStreamSession` avoid recomputing the whole
    /// mel on every step (O(buffer²) total) and instead pay O(buffer) overall.
    ///
    /// The rows are built with the SAME `asStrided` view the offline path uses
    /// (not a copied stack), keeping the numerics as close as the differing rfft
    /// batch counts allow.
    static func melFrames(
        _ preemphSignal: [Float],
        frames: Range<Int>,
        window: MLXArray,
        filters: MLXArray,
        config: NemotronASRPreprocessConfig
    ) -> MLXArray {
        let nFft = config.nFft
        let hop = config.hopLength
        let half = nFft / 2
        let m0 = frames.lowerBound
        let count = frames.count
        let N = preemphSignal.count

        // padded = [0 × half] ++ signal ++ [0 × half]; frame f == padded[f·hop ..< f·hop+nFft].
        // Build the minimal padded slice [rowStart, hi) covering the new frames.
        let rowStart = m0 * hop
        let hi = (m0 + count - 1) * hop + nFft

        var samples: [Float] = []
        samples.reserveCapacity(hi - rowStart)
        // padded[i] == 0 for i < half and i >= half + N, else signal[i - half].
        let frontZeros = max(0, min(half, hi) - rowStart)
        if frontZeros > 0 {
            samples.append(contentsOf: [Float](repeating: 0, count: frontZeros))
        }
        let preLo = max(0, rowStart - half)
        let preHi = max(0, min(hi, half + N) - half)
        if preLo < preHi {
            samples.append(contentsOf: preemphSignal[preLo..<preHi])
        }
        let backZeros = max(0, hi - (half + N))
        if backZeros > 0 {
            samples.append(contentsOf: [Float](repeating: 0, count: backZeros))
        }

        let paddedSlice = MLXArray(samples).asType(.float32)
        let framesStacked = asStrided(paddedSlice, [count, nFft], strides: [hop, 1], offset: 0)
        let windowed = framesStacked * window.asType(.float32)
        let fft = MLXFFT.rfft(windowed, axis: 1)
        let power = MLX.abs(fft).square().asType(.float32)
        let mel = MLX.matmul(power, filters.asType(power.dtype))
        return MLX.log(mel + MLXArray(config.logZeroGuardValue, dtype: mel.dtype))
    }
}
