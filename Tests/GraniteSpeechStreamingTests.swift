import Testing

@testable import MLXAudioSTT

struct GraniteSpeechStreamingTests {
    @Test func finalOnlyModeDefersInferenceUntilFinish() {
        #expect(
            GraniteSpeechStreamSession.shouldDecodeIntermediate(mode: .finalOnly) == false
        )
        #expect(
            GraniteSpeechStreamSession.shouldDecodeIntermediate(mode: .growingWindow) == true
        )
        #expect(
            GraniteSpeechStreamSession.shouldEmitSnapshots(mode: .revisableOverlay) == true
        )
        #expect(
            GraniteSpeechStreamSession.shouldEmitSnapshots(mode: .finalOnly) == false
        )
    }

    @Test func revisableOverlayKeepsTheLatestFullText() {
        #expect(
            GraniteSpeechStreamSession.snapshotText(
                previous: "hello world",
                latest: "hello"
            ) == "hello"
        )
        #expect(
            GraniteSpeechStreamSession.snapshotText(
                previous: "hello",
                latest: "hello world"
        ) == "hello world"
        )
    }

    @Test func decodeGatePreservesWindowCadenceAndFinalFlush() {
        #expect(
            GraniteSpeechStreamSession.shouldDecode(
                mode: .revisableOverlay,
                isFinal: false,
                numAudioTokens: 3,
                lastNumAudioTokens: 3
            ) == false
        )
        #expect(
            GraniteSpeechStreamSession.shouldDecode(
                mode: .revisableOverlay,
                isFinal: false,
                numAudioTokens: 4,
                lastNumAudioTokens: 3
            ) == true
        )
        #expect(
            GraniteSpeechStreamSession.shouldDecode(
                mode: .finalOnly,
                isFinal: false,
                numAudioTokens: 99,
                lastNumAudioTokens: 0
            ) == false
        )
        #expect(
            GraniteSpeechStreamSession.shouldDecode(
                mode: .finalOnly,
                isFinal: true,
                numAudioTokens: 1,
                lastNumAudioTokens: 1
            ) == true
        )
    }

    @Test func shorterReDecodeDoesNotEmitARewriteOrTrap() {
        let priorText = "hello world"
        let priorTokenCount = 3
        let latestText = "hello"
        let latestTokenIds = [10, 11]

        #expect(
            GraniteSpeechStreamSession.appendOnlyTextDelta(
                previous: priorText,
                latest: latestText
            ).isEmpty
        )
        #expect(
            GraniteSpeechStreamSession.appendOnlyTokenDelta(
                previousCount: priorTokenCount,
                latest: latestTokenIds
            ).isEmpty
        )
    }

    @Test func growingReDecodeEmitsOnlyTheNewSuffix() {
        #expect(
            GraniteSpeechStreamSession.appendOnlyTextDelta(
                previous: "hello",
                latest: "hello world"
            ) == " world"
        )
        #expect(
            GraniteSpeechStreamSession.appendOnlyTokenDelta(
                previousCount: 2,
                latest: [10, 11, 12]
            ) == [12]
        )
    }
}
