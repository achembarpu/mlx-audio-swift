import Testing

@testable import MLXAudioSTT

struct GraniteSpeechStreamingTests {
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
