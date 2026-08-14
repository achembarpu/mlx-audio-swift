import Testing

@testable import MLXAudioSTT

/// Granite Speech 4.1 keeps the 4.0 model topology, but its converted MLX
/// config adds dtype/conversion/rope metadata and uses a 2B checkpoint name.
/// Those fields must remain forward-compatible with the shared loader.
struct GraniteSpeech41CompatibilityTests {
    @Test func converted41ConfigDecodesWithSharedTopology() throws {
        let json = """
        {
          "model_type": "granite_speech",
          "audio_token_index": 100352,
          "downsample_rate": 5,
          "window_size": 15,
          "dtype": "bfloat16",
          "conversion": {"base_model": "ibm-granite/granite-speech-4.1-2b"},
          "quantization": {"group_size": 64, "bits": 4, "mode": "affine"},
          "encoder_config": {
            "model_type": "granite_speech_encoder",
            "input_dim": 160, "num_layers": 16, "hidden_dim": 1024,
            "feedforward_mult": 4, "num_heads": 8, "dim_head": 128,
            "output_dim": 348, "context_size": 200, "max_pos_emb": 512,
            "conv_kernel_size": 15, "conv_expansion_factor": 2,
            "dtype": "bfloat16"
          },
          "projector_config": {
            "model_type": "blip_2_qformer", "hidden_size": 1024,
            "num_hidden_layers": 2, "num_attention_heads": 16,
            "intermediate_size": 4096, "encoder_hidden_size": 1024,
            "layer_norm_eps": 1e-12
          },
          "text_config": {
            "model_type": "granite", "vocab_size": 100353,
            "hidden_size": 2048, "intermediate_size": 4096,
            "num_hidden_layers": 40, "num_attention_heads": 16,
            "num_key_value_heads": 4, "rope_theta": 10000.0,
            "rms_norm_eps": 1e-5, "attention_multiplier": 0.0078125,
            "embedding_multiplier": 12.0, "residual_multiplier": 0.22,
            "logits_scaling": 8.0, "tie_word_embeddings": false,
            "dtype": "bfloat16",
            "rope_parameters": {"rope_theta": 10000, "rope_type": "default"}
          }
        }
        """

        let config = try JSONDecoder().decode(
            GraniteSpeechModelConfig.self,
            from: Data(json.utf8)
        )
        #expect(config.encoderConfig.numLayers == 16)
        #expect(config.encoderConfig.outputDim == 348)
        #expect(config.textConfig.hiddenSize == 2048)
        #expect(config.textConfig.numKeyValueHeads == 4)
        #expect(config.windowSize == 15)
    }
}
