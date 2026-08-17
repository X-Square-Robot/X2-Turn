def register():
    from vllm import ModelRegistry

    if "CosyVoice2ForCausalLM" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            "CosyVoice2ForCausalLM",
            "cosyvoice.vllm.cosyvoice2:CosyVoice2ForCausalLM",
        )
