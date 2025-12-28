GEMMA_TEMPLATE = (
        "<bos><start_of_turn>user\n"
        "{instruction}"
        "<end_of_turn>\n"
        "<start_of_turn>model\n"
    )

LLAMA_TEMPLATE = (
    "<s>[INST] "
    "{instruction} "
    "[/INST]"
)

BASE_TEMPLATE = (
    "You are a helpful assistant who answers questions."
    "{instruction} "
    "The answer is "
)