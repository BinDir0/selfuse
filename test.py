from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "/share_data/checkpoints/paligemma-3b-pt-224", padding_side="right"
)
print(tokenizer)
tokenizer.add_bos_token = False
tokenizer.add_eos_token = False

print(f"bos_token_id: {tokenizer.bos_token_id}")
print(f"eos_token_id: {tokenizer.eos_token_id}")
print(f"pad_token_id: {tokenizer.pad_token_id}")

input_string = "\n"

output = tokenizer(
    input_string,
    max_length=16,
    padding="max_length",
    truncation=True,
)

for key, value in output.items():
    print(f"{key}: {value}")
