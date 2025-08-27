import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


def load_model_and_tokenizer(base_model_name_or_path: str, adapter_path: str):
    """Load base model in 4-bit and merge LoRA adapters for inference."""
    # Match training quantization for compatibility
    compute_dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
        else torch.float16
    )
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )

    # Prefer tokenizer from adapter folder (it was saved during training)
    try:
        tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name_or_path,
        quantization_config=bnb_config,
        device_map="auto",
    )

    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    return model, tokenizer


def build_prompt(history):
    """Build prompt matching the fine-tuning format (role-inverted dataset).

    In training, targets begin with "User:". So we always cue the next turn as "User:".
    """
    parts = []
    for turn in history:
        role = turn["role"].capitalize()
        parts.append(f"{role}: {turn['content'].strip()}")
    # Cue model to speak as the target role used during training ("User:")
    parts.append("User:")
    return "\n".join(parts)


@torch.inference_mode()
def generate_reply(model, tokenizer, history, max_new_tokens=256, temperature=0.7, top_p=0.9):
    prompt = build_prompt(history)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        no_repeat_ngram_size=4,
        repetition_penalty=1.15,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )[0]

    full_text = tokenizer.decode(output_ids, skip_special_tokens=True)
    # Extract only the newly generated User: segment
    # Find the last occurrence of "User:" and take what follows, until next role label if present
    last_user = full_text.rfind("User:")
    candidate = full_text[last_user + len("User:") :].strip() if last_user != -1 else full_text
    # Cut off if model starts the next role label
    for stop_label in ["\nAssistant:", "\nUser:"]:
        idx = candidate.find(stop_label)
        if idx != -1:
            candidate = candidate[:idx].strip()
    return candidate


def chat_loop(model, tokenizer, assistant_starts: bool, system_prompt: str, max_new_tokens: int, temperature: float, top_p: float):
    history = []
    if system_prompt:
        history.append({"role": "system", "content": system_prompt})

    if assistant_starts:
        reply = generate_reply(
            model,
            tokenizer,
            history,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        print(f"User: {reply}")
        # Model produced the "User:" role (assistant in inverted scheme)
        history.append({"role": "user", "content": reply})

    print("Type your messages. Enter 'exit' to quit. Press Enter on an empty line to submit.")
    buffer = []
    while True:
        try:
            line = input("You > ")
        except (EOFError, KeyboardInterrupt):
            print()  # newline
            break

        if line.strip().lower() == "exit":
            break

        # Support multi-line until empty line
        if line.strip() == "":
            user_msg = "\n".join(buffer).strip()
            buffer = []
            if not user_msg:
                continue
            # Map human input to "Assistant:" in the inverted training format
            history.append({"role": "assistant", "content": user_msg})
            reply = generate_reply(
                model,
                tokenizer,
                history,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            print(f"User: {reply}")
            # Model reply is "User:" in the stored history
            history.append({"role": "user", "content": reply})
        else:
            buffer.append(line)


def main():
    parser = argparse.ArgumentParser(description="Chat with a QLoRA-adapted Qwen model.")
    parser.add_argument("--adapter_path", type=str, default="./qlora-adapters/checkpoint-1500", help="Path to saved LoRA adapters (e.g., ./qlora-adapters/checkpoint-1500)")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-7B", help="Base model name or path")
    parser.add_argument("--assistant_starts", action="store_true", help="If set, assistant sends the first message")
    parser.add_argument("--system", type=str, default="You are a helpful AI assistant.", help="Optional system instruction")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)

    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.base_model, args.adapter_path)

    prompt = "System: You are helpful.\nAssistant: Hello!\nUser:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=64)
    print(tokenizer.decode(outputs[0]))
    print(model.peft_config)
    chat_loop(
        model,
        tokenizer,
        assistant_starts=args.assistant_starts,
        system_prompt=args.system,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    



if __name__ == "__main__":
    main()


