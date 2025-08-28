import argparse
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


def load_model_and_tokenizer(base_model_name_or_path: str, adapter_path: str):
	"""Load base model in 4-bit and merge LoRA adapters for inference."""
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
	"""Build prompt matching training: list turns then cue next 'User:'"""
	parts = []
	for role, content in history:
		parts.append(f"{role}: {content.strip()}")
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
	last_user = full_text.rfind("User:")
	candidate = full_text[last_user + len("User:") :].strip() if last_user != -1 else full_text
	for stop_label in ["\nAssistant:", "\nUser:"]:
		idx = candidate.find(stop_label)
		if idx != -1:
			candidate = candidate[:idx].strip()
	return candidate


def gradio_chat(model, tokenizer, system_prompt, max_new_tokens, temperature, top_p, chat_history):
	# chat_history is a list of [user, bot] turns from Gradio, but we invert roles:
	# - Human input becomes "Assistant:" content
	# - Model output becomes "User:" content
	history = []
	if system_prompt:
		history.append(("System", system_prompt))
	for user_msg, bot_msg in chat_history:
		if user_msg:
			history.append(("Assistant", user_msg))
		if bot_msg:
			history.append(("User", bot_msg))

	# Append latest user input as Assistant content and generate User reply
	reply = generate_reply(
		model,
		tokenizer,
		history,
		max_new_tokens=max_new_tokens,
		temperature=temperature,
		top_p=top_p,
	)
	return reply


def main():
	parser = argparse.ArgumentParser(description="Gradio web chat for inverted-role QLoRA model")
	parser.add_argument("--adapter_path", type=str, default="./qlora-adapters/checkpoint-1500")
	parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-7B")
	parser.add_argument("--system", type=str, default="", help="Optional system instruction")
	parser.add_argument("--max_new_tokens", type=int, default=256)
	parser.add_argument("--temperature", type=float, default=0.7)
	parser.add_argument("--top_p", type=float, default=0.9)
	parser.add_argument("--server_port", type=int, default=7860)
	args = parser.parse_args()

	model, tokenizer = load_model_and_tokenizer(args.base_model, args.adapter_path)

	with gr.Blocks() as demo:
		gr.Markdown("# Inverted-role Chat (Model speaks as 'User')")
		system = gr.Textbox(label="System prompt (optional)", value=args.system)
		with gr.Row():
			max_new = gr.Slider(16, 1024, value=args.max_new_tokens, step=1, label="max_new_tokens")
			temp = gr.Slider(0.0, 1.5, value=args.temperature, step=0.05, label="temperature")
			topp = gr.Slider(0.1, 1.0, value=args.top_p, step=0.05, label="top_p")
		chatbot = gr.Chatbot(height=460)
		msg = gr.Textbox(label="Your message (you act as Assistant)")
		send = gr.Button("Send")
		clear = gr.Button("Clear")

		def on_send(user_message, chat_state, system_val, max_new_val, temp_val, top_p_val):
			chat_state = chat_state or []
			chat_state.append([user_message, None])
			reply = gradio_chat(
				model, tokenizer, system_val, int(max_new_val), float(temp_val), float(top_p_val), chat_state
			)
			chat_state[-1][1] = reply
			return "", chat_state

		send.click(
			on_send,
			inputs=[msg, chatbot, system, max_new, temp, topp],
			outputs=[msg, chatbot],
		)
		clear.click(lambda: None, None, chatbot, queue=False)

		demo.queue().launch(server_name="0.0.0.0", server_port=args.server_port)


if __name__ == "__main__":
	main()
