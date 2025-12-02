import argparse
from .inference import BumblebeeChat, HFStreamChat

def start_chat_session(
    model_path,
    device_map,
    dtype,
    max_new_tokens,
    temperature,
    top_k,
    top_p,
    repetition_penalty,
    system_prompt,
    do_sample,
    enable_history,
    training_stage
):
    print(f"正在加载模型: {model_path}，请稍候...")
    bot = BumblebeeChat(
        model_path=model_path,
        device_map=device_map,
        dtype=dtype
    )
    print("模型加载完成！输入 'quit' 或 'exit' 退出聊天。\n")

    messages = []

    while True:
        user_input = input("👤 用户: ").strip()
        if user_input.lower() in {"quit", "exit"}:
            print("👋 再见！")
            break

        if training_stage == "pretrain":
            print("🤖 助手: ", end="", flush=True)
            response_chunks = []
            for text in bot.stream_chat(
                messages=user_input,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample
            ):
                print(text, end="", flush=True)
                response_chunks.append(text)
            print("\n"+ "="*120)
            continue

        if enable_history:
            messages.append({"role": "user", "content": user_input})
            current_messages = messages
        else:
            current_messages = [{"role": "user", "content": user_input}]

        print("🤖 助手: ", end="", flush=True)

        response_chunks = []
        for text in bot.stream_chat(
            messages=current_messages,
            max_new_tokens=max_new_tokens,
            system_prompt=system_prompt,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample
        ):
            print(text, end="", flush=True)
            response_chunks.append(text)
        print("\n"+ "="*120)

        full_response = "".join(response_chunks)

        if enable_history:
            messages.append({"role": "assistant", "content": full_response})

        print()


def bumblebee_streaming_chat():
    parser = argparse.ArgumentParser(description="启动 Bumblebee 聊天会话")
    parser.add_argument("--model_path", type=str, required=True,
                        help="模型路径")
    parser.add_argument("--device_map", type=str, default="auto",
                        help="设备映射（如 'auto', 'cpu', 'cuda:0' 等）")
    parser.add_argument("--dtype", type=str, default="auto",
                        help="模型数据类型（如 'torch.float16', 'torch.bfloat16', 'auto'）")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="最大生成 token 数")
    parser.add_argument("--system_prompt", type=str, default=None,
                        help="设置 system prompt（不传使用默认）")
    parser.add_argument("--temperature", type=float, default=None,
                        help="采样温度（不传使用模型默认）")
    parser.add_argument("--top_k", type=int, default=None,
                        help="Top_k 采样（不传使用模型默认）")
    parser.add_argument("--top_p", type=float, default=None,
                        help="Top_p (nucleus) 采样（不传使用模型默认）")
    parser.add_argument("--repetition_penalty", type=float, default=None,
                        help="重复惩罚系数（不传使用模型默认）")
    parser.add_argument("--do_sample", action="store_true",
                        help="启用采样（否则使用贪婪解码）")
    parser.add_argument("--enable_history", action="store_true",
                        help="启用多轮对话历史")
    
    parser.add_argument("--training_stage", type=str, required=True,
                        choices=["sft", "dpo", "pretrain"],
                        help="模型训练阶段：sft（指令微调）、dpo（偏好优化）、pretrain（预训练）。"
                             "若为 pretrain，则不使用对话格式。")

    args = parser.parse_args()
    args_dict = vars(args)

    start_chat_session(**args_dict)