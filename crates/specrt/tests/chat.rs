use specrt::chat::ChatTemplate;
use std::path::Path;

fn template(file: &str) -> ChatTemplate {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(file);
    ChatTemplate::from_tokenizer_config(&std::fs::read_to_string(path).unwrap()).unwrap().unwrap()
}

#[test]
fn qwen3_wraps_a_user_turn_and_closes_the_thinking_block() {
    assert_eq!(
        template("Qwen3-0.6B.tokenizer_config.json").render_user("hi").unwrap(),
        "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    );
}

#[test]
fn qwen25_adds_the_default_system_prompt() {
    assert_eq!(
        template("Qwen2.5-0.5B.tokenizer_config.json").render_user("hi").unwrap(),
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
    );
}
