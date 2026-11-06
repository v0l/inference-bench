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

#[test]
fn a_multi_turn_conversation_renders_each_message_with_its_role() {
    let messages = vec![
        serde_json::json!({"role": "system", "content": "Be brief."}),
        serde_json::json!({"role": "user", "content": "What is a KV cache?"}),
        serde_json::json!({"role": "assistant", "content": "Stored keys and values."}),
        serde_json::json!({"role": "user", "content": "Show Rust."}),
    ];
    let text = template("Qwen3-0.6B.tokenizer_config.json").render(&messages).unwrap();
    assert!(text.starts_with("<|im_start|>system\nBe brief.<|im_end|>\n"), "{text}");
    assert!(
        text.contains("<|im_start|>user\nWhat is a KV cache?<|im_end|>\n<|im_start|>assistant\n"),
        "{text}"
    );
    assert!(
        text.contains(
            "Stored keys and values.<|im_end|>\n<|im_start|>user\nShow Rust.<|im_end|>\n"
        ),
        "{text}"
    );
    assert!(text.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"), "{text}");
    assert!(!text.contains("Human:") && !text.contains("Assistant:"));
}
