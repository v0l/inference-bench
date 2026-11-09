use specrt::chat::ChatTemplate;
use std::path::Path;

fn template(file: &str) -> ChatTemplate {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(file);
    ChatTemplate::from_tokenizer_config(&std::fs::read_to_string(path).unwrap()).unwrap().unwrap()
}

#[test]
fn the_recorded_agent_request_renders_to_the_vllm_token_ids() {
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let read = |f: &str| std::fs::read_to_string(fixtures.join(f)).unwrap();
    let t = ChatTemplate::from_files(&read("qwen3.8-27b.tokenizer_config.json"), Some(&read("qwen3.8-27b.chat_template.jinja")))
        .unwrap()
        .unwrap();
    let req: serde_json::Value = serde_json::from_str(&read("agent-request.json")).unwrap();
    let want: serde_json::Value = serde_json::from_str(&read("agent-request.vllm.json")).unwrap();
    let tokenizer = specrt::weights::find_in_snapshot("orcarouter/Qwen3.8-27B-Uncensored-NVFP4", "tokenizer.json")
        .ok()
        .and_then(|p| tokenizers::Tokenizer::from_file(p).ok());
    let cases = [
        ("medium", serde_json::json!({"reasoning_effort": "medium"})),
        ("no_thinking", serde_json::json!({"enable_thinking": false})),
        ("low", serde_json::json!({"reasoning_effort": "low"})),
    ];
    for (name, mut vars) in cases {
        vars["tools"] = req["tools"].clone();
        let text = t.render_with(req["messages"].as_array().unwrap(), &vars).unwrap();
        assert_eq!(text, want[name]["text"].as_str().unwrap(), "{name}");
        if let Some(tok) = &tokenizer {
            let ids: Vec<u64> = tok.encode(text.as_str(), false).unwrap().get_ids().iter().map(|&i| i as u64).collect();
            let expected: Vec<u64> = want[name]["tokens"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap()).collect();
            assert_eq!(ids, expected, "{name}");
        }
    }
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
