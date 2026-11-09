use specrt::runtime::{Runtime, Sampling};
use specrt::spec::Spec;
use serde_json::json;
use std::path::Path;

fn spec() -> Spec {
    Spec::load(&Path::new(env!("CARGO_MANIFEST_DIR")).join("specs/qwen3-0.6b.toml")).unwrap()
}

fn prompt(n: usize) -> Vec<u32> {
    (0..n as u32).map(|i| 1000 + i * 7 % 5000).collect()
}

#[test]
#[ignore = "needs a CUDA GPU and the HF weights"]
fn seeded_sampling_reproduces_and_greedy_is_the_default() {
    let mut rt = Runtime::load(&spec(), 256).unwrap();
    let p = prompt(32);
    let hot = |seed| Sampling { temperature: 1.5, seed };
    let a = rt.generate(&p, 24, hot(1)).unwrap().tokens;
    let b = rt.generate(&p, 24, hot(1)).unwrap().tokens;
    let c = rt.generate(&p, 24, hot(2)).unwrap().tokens;
    assert_eq!(a, b);
    assert_ne!(a, c);
    let greedy = rt.generate(&p, 24, Sampling::GREEDY).unwrap().tokens;
    assert_eq!(greedy, rt.generate(&p, 24, Sampling::GREEDY).unwrap().tokens);
    assert_ne!(greedy, a);
}

#[test]
#[ignore = "needs a CUDA GPU and the HF weights"]
fn batch_decode_matches_single_sequence_greedy() {
    for file in ["specs/qwen3-0.6b.toml", "specs/qwen2.5-0.5b.toml"] {
        let spec = Spec::load(&Path::new(env!("CARGO_MANIFEST_DIR")).join(file)).unwrap();
        let prompts: Vec<Vec<u32>> = (0..4u32)
            .map(|s| (0..40u32).map(|i| 1000 + (i * 7 + s * 131) % 5000).collect())
            .collect();
        let mut single = Runtime::load(&spec, 256).unwrap();
        let expected: Vec<Vec<u32>> = prompts
            .iter()
            .map(|p| single.generate(p, 24, Sampling::GREEDY).unwrap().tokens)
            .collect();
        drop(single);
        let mut batch = Runtime::load_batched(&spec, 256, 4).unwrap();
        let got = batch.generate_batch(&prompts, 24).unwrap().tokens;
        assert_eq!(got, expected, "{file}");
    }
}

#[test]
#[ignore = "needs a CUDA GPU and the HF weights"]
fn contexts_above_4096_generate_and_match_the_short_context_path() {
    let p = prompt(4500);
    let mut long = Runtime::load(&spec(), 4700).unwrap();
    let g = long.generate(&p, 16, Sampling::GREEDY).unwrap();
    assert_eq!(g.tokens.len(), 16);
    let mut short = Runtime::load(&spec(), 4096).unwrap();
    let head = &p[..3000];
    let a = long.generate(head, 16, Sampling::GREEDY).unwrap().tokens;
    let b = short.generate(head, 16, Sampling::GREEDY).unwrap().tokens;
    assert_eq!(a, b);
}

#[test]
#[ignore = "needs a CUDA GPU and the HF weights"]
fn a_chat_reply_stops_at_the_end_of_turn_and_streams_in_pieces() {
    let mut engine = specrt::Engine::load("Qwen/Qwen3-0.6B", Some(2048)).unwrap();
    let turns = [
        json!([{"role": "user", "content": "Say hello in five words."}]),
        json!([
            {"role": "user", "content": "Say hello in five words."},
            {"role": "assistant", "content": "Hello there, how are you?"},
            {"role": "user", "content": "Now say goodbye."}
        ]),
    ];
    let mut pieces = 0;
    for messages in turns {
        let reply = engine
            .chat(messages.as_array().unwrap(), Some(&json!({"enable_thinking": false})), 400, Sampling::GREEDY, &mut |_| {
                pieces += 1;
                true
            })
            .unwrap();
        assert!(reply.completion_tokens < 400, "never stopped: {}", reply.text);
        for leak in ["Human:", "User:", "<|im_start|>", "<|im_end|>"] {
            assert!(!reply.text.contains(leak), "{leak} in {}", reply.text);
        }
    }
    assert!(pieces > 2, "the replies arrived as {pieces} pieces");
}

#[test]
#[ignore = "needs a CUDA GPU and the HF weights"]
fn a_callback_returning_false_stops_the_stream() {
    let mut engine = specrt::Engine::load("Qwen/Qwen3-0.6B", Some(2048)).unwrap();
    let messages = json!([{"role": "user", "content": "Count from one to two hundred in words."}]);
    let mut pieces = 0;
    let reply = engine
        .chat(messages.as_array().unwrap(), Some(&json!({"enable_thinking": false})), 1000, Sampling::GREEDY, &mut |_| {
            pieces += 1;
            pieces < 3
        })
        .unwrap();
    assert!(reply.completion_tokens < 100, "kept going for {} tokens", reply.completion_tokens);
}

#[test]
#[ignore = "needs a CUDA GPU and the HF weights"]
fn a_context_too_large_for_the_device_is_an_error_not_an_abort() {
    let oom = Runtime::load(&spec(), 1 << 23).err().expect("an 8M token KV cache cannot fit");
    assert!(format!("{oom:#}").to_lowercase().contains("memory"), "{oom:#}");
    let mut engine = specrt::Engine::load("Qwen/Qwen3-0.6B", Some(256)).unwrap();
    assert!(engine.generate("hello", 8, Sampling::GREEDY).is_ok());
}
