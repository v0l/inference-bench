use specrt::wire::{hmac_sha256, read_frame, write_frame, Chain, Input, Message, StageClient, StageHandler, StageServer, PROTOCOL_VERSION};
use std::io::{Read, Write};
use std::net::TcpStream;
use std::sync::{Arc, Mutex};
use std::time::Duration;

struct Fake {
    start: u32,
    end: u32,
    last: bool,
    seen: Mutex<Vec<Message>>,
}

impl Fake {
    fn new(start: u32, end: u32, last: bool) -> Arc<Fake> {
        Arc::new(Fake { start, end, last, seen: Mutex::new(Vec::new()) })
    }
}

impl StageHandler for Fake {
    fn ready(&self) -> Message {
        Message::Ready { layer_start: self.start, layer_end: self.end, hidden: 4 }
    }

    fn handle(&self, message: Message) -> Message {
        self.seen.lock().unwrap().push(message.clone());
        match message {
            Message::Forward { input, .. } => {
                let sum: u32 = match &input {
                    Input::Tokens(t) => t.iter().sum(),
                    Input::Hidden { data, .. } => data.iter().map(|b| *b as u32).sum(),
                };
                if self.last {
                    Message::Token { token: sum + 1, compute_us: 7 }
                } else {
                    Message::Output { rows: 1, data: vec![sum as u8; 16], compute_us: 7 }
                }
            }
            Message::DecodeBatch { slots, .. } => {
                if self.last {
                    Message::Tokens { tokens: slots.clone(), compute_us: 7 }
                } else {
                    Message::Output { rows: slots.len() as u32, data: vec![1; 16 * slots.len()], compute_us: 7 }
                }
            }
            Message::Free { .. } | Message::Cancel { .. } => Message::Ack,
            _ => Message::Error { message: "unexpected".into() },
        }
    }
}

const KEY: &[u8] = b"pre-shared";
const T: Duration = Duration::from_secs(5);

#[test]
fn every_message_round_trips_through_the_codec() {
    let messages = vec![
        Message::Hello { version: PROTOCOL_VERSION },
        Message::Ready { layer_start: 3, layer_end: 40, hidden: 5120 },
        Message::Forward { slot: 2, position: 77, input: Input::Tokens(vec![1, 2, 3]), decode: false },
        Message::Forward { slot: 2, position: 0, input: Input::Hidden { rows: 2, data: vec![9; 40] }, decode: true },
        Message::Output { rows: 1, data: vec![1, 2, 3], compute_us: 99 },
        Message::Token { token: 248319, compute_us: 5 },
        Message::Free { slot: 7 },
        Message::Cancel { slot: 0 },
        Message::Ack,
        Message::Error { message: "no".into() },
        Message::Assign { layer_start: 8, layer_end: 16 },
        Message::DecodeBatch { slots: vec![0, 3, 5], positions: vec![10, 20, 30], input: Input::Tokens(vec![7, 8, 9]) },
        Message::DecodeBatch { slots: vec![1, 2], positions: vec![4, 5], input: Input::Hidden { rows: 2, data: vec![3; 32] } },
        Message::Tokens { tokens: vec![5, 6, 7], compute_us: 12 },
    ];
    for m in messages {
        let mut buf = Vec::new();
        write_frame(&mut buf, &m).unwrap();
        assert_eq!(read_frame(&mut buf.as_slice()).unwrap(), m);
    }
}

#[test]
fn truncated_trailing_and_unknown_frames_are_rejected() {
    let good = Message::Forward { slot: 1, position: 2, input: Input::Tokens(vec![1, 2]), decode: true }.encode();
    for cut in 0..good.len() {
        assert!(Message::decode(&good[..cut]).is_err(), "cut at {cut}");
    }
    let mut trailing = good.clone();
    trailing.push(0);
    assert!(Message::decode(&trailing).is_err());
    assert!(Message::decode(&[200]).is_err());
    let huge_count = [3u8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 0];
    assert!(Message::decode(&huge_count).is_err());
    let oversized = (u32::MAX).to_le_bytes();
    assert!(read_frame(&mut oversized.as_slice()).is_err());
}

#[test]
fn hmac_matches_the_rfc_4231_vectors() {
    let hex = |b: [u8; 32]| b.iter().map(|x| format!("{x:02x}")).collect::<String>();
    assert_eq!(hex(hmac_sha256(&[0x0b; 20], b"Hi There")), "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7");
    assert_eq!(hex(hmac_sha256(b"Jefe", b"what do ya want for nothing?")), "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843");
    assert_eq!(
        hex(hmac_sha256(&[0xaa; 131], b"Test Using Larger Than Block-Size Key - Hash Key First")),
        "60e431591ee0b67f0d8a26aacbf5b77f8e0bc6213728c5140546040f0ee37f54"
    );
}

#[test]
fn a_chain_runs_downstream_first_and_passes_hidden_rows_between_stages() {
    let a = Fake::new(0, 10, false);
    let b = Fake::new(10, 20, false);
    let c = Fake::new(20, 30, true);
    let servers: Vec<_> = [&a, &b, &c].iter().map(|f| StageServer::start("127.0.0.1:0", KEY, (*f).clone()).unwrap()).collect();
    let addrs: Vec<String> = servers.iter().map(|s| s.addr().to_string()).collect();
    let mut chain = Chain::connect(&addrs, KEY, T).unwrap();
    assert_eq!(chain.len(), 3);
    assert_eq!(chain.layers(), 30);
    let (token, times) = chain.forward(1, 0, vec![1, 2, 3], false).unwrap();
    assert_eq!(token, 96 * 16 + 1);
    assert_eq!(times.stage.len(), 3);
    assert_eq!(times.compute, vec![Duration::from_micros(7); 3]);
    assert!(matches!(a.seen.lock().unwrap()[0], Message::Forward { position: 0, decode: false, input: Input::Tokens(_), .. }));
    assert!(matches!(c.seen.lock().unwrap()[0], Message::Forward { decode: false, input: Input::Hidden { rows: 1, .. }, .. }));
    chain.cancel(1).unwrap();
    chain.free(1).unwrap();
    for f in [&a, &b, &c] {
        let seen = f.seen.lock().unwrap();
        assert!(matches!(seen[1], Message::Cancel { slot: 1 }) && matches!(seen[2], Message::Free { slot: 1 }));
    }
    assert_eq!(chain.ping_all().unwrap().len(), 3);
}

#[test]
fn a_decode_batch_makes_one_round_trip_per_stage_and_returns_a_token_per_row() {
    let a = Fake::new(0, 10, false);
    let b = Fake::new(10, 20, true);
    let servers: Vec<_> = [&a, &b].iter().map(|f| StageServer::start("127.0.0.1:0", KEY, (*f).clone()).unwrap()).collect();
    let addrs: Vec<String> = servers.iter().map(|s| s.addr().to_string()).collect();
    let mut chain = Chain::connect(&addrs, KEY, T).unwrap();
    let (tokens, times) = chain.forward_batch(&[0, 1, 2], &[5, 6, 7], &[1, 2, 3]).unwrap();
    assert_eq!(tokens.len(), 3);
    assert_eq!(times.stage.len(), 2);
    assert_eq!(a.seen.lock().unwrap().len(), 1);
    assert_eq!(b.seen.lock().unwrap().len(), 1);
    assert!(matches!(&b.seen.lock().unwrap()[0], Message::DecodeBatch { input: Input::Hidden { rows: 3, .. }, .. }));
}

#[test]
fn a_decode_batch_with_mismatched_rows_is_rejected() {
    let bad = Message::DecodeBatch { slots: vec![0, 1], positions: vec![1, 2], input: Input::Tokens(vec![9]) };
    let mut frame = bad.encode();
    frame.truncate(frame.len() - 4);
    assert!(Message::decode(&frame).is_err());
}

#[test]
fn a_chain_with_a_gap_or_an_unreachable_stage_fails_clearly() {
    let a = Fake::new(0, 10, false);
    let b = Fake::new(12, 20, true);
    let sa = StageServer::start("127.0.0.1:0", KEY, a).unwrap();
    let sb = StageServer::start("127.0.0.1:0", KEY, b).unwrap();
    let err = Chain::connect(&[sa.addr().to_string(), sb.addr().to_string()], KEY, T).err().unwrap();
    assert!(format!("{err:#}").contains("starts at layer 12, expected 10"), "{err:#}");
    let dead = {
        let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        l.local_addr().unwrap().to_string()
    };
    let err = Chain::connect(&[sa.addr().to_string(), dead.clone()], KEY, Duration::from_millis(500)).err().unwrap();
    assert!(format!("{err:#}").contains(&dead), "{err:#}");
}

#[test]
fn the_wrong_key_is_refused_and_the_server_keeps_serving() {
    let server = StageServer::start("127.0.0.1:0", KEY, Fake::new(0, 1, true)).unwrap();
    let addr = server.addr().to_string();
    assert!(StageClient::connect(&addr, b"wrong", T).is_err());
    let mut raw = TcpStream::connect(&addr).unwrap();
    raw.set_read_timeout(Some(T)).unwrap();
    let mut nonce = [0u8; 32];
    raw.read_exact(&mut nonce).unwrap();
    raw.write_all(&[0u8; 32]).unwrap();
    assert!(matches!(read_frame(&mut raw), Ok(Message::Error { .. }) | Err(_)));
    assert!(StageClient::connect(&addr, KEY, T).is_ok());
}

#[test]
fn a_replayed_answer_to_an_old_nonce_does_not_authenticate() {
    let server = StageServer::start("127.0.0.1:0", KEY, Fake::new(0, 1, true)).unwrap();
    let addr = server.addr().to_string();
    let mut first = TcpStream::connect(&addr).unwrap();
    first.set_read_timeout(Some(T)).unwrap();
    let mut nonce = [0u8; 32];
    first.read_exact(&mut nonce).unwrap();
    let answer = hmac_sha256(KEY, &nonce);
    let mut second = TcpStream::connect(&addr).unwrap();
    second.set_read_timeout(Some(T)).unwrap();
    let mut nonce2 = [0u8; 32];
    second.read_exact(&mut nonce2).unwrap();
    assert_ne!(nonce, nonce2);
    second.write_all(&answer).unwrap();
    assert!(matches!(read_frame(&mut second), Ok(Message::Error { .. }) | Err(_)));
}

#[test]
fn a_version_mismatch_is_refused_with_the_versions_named() {
    let server = StageServer::start("127.0.0.1:0", KEY, Fake::new(0, 1, true)).unwrap();
    let mut raw = TcpStream::connect(server.addr()).unwrap();
    raw.set_read_timeout(Some(T)).unwrap();
    let mut nonce = [0u8; 32];
    raw.read_exact(&mut nonce).unwrap();
    raw.write_all(&hmac_sha256(KEY, &nonce)).unwrap();
    write_frame(&mut raw, &Message::Hello { version: 99 }).unwrap();
    match read_frame(&mut raw).unwrap() {
        Message::Error { message } => assert!(message.contains("99"), "{message}"),
        other => panic!("{other:?}"),
    }
}

#[test]
fn a_stage_that_goes_away_fails_the_next_call_instead_of_hanging() {
    let server = StageServer::start("127.0.0.1:0", KEY, Fake::new(0, 1, true)).unwrap();
    let mut client = StageClient::connect(&server.addr().to_string(), KEY, Duration::from_secs(1)).unwrap();
    server.shutdown();
    std::thread::sleep(Duration::from_millis(500));
    assert!(client.call(&Message::Free { slot: 0 }).is_err());
}

#[test]
fn an_empty_key_is_refused_at_start() {
    assert!(StageServer::start("127.0.0.1:0", b"", Fake::new(0, 1, true)).is_err());
}
