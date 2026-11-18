use chacha20poly1305::aead::{Aead, KeyInit, Payload};
use chacha20poly1305::{ChaCha20Poly1305, Nonce};
use hkdf::Hkdf;
use sha2::Sha256;
use spake2::{Ed25519Group, Identity, Password, Spake2};
use anyhow::{Context, Result, anyhow, ensure};
use std::path::Path;
use std::time::{Duration, Instant};

const ALPHABET: &[u8; 32] = b"ABCDEFGHJKLMNPQRSTUVWXYZ23456789";
pub const CODE_CHARS: usize = 8;
pub const MAX_ATTEMPTS: u32 = 3;
pub const CODE_LIFETIME: Duration = Duration::from_secs(300);
pub const KEY_BYTES: usize = 32;

pub fn random(n: usize) -> Result<Vec<u8>> {
    let mut out = vec![0u8; n];
    getrandom::fill(&mut out).map_err(|e| anyhow::anyhow!("system random source: {e}"))?;
    Ok(out)
}

pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

pub fn unhex(text: &str) -> Result<Vec<u8>> {
    ensure!(text.len().is_multiple_of(2), "odd hex length");
    (0..text.len()).step_by(2).map(|i| u8::from_str_radix(&text[i..i + 2], 16).context("bad hex")).collect()
}

pub fn generate_key() -> Result<String> {
    Ok(hex(&random(KEY_BYTES)?))
}

pub fn write_key(path: &Path, key: &str) -> Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(path, format!("{key}\n")).with_context(|| path.display().to_string())?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))?;
    }
    Ok(())
}

pub fn read_key(path: &Path) -> Result<String> {
    let key = std::fs::read_to_string(path).with_context(|| path.display().to_string())?.trim().to_string();
    ensure!(!key.is_empty(), "the key file {} is empty", path.display());
    Ok(key)
}

pub fn ensure_key(path: &Path) -> Result<String> {
    match read_key(path) {
        Ok(k) => Ok(k),
        Err(_) => {
            let key = generate_key()?;
            write_key(path, &key)?;
            Ok(key)
        }
    }
}

pub fn normalize_code(text: &str) -> String {
    text.chars().filter(|c| c.is_ascii_alphanumeric()).map(|c| c.to_ascii_uppercase()).collect()
}

pub fn display_code(code: &str) -> String {
    let n = code.len() / 2;
    format!("{}-{}", &code[..n], &code[n..])
}

const CLIENT_ID: &[u8] = b"inference-bench-client";
const HOST_ID: &[u8] = b"inference-bench-host";
const NONCE_LEN: usize = 12;

pub fn fresh_nonce() -> Result<Vec<u8>> {
    random(16)
}

pub struct Offer {
    code: String,
    expires: Instant,
    attempts: u32,
}

impl Offer {
    pub fn new(_peer: &str) -> Result<Offer> {
        let code = random(CODE_CHARS)?.iter().map(|b| ALPHABET[(*b & 31) as usize] as char).collect();
        Ok(Offer { code, expires: Instant::now() + CODE_LIFETIME, attempts: 0 })
    }

    pub fn with_code(code: &str, lifetime: Duration) -> Offer {
        Offer { code: normalize_code(code), expires: Instant::now() + lifetime, attempts: 0 }
    }

    pub fn code(&self) -> String {
        display_code(&self.code)
    }

    pub fn live(&self) -> bool {
        Instant::now() < self.expires && self.attempts < MAX_ATTEMPTS
    }

    pub fn seconds_left(&self) -> u64 {
        self.expires.saturating_duration_since(Instant::now()).as_secs()
    }

    pub fn start(&mut self, msg_a: &[u8]) -> Result<(Vec<u8>, Vec<u8>, HostSession)> {
        ensure!(self.live(), "the pairing code is no longer valid");
        self.attempts += 1;
        let (state, msg_b) = Spake2::<Ed25519Group>::start_b(&Password::new(self.code.as_bytes()), &Identity::new(CLIENT_ID), &Identity::new(HOST_ID));
        let shared = state.finish(msg_a).map_err(|e| anyhow!("pairing message rejected: {e:?}"))?;
        let keys = Keys::derive(&shared, msg_a, &msg_b);
        let confirm_b = keys.host_confirm.to_vec();
        Ok((msg_b, confirm_b, HostSession { keys }))
    }

    pub fn consume(&mut self) {
        self.attempts = MAX_ATTEMPTS;
    }
}

struct Keys {
    host_confirm: [u8; 32],
    client_confirm: [u8; 32],
    wrap: [u8; 32],
    transcript: Vec<u8>,
}

impl Keys {
    fn derive(shared: &[u8], msg_a: &[u8], msg_b: &[u8]) -> Keys {
        let mut transcript = msg_a.to_vec();
        transcript.extend_from_slice(msg_b);
        let hk = Hkdf::<Sha256>::new(Some(&transcript), shared);
        let mut okm = [0u8; 96];
        hk.expand(b"inference-bench pairing v1", &mut okm).expect("96 bytes is a valid hkdf length");
        let mut host_confirm = [0u8; 32];
        let mut client_confirm = [0u8; 32];
        let mut wrap = [0u8; 32];
        host_confirm.copy_from_slice(&okm[..32]);
        client_confirm.copy_from_slice(&okm[32..64]);
        wrap.copy_from_slice(&okm[64..]);
        Keys { host_confirm, client_confirm, wrap, transcript }
    }
}

pub struct HostSession {
    keys: Keys,
}

impl HostSession {
    pub fn open(&self, confirm: &[u8], nonce: &[u8], ciphertext: &[u8]) -> Result<String> {
        ensure!(equal(&self.keys.client_confirm, confirm), "wrong pairing code");
        ensure!(nonce.len() == NONCE_LEN, "bad nonce length");
        let cipher = ChaCha20Poly1305::new((&self.keys.wrap).into());
        let raw = cipher
            .decrypt(Nonce::from_slice(nonce), Payload { msg: ciphertext, aad: &self.keys.transcript })
            .map_err(|_| anyhow!("the sealed key did not decrypt"))?;
        ensure!(raw.len() == KEY_BYTES, "sealed key has {} bytes", raw.len());
        Ok(hex(&raw))
    }
}

pub struct ClientSession {
    state: Spake2<Ed25519Group>,
    msg_a: Vec<u8>,
}

#[derive(Debug)]
pub struct Sealed {
    pub confirm: Vec<u8>,
    pub nonce: Vec<u8>,
    pub ciphertext: Vec<u8>,
}

impl ClientSession {
    pub fn start(code: &str) -> (ClientSession, Vec<u8>) {
        let (state, msg_a) = Spake2::<Ed25519Group>::start_a(&Password::new(normalize_code(code).as_bytes()), &Identity::new(CLIENT_ID), &Identity::new(HOST_ID));
        (ClientSession { state, msg_a: msg_a.clone() }, msg_a)
    }

    pub fn finish(self, msg_b: &[u8], host_confirm: &[u8], key: &str) -> Result<Sealed> {
        let raw = unhex(key)?;
        ensure!(raw.len() == KEY_BYTES, "the cluster key must be a generated key");
        let shared = self.state.finish(msg_b).map_err(|e| anyhow!("pairing message rejected: {e:?}"))?;
        let keys = Keys::derive(&shared, &self.msg_a, msg_b);
        ensure!(equal(&keys.host_confirm, host_confirm), "the peer did not prove it knows the pairing code");
        let nonce = random(NONCE_LEN)?;
        let cipher = ChaCha20Poly1305::new((&keys.wrap).into());
        let ciphertext = cipher
            .encrypt(Nonce::from_slice(&nonce), Payload { msg: &raw, aad: &keys.transcript })
            .map_err(|_| anyhow!("could not seal the key"))?;
        Ok(Sealed { confirm: keys.client_confirm.to_vec(), nonce, ciphertext })
    }
}

fn equal(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}


#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct Transcript {
        msg_a: Vec<u8>,
        msg_b: Vec<u8>,
        confirm_b: Vec<u8>,
        sealed: Sealed,
    }

    fn exchange(offer: &mut Offer, typed: &str, key: &str) -> Result<(String, Transcript)> {
        let (client, msg_a) = ClientSession::start(typed);
        let (msg_b, confirm_b, session) = offer.start(&msg_a)?;
        let sealed = client.finish(&msg_b, &confirm_b, key)?;
        let got = session.open(&sealed.confirm, &sealed.nonce, &sealed.ciphertext)?;
        offer.consume();
        Ok((got, Transcript { msg_a, msg_b, confirm_b, sealed }))
    }

    #[test]
    fn the_right_code_delivers_the_key_and_the_wire_never_carries_it() {
        let key = generate_key().unwrap();
        let mut offer = Offer::new("peer").unwrap();
        let code = offer.code();
        let (got, t) = exchange(&mut offer, &code.to_lowercase(), &key).unwrap();
        assert_eq!(got, key);
        let raw = unhex(&key).unwrap();
        for part in [&t.msg_a, &t.msg_b, &t.confirm_b, &t.sealed.ciphertext, &t.sealed.confirm] {
            assert!(!part.windows(raw.len()).any(|w| w == raw.as_slice()));
        }
    }

    #[test]
    fn a_recorded_transcript_cannot_be_opened_by_guessing_codes_offline_even_the_right_one() {
        let key = generate_key().unwrap();
        let mut offer = Offer::with_code("ABCD-2345", Duration::from_secs(60));
        let (_, t) = exchange(&mut offer, "ABCD-2345", &key).unwrap();
        for guess in ["ABCD-2345", "ABCD-2346", "AAAA-AAAA", "ZZZZ-9999"] {
            let (attacker, _) = ClientSession::start(guess);
            let attempt = attacker.finish(&t.msg_b, &t.confirm_b, &key);
            assert!(attempt.is_err(), "guess {guess} validated against a recorded transcript");
        }
        let (second, _) = ClientSession::start("ABCD-2345");
        assert!(second.finish(&t.msg_b, &t.confirm_b, &key).is_err());
    }

    #[test]
    fn a_wrong_code_fails_and_three_handshakes_kill_the_offer_even_for_the_right_code() {
        let key = generate_key().unwrap();
        let mut offer = Offer::with_code("ABCD-2345", Duration::from_secs(60));
        for _ in 0..MAX_ATTEMPTS {
            assert!(exchange(&mut offer, "AAAA-AAAA", &key).is_err());
        }
        assert!(!offer.live());
        assert!(exchange(&mut offer, "ABCD-2345", &key).unwrap_err().to_string().contains("no longer valid"));
    }

    #[test]
    fn a_code_works_once() {
        let key = generate_key().unwrap();
        let mut offer = Offer::with_code("ABCD2345", Duration::from_secs(60));
        assert_eq!(exchange(&mut offer, "abcd-2345", &key).unwrap().0, key);
        assert!(exchange(&mut offer, "ABCD2345", &key).is_err());
    }

    #[test]
    fn an_expired_code_is_refused() {
        let key = generate_key().unwrap();
        let mut offer = Offer::with_code("ABCD2345", Duration::ZERO);
        assert!(exchange(&mut offer, "ABCD2345", &key).is_err());
    }

    #[test]
    fn a_host_that_does_not_know_the_code_is_detected_by_the_client() {
        let key = generate_key().unwrap();
        let mut rogue = Offer::with_code("WXYZ6789", Duration::from_secs(60));
        let (client, msg_a) = ClientSession::start("ABCD2345");
        let (msg_b, confirm_b, _) = rogue.start(&msg_a).unwrap();
        assert!(client.finish(&msg_b, &confirm_b, &key).is_err());
    }

    #[test]
    fn a_tampered_ciphertext_is_rejected() {
        let key = generate_key().unwrap();
        let mut offer = Offer::with_code("ABCD2345", Duration::from_secs(60));
        let (client, msg_a) = ClientSession::start("ABCD2345");
        let (msg_b, confirm_b, session) = offer.start(&msg_a).unwrap();
        let mut sealed = client.finish(&msg_b, &confirm_b, &key).unwrap();
        sealed.ciphertext[0] ^= 1;
        assert!(session.open(&sealed.confirm, &sealed.nonce, &sealed.ciphertext).is_err());
    }

    #[test]
    fn randomness_comes_from_the_system_source_not_a_unix_device_file() {
        let (a, b) = (random(32).unwrap(), random(32).unwrap());
        assert_eq!((a.len(), b.len()), (32, 32));
        assert_ne!(a, b);
        assert!(a.iter().any(|x| *x != 0));
        let source = include_str!("pairing.rs").to_string() + include_str!("wire.rs");
        assert!(!source.contains(&format!("/dev/{}", "urandom")));
    }

    #[test]
    fn the_key_file_is_created_once_private_and_reread() {
        let dir = std::env::temp_dir().join(format!("pair-{}", std::process::id()));
        let path = dir.join("cluster.key");
        let first = ensure_key(&path).unwrap();
        assert_eq!(first.len(), KEY_BYTES * 2);
        assert_eq!(ensure_key(&path).unwrap(), first);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(std::fs::metadata(&path).unwrap().permissions().mode() & 0o777, 0o600);
        }
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn codes_are_eight_unambiguous_characters() {
        let offer = Offer::new("x").unwrap();
        let shown = offer.code();
        assert_eq!(shown.len(), 9);
        assert!(!shown.contains(['0', 'O', '1', 'I']));
        assert_eq!(normalize_code(&shown).len(), CODE_CHARS);
    }
}
