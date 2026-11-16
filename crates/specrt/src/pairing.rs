use crate::wire::hmac_sha256;
use anyhow::{Context, Result, bail, ensure};
use std::io::Read;
use std::path::Path;
use std::time::{Duration, Instant};

const ALPHABET: &[u8; 32] = b"ABCDEFGHJKLMNPQRSTUVWXYZ23456789";
pub const CODE_CHARS: usize = 8;
pub const MAX_ATTEMPTS: u32 = 3;
pub const CODE_LIFETIME: Duration = Duration::from_secs(300);
pub const KEY_BYTES: usize = 32;

fn random(n: usize) -> Result<Vec<u8>> {
    let mut out = vec![0u8; n];
    std::fs::File::open("/dev/urandom").and_then(|mut f| f.read_exact(&mut out)).context("read /dev/urandom")?;
    Ok(out)
}

pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

pub fn unhex(text: &str) -> Result<Vec<u8>> {
    ensure!(text.len() % 2 == 0, "odd hex length");
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

    pub fn accept(&mut self, nonce_client: &[u8], nonce_server: &[u8], client_proof: &[u8], sealed_key: &[u8]) -> Result<Accepted> {
        ensure!(self.live(), "the pairing code is no longer valid");
        let expected = proof(self.code.as_bytes(), b"client", nonce_client, nonce_server);
        if !equal(&expected, client_proof) {
            self.attempts += 1;
            bail!("wrong pairing code");
        }
        ensure!(sealed_key.len() == KEY_BYTES, "sealed key has {} bytes", sealed_key.len());
        let mask = proof(self.code.as_bytes(), b"key", nonce_client, nonce_server);
        let raw: Vec<u8> = sealed_key.iter().zip(mask).map(|(a, b)| a ^ b).collect();
        self.attempts = MAX_ATTEMPTS;
        Ok(Accepted { key: hex(&raw), server_proof: proof(self.code.as_bytes(), b"server", nonce_client, nonce_server).to_vec() })
    }
}

pub struct Accepted {
    pub key: String,
    pub server_proof: Vec<u8>,
}

pub struct Sealed {
    pub proof: Vec<u8>,
    pub sealed_key: Vec<u8>,
}

pub fn seal(code: &str, key: &str, nonce_client: &[u8], nonce_server: &[u8]) -> Result<Sealed> {
    let code = normalize_code(code);
    let raw = unhex(key)?;
    ensure!(raw.len() == KEY_BYTES, "the cluster key must be a generated key");
    let mask = proof(code.as_bytes(), b"key", nonce_client, nonce_server);
    Ok(Sealed {
        proof: proof(code.as_bytes(), b"client", nonce_client, nonce_server).to_vec(),
        sealed_key: raw.iter().zip(mask).map(|(a, b)| a ^ b).collect(),
    })
}

pub fn verify_server(code: &str, nonce_client: &[u8], nonce_server: &[u8], server_proof: &[u8]) -> Result<()> {
    let expected = proof(normalize_code(code).as_bytes(), b"server", nonce_client, nonce_server);
    ensure!(equal(&expected, server_proof), "the peer did not prove it knows the pairing code");
    Ok(())
}

fn equal(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

fn proof(code: &[u8], role: &[u8], nonce_client: &[u8], nonce_server: &[u8]) -> [u8; 32] {
    let mut data = role.to_vec();
    data.extend_from_slice(nonce_client);
    data.extend_from_slice(nonce_server);
    hmac_sha256(code, &data)
}

pub fn fresh_nonce() -> Result<Vec<u8>> {
    random(16)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn exchange(offer: &mut Offer, typed: &str, key: &str) -> Result<String> {
        let (nc, ns) = (fresh_nonce()?, fresh_nonce()?);
        let sealed = seal(typed, key, &nc, &ns)?;
        let accepted = offer.accept(&nc, &ns, &sealed.proof, &sealed.sealed_key)?;
        verify_server(typed, &nc, &ns, &accepted.server_proof)?;
        Ok(accepted.key)
    }

    #[test]
    fn the_right_code_delivers_the_key_without_it_appearing_in_the_exchange() {
        let key = generate_key().unwrap();
        let mut offer = Offer::new("peer").unwrap();
        let code = offer.code();
        let (nc, ns) = (fresh_nonce().unwrap(), fresh_nonce().unwrap());
        let sealed = seal(&code.to_lowercase(), &key, &nc, &ns).unwrap();
        assert_ne!(sealed.sealed_key, unhex(&key).unwrap());
        let accepted = offer.accept(&nc, &ns, &sealed.proof, &sealed.sealed_key).unwrap();
        assert_eq!(accepted.key, key);
    }

    #[test]
    fn a_wrong_code_fails_three_times_then_the_offer_is_dead_even_for_the_right_code() {
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
        assert_eq!(exchange(&mut offer, "abcd-2345", &key).unwrap(), key);
        assert!(exchange(&mut offer, "ABCD2345", &key).is_err());
    }

    #[test]
    fn an_expired_code_is_refused() {
        let key = generate_key().unwrap();
        let mut offer = Offer::with_code("ABCD2345", Duration::ZERO);
        assert!(exchange(&mut offer, "ABCD2345", &key).is_err());
    }

    #[test]
    fn a_peer_that_does_not_know_the_code_is_detected() {
        let key = generate_key().unwrap();
        let mut rogue = Offer::with_code("WXYZ6789", Duration::from_secs(60));
        let (nc, ns) = (fresh_nonce().unwrap(), fresh_nonce().unwrap());
        let sealed = seal("WXYZ6789", &key, &nc, &ns).unwrap();
        let forged = rogue.accept(&nc, &ns, &sealed.proof, &sealed.sealed_key).unwrap();
        assert!(verify_server("ABCD2345", &nc, &ns, &forged.server_proof).is_err());
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
