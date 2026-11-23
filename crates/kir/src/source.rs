use std::fmt;
use std::ops::Deref;

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct KirSource(String);

impl KirSource {
    pub(crate) fn emitted(text: String) -> Self {
        KirSource(text)
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn into_string(self) -> String {
        self.0
    }

    pub fn join<'a>(parts: impl IntoIterator<Item = &'a KirSource>, sep: &str) -> KirSource {
        let mut out = String::new();
        for (i, p) in parts.into_iter().enumerate() {
            if i > 0 {
                out.push_str(sep);
            }
            out.push_str(&p.0);
        }
        KirSource(out)
    }

    pub fn append(&mut self, sep: &str, other: &KirSource) {
        self.0.push_str(sep);
        self.0.push_str(&other.0);
    }

    #[cfg(feature = "test-text")]
    pub fn test_text(text: impl Into<String>) -> Self {
        KirSource(text.into())
    }

    #[cfg(feature = "test-text")]
    pub fn edit_text(&mut self, f: impl FnOnce(&mut String)) {
        f(&mut self.0)
    }
}

impl Deref for KirSource {
    type Target = str;
    fn deref(&self) -> &str {
        &self.0
    }
}

impl AsRef<str> for KirSource {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl AsRef<[u8]> for KirSource {
    fn as_ref(&self) -> &[u8] {
        self.0.as_bytes()
    }
}

impl fmt::Display for KirSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl PartialEq<str> for KirSource {
    fn eq(&self, other: &str) -> bool {
        self.0 == other
    }
}

impl PartialEq<&str> for KirSource {
    fn eq(&self, other: &&str) -> bool {
        self.0 == *other
    }
}
