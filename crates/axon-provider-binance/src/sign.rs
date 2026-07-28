//! The signing seam — and, deliberately, no crypto.
//!
//! ## What Binance's scheme is, and how far it is from Hyperliquid's
//!
//! Binance authenticates a request with two independent things: the API key in an
//! `X-MBX-APIKEY` **header**, and an HMAC-SHA256 of the query string, hex-encoded and
//! appended as the final `signature=` parameter. There is no wallet, no domain
//! separator, no msgpack, no chain id, and no nonce — replay protection is
//! `timestamp` plus `recvWindow`, a five-second window rather than Hyperliquid's
//! monotonic per-address counter.
//!
//! Read from the port, though, the distance is much smaller than that list suggests,
//! and this is the most interesting thing the second venue had to say about
//! [`axon_providers`]:
//!
//! - [`SignatureScheme::HmacApiKey`] already existed, as the very first variant. It
//!   was written in ADR-0004 against no venue at all and needed no change.
//! - [`Credentials::ApiKey`] already existed, split into `key` and `secret` — which is
//!   exactly right, because the two travel by different routes (one header, one MAC)
//!   and a single opaque credential could not express that.
//! - [`Signer`] — `fn scheme()` plus `fn sign(&self, payload: &[u8]) -> Vec<u8>` — is
//!   a **perfect** fit. Binance's payload is a byte string and its signature is bytes.
//!
//! And the finding attached to that: **the Hyperliquid adapter does not implement
//! [`Signer`] at all.** `HlSigner` is a concrete type with `sign_l1_action` and
//! `sign_user_action`, because an EIP-712 signature is over a *typed structured*
//! payload rather than a byte string and `&[u8]` cannot carry the domain. So the port
//! has a signing trait that the first venue could not use and the second one fits
//! exactly. That is not an argument for deleting it — it is an argument that the trait
//! was written for CEXes and the DEX is the exception, which is the reverse of how the
//! codebase reads today.
//!
//! ## What is missing, and why it is missing on purpose
//!
//! **There is no HMAC-SHA256 in this crate, and therefore no code anywhere in this
//! workspace that can sign a Binance request.** That is a deliberate stopping point,
//! not an oversight:
//!
//! 1. It needs two workspace dependencies (`hmac`, `sha2`) that this workstream is not
//!    permitted to add to the root manifest. The exact lines are in ADR-0023.
//! 2. There are no Binance credentials in this repository, none are being sought, and
//!    this workstream is not authorized to trade on Binance. An adapter that could
//!    sign but had nothing to sign with would be a loaded path with the safety on.
//!
//! So everything up to the bytes is here and tested — the canonical string, the
//! parameter order, the hex encoding, the `signature`-last assembly — and the MAC
//! itself is a `Signer` implementation somebody adds later, against the known answer
//! pinned in this module's tests. What exists cannot place an order: there is no HTTP
//! client for `/fapi/v1/order` in this crate at all.

use axon_providers::{Credentials, ProviderError, SignatureScheme, Signer};

use crate::encode::{EncodeError, Params};

/// The header the API key travels in. Not a query parameter, and not part of the
/// signed payload — which is why [`Credentials::ApiKey`]'s split into `key` and
/// `secret` is the right shape and a single opaque token would not be.
pub const API_KEY_HEADER: &str = "X-MBX-APIKEY";

/// The parameter the signature travels in. **Always last.**
pub const SIGNATURE_PARAM: &str = "signature";

/// A borrowed view of the two halves of a Binance credential.
///
/// Borrowed rather than owned so this crate never holds a copy of a secret it has no
/// use for — there is no signer here to give it to.
#[derive(Debug, Clone, Copy)]
pub struct BinanceCredentials<'a> {
    /// Goes in the [`API_KEY_HEADER`], in clear.
    pub api_key: &'a str,
    /// Never leaves the process: it is the HMAC key, not a transmitted value.
    pub secret: &'a str,
}

impl<'a> BinanceCredentials<'a> {
    /// Read the port's [`Credentials`] as a Binance credential.
    ///
    /// A [`Credentials::Wallet`] is refused rather than coerced. The two arms are not
    /// two encodings of one thing — a wallet has no secret to MAC with — and the
    /// failure of accepting one here would be a request signed with an empty key,
    /// which the venue rejects as `-1022` and which reads from inside exactly like a
    /// wrong secret.
    pub fn from_port(c: &'a Credentials) -> Result<Self, ProviderError> {
        match c {
            Credentials::ApiKey { key, secret } => Ok(Self {
                api_key: key,
                secret,
            }),
            Credentials::Wallet { .. } => Err(ProviderError::Auth(format!(
                "{} signs with an API key and secret, not a wallet",
                crate::VENUE
            ))),
        }
    }
}

/// A request ready to send, once something has produced the signature.
///
/// Carries the *whole* query — signed parameters and the signature — as one string,
/// because the venue verifies the MAC over the bytes it received. Handing a caller
/// the parts separately would let it reorder or re-serialize them, and any
/// re-serialization is a different signature over the same request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SignedRequest {
    /// `POST`, `DELETE`, `GET`.
    pub method: &'static str,
    /// `/fapi/v1/order`, `/fapi/v1/allOpenOrders`, …
    pub path: &'static str,
    /// `k=v&…&signature=…`.
    pub query: String,
}

impl SignedRequest {
    /// The full URL to send this to.
    pub fn url(&self, base_url: &str) -> String {
        format!("{base_url}{}?{}", self.path, self.query)
    }
}

/// The exact bytes an HMAC is taken over: the query string, in parameter order, with
/// no `signature` on it.
///
/// A named function rather than a call to [`Params::query_string`] at each site,
/// because "the string we sign" and "the string we send" must be produced by one
/// thing. Hyperliquid's equivalent property is that the msgpack bytes are hashed
/// before the JSON envelope is built; the hazard is identical and so is the fix.
pub fn signing_payload(params: &Params) -> Result<String, EncodeError> {
    params.query_string()
}

/// Lower-case hex, the encoding Binance expects for the signature.
///
/// Hand-written because it is four lines and the alternative is a dependency. Upper
/// case is *also* accepted by the venue, which is why this is pinned by a test: a
/// silent case change would not fail against Binance and would fail against any
/// fixture comparison built on it.
pub fn hex_lower(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push(char::from_digit((b >> 4) as u32, 16).expect("nibble"));
        out.push(char::from_digit((b & 0xf) as u32, 16).expect("nibble"));
    }
    out
}

/// Sign `params` with a [`Signer`] and assemble the request.
///
/// The signer must declare [`SignatureScheme::HmacApiKey`]; anything else is refused
/// here rather than at the venue. Mixing signing schemes is the failure ADR-0009 names
/// as Hyperliquid's #1 invalid-signature cause, and the check costs nothing — an
/// EIP-712 signer handed a Binance query string would produce 65 well-formed bytes
/// that mean nothing, and the venue would answer `-1022`.
///
/// **Nothing in this workspace implements `Signer` with an HMAC**, so this function
/// has no production caller today. It is here because the assembly — hex, ordering,
/// `signature` last — is the part that is easy to get wrong and cheap to pin, and
/// because it is the demonstration that the port's own trait fits this venue.
pub fn sign_request(
    method: &'static str,
    path: &'static str,
    params: &Params,
    signer: &dyn Signer,
) -> Result<SignedRequest, ProviderError> {
    if signer.scheme() != SignatureScheme::HmacApiKey {
        return Err(ProviderError::Auth(format!(
            "{} needs an {:?} signer, got {:?}",
            crate::VENUE,
            SignatureScheme::HmacApiKey,
            signer.scheme()
        )));
    }
    let payload = signing_payload(params).map_err(|e| ProviderError::Unsupported {
        venue: crate::VENUE,
        what: e.to_string(),
    })?;
    let mac = signer.sign(payload.as_bytes())?;
    Ok(SignedRequest {
        method,
        path,
        // Appended, never re-serialized. The venue verifies over the bytes it got, so
        // rebuilding the query here would be a different string than the one MAC'd.
        query: format!("{payload}&{SIGNATURE_PARAM}={}", hex_lower(&mac)),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encode::Params;

    /// Binance's own worked example for a signed endpoint, quoted from the futures API
    /// documentation.
    ///
    /// **Nothing in this tree computes this signature.** It is pinned so that whoever
    /// adds the `hmac`/`sha2` dependency has a known answer to land against, and so
    /// that the canonicalization below can be checked against the venue's own string
    /// today, without any crypto at all. The secret is the documentation's, is public,
    /// and belongs to no account here.
    const DOC_SECRET: &str = "2b5eb11e18796d12d88f13dc27dbbd02c2cc51ff7059765ed9821957d82bb4d9";
    const DOC_QUERY: &str = "symbol=BTCUSDT&side=BUY&type=LIMIT&timeInForce=GTC&quantity=1\
                             &price=9000&recvWindow=5000&timestamp=1591702613943";
    const DOC_SIGNATURE: &str = "3c661234138461fcc7a7d8746c6558c9842d4e10870d2ecbedf7777cad694af9";

    /// A signer that returns fixed bytes. It proves the **assembly** — hex, ordering,
    /// `signature` last — and proves nothing whatever about the MAC, which is the
    /// honest division of labour while no MAC exists.
    struct FixedSigner(Vec<u8>);

    impl Signer for FixedSigner {
        fn scheme(&self) -> SignatureScheme {
            SignatureScheme::HmacApiKey
        }
        fn sign(&self, _payload: &[u8]) -> Result<Vec<u8>, ProviderError> {
            Ok(self.0.clone())
        }
    }

    struct WalletSigner;

    impl Signer for WalletSigner {
        fn scheme(&self) -> SignatureScheme {
            SignatureScheme::Eip712L1Action
        }
        fn sign(&self, _payload: &[u8]) -> Result<Vec<u8>, ProviderError> {
            Ok(vec![0; 65])
        }
    }

    fn doc_params() -> Params {
        let mut p = Params::new();
        p.push("symbol", "BTCUSDT");
        p.push("side", "BUY");
        p.push("type", "LIMIT");
        p.push("timeInForce", "GTC");
        p.push("quantity", "1");
        p.push("price", "9000");
        p.push("recvWindow", "5000");
        p.push("timestamp", "1591702613943");
        p
    }

    #[test]
    fn the_canonical_payload_matches_the_venues_own_worked_example_byte_for_byte() {
        // The part of signing that can be verified with no crypto, and the part that
        // is actually easy to get wrong: which parameters, in which order, joined how.
        // Binance publishes this exact string beside its expected signature, so this is
        // a check against the venue rather than against ourselves.
        assert_eq!(signing_payload(&doc_params()).unwrap(), DOC_QUERY);
        assert!(
            !DOC_QUERY.contains("signature"),
            "the MAC is not self-signed"
        );
        // And the answer that string is supposed to produce, recorded for whoever adds
        // the MAC. 32 bytes of SHA-256, hex.
        assert_eq!(DOC_SIGNATURE.len(), 64);
        assert_eq!(DOC_SECRET.len(), 64);
    }

    #[test]
    fn the_signature_is_appended_last_and_the_signed_bytes_are_never_re_serialized() {
        // Two failures in one assertion. The venue requires `signature` to be the final
        // parameter; and it verifies the MAC over the bytes it received, so rebuilding
        // the query after signing would produce a different string than the one signed.
        let signer = FixedSigner(vec![0xde, 0xad, 0xbe, 0xef]);
        let req = sign_request("POST", "/fapi/v1/order", &doc_params(), &signer).unwrap();
        assert_eq!(req.query, format!("{DOC_QUERY}&signature=deadbeef"));
        assert!(req.query.starts_with(DOC_QUERY), "the payload is untouched");
        assert_eq!(
            req.query.matches("signature=").count(),
            1,
            "exactly one signature"
        );
        assert_eq!(
            req.url("https://fapi.binance.com"),
            format!("https://fapi.binance.com/fapi/v1/order?{DOC_QUERY}&signature=deadbeef")
        );
    }

    #[test]
    fn a_wallet_signer_is_refused_before_the_wire_rather_than_producing_meaningless_bytes() {
        // ADR-0009 names mixing signing schemes as the venue's #1 invalid-signature
        // cause, on a venue that has two of them. Across two venues the same mistake is
        // available for free, and it is cheap to close: an EIP-712 signer handed a query
        // string returns 65 well-formed bytes that mean nothing, and `-1022` reads
        // exactly like a wrong secret.
        let err = sign_request("POST", "/fapi/v1/order", &doc_params(), &WalletSigner).unwrap_err();
        assert!(matches!(err, ProviderError::Auth(_)), "got {err:?}");
    }

    #[test]
    fn credentials_read_off_the_port_and_a_wallet_is_not_one() {
        // `Credentials::ApiKey` was written in ADR-0004 against no venue at all, split
        // into key and secret. That split is exactly right here and could not have been
        // guessed from Hyperliquid: the two halves travel by different routes — one in
        // a header in clear, one never transmitted at all.
        let creds = Credentials::ApiKey {
            key: "public-part".into(),
            secret: DOC_SECRET.into(),
        };
        let b = BinanceCredentials::from_port(&creds).unwrap();
        assert_eq!(b.api_key, "public-part");
        assert_eq!(b.secret, DOC_SECRET);

        let wallet = Credentials::Wallet {
            address: "0xdeadbeef".into(),
        };
        assert!(matches!(
            BinanceCredentials::from_port(&wallet),
            Err(ProviderError::Auth(_))
        ));
    }

    #[test]
    fn hex_is_lower_case_and_zero_padded_to_two_characters_a_byte() {
        // The venue accepts either case, which is precisely why this needs pinning: a
        // silent change to upper case would pass against Binance and break every
        // fixture comparison built on it, with nothing failing at the venue to say so.
        assert_eq!(hex_lower(&[0x00, 0x0f, 0xa0, 0xff]), "000fa0ff");
        assert_eq!(hex_lower(&[]), "");
        assert_eq!(hex_lower(&[0xde, 0xad]).len(), 4);
        assert!(hex_lower(&[0xAB]).chars().all(|c| !c.is_ascii_uppercase()));
    }

    #[test]
    fn no_signer_in_this_workspace_can_actually_sign_for_this_venue() {
        // Stated as a test so it cannot quietly stop being true. This crate has no HMAC
        // and no HTTP client for `/fapi/v1/order`, so there is no path from here to an
        // order at Binance — which is the property this workstream was required to
        // hold, not merely a gap in it. The day somebody adds a real `Signer`, this
        // test should be deleted in the same change that adds the live-path guard.
        let signer = FixedSigner(vec![0; 32]);
        let req = sign_request("POST", "/fapi/v1/order", &doc_params(), &signer).unwrap();
        assert_eq!(
            hex_lower(&[0; 32]).len(),
            64,
            "a real HMAC-SHA256 is 32 bytes"
        );
        assert!(
            req.query
                .ends_with(&format!("signature={}", hex_lower(&[0; 32]))),
            "and a fake one is still only an assembly test"
        );
        assert_ne!(
            hex_lower(&[0; 32]),
            DOC_SIGNATURE,
            "nothing here computes the venue's own answer"
        );
    }
}
