//! `approve_agent` — the one-off ceremony that moves Axon off the master key.
//!
//! Under the agent-wallet model ([ADR-0009])
//! the process that trades holds a key that **cannot withdraw**. Today it does not:
//! `wallet_info` reports the signing address and the account address as the same
//! value, so `AXON_HL_SECRET_KEY` *is* the account and the containment property is
//! only a claim. This tool closes that gap — generate a fresh agent key locally,
//! have the master account sign one `approveAgent`, verify the venue lists it, and
//! print the exact `.env` edit that completes the move.
//!
//! Four properties it is built around, each preventing a specific way this goes
//! wrong:
//!
//! - **Dry run by default.** A bare invocation prints the action it *would* sign
//!   and exits. Approval is destructive in the way that matters: an account holds
//!   one unnamed agent and three named ones, so a careless run spends a slot and
//!   deregisters whoever occupied it.
//! - **The agent key is generated here, never derived from the master key.** A key
//!   recomputable from the account key is not a containment boundary, it is the
//!   master key wearing a hat.
//! - **`approveAgent` is a *user-signed* action** — EIP-712 over the action's own
//!   fields under the `HyperliquidSignTransaction` domain — not an L1
//!   phantom-agent action. Signed under the L1 domain it still produces a
//!   well-formed signature; it just recovers to a different address, which the
//!   venue reports as a bare "invalid signature". That is the documented number-one
//!   cause, so the tool signs and *recovers* locally before it will send anything.
//! - **The secret has exactly one destination.** A `0600` file, or one deliberate
//!   print behind `--print-secret`. Never a log line, and never argv — argv is
//!   world-readable through `ps`, so the tool refuses to start if anything on the
//!   command line looks like a private key.
//!
//! ```text
//! # look, change nothing (the default):
//! bash scripts/with-env.sh cargo run -p axon-provider-hyperliquid --example approve_agent
//!
//! # do it, on testnet:
//! bash scripts/with-env.sh cargo run -p axon-provider-hyperliquid --example approve_agent -- --submit
//! ```
//!
//! The signing key is read from `AXON_HL_MASTER_KEY`, falling back to
//! `AXON_HL_SECRET_KEY` for exactly as long as that variable still holds the
//! account key — which is the situation this tool exists to end.
//!
//! [ADR-0009]: ../../../docs/adr/0009-hyperliquid-signing.md

use std::fmt::Write as _;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use std::{env, fs, io};

use alloy_primitives::{hex, Address, Signature, B256, U256};
use alloy_signer_local::PrivateKeySigner;
use axon_provider_hyperliquid::sign::user_signed::MAX_VALIDITY_MARGIN_MS;
use axon_provider_hyperliquid::sign::user_signed_hash;
use axon_provider_hyperliquid::ws::{MAINNET_INFO, TESTNET_INFO};
use axon_provider_hyperliquid::{
    fetch_extra_agents, AgentName, ApproveAgent, ExchangeClient, ExtraAgent, HlSigner,
    NonceManager, RpcSignature, UserSignedAction, MAX_AGENT_VALIDITY_MS,
};

/// Where the ceremony reads the **master account** key from.
///
/// Separate from `AXON_HL_SECRET_KEY` because after a successful migration that
/// variable holds the *agent* key, which by construction cannot sign an approval.
/// Sharing one variable would make rotation impossible without temporarily putting
/// the master key back where the trading process reads it.
const MASTER_KEY_ENV: &str = "AXON_HL_MASTER_KEY";
const ACCOUNT_ENV: &str = "AXON_HL_ACCOUNT_ADDRESS";
const NETWORK_ENV: &str = "AXON_HL_NETWORK";

const DEFAULT_AGENT_NAME: &str = "axon";
const MS_PER_DAY: u64 = 24 * 60 * 60 * 1_000;

/// How many times the post-submit check re-reads `extraAgents` before giving up.
///
/// The venue acknowledges the write before the read path is guaranteed to show it,
/// and a single miss reported as failure is worse than a slow success: it invites a
/// second approval, and a second approval of the same name deregisters the agent
/// that did land.
const VERIFY_ATTEMPTS: u32 = 5;
const VERIFY_DELAY: Duration = Duration::from_secs(1);

const USAGE: &str = "\
approve_agent — authorize a fresh agent (API) wallet for the master account.

  bash scripts/with-env.sh cargo run -p axon-provider-hyperliquid \\
      --example approve_agent -- [OPTIONS]

Options:
  --name <NAME>       named agent slot (default: \"axon\"). The name is the agent's
                      identity — approving an existing name replaces that agent.
  --unnamed           use the account's single unnamed slot instead. Carries no
                      expiry: the deadline rides inside the name.
  --valid-days <N>    lifetime in days (max 180). Default: the venue's maximum.
  --out <PATH>        where to write the new key (default: secrets/agent-<addr>.key)
  --print-secret      print the key once instead of writing a file
  --submit            actually send it. Without this, nothing leaves the machine.
  --confirm-mainnet <ADDRESS>
                      required to --submit on mainnet; must be the account address
  -h, --help          this text

The master key comes from AXON_HL_MASTER_KEY (falling back to AXON_HL_SECRET_KEY),
the account from AXON_HL_ACCOUNT_ADDRESS, the network from AXON_HL_NETWORK.
Never pass a key on the command line: argv is visible to every user via `ps`.";

// ── network ──────────────────────────────────────────────────────────────────

/// The endpoint pair for one Hyperliquid network, resolved once so that no later
/// step can read `/info` from testnet while posting `/exchange` to mainnet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Network {
    label: &'static str,
    is_mainnet: bool,
    info_url: &'static str,
    exchange_url: &'static str,
}

impl Network {
    const TESTNET: Self = Self {
        label: "testnet",
        is_mainnet: false,
        info_url: TESTNET_INFO,
        exchange_url: ExchangeClient::TESTNET,
    };
    const MAINNET: Self = Self {
        label: "mainnet",
        is_mainnet: true,
        info_url: MAINNET_INFO,
        exchange_url: ExchangeClient::MAINNET,
    };

    fn parse(value: &str) -> Option<Self> {
        match value {
            "testnet" => Some(Self::TESTNET),
            "mainnet" => Some(Self::MAINNET),
            _ => None,
        }
    }
}

// ── argument parsing ─────────────────────────────────────────────────────────

/// Which agent slot the approval targets.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Slot {
    /// One of the three named slots.
    Named(String),
    /// The account's single unnamed slot.
    Unnamed,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Args {
    slot: Slot,
    valid_days: Option<u64>,
    out: Option<PathBuf>,
    print_secret: bool,
    submit: bool,
    confirm_mainnet: Option<String>,
    help: bool,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            slot: Slot::Named(DEFAULT_AGENT_NAME.to_string()),
            valid_days: None,
            out: None,
            print_secret: false,
            submit: false,
            confirm_mainnet: None,
            help: false,
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
enum ArgError {
    SecretOnArgv,
    Unknown(String),
    MissingValue(&'static str),
    NotANumber(String),
    UnnamedCannotExpire,
    NameAndUnnamed,
    TwoDestinations,
}

impl std::fmt::Display for ArgError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SecretOnArgv => write!(
                f,
                "something on the command line looks like a private key. argv is \
                 world-readable via `ps`, so that key must now be treated as public: \
                 generate a new one and pass keys through the environment (.env + \
                 scripts/with-env.sh) instead"
            ),
            Self::Unknown(a) => write!(f, "unknown argument {a:?} (try --help)"),
            Self::MissingValue(flag) => write!(f, "{flag} needs a value"),
            Self::NotANumber(v) => write!(f, "expected a number, got {v:?}"),
            Self::UnnamedCannotExpire => write!(
                f,
                "--unnamed cannot take --valid-days: Hyperliquid has no expiry field, \
                 the deadline rides inside the agent's name"
            ),
            Self::NameAndUnnamed => write!(f, "--name and --unnamed are mutually exclusive"),
            Self::TwoDestinations => write!(
                f,
                "--print-secret and --out both name a destination for the key; pick one"
            ),
        }
    }
}

/// Whether `s` could be a 32-byte secp256k1 key in hex.
///
/// A match aborts the run rather than warning. argv is world-readable through
/// `ps`, so by the time we could warn the key has already leaked, and continuing
/// would hand the operator an approval they believe is contained.
fn looks_like_a_private_key(s: &str) -> bool {
    let body = s
        .strip_prefix("0x")
        .or_else(|| s.strip_prefix("0X"))
        .unwrap_or(s);
    body.len() == 64 && body.bytes().all(|b| b.is_ascii_hexdigit())
}

fn parse_args<I: IntoIterator<Item = String>>(argv: I) -> Result<Args, ArgError> {
    let mut args = Args::default();
    let (mut saw_name, mut saw_unnamed) = (false, false);
    let mut it = argv.into_iter();
    while let Some(arg) = it.next() {
        if looks_like_a_private_key(&arg) {
            return Err(ArgError::SecretOnArgv);
        }
        // Every flag that takes a value re-checks it, because `--name <key>` puts a
        // secret on argv just as effectively as a bare positional would.
        let mut value = |flag| -> Result<String, ArgError> {
            let v = it.next().ok_or(ArgError::MissingValue(flag))?;
            if looks_like_a_private_key(&v) {
                return Err(ArgError::SecretOnArgv);
            }
            Ok(v)
        };
        match arg.as_str() {
            "-h" | "--help" => args.help = true,
            "--submit" => args.submit = true,
            "--print-secret" => args.print_secret = true,
            "--unnamed" => {
                saw_unnamed = true;
                args.slot = Slot::Unnamed;
            }
            "--name" => {
                saw_name = true;
                args.slot = Slot::Named(value("--name")?);
            }
            "--valid-days" => {
                let v = value("--valid-days")?;
                args.valid_days = Some(v.parse().map_err(|_| ArgError::NotANumber(v))?);
            }
            "--out" => args.out = Some(PathBuf::from(value("--out")?)),
            "--confirm-mainnet" => args.confirm_mainnet = Some(value("--confirm-mainnet")?),
            other => return Err(ArgError::Unknown(other.to_string())),
        }
    }
    if saw_name && saw_unnamed {
        return Err(ArgError::NameAndUnnamed);
    }
    if saw_unnamed && args.valid_days.is_some() {
        return Err(ArgError::UnnamedCannotExpire);
    }
    if args.print_secret && args.out.is_some() {
        return Err(ArgError::TwoDestinations);
    }
    Ok(args)
}

// ── the mainnet guard ────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    DryRun,
    Submit,
}

#[derive(Debug, PartialEq, Eq)]
enum GuardError {
    MainnetUnconfirmed,
    MainnetConfirmationMismatch { given: String, account: String },
    ConfirmationWithoutMainnet,
}

impl std::fmt::Display for GuardError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MainnetUnconfirmed => write!(
                f,
                "refusing to submit on MAINNET without a second confirmation. Re-run with \
                 `--confirm-mainnet <the account address>` once you are certain this is the \
                 account you mean to change."
            ),
            Self::MainnetConfirmationMismatch { given, account } => write!(
                f,
                "--confirm-mainnet names {given}, but the approval would modify {account}. \
                 Refusing: the two disagreeing means one of them is not the account you think."
            ),
            Self::ConfirmationWithoutMainnet => write!(
                f,
                "--confirm-mainnet was passed but {NETWORK_ENV} is testnet. That mismatch \
                 usually means the network variable is not what you think it is, so nothing \
                 is done until the two agree."
            ),
        }
    }
}

/// Decide whether this invocation may actually send the approval.
///
/// Mainnet needs a second, *separate* acknowledgement, and it is deliberately not a
/// bare `--yes`: the operator retypes the address the approval modifies. A flag that
/// can be added from muscle memory is not a confirmation. The specific mistake being
/// guarded is a testnet ceremony run with `AXON_HL_NETWORK=mainnet` still exported
/// from an earlier shell — which is why a mainnet confirmation on testnet is also an
/// error rather than a harmless extra.
fn decide_mode(args: &Args, net: Network, account: &str) -> Result<Mode, GuardError> {
    if let Some(given) = &args.confirm_mainnet {
        if !net.is_mainnet {
            return Err(GuardError::ConfirmationWithoutMainnet);
        }
        if !given.eq_ignore_ascii_case(account) {
            return Err(GuardError::MainnetConfirmationMismatch {
                given: given.clone(),
                account: account.to_string(),
            });
        }
    }
    if !args.submit {
        return Ok(Mode::DryRun);
    }
    if net.is_mainnet && args.confirm_mainnet.is_none() {
        return Err(GuardError::MainnetUnconfirmed);
    }
    Ok(Mode::Submit)
}

// ── the freshly generated agent key ──────────────────────────────────────────

/// A newly generated agent wallet.
///
/// Deliberately no `Debug`: the secret is supposed to have exactly one
/// destination, and a stray `{:?}` in some error path is the easiest way to give
/// it a second.
struct AgentKey {
    address: Address,
    secret_hex: String,
}

impl AgentKey {
    /// Generate locally from the OS CSPRNG.
    ///
    /// Never derived from the master key, and never reused between approvals: the
    /// venue warns that once an agent is pruned, actions it signed earlier can be
    /// replayed against a fresh approval of the *same* address.
    fn generate() -> Self {
        let wallet = PrivateKeySigner::random();
        Self {
            address: wallet.address(),
            secret_hex: hex::encode_prefixed(wallet.to_bytes()),
        }
    }

    /// The lowercase form the venue echoes back in `extraAgents`.
    fn address_lower(&self) -> String {
        hex::encode_prefixed(self.address.as_slice())
    }
}

const RESTRICTS_FILE_PERMISSIONS: bool = cfg!(unix);

#[cfg(unix)]
fn create_private_dir(dir: &Path) -> io::Result<()> {
    use std::os::unix::fs::DirBuilderExt;
    if dir.is_dir() {
        return Ok(());
    }
    fs::DirBuilder::new()
        .recursive(true)
        .mode(0o700)
        .create(dir)
}

#[cfg(not(unix))]
fn create_private_dir(dir: &Path) -> io::Result<()> {
    fs::create_dir_all(dir)
}

/// The mode is set at `open` time, not with a later `chmod`: a file that exists
/// world-readable for even a moment has already been readable for a moment.
#[cfg(unix)]
fn create_private_file(path: &Path) -> io::Result<fs::File> {
    use std::os::unix::fs::OpenOptionsExt;
    fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(path)
}

/// Windows has no mode bits here; the file inherits the directory's ACL. Creating
/// it anyway is right — refusing would push the operator to `--print-secret` and a
/// terminal scrollback — but [`RESTRICTS_FILE_PERMISSIONS`] makes the caller say so
/// out loud rather than let the difference pass unnoticed.
#[cfg(not(unix))]
fn create_private_file(path: &Path) -> io::Result<fs::File> {
    fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
}

/// Write the agent secret where only this user can read it.
///
/// `create_new`, so an existing file is never clobbered: that file may be the only
/// copy of a key the account still authorizes, and overwriting it leaves the
/// account approving a key nobody holds.
fn write_secret_file(path: &Path, secret_hex: &str) -> io::Result<()> {
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        create_private_dir(dir)?;
    }
    let mut file = create_private_file(path)?;
    writeln!(file, "{secret_hex}")?;
    file.sync_all()
}

fn default_key_path(agent: &AgentKey) -> PathBuf {
    // `secrets/` and `*.key` are both gitignored, so the default destination cannot
    // be committed by an absent-minded `git add -A`.
    PathBuf::from("secrets").join(format!("agent-{}.key", agent.address_lower()))
}

// ── the ceremony ─────────────────────────────────────────────────────────────

/// Everything one run needs, resolved before anything is signed or shown.
struct Ceremony {
    net: Network,
    /// The account the approval modifies — the master key's own address.
    account: String,
    master: HlSigner,
    /// Which env var the master key came from, so the output can say whether the
    /// pre-migration fallback is still in use.
    master_source: &'static str,
    agent: AgentKey,
    slot: Slot,
    valid_days: Option<u64>,
    now_ms: u64,
    nonce: u64,
}

/// The deadline an approval asks for, in epoch ms.
///
/// One function, called both by the plan and by the action it describes: a
/// displayed expiry that disagrees with the signed one is worse than showing none,
/// because the operator schedules the next rotation off the number they were shown.
fn expiry_ms(valid_days: Option<u64>, now_ms: u64) -> u64 {
    match valid_days {
        Some(days) => now_ms.saturating_add(days.saturating_mul(MS_PER_DAY)),
        // What `AgentName::max_validity` picks: the venue's 180-day cap less a
        // margin, so its own clock-side re-check cannot reject us over skew.
        None => now_ms.saturating_add(MAX_AGENT_VALIDITY_MS - MAX_VALIDITY_MARGIN_MS),
    }
}

impl Ceremony {
    /// The `approveAgent` action this run would send.
    fn action(&self) -> Result<ApproveAgent, String> {
        Ok(match &self.slot {
            Slot::Unnamed => {
                ApproveAgent::unnamed(self.agent.address, self.nonce, self.net.is_mainnet)
            }
            Slot::Named(base) => {
                let until = expiry_ms(self.valid_days, self.now_ms);
                let name =
                    AgentName::valid_until(base, until, self.now_ms).map_err(|e| e.to_string())?;
                ApproveAgent::named(self.agent.address, name, self.nonce, self.net.is_mainnet)
            }
        })
    }

    /// The expiry baked into the agent's name — `None` for the unnamed slot, which
    /// has no name to carry one and therefore gets the venue's own default.
    fn valid_until_ms(&self) -> Option<u64> {
        match self.slot {
            Slot::Unnamed => None,
            Slot::Named(_) => Some(expiry_ms(self.valid_days, self.now_ms)),
        }
    }

    /// The 32-byte EIP-712 digest the master key signs.
    fn digest(&self, action: &ApproveAgent) -> B256 {
        user_signed_hash(&action.eip712_payload(), action.signature_chain_id().get())
    }

    /// Sign, then recover, and confirm the recovered address is the master.
    ///
    /// This is the offline proof that the *user-signed* scheme was used: an action
    /// hashed under the L1 phantom-agent domain signs perfectly well and simply
    /// recovers to a different address, which the venue reports as an unexplained
    /// "invalid signature" after the nonce is already spent.
    fn self_check(&self, action: &ApproveAgent) -> Result<(), String> {
        let signature = self
            .master
            .sign_user_signed_action(action)
            .map_err(|e| format!("signing failed: {e}"))?;
        let digest = self.digest(action);
        match recovered_signer(&signature, &digest) {
            Some(addr) if addr == self.master.address() => Ok(()),
            Some(addr) => Err(format!(
                "the signature recovers to {addr}, not the master {}. Refusing to send \
                 an approval the venue would reject.",
                self.master.address()
            )),
            None => Err("the signature could not be recovered at all".to_string()),
        }
    }
}

/// Recover the address that produced `sig` over `digest`.
fn recovered_signer(sig: &RpcSignature, digest: &B256) -> Option<Address> {
    let r: U256 = sig.r.parse().ok()?;
    let s: U256 = sig.s.parse().ok()?;
    let parity = match sig.v {
        27 => false,
        28 => true,
        _ => return None,
    };
    Signature::new(r, s, parity)
        .recover_address_from_prehash(digest)
        .ok()
}

/// What the approval would displace, given the account's current agents.
///
/// Shown before submitting because the venue applies these rules silently: a name
/// collision replaces that agent, and approving an unnamed agent replaces the
/// previous unnamed one. An operator who thinks they are *adding* an agent and is
/// in fact *replacing* their live one only finds out when the running process
/// starts failing to sign.
fn displaced_by(slot: &Slot, agents: &[ExtraAgent]) -> Vec<String> {
    agents
        .iter()
        .filter(|a| match slot {
            Slot::Named(base) => a.name.split(" valid_until ").next() == Some(base.as_str()),
            Slot::Unnamed => a.name.is_empty(),
        })
        .map(|a| {
            format!(
                "{} ({})",
                if a.name.is_empty() {
                    "<unnamed>"
                } else {
                    &a.name
                },
                a.address
            )
        })
        .collect()
}

fn render_agents(agents: &[ExtraAgent], now_ms: u64) -> String {
    if agents.is_empty() {
        return "  (none — the account has never approved an agent)\n".to_string();
    }
    let now_ns = (now_ms as i64).saturating_mul(1_000_000);
    let mut out = String::new();
    for a in agents {
        let name = if a.name.is_empty() {
            "<unnamed>"
        } else {
            &a.name
        };
        let state = if a.is_valid_at(now_ns) {
            "valid"
        } else {
            "EXPIRED"
        };
        let _ = writeln!(out, "  {name:<34} {}  {state}", a.address);
    }
    out
}

/// The whole "here is what I will do" block, built from the agent's **address**
/// only — the secret is not an input, which is what makes it impossible for this
/// output to carry it.
fn render_plan(c: &Ceremony, action: &ApproveAgent, mode: Mode, destination: &str) -> String {
    let headline = match mode {
        Mode::DryRun => "DRY RUN, nothing will be sent",
        Mode::Submit => "WILL BE SUBMITTED",
    };
    let mut out = String::new();
    let w = &mut out;
    let _ = writeln!(w, "\n=== approveAgent — {headline} ===\n");
    let _ = writeln!(w, "network          : {}", c.net.label);
    let _ = writeln!(w, "exchange endpoint: {}", c.net.exchange_url);
    let _ = writeln!(w, "account modified : {}", c.account);
    let _ = writeln!(
        w,
        "signed by        : {}  (master key, from {})",
        c.master.address(),
        c.master_source
    );
    let _ = writeln!(w, "\nnew agent wallet");
    let _ = writeln!(w, "  address        : {}", c.agent.address);
    let _ = writeln!(
        w,
        "  name           : {}",
        action.agent_name().unwrap_or("<unnamed slot>")
    );
    match c.valid_until_ms() {
        Some(until) => {
            let days = (until.saturating_sub(c.now_ms)) / MS_PER_DAY;
            let _ = writeln!(w, "  valid until    : {until} (epoch ms, ~{days} days)");
        }
        None => {
            let _ = writeln!(
                w,
                "  valid until    : venue default (the unnamed slot has no expiry field)"
            );
        }
    }
    let _ = writeln!(w, "  secret goes to : {destination}");
    let _ = writeln!(
        w,
        "\nsigning scheme   : user-signed EIP-712, domain HyperliquidSignTransaction"
    );
    let _ = writeln!(
        w,
        "  (NOT the L1 phantom-agent scheme orders use — mixing them is the"
    );
    let _ = writeln!(
        w,
        "   venue's number-one cause of an unexplained \"invalid signature\")"
    );
    let _ = writeln!(w, "  digest         : {}", c.digest(action));
    let _ = writeln!(w, "  nonce          : {}", c.nonce);
    let json = serde_json::to_string_pretty(action).unwrap_or_else(|e| format!("<{e}>"));
    let _ = writeln!(w, "\naction that would be signed:\n{json}");
    out
}

// ── env plumbing ─────────────────────────────────────────────────────────────

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Read the master key, preferring the dedicated variable.
///
/// Returns which variable it came from so the plan can say so: falling back to
/// `AXON_HL_SECRET_KEY` only works while that key still *is* the account, i.e.
/// only before the migration this tool performs.
fn master_key() -> Result<(String, &'static str), String> {
    if let Ok(k) = env::var(MASTER_KEY_ENV) {
        return Ok((k, MASTER_KEY_ENV));
    }
    match env::var(HlSigner::ENV_KEY) {
        Ok(k) => Ok((k, HlSigner::ENV_KEY)),
        Err(_) => Err(format!(
            "no master key: set {MASTER_KEY_ENV} (or, pre-migration, {}) in .env and run \
             through `bash scripts/with-env.sh`",
            HlSigner::ENV_KEY
        )),
    }
}

/// Resolve the master signer and the account it acts for, refusing any
/// disagreement between them.
///
/// `approveAgent` authorizes an agent for **the signer's own account**, so a
/// configured `AXON_HL_ACCOUNT_ADDRESS` that does not match the signing address
/// means one of two things, both fatal: the operator is about to approve an agent
/// on a different account than they believe, or `AXON_HL_SECRET_KEY` is already an
/// agent key — which cannot sign this action at all, and whose rejection at the
/// venue says nothing about why.
fn resolve_master(net: Network) -> Result<(HlSigner, String, &'static str), String> {
    let (key, source) = master_key()?;
    let signer = HlSigner::from_hex(&key, net.is_mainnet).map_err(|e| format!("{source}: {e}"))?;
    let signing = signer.address().to_string();
    if let Ok(account) = env::var(ACCOUNT_ENV) {
        if !account.eq_ignore_ascii_case(&signing) {
            // The likeliest cause depends on where the key came from and the two need
            // opposite fixes, so the hint is chosen rather than hedged.
            let hint = if source == MASTER_KEY_ENV {
                format!(
                    "{MASTER_KEY_ENV} holds a key for some other account; use the one that \
                     owns {ACCOUNT_ENV}."
                )
            } else {
                format!(
                    "{source} is most likely already an agent key — an agent cannot approve \
                     another agent. Put the MASTER key in {MASTER_KEY_ENV} for this one \
                     ceremony."
                )
            };
            return Err(format!(
                "{ACCOUNT_ENV} is {account} but {source} signs as {signing}.\n\
                 approveAgent is signed by the account itself, so these must match. {hint}"
            ));
        }
    }
    Ok((signer, signing, source))
}

// ── main ─────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() {
    if let Err(e) = run().await {
        eprintln!("\napprove_agent: {e}");
        std::process::exit(1);
    }
}

async fn run() -> Result<(), String> {
    let args = parse_args(env::args().skip(1)).map_err(|e| e.to_string())?;
    if args.help {
        println!("{USAGE}");
        return Ok(());
    }

    let raw_network = env::var(NETWORK_ENV).unwrap_or_else(|_| "testnet".to_string());
    let net = Network::parse(&raw_network).ok_or_else(|| {
        format!("{NETWORK_ENV} must be 'testnet' or 'mainnet', got {raw_network:?}")
    })?;
    let (master, account, master_source) = resolve_master(net)?;
    let mode = decide_mode(&args, net, &account).map_err(|e| e.to_string())?;

    let now = now_ms();
    let ceremony = Ceremony {
        net,
        account,
        master,
        master_source,
        agent: AgentKey::generate(),
        slot: args.slot.clone(),
        valid_days: args.valid_days,
        now_ms: now,
        nonce: NonceManager::new().next(now),
    };
    let action = ceremony.action()?;
    // Prove the signature before showing a plan that claims it will work.
    ceremony.self_check(&action)?;

    let key_path = args
        .out
        .clone()
        .unwrap_or_else(|| default_key_path(&ceremony.agent));
    let destination = if args.print_secret {
        "stdout, once (--print-secret)".to_string()
    } else {
        key_path.display().to_string()
    };
    print!("{}", render_plan(&ceremony, &action, mode, &destination));

    // A read-only look at what is already approved. Best-effort: being unable to
    // reach `/info` must not block a dry run, but the displacement warning is the
    // part of the plan an operator is most likely to be surprised by, so its
    // absence is stated rather than silently skipped.
    println!("\ncurrently approved agents for {}:", ceremony.account);
    match fetch_extra_agents(net.info_url, &ceremony.account).await {
        Ok(agents) => {
            print!("{}", render_agents(&agents, now));
            let displaced = displaced_by(&ceremony.slot, &agents);
            if !displaced.is_empty() {
                println!("\n  !! this approval DEREGISTERS: {}", displaced.join(", "));
                println!("     Anything still signing with that key stops working immediately.");
            }
        }
        Err(e) => println!("  (could not read extraAgents: {e})"),
    }

    if mode == Mode::DryRun {
        println!(
            "\nDRY RUN — nothing was signed onto the wire and nothing was written.\n\
             The agent key shown above was generated only to build this preview and is\n\
             being discarded; a real run generates a different one. Re-run with --submit."
        );
        return Ok(());
    }

    // The secret is persisted *before* the approval goes out. The other order loses
    // a race we cannot re-run: if the submit lands and this process then dies, the
    // account authorizes a key nobody has, and the slot has to be burned to recover.
    if args.print_secret {
        println!("\n{}", secret_banner(&ceremony.agent));
    } else {
        write_secret_file(&key_path, &ceremony.agent.secret_hex)
            .map_err(|e| format!("could not write {}: {e}", key_path.display()))?;
        println!("\nwrote the agent key to {}", key_path.display());
        if RESTRICTS_FILE_PERMISSIONS {
            println!("  mode 0600 — readable only by you.");
        } else {
            println!(
                "  WARNING: this platform has no mode bits here; the file inherited the\n\
                 directory's permissions. Restrict it before doing anything else."
            );
        }
    }

    println!("\nsubmitting approveAgent to {} …", net.exchange_url);
    let agent_signer = HlSigner::from_hex(&ceremony.agent.secret_hex, net.is_mainnet)
        .map_err(|e| format!("generated key rejected by the signer: {e}"))?;
    // The client is built around the agent being approved, matching the shape of the
    // steady state; `approve_agent` takes the master signer explicitly precisely so a
    // trading key can never quietly grant authority to itself.
    let client = if net.is_mainnet {
        ExchangeClient::mainnet(agent_signer)
    } else {
        ExchangeClient::testnet(agent_signer)
    }
    .map_err(|e| e.to_string())?;
    client
        .approve_agent(&ceremony.master, action)
        .await
        .map_err(|e| format!("the venue rejected the approval: {e}"))?;
    println!("venue accepted the action.");

    let agents = verify_listed(&ceremony).await?;
    println!("\nverified against extraAgents:");
    print!("{}", render_agents(&agents, now_ms()));
    println!("{}", env_edit(&ceremony, &key_path, args.print_secret));
    Ok(())
}

/// Re-read `extraAgents` until the new agent appears.
async fn verify_listed(c: &Ceremony) -> Result<Vec<ExtraAgent>, String> {
    let wanted = c.agent.address_lower();
    let mut last_error = None;
    for attempt in 1..=VERIFY_ATTEMPTS {
        match fetch_extra_agents(c.net.info_url, &c.account).await {
            Ok(agents)
                if agents
                    .iter()
                    .any(|a| a.address.eq_ignore_ascii_case(&wanted)) =>
            {
                return Ok(agents)
            }
            Ok(_) => last_error = Some("the agent is not listed yet".to_string()),
            Err(e) => last_error = Some(e.to_string()),
        }
        if attempt < VERIFY_ATTEMPTS {
            tokio::time::sleep(VERIFY_DELAY).await;
        }
    }
    Err(format!(
        "the venue accepted the action but {wanted} is still not in extraAgents after \
         {VERIFY_ATTEMPTS} reads ({}). Do NOT re-approve on that basis — re-approving the \
         same name deregisters an agent that may in fact have landed. Check with \
         `wallet_info` first.",
        last_error.unwrap_or_default()
    ))
}

/// The one-time print, when the operator asked for it instead of a file.
fn secret_banner(agent: &AgentKey) -> String {
    format!(
        "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\
         !! AGENT PRIVATE KEY — SHOWN ONCE, NOT STORED ANYWHERE BY THIS TOOL\n\
         !! It is now in your terminal scrollback. Anyone who reads it can\n\
         !! trade this account (they cannot withdraw). Copy it into .env, then\n\
         !! clear the scrollback.\n\
         !!\n\
         !!   address: {}\n\
         !!   key    : {}\n\
         !!\n\
         !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!",
        agent.address, agent.secret_hex
    )
}

/// The exact `.env` edit that completes the migration.
fn env_edit(c: &Ceremony, key_path: &Path, printed: bool) -> String {
    let source = if printed {
        "the key printed above".to_string()
    } else {
        format!("the key in {}", key_path.display())
    };
    format!(
        "\n=== finish the migration — edit .env ===\n\n\
         1. Replace the trading key with the new agent key:\n\n\
         \x20      {} = <{}>\n\
         \x20      {ACCOUNT_ENV} = {}   (unchanged — the master account)\n\
         \x20      {NETWORK_ENV} = {}   (unchanged)\n\n\
         2. Move the MASTER key out of .env entirely. Axon does not need it to trade,\n\
         \x20  and this tool reads it from {MASTER_KEY_ENV} when you next rotate:\n\n\
         \x20      {MASTER_KEY_ENV}=\"$(cat secrets/master.key)\" bash scripts/with-env.sh \\\n\
         \x20          cargo run -p axon-provider-hyperliquid --example approve_agent\n\n\
         3. Confirm the containment property now actually holds:\n\n\
         \x20      bash scripts/with-env.sh cargo run -p axon-provider-hyperliquid \\\n\
         \x20          --example wallet_info\n\n\
         \x20  It must report the signing address DIFFERS from the account address, and\n\
         \x20  list {} as a valid agent.\n",
        HlSigner::ENV_KEY,
        source,
        c.account,
        c.net.label,
        c.agent.address_lower(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use axon_provider_hyperliquid::sign::{l1_action_hash, l1_signing_hash};

    /// Hardhat/anvil account #0 — a public vector, never a real key. Standing in
    /// for the master account throughout.
    const MASTER_KEY: &str = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80";
    const MASTER_ADDR: &str = "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266";
    const NOW: u64 = 1_784_976_929_583;

    fn args(argv: &[&str]) -> Result<Args, ArgError> {
        parse_args(argv.iter().map(|s| s.to_string()))
    }

    fn ceremony(net: Network, slot: Slot, valid_days: Option<u64>) -> Ceremony {
        Ceremony {
            net,
            account: MASTER_ADDR.to_string(),
            master: HlSigner::from_hex(MASTER_KEY, net.is_mainnet).unwrap(),
            master_source: MASTER_KEY_ENV,
            agent: AgentKey::generate(),
            slot,
            valid_days,
            now_ms: NOW,
            nonce: NOW,
        }
    }

    fn named_testnet() -> Ceremony {
        ceremony(Network::TESTNET, Slot::Named("axon".into()), None)
    }

    // ── the dry-run default and the argument surface ──────────────────────────

    #[test]
    fn dry_run_is_the_default_so_a_bare_invocation_cannot_approve_anything() {
        let a = args(&[]).unwrap();
        assert!(!a.submit);
        assert_eq!(
            decide_mode(&a, Network::TESTNET, MASTER_ADDR),
            Ok(Mode::DryRun)
        );
        assert_eq!(
            decide_mode(&a, Network::MAINNET, MASTER_ADDR),
            Ok(Mode::DryRun)
        );
        // ...and the default slot is a *named* one, so a bare run can never take the
        // single unnamed slot, which is the one an unrelated tool is most likely to
        // also be using.
        assert_eq!(a.slot, Slot::Named("axon".into()));
    }

    #[test]
    fn an_unknown_flag_is_refused_rather_than_silently_ignored() {
        // `--sumbit` must not quietly become a dry run that the operator then
        // "fixes" by adding more flags until something goes out.
        assert_eq!(
            args(&["--sumbit"]),
            Err(ArgError::Unknown("--sumbit".into()))
        );
        assert_eq!(args(&["--name"]), Err(ArgError::MissingValue("--name")));
        assert_eq!(
            args(&["--valid-days", "soon"]),
            Err(ArgError::NotANumber("soon".into()))
        );
    }

    #[test]
    fn a_private_key_anywhere_on_argv_aborts_the_run() {
        // argv is world-readable through `ps`; by the time we could warn, it has
        // leaked. Both a bare positional and a flag value have to be caught.
        assert_eq!(args(&[MASTER_KEY]), Err(ArgError::SecretOnArgv));
        assert_eq!(args(&["--name", MASTER_KEY]), Err(ArgError::SecretOnArgv));
        assert_eq!(args(&["--out", MASTER_KEY]), Err(ArgError::SecretOnArgv));
        // Bare (un-prefixed) hex counts too.
        assert_eq!(args(&[&MASTER_KEY[2..]]), Err(ArgError::SecretOnArgv));
        // An ordinary name is obviously fine.
        assert!(args(&["--name", "axon-live"]).is_ok());
    }

    #[test]
    fn an_unnamed_agent_cannot_carry_an_expiry() {
        // Hyperliquid has no expiry field: the deadline lives inside the name, so an
        // unnamed agent with --valid-days would silently get no expiry at all.
        assert_eq!(
            args(&["--unnamed", "--valid-days", "30"]),
            Err(ArgError::UnnamedCannotExpire)
        );
        assert_eq!(
            args(&["--unnamed", "--name", "x"]),
            Err(ArgError::NameAndUnnamed)
        );
        assert_eq!(
            args(&["--print-secret", "--out", "k.key"]),
            Err(ArgError::TwoDestinations)
        );
    }

    // ── the mainnet guard ─────────────────────────────────────────────────────

    #[test]
    fn mainnet_submit_requires_retyping_the_account_address() {
        let a = args(&["--submit"]).unwrap();
        assert_eq!(
            decide_mode(&a, Network::TESTNET, MASTER_ADDR),
            Ok(Mode::Submit)
        );
        assert_eq!(
            decide_mode(&a, Network::MAINNET, MASTER_ADDR),
            Err(GuardError::MainnetUnconfirmed)
        );
        let confirmed = args(&["--submit", "--confirm-mainnet", MASTER_ADDR]).unwrap();
        assert_eq!(
            decide_mode(&confirmed, Network::MAINNET, MASTER_ADDR),
            Ok(Mode::Submit)
        );
        // Case is not the point; the address is.
        let lower = MASTER_ADDR.to_lowercase();
        let confirmed = args(&["--submit", "--confirm-mainnet", &lower]).unwrap();
        assert_eq!(
            decide_mode(&confirmed, Network::MAINNET, MASTER_ADDR),
            Ok(Mode::Submit)
        );
    }

    #[test]
    fn a_mainnet_confirmation_naming_the_wrong_account_is_refused() {
        let other = "0x1234567890abcdef1234567890abcdef12345678";
        let a = args(&["--submit", "--confirm-mainnet", other]).unwrap();
        assert_eq!(
            decide_mode(&a, Network::MAINNET, MASTER_ADDR),
            Err(GuardError::MainnetConfirmationMismatch {
                given: other.to_string(),
                account: MASTER_ADDR.to_string(),
            })
        );
    }

    #[test]
    fn a_mainnet_confirmation_on_testnet_is_refused_as_a_network_mixup() {
        // The operator believes they are on mainnet and the process disagrees. Which
        // one is wrong is unknowable from here, so neither a dry run nor a submit
        // proceeds until they agree.
        for argv in [
            vec!["--confirm-mainnet", MASTER_ADDR],
            vec!["--submit", "--confirm-mainnet", MASTER_ADDR],
        ] {
            let a = args(&argv).unwrap();
            assert_eq!(
                decide_mode(&a, Network::TESTNET, MASTER_ADDR),
                Err(GuardError::ConfirmationWithoutMainnet)
            );
        }
    }

    // ── key generation ────────────────────────────────────────────────────────

    #[test]
    fn generated_agent_keys_are_fresh_and_unrelated_to_the_master() {
        let master = HlSigner::from_hex(MASTER_KEY, false).unwrap();
        let a = AgentKey::generate();
        let b = AgentKey::generate();
        assert_ne!(a.address, b.address, "every approval must get new material");
        assert_ne!(a.secret_hex, b.secret_hex);
        assert_ne!(a.address, master.address(), "an agent is never the account");
        // 0x + 64 hex, and it round-trips into a signer whose address matches — i.e.
        // the printed address really is the address of the key that was written.
        assert_eq!(a.secret_hex.len(), 66);
        assert!(a.secret_hex.starts_with("0x"));
        assert_eq!(
            HlSigner::from_hex(&a.secret_hex, false).unwrap().address(),
            a.address
        );
        assert_eq!(a.address_lower(), a.address.to_string().to_lowercase());
    }

    // ── the signed payload ────────────────────────────────────────────────────

    #[test]
    fn the_approval_is_signed_under_the_user_signed_domain_not_the_l1_scheme() {
        let c = named_testnet();
        let action = c.action().unwrap();
        // The self-check is the tool's own proof, and it must pass.
        c.self_check(&action).unwrap();

        // Spelled out: the digest is the user-signed one, and the L1 digest over the
        // same action is a different value that would recover to a different address.
        let user = user_signed_hash(&action.eip712_payload(), action.signature_chain_id().get());
        assert_eq!(c.digest(&action), user);
        let l1 = l1_signing_hash(l1_action_hash(&action, c.nonce, None, None).unwrap(), false);
        assert_ne!(user, l1, "the two schemes must never coincide");

        let sig = c.master.sign_user_signed_action(&action).unwrap();
        assert_eq!(recovered_signer(&sig, &user), Some(c.master.address()));
        assert_ne!(recovered_signer(&sig, &l1), Some(c.master.address()));
    }

    #[test]
    fn the_signed_action_matches_the_plan_that_was_shown() {
        let c = ceremony(Network::MAINNET, Slot::Named("axon-live".into()), Some(30));
        let action = c.action().unwrap();
        let json = serde_json::to_value(&action).unwrap();
        assert_eq!(json["type"], "approveAgent");
        assert_eq!(json["hyperliquidChain"], "Mainnet");
        assert_eq!(json["nonce"], NOW);
        assert_eq!(json["agentAddress"], c.agent.address_lower());
        // The name shown in the plan is the name that gets hashed.
        assert_eq!(
            json["agentName"].as_str().unwrap(),
            action.agent_name().unwrap()
        );
        // The unnamed slot omits the key entirely (it still hashes as "").
        let unnamed = ceremony(Network::TESTNET, Slot::Unnamed, None);
        let action = unnamed.action().unwrap();
        let json = serde_json::to_value(&action).unwrap();
        assert!(json.get("agentName").is_none());
        assert_eq!(unnamed.valid_until_ms(), None);
    }

    #[test]
    fn the_displayed_expiry_is_the_one_baked_into_the_name() {
        // Two independent computations of the deadline — the plan's and
        // `AgentName`'s — that must not drift apart, because the operator decides
        // when to rotate from the number they were shown.
        for days in [None, Some(1u64), Some(30), Some(180)] {
            let c = ceremony(Network::TESTNET, Slot::Named("axon".into()), days);
            let until = c.valid_until_ms().unwrap();
            let name = c.action().unwrap().agent_name().unwrap().to_string();
            assert_eq!(name, format!("axon valid_until {until}"), "days={days:?}");
            assert!(until > NOW && until - NOW <= MAX_AGENT_VALIDITY_MS);
        }
        // Past the venue's cap the action fails locally instead of burning a nonce.
        let c = ceremony(Network::TESTNET, Slot::Named("axon".into()), Some(181));
        assert!(c.action().is_err());
    }

    #[test]
    fn the_plan_never_contains_the_agent_secret() {
        // The plan is built from the address, so this is a property of the data flow
        // rather than of careful formatting — but it is the one leak that would be
        // catastrophic and invisible, so it is asserted directly.
        let c = named_testnet();
        let action = c.action().unwrap();
        for mode in [Mode::DryRun, Mode::Submit] {
            let text = render_plan(&c, &action, mode, "secrets/agent.key");
            assert!(
                !text.contains(&c.agent.secret_hex),
                "the plan leaked the key"
            );
            assert!(
                !text.contains(&c.agent.secret_hex[2..]),
                "leaked it unprefixed"
            );
            assert!(
                text.contains(&c.agent.address.to_string()),
                "address must be shown"
            );
            assert!(text.contains("HyperliquidSignTransaction"));
        }
        // Same for the follow-up instructions: they point at the file, never inline
        // the key.
        let edit = env_edit(&c, Path::new("secrets/agent.key"), false);
        assert!(!edit.contains(&c.agent.secret_hex));
        assert!(edit.contains("secrets/agent.key"));
        // `--print-secret` is the single sanctioned exception, and it is unmissable.
        let banner = secret_banner(&c.agent);
        assert!(banner.contains(&c.agent.secret_hex));
        assert!(banner.contains("SHOWN ONCE"));
    }

    // ── what an approval displaces ────────────────────────────────────────────

    #[test]
    fn a_name_collision_is_reported_as_a_deregistration() {
        let agents = vec![
            ExtraAgent {
                name: "axon valid_until 1800000000000".into(),
                address: "0x1111111111111111111111111111111111111111".into(),
                valid_until: 1_800_000_000_000 * 1_000_000,
            },
            ExtraAgent {
                name: String::new(),
                address: "0x2222222222222222222222222222222222222222".into(),
                valid_until: 1_800_000_000_000 * 1_000_000,
            },
        ];
        // A named approval displaces the same *base* name, expiry suffix and all.
        let hit = displaced_by(&Slot::Named("axon".into()), &agents);
        assert_eq!(hit.len(), 1);
        assert!(hit[0].contains("0x1111111111111111111111111111111111111111"));
        // An unnamed approval displaces the unnamed slot, not the named one.
        let hit = displaced_by(&Slot::Unnamed, &agents);
        assert_eq!(hit.len(), 1);
        assert!(hit[0].contains("0x2222222222222222222222222222222222222222"));
        // A fresh name displaces nothing.
        assert!(displaced_by(&Slot::Named("axon-2".into()), &agents).is_empty());

        let table = render_agents(&agents, NOW);
        assert!(table.contains("<unnamed>") && table.contains("valid"));
        assert!(render_agents(&[], NOW).contains("none"));
    }

    // ── the key file ──────────────────────────────────────────────────────────

    fn scratch(tag: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        env::temp_dir().join(format!("axon-approve-agent-{tag}-{unique}"))
    }

    #[test]
    fn an_existing_key_file_is_never_clobbered() {
        // That file can be the only copy of a key the account still authorizes;
        // overwriting it leaves the account approving a key nobody holds.
        let dir = scratch("clobber");
        let path = dir.join("agent.key");
        write_secret_file(&path, "0xdeadbeef").unwrap();
        let err = write_secret_file(&path, "0xfeedface").unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::AlreadyExists);
        assert!(fs::read_to_string(&path).unwrap().contains("0xdeadbeef"));
        let _ = fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    #[test]
    fn the_key_file_is_unreadable_to_other_users() {
        use std::os::unix::fs::PermissionsExt;
        let dir = scratch("mode");
        let path = dir.join("nested").join("agent.key");
        let agent = AgentKey::generate();
        write_secret_file(&path, &agent.secret_hex).unwrap();
        assert_eq!(
            fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o600
        );
        // The directory the tool creates must not be a way around the file's mode.
        let parent = path.parent().unwrap();
        assert_eq!(
            fs::metadata(parent).unwrap().permissions().mode() & 0o777,
            0o700
        );
        assert_eq!(fs::read_to_string(&path).unwrap().trim(), agent.secret_hex);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_default_key_path_lands_somewhere_gitignored() {
        let agent = AgentKey::generate();
        let path = default_key_path(&agent);
        // `secrets/` and `*.key` are both in .gitignore, so `git add -A` after the
        // ceremony cannot commit the key.
        assert!(path.starts_with("secrets"));
        assert_eq!(path.extension().unwrap(), "key");
        assert!(path.to_string_lossy().contains(&agent.address_lower()));
    }
}
