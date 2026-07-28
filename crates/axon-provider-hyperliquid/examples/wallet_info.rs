//! `wallet_info` — the pre-flight check before any live order flow.
//!
//! Derives the signing address from `AXON_HL_SECRET_KEY` and answers the one
//! question that decides how much a leak of that key costs: **is it an agent
//! wallet, or is it the account?** Under the model ADR-0009 describes, the trading
//! process holds a key that can trade and cannot withdraw. That is a property of the *venue's*
//! `extraAgents` list, not of our intentions, so this tool reads the list and
//! reports what it actually finds — including the case where the two addresses are
//! the same and the containment property is simply absent.
//!
//! It also reports each approved agent's **validity window**. An agent's deadline
//! is baked into its name at approval time and the venue silently stops accepting
//! its signatures when the deadline passes; discovering that from a stream of
//! rejected orders mid-session is the failure this read prevents.
//!
//! Run it through the env loader so the key never appears on a command line:
//!
//! ```text
//! bash scripts/with-env.sh cargo run -p axon-provider-hyperliquid --example wallet_info
//! ```
//!
//! Read-only: it signs nothing and POSTs nothing but `/info` queries.

use std::env;

use axon_core::Decimal;
use axon_core::Nanos;
use axon_provider_hyperliquid::ws::{MAINNET_INFO, TESTNET_INFO};
use axon_provider_hyperliquid::{fetch_extra_agents, ExtraAgent, HlSigner};

const MS_TO_NS: i64 = 1_000_000;

/// Hyperliquid's minimum order notional, in USDC. An account below it cannot rest
/// even a throwaway test order, which is why fill verification is gated on someone
/// funding the account rather than on any remaining code.
const MIN_ORDER_NOTIONAL: i64 = 10;

/// How close to expiry an agent has to be before this tool starts nagging.
///
/// Rotation is a manual ceremony with a human in it, so the warning has to lead the
/// deadline by more than a working week — an agent that expires on a Saturday is
/// still an outage.
const ROTATE_WITHIN_DAYS: i64 = 14;

const MS_PER_DAY: i64 = 24 * 60 * 60 * 1_000;

/// What the configured key **is**, as distinct from what the setup intends it to be.
#[derive(Debug, Clone, PartialEq, Eq)]
enum KeyModel {
    /// The signing key is the account itself. It can withdraw; the containment
    /// property does not hold.
    MasterKey,
    /// The signer is an approved agent inside its validity window — the intended
    /// state, and the only one that makes a leak survivable.
    ApprovedAgent { name: String, valid_until_ms: i64 },
    /// Approved once, but the window has closed. Every action it signs now fails,
    /// and the venue's error does not say why.
    ExpiredAgent { name: String, valid_until_ms: i64 },
    /// Neither the account nor an approved agent: nothing it signs will be accepted.
    Unapproved,
    /// `extraAgents` could not be read. Deliberately its own state rather than
    /// collapsing into [`Unapproved`]: telling an operator their agent is
    /// unapproved because `/info` was unreachable sends them to re-run the approval
    /// ceremony, which deregisters the agent that was working fine.
    Unknown,
}

/// Classify the configured key against the venue's own view of the account.
fn classify(
    signing: &str,
    account: &str,
    agents: Option<&[ExtraAgent]>,
    now_ns: Nanos,
) -> KeyModel {
    // Local knowledge, and it outranks the agent list: an address that *is* the
    // account can withdraw whether or not it also happens to be listed.
    if signing.eq_ignore_ascii_case(account) {
        return KeyModel::MasterKey;
    }
    let Some(agents) = agents else {
        return KeyModel::Unknown;
    };
    match agents
        .iter()
        .find(|a| a.address.eq_ignore_ascii_case(signing))
    {
        Some(a) if a.is_valid_at(now_ns) => KeyModel::ApprovedAgent {
            name: agent_label(a),
            valid_until_ms: a.valid_until / MS_TO_NS,
        },
        Some(a) => KeyModel::ExpiredAgent {
            name: agent_label(a),
            valid_until_ms: a.valid_until / MS_TO_NS,
        },
        None => KeyModel::Unapproved,
    }
}

fn agent_label(a: &ExtraAgent) -> String {
    if a.name.is_empty() {
        "<unnamed>".to_string()
    } else {
        a.name.clone()
    }
}

/// A signed, human-scale gap between two epoch-ms instants.
///
/// Relative rather than a calendar date on purpose: the decision an operator makes
/// from this number is "do I have to rotate before the weekend", and a relative
/// figure answers that without any timezone reasoning.
fn until(now_ms: i64, then_ms: i64) -> String {
    let delta = then_ms - now_ms;
    let mag = delta.unsigned_abs();
    let (d, h, m) = (
        mag / 86_400_000,
        (mag % 86_400_000) / 3_600_000,
        (mag % 3_600_000) / 60_000,
    );
    if delta >= 0 {
        format!("in {d}d {h}h {m}m")
    } else {
        format!("EXPIRED {d}d {h}h {m}m ago")
    }
}

/// The `extraAgents` table, with the configured signer called out in place.
fn render_agents(agents: &[ExtraAgent], signing: &str, now_ms: i64) -> String {
    if agents.is_empty() {
        return "  (none — this account has never approved an agent wallet)\n".to_string();
    }
    let mut out = String::new();
    for a in agents {
        let valid_until_ms = a.valid_until / MS_TO_NS;
        let mine = if a.address.eq_ignore_ascii_case(signing) {
            "  <<< THIS IS THE CONFIGURED SIGNER"
        } else {
            ""
        };
        out.push_str(&format!(
            "  {:<32} {}\n      valid until {valid_until_ms} ({}){mine}\n",
            agent_label(a),
            a.address,
            until(now_ms, valid_until_ms),
        ));
    }
    out
}

/// The verdict block. Loud for every state that is not the intended one, because
/// the whole point of this tool is that the bad states are the ones that otherwise
/// look like nothing.
fn verdict(model: &KeyModel, now_ms: i64) -> String {
    let rule = "!".repeat(78);
    match model {
        KeyModel::MasterKey => format!(
            "{rule}\n\
             !! THE CONFIGURED KEY *IS* THE MASTER ACCOUNT — NOT AN AGENT WALLET.\n\
             !!\n\
             !! AXON_HL_SECRET_KEY signs as the account address itself, so this key can\n\
             !! WITHDRAW. The containment property ADR-0009 describes\n\
             !! — \"a leaked hot key costs you bad trades, not the balance\" — DOES NOT\n\
             !! HOLD for this configuration. It is not a warning about the future; it is\n\
             !! a statement about right now.\n\
             !!\n\
             !! Tolerable only on testnet with play money. Never fund this address on\n\
             !! mainnet, and never run Phase 6 like this.\n\
             !!\n\
             !! Fix it — the ceremony is one command, and dry-runs by default:\n\
             !!     bash scripts/with-env.sh cargo run -p axon-provider-hyperliquid \\\n\
             !!         --example approve_agent\n\
             !! Then re-run this check: the venue must list the agent.\n\
             {rule}"
        ),
        KeyModel::ApprovedAgent {
            name,
            valid_until_ms,
        } => {
            let days_left = (valid_until_ms - now_ms) / MS_PER_DAY;
            let mut s = format!(
                "OK — agent-wallet model holds. The signer is approved agent {name:?}, it can\n\
                 trade this account and it cannot withdraw. Expires {} ({valid_until_ms}).",
                until(now_ms, *valid_until_ms)
            );
            if days_left <= ROTATE_WITHIN_DAYS {
                s.push_str(&format!(
                    "\n\n!! ROTATE SOON: {days_left} day(s) of validity left. When it lapses the venue\n\
                     !! stops accepting this signer with no explanation attached to the rejection.\n\
                     !! Approve a replacement before then:\n\
                     !!     bash scripts/with-env.sh cargo run -p axon-provider-hyperliquid \\\n\
                     !!         --example approve_agent -- --submit"
                ));
            }
            s
        }
        KeyModel::ExpiredAgent {
            name,
            valid_until_ms,
        } => format!(
            "{rule}\n\
             !! THE CONFIGURED AGENT HAS EXPIRED — nothing it signs will be accepted.\n\
             !!\n\
             !! Agent {name:?} lapsed {} ({valid_until_ms}). The venue rejects its\n\
             !! signatures without saying that expiry is the reason, so this reads as a\n\
             !! generic \"invalid signature\" from inside the trading process.\n\
             !!\n\
             !! Approve a replacement (a fresh key — never reuse an expired agent's\n\
             !! address, its old signatures become replayable against a new approval):\n\
             !!     bash scripts/with-env.sh cargo run -p axon-provider-hyperliquid \\\n\
             !!         --example approve_agent -- --submit\n\
             {rule}",
            until(now_ms, *valid_until_ms)
        ),
        KeyModel::Unapproved => format!(
            "{rule}\n\
             !! THE SIGNING KEY IS NEITHER THE ACCOUNT NOR AN APPROVED AGENT.\n\
             !!\n\
             !! It is not in this account's extraAgents list, so every order it signs will\n\
             !! be rejected. Either AXON_HL_ACCOUNT_ADDRESS names the wrong account, or the\n\
             !! approval was never made (or was displaced — approving an agent under an\n\
             !! existing name deregisters the previous holder of that name).\n\
             {rule}"
        ),
        KeyModel::Unknown => format!(
            "{rule}\n\
             !! COULD NOT READ extraAgents — the key model is UNVERIFIED.\n\
             !!\n\
             !! The signer differs from the account, which is the right shape, but whether\n\
             !! the venue actually authorizes it is unknown. Do not treat this as a pass,\n\
             !! and do not re-run the approval ceremony on the strength of it: re-approving\n\
             !! the same name deregisters an agent that may be working perfectly.\n\
             {rule}"
        ),
    }
}

/// One line, printed last, so the verdict survives a long JSON dump above it.
fn status_line(model: &KeyModel) -> &'static str {
    match model {
        KeyModel::MasterKey => "STATUS: master key in use — NOT contained (see the block above)",
        KeyModel::ApprovedAgent { .. } => "STATUS: agent wallet, approved and valid",
        KeyModel::ExpiredAgent { .. } => "STATUS: agent wallet EXPIRED — it cannot sign",
        KeyModel::Unapproved => "STATUS: signer NOT approved for this account",
        KeyModel::Unknown => "STATUS: unverified — extraAgents could not be read",
    }
}

/// Pull `marginSummary.accountValue` out of a `clearinghouseState` body.
///
/// Parsed as a `Decimal`. This number gates whether an order can be placed at all,
/// and the repo's rule is that nothing on the money path goes through binary
/// floating point — an equity of `10.00` that compares as `9.999999999999998`
/// against the minimum notional is exactly the class of bug that rule exists for.
fn account_value(body: &str) -> Option<Decimal> {
    let v: serde_json::Value = serde_json::from_str(body).ok()?;
    v.get("marginSummary")?
        .get("accountValue")?
        .as_str()?
        .parse()
        .ok()
}

/// Whether this account can place anything at all.
fn funding_note(equity: Option<Decimal>) -> String {
    let minimum = Decimal::from(MIN_ORDER_NOTIONAL);
    match equity {
        Some(e) if e >= minimum => {
            format!("funding: accountValue {e} USDC — orders can be placed.")
        }
        Some(e) => format!(
            "funding: accountValue {e} USDC is under the venue's ${MIN_ORDER_NOTIONAL} minimum \
             order notional.\n         Nothing can be placed, so fill verification is BLOCKED \
             on a human funding\n         this account — see \
             this account at the venue's testnet faucet."
        ),
        None => "funding: could not read accountValue from clearinghouseState.".to_string(),
    }
}

fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

#[tokio::main]
async fn main() {
    let network = env::var("AXON_HL_NETWORK").unwrap_or_else(|_| "testnet".into());
    let is_mainnet = match network.as_str() {
        "mainnet" => true,
        "testnet" => false,
        other => {
            eprintln!("AXON_HL_NETWORK must be 'testnet' or 'mainnet', got {other:?}");
            std::process::exit(2);
        }
    };
    let info_url = if is_mainnet {
        MAINNET_INFO
    } else {
        TESTNET_INFO
    };

    let signer = match HlSigner::from_env(is_mainnet) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("{e}");
            eprintln!("hint: run via `bash scripts/with-env.sh` after filling in .env");
            std::process::exit(1);
        }
    };
    let signing_address = signer.address().to_string();

    println!("network          : {network}");
    println!("info endpoint    : {info_url}");
    println!("signing address  : {signing_address}  (derived from AXON_HL_SECRET_KEY)");

    // The account whose funds/positions the signer acts on. Equal to the signing
    // address when the key IS the master account; different under an agent wallet.
    let account = match env::var("AXON_HL_ACCOUNT_ADDRESS") {
        Ok(a) => {
            println!("account address  : {a}  (from AXON_HL_ACCOUNT_ADDRESS)");
            a
        }
        Err(_) => {
            println!("account address  : (unset) - assuming the signer is the master account");
            signing_address.clone()
        }
    };

    let now = now_ms();
    println!("\n--- approved agents (extraAgents) ---");
    let agents = match fetch_extra_agents(info_url, &account).await {
        Ok(a) => {
            print!("{}", render_agents(&a, &signing_address, now));
            Some(a)
        }
        Err(e) => {
            println!("  (read failed: {e})");
            None
        }
    };

    let model = classify(
        &signing_address,
        &account,
        agents.as_deref(),
        now * MS_TO_NS,
    );
    println!("\n{}\n", verdict(&model, now));

    let http = reqwest::Client::new();
    let mut equity = None;
    for (label, body) in [
        (
            "clearinghouseState",
            serde_json::json!({ "type": "clearinghouseState", "user": account }),
        ),
        (
            "openOrders",
            serde_json::json!({ "type": "openOrders", "user": account }),
        ),
    ] {
        println!("--- {label} ---");
        match http.post(info_url).json(&body).send().await {
            Ok(resp) => match resp.text().await {
                Ok(text) => {
                    if label == "clearinghouseState" {
                        equity = account_value(&text);
                    }
                    println!("{text}");
                }
                Err(e) => println!("(body read failed: {e})"),
            },
            Err(e) => println!("(request failed: {e})"),
        }
    }

    println!("\n{}", funding_note(equity));
    println!("{}", status_line(&model));
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    const ACCOUNT: &str = "0xB7c4D2e8F1a09B3c5D7e2F4a6B8c0D1e3F5a7B9c";
    const AGENT: &str = "0xa1b2c3d4e5f607182930a1b2c3d4e5f607182930";
    const NOW_MS: i64 = 1_784_977_518_165;

    fn agent(name: &str, address: &str, valid_until_ms: i64) -> ExtraAgent {
        ExtraAgent {
            name: name.to_string(),
            address: address.to_string(),
            valid_until: valid_until_ms * MS_TO_NS,
        }
    }

    fn live_agent() -> ExtraAgent {
        agent("axon valid_until 1793708044866", AGENT, 1_793_708_044_866)
    }

    // ── what the key actually is ──────────────────────────────────────────────

    #[test]
    fn an_identical_signer_and_account_is_reported_as_the_master_key() {
        // The state the repo is in today. It has to be detected from the addresses
        // alone: a master key is often *also* absent from extraAgents, which would
        // otherwise classify as merely "unapproved" and understate the problem.
        let m = classify(ACCOUNT, ACCOUNT, Some(&[]), NOW_MS * MS_TO_NS);
        assert_eq!(m, KeyModel::MasterKey);
        // Case differences in how the address was pasted must not hide it.
        let m = classify(&ACCOUNT.to_lowercase(), ACCOUNT, None, NOW_MS * MS_TO_NS);
        assert_eq!(m, KeyModel::MasterKey);
        // Nor does being listed as an agent make an account key contained.
        let self_listed = agent("weird", &ACCOUNT.to_lowercase(), NOW_MS + MS_PER_DAY);
        assert_eq!(
            classify(ACCOUNT, ACCOUNT, Some(&[self_listed]), NOW_MS * MS_TO_NS),
            KeyModel::MasterKey
        );
    }

    #[test]
    fn a_valid_approved_agent_is_the_only_state_that_reads_as_contained() {
        let m = classify(AGENT, ACCOUNT, Some(&[live_agent()]), NOW_MS * MS_TO_NS);
        assert_eq!(
            m,
            KeyModel::ApprovedAgent {
                name: "axon valid_until 1793708044866".into(),
                valid_until_ms: 1_793_708_044_866,
            }
        );
        assert!(verdict(&m, NOW_MS).starts_with("OK"));
        assert!(status_line(&m).contains("approved and valid"));
        // Every other state must be shouted, not mentioned.
        for other in [
            KeyModel::MasterKey,
            KeyModel::ExpiredAgent {
                name: "axon".into(),
                valid_until_ms: NOW_MS - MS_PER_DAY,
            },
            KeyModel::Unapproved,
            KeyModel::Unknown,
        ] {
            assert!(verdict(&other, NOW_MS).starts_with("!!!"), "{other:?}");
        }
    }

    #[test]
    fn an_expired_agent_is_not_mistaken_for_a_working_one() {
        // One millisecond past the deadline the venue stops accepting the signature,
        // so the boundary is where this has to be right.
        let expired = agent("axon", AGENT, NOW_MS);
        assert_eq!(
            classify(AGENT, ACCOUNT, Some(&[expired]), NOW_MS * MS_TO_NS),
            KeyModel::ExpiredAgent {
                name: "axon".into(),
                valid_until_ms: NOW_MS,
            }
        );
        let live = agent("axon", AGENT, NOW_MS + 1);
        assert!(matches!(
            classify(AGENT, ACCOUNT, Some(&[live]), NOW_MS * MS_TO_NS),
            KeyModel::ApprovedAgent { .. }
        ));
    }

    #[test]
    fn a_signer_absent_from_extra_agents_is_reported_as_unapproved() {
        let others = [agent(
            "someone-else",
            "0x1111111111111111111111111111111111111111",
            NOW_MS + MS_PER_DAY,
        )];
        assert_eq!(
            classify(AGENT, ACCOUNT, Some(&others), NOW_MS * MS_TO_NS),
            KeyModel::Unapproved
        );
        assert_eq!(
            classify(AGENT, ACCOUNT, Some(&[]), NOW_MS * MS_TO_NS),
            KeyModel::Unapproved
        );
    }

    #[test]
    fn an_unreadable_agent_list_is_unverified_rather_than_unapproved() {
        // Reporting "unapproved" after a failed /info read would send the operator
        // into the approval ceremony, and re-approving an existing name deregisters
        // the agent that was working.
        assert_eq!(
            classify(AGENT, ACCOUNT, None, NOW_MS * MS_TO_NS),
            KeyModel::Unknown
        );
        assert!(verdict(&KeyModel::Unknown, NOW_MS).contains("UNVERIFIED"));
    }

    #[test]
    fn the_master_key_verdict_cannot_be_skimmed_past() {
        let text = verdict(&KeyModel::MasterKey, NOW_MS);
        assert!(text.contains("*IS* THE MASTER ACCOUNT"));
        assert!(text.contains("WITHDRAW"));
        assert!(
            text.contains("DOES NOT"),
            "it must deny the property, not hedge it"
        );
        // It has to say what to do next, or it is just noise the operator learns to
        // scroll past.
        assert!(text.contains("approve_agent"));
        assert!(text.contains("re-run this check"));
        assert!(!status_line(&KeyModel::MasterKey).contains("OK"));
    }

    #[test]
    fn an_agent_close_to_expiry_is_flagged_while_it_still_works() {
        let soon = KeyModel::ApprovedAgent {
            name: "axon".into(),
            valid_until_ms: NOW_MS + 3 * MS_PER_DAY,
        };
        let text = verdict(&soon, NOW_MS);
        assert!(
            text.starts_with("OK"),
            "it is still valid, so it is still a pass"
        );
        assert!(text.contains("ROTATE SOON"));
        let far = KeyModel::ApprovedAgent {
            name: "axon".into(),
            valid_until_ms: NOW_MS + 90 * MS_PER_DAY,
        };
        assert!(!verdict(&far, NOW_MS).contains("ROTATE SOON"));
    }

    // ── rendering ─────────────────────────────────────────────────────────────

    #[test]
    fn validity_windows_are_rendered_relative_to_now() {
        assert_eq!(until(0, 0), "in 0d 0h 0m");
        assert_eq!(until(0, MS_PER_DAY + 3_600_000 + 60_000), "in 1d 1h 1m");
        assert_eq!(until(MS_PER_DAY, 0), "EXPIRED 1d 0h 0m ago");

        let table = render_agents(&[live_agent()], AGENT, NOW_MS);
        assert!(table.contains(AGENT));
        assert!(table.contains("valid until 1793708044866"));
        assert!(table.contains("THIS IS THE CONFIGURED SIGNER"));
        // An agent that is not ours must not be marked as ours.
        let table = render_agents(&[live_agent()], ACCOUNT, NOW_MS);
        assert!(!table.contains("THIS IS THE CONFIGURED SIGNER"));
        // The unnamed slot has an empty name on the wire; it still needs a label.
        let table = render_agents(&[agent("", AGENT, NOW_MS + 1)], AGENT, NOW_MS);
        assert!(table.contains("<unnamed>"));
        assert!(render_agents(&[], AGENT, NOW_MS).contains("never approved"));
    }

    // ── funding ───────────────────────────────────────────────────────────────

    #[test]
    fn an_unfunded_account_is_called_out_as_blocking_fill_verification() {
        // The live testnet payload captured on 2026-07-25 — the state this account
        // is actually in.
        let empty = r#"{"marginSummary":{"accountValue":"0.0","totalNtlPos":"0.0",
            "totalRawUsd":"0.0","totalMarginUsed":"0.0"},"withdrawable":"0.0",
            "assetPositions":[],"time":1784977518165}"#;
        let equity = account_value(empty);
        assert_eq!(equity, Some(Decimal::ZERO));
        let note = funding_note(equity);
        assert!(note.contains("BLOCKED"));
        assert!(note.contains("$10"));

        // Just under the minimum is still blocked; exactly at it is not.
        let at_min = funding_note(Some(Decimal::from(MIN_ORDER_NOTIONAL)));
        assert!(!at_min.contains("BLOCKED"));
        let under = funding_note(Some(Decimal::from_str("9.99").unwrap()));
        assert!(under.contains("BLOCKED"));
        // A body we cannot parse must not read as funded.
        assert!(funding_note(account_value("not json")).contains("could not read"));
        assert_eq!(account_value(r#"{"marginSummary":{}}"#), None);
    }

    #[test]
    fn account_value_is_parsed_as_a_decimal_not_a_float() {
        // 20 significant digits: f64 carries ~15–16, so a float round-trip would
        // silently change this number. Balances are money and money is `Decimal`.
        let body = r#"{"marginSummary":{"accountValue":"12345678901234567890.1"}}"#;
        assert_eq!(
            account_value(body),
            Some(Decimal::from_str("12345678901234567890.1").unwrap())
        );
    }
}
