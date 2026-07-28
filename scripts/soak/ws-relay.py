#!/usr/bin/env python3
"""A loopback TCP relay that can sever a WebSocket at will — the soak's outage lever.

Why this exists
---------------
`docs/08-roadmap.md` has owed a soak since Phase 2: reconnect and resubscribe are
implemented and unit-tested, and had never been *endured*. Enduring them needs real
disconnects, and the three obvious ways to get one on a developer box all fail:

- asking the client to simulate a disconnect tests the simulator, not the client;
- `iptables`/`ss -K` need CAP_NET_ADMIN, which the soak account does not have;
- killing the process is a restart, not a reconnect.

So the session is pointed at `ws://127.0.0.1:PORT/ws` via `venue.ws_url` (a documented
override — see `VenueConfig`) and this relay carries the bytes to
`wss://api.hyperliquid-testnet.xyz/ws`. Severing the loopback half drops the TCP
connection *underneath* the client's socket, which is the failure the reconnect loop
claims to survive, and nothing about the outage touches Hyperliquid: during a blackout
the client's retries land on this box's own loopback and never reach the venue.

What it deliberately does NOT do: it never rewrites, reorders or delays a WebSocket
*frame*. It is a byte pipe from the upgrade onwards. A relay that understood WebSocket
would be a second implementation of the thing under test.

The one exception is the `Host:` header of the HTTP upgrade, and it is not optional:
tungstenite derives `Host` from the URL, so pointing it at loopback sends
`Host: 127.0.0.1:8765` and Hyperliquid's edge answers **403 Forbidden** before a socket
is ever established — measured, not assumed. Rewriting that single header (and only
inside the request head, never past the blank line) is what makes the relayed session
the same session. Everything after `\r\n\r\n` is untouched bytes.

Modes, and the failure each one is aimed at
-------------------------------------------
- ``open``   — relay normally.
- ``cut``    — **close the listening socket** and force-close every live connection.
               The port then answers with RST, so the client sees ``ECONNREFUSED``:
               a hard, honest outage that exercises the error path and the capped
               backoff. Not "accept and go quiet" — that variant hangs
               ``connect_async``, which has no handshake timeout, and would end the
               soak rather than test it.
- ``churn``  — relay normally but close each connection ``life_ms`` after it opens,
               *after* the WS handshake and the subscribe frames have gone through.
               This is the case `reconnect_forever` treats as a **clean** close, and
               the gap between one connection closing and the next opening is the
               measurement that says whether a clean close is backed off at all.

Control is a one-line text file polled every 50 ms, so a soak driver can change the
weather without a signal, a port or a client library.

    open
    cut <until_unix_epoch_seconds>
    churn <until_unix_epoch_seconds> <life_ms>

Every connection is journalled to a JSONL file with wall and monotonic clocks, so the
count of reconnects is measured by something other than the process being tested.

Usage:
    ws-relay.py --port 8765 --upstream api.hyperliquid-testnet.xyz:443 \
                --control /dev/shm/m7-relay.ctl --journal /tmp/m7-relay.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import select
import socket
import ssl
import struct
import sys
import threading
import time

# Force-close with RST rather than FIN. A FIN is a *clean* close, which is a different
# outage: the client would see the stream end normally. RST is what a severed path
# looks like, and keeping the two distinguishable is the entire point of `churn`
# existing separately from `cut`.
LINGER_RST = struct.pack("ii", 1, 0)

BUF = 65536


class Journal:
    """Append-only JSONL, flushed per line.

    Per line because the interesting record is always the last one before something
    went wrong, and a buffered journal loses exactly that.
    """

    def __init__(self, path: str):
        self.f = open(path, "a", buffering=1)
        self.lock = threading.Lock()

    def write(self, **kw):
        kw["wall"] = time.time()
        kw["mono"] = time.monotonic()
        with self.lock:
            self.f.write(json.dumps(kw) + "\n")


class Control:
    """The weather. Polled from a file so nothing needs to be signalled or dialled."""

    def __init__(self, path: str):
        self.path = path
        self.mode = "open"
        self.until = 0.0
        self.life_ms = 0
        self._mtime = 0.0

    def poll(self) -> str:
        try:
            st = os.stat(self.path)
        except FileNotFoundError:
            return self.effective()
        if st.st_mtime != self._mtime:
            self._mtime = st.st_mtime
            try:
                parts = open(self.path).read().split()
            except OSError:
                return self.effective()
            if parts:
                self.mode = parts[0]
                self.until = float(parts[1]) if len(parts) > 1 else 0.0
                self.life_ms = int(parts[2]) if len(parts) > 2 else 0
        return self.effective()

    def effective(self) -> str:
        # A mode with an expired deadline decays to `open` on its own. A soak driver
        # that died mid-outage must not leave the session permanently disconnected —
        # the artifact would then be a recording of a broken harness.
        if self.mode != "open" and self.until and time.time() >= self.until:
            self.mode, self.until, self.life_ms = "open", 0.0, 0
        return self.mode


class Relay:
    def __init__(self, args):
        self.port = args.port
        host, _, port = args.upstream.partition(":")
        self.up_host, self.up_port = host, int(port or 443)
        self.control = Control(args.control)
        self.journal = Journal(args.journal)
        self.ctx = ssl.create_default_context()
        self.listener: socket.socket | None = None
        self.conns: set[socket.socket] = set()
        self.conn_lock = threading.Lock()
        self.n = 0
        self.stop = threading.Event()

    # -- listener ---------------------------------------------------------------
    def open_listener(self):
        if self.listener is not None:
            return
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", self.port))
        s.listen(16)
        s.settimeout(0.2)
        self.listener = s
        self.journal.write(ev="listener_open", port=self.port)

    def close_listener(self):
        if self.listener is None:
            return
        try:
            self.listener.close()
        finally:
            self.listener = None
        self.journal.write(ev="listener_closed", port=self.port)

    def kill_conns(self, why: str):
        with self.conn_lock:
            victims = list(self.conns)
            self.conns.clear()
        for c in victims:
            try:
                c.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, LINGER_RST)
                c.close()
            except OSError:
                pass
        if victims:
            self.journal.write(ev="conns_killed", n=len(victims), why=why)

    # -- one connection ---------------------------------------------------------
    def serve(self, client: socket.socket, cid: int, life_ms: int):
        opened = time.monotonic()
        up = None
        try:
            raw = socket.create_connection((self.up_host, self.up_port), timeout=10)
            up = self.ctx.wrap_socket(raw, server_hostname=self.up_host)
            up.settimeout(None)
            self.journal.write(ev="upstream_open", cid=cid)
        except Exception as e:  # noqa: BLE001 - report anything, never crash the relay
            self.journal.write(ev="upstream_failed", cid=cid, err=repr(e))
            try:
                client.close()
            except OSError:
                pass
            return

        with self.conn_lock:
            self.conns.add(client)
            self.conns.add(up)

        c_to_u = u_to_c = 0
        deadline = opened + life_ms / 1000.0 if life_ms else None
        try:
            # The HTTP upgrade head, and only it. Read until the blank line, swap the
            # loopback `Host` for the one the venue's edge will accept, forward. A
            # `Host: 127.0.0.1:8765` is answered with 403 before any WebSocket exists,
            # so without this the relay tests nothing but the error path.
            client.settimeout(10)
            head = b""
            while b"\r\n\r\n" not in head:
                chunk = client.recv(BUF)
                if not chunk:
                    raise StopIteration
                head += chunk
                if len(head) > 65536:
                    raise StopIteration
            client.settimeout(None)
            sep = head.index(b"\r\n\r\n") + 4
            lines = head[:sep].split(b"\r\n")
            for i, ln in enumerate(lines):
                if ln.lower().startswith(b"host:"):
                    lines[i] = b"Host: " + self.up_host.encode()
            rewritten = b"\r\n".join(lines) + head[sep:]
            c_to_u += len(head)
            up.sendall(rewritten)
            # Both sockets stay **blocking**, and `select` is used only to ask which one
            # has something to read. The soak also freezes the session with SIGSTOP, and
            # a frozen client stops draining its receive buffer: with non-blocking
            # sockets the relay's `sendall` would raise and tear the connection down, so
            # every process freeze would arrive as a relay-induced disconnect and the two
            # experiments could never be told apart.
            while not self.stop.is_set():
                if deadline and time.monotonic() >= deadline:
                    # A *clean* close: FIN in both directions once the handshake and
                    # the subscribe frames have long since gone through. This is the
                    # arm that tells us whether `run_once` returning Ok is backed off.
                    self.journal.write(ev="churn_close", cid=cid)
                    break
                r, _, x = select.select([client, up], [], [client, up], 0.2)
                if x:
                    break
                for s in r:
                    try:
                        data = s.recv(BUF)
                    except ssl.SSLWantReadError:
                        continue
                    except OSError:
                        data = b""
                    if not data:
                        raise StopIteration
                    if s is client:
                        c_to_u += len(data)
                        up.sendall(data)
                    else:
                        u_to_c += len(data)
                        client.sendall(data)
                # `up` is TLS: recv() can leave decrypted bytes buffered that select()
                # will never report, because select watches the *socket*, not the SSL
                # object. Draining pending() is the difference between a working relay
                # and one that stalls on a burst.
                while up.pending():
                    data = up.recv(BUF)
                    if not data:
                        raise StopIteration
                    u_to_c += len(data)
                    client.sendall(data)
        except StopIteration:
            pass
        except Exception as e:  # noqa: BLE001
            self.journal.write(ev="relay_error", cid=cid, err=repr(e))
        finally:
            with self.conn_lock:
                self.conns.discard(client)
                self.conns.discard(up)
            for s in (client, up):
                try:
                    s.close()
                except OSError:
                    pass
            self.journal.write(
                ev="conn_closed",
                cid=cid,
                secs=round(time.monotonic() - opened, 3),
                up_bytes=u_to_c,
                down_bytes=c_to_u,
            )

    # -- main loop --------------------------------------------------------------
    def run(self):
        mode = "open"
        self.open_listener()
        self.journal.write(ev="relay_up", upstream=f"{self.up_host}:{self.up_port}")
        while not self.stop.is_set():
            new = self.control.poll()
            if new != mode:
                self.journal.write(ev="mode", frm=mode, to=new)
                mode = new
                if mode == "cut":
                    self.close_listener()
                    self.kill_conns("cut")
                else:
                    self.open_listener()
                    if mode == "churn":
                        # Entering churn has to end the connection that is already up,
                        # or the mode is a no-op: `life_ms` is stamped on a connection
                        # when it is *accepted*, so a session that never reconnects
                        # during the window is never churned. Found by a churn window
                        # passing with the session none the wiser — a harness bug that
                        # would have been reported as "clean closes were not observed".
                        self.kill_conns("churn-entry")
            if mode == "cut" or self.listener is None:
                time.sleep(0.05)
                continue
            try:
                client, _ = self.listener.accept()
            except (socket.timeout, TimeoutError):
                continue
            except OSError:
                continue
            self.n += 1
            cid = self.n
            client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.journal.write(ev="conn_open", cid=cid, mode=mode)
            life = self.control.life_ms if mode == "churn" else 0
            threading.Thread(
                target=self.serve, args=(client, cid, life), daemon=True
            ).start()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--upstream", default="api.hyperliquid-testnet.xyz:443")
    ap.add_argument("--control", default="/dev/shm/m7-relay.ctl")
    ap.add_argument("--journal", default="/dev/shm/m7-relay.jsonl")
    # So a relay restarted mid-soak keeps appending to one journal without reusing
    # connection ids. A soak's journal is the only independent record of how many
    # reconnects happened; restarting the recorder and starting the numbering again
    # would silently merge two connections into one row.
    ap.add_argument("--cid-base", type=int, default=0)
    args = ap.parse_args()
    if not os.path.exists(args.control):
        with open(args.control, "w") as f:
            f.write("open\n")
    relay = Relay(args)
    relay.n = args.cid_base
    relay.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
