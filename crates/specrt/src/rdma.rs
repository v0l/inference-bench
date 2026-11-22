use anyhow::{Context, Result, bail, ensure};
use libloading::Library;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::ffi::{CStr, c_char, c_int, c_void};
use std::net::{IpAddr, TcpStream};
use std::os::fd::AsRawFd;
use std::ptr::{null, null_mut};
use std::sync::Arc;
use std::time::{Duration, Instant};

pub const CHUNK: usize = 256 << 10;
pub const DEFAULT_LIBRARY: &str = "libibverbs.so.1";

const LAST: u32 = 1 << 31;
const CREDIT: u32 = 1 << 30;
const LEN_MASK: u32 = CREDIT - 1;
const RECV_DEPTH: usize = 64;
const SEND_DEPTH: usize = 64;
const CQ_DEPTH: i32 = 256;
const WR_DATA: u64 = 1;
const WR_CREDIT: u64 = 2;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Transport {
    #[default]
    Tcp,
    Rdma,
    Auto,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct LinkOptions {
    pub transport: Transport,
    pub library: Option<String>,
}

impl LinkOptions {
    pub fn library(&self) -> &str {
        self.library.as_deref().unwrap_or(DEFAULT_LIBRARY)
    }
}

#[derive(Debug)]
pub struct PeerClosed;

impl std::fmt::Display for PeerClosed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "peer closed the connection")
    }
}

impl std::error::Error for PeerClosed {}

#[derive(Clone, Debug, PartialEq)]
pub struct ConnInfo {
    pub qpn: u32,
    pub psn: u32,
    pub gid: [u8; 16],
    pub addr: u64,
    pub rkey: u32,
}

#[repr(C)]
#[derive(Default, Clone, Copy)]
struct QpCap {
    max_send_wr: u32,
    max_recv_wr: u32,
    max_send_sge: u32,
    max_recv_sge: u32,
    max_inline_data: u32,
}

#[repr(C)]
struct QpInitAttr {
    qp_context: *mut c_void,
    send_cq: *mut c_void,
    recv_cq: *mut c_void,
    srq: *mut c_void,
    cap: QpCap,
    qp_type: u32,
    sq_sig_all: c_int,
}

#[repr(C)]
#[derive(Default, Clone, Copy)]
struct AhAttr {
    dgid: [u8; 16],
    flow_label: u32,
    sgid_index: u8,
    hop_limit: u8,
    traffic_class: u8,
    dlid: u16,
    sl: u8,
    src_path_bits: u8,
    static_rate: u8,
    is_global: u8,
    port_num: u8,
}

#[repr(C)]
#[derive(Default)]
struct QpAttr {
    qp_state: u32,
    cur_qp_state: u32,
    path_mtu: u32,
    path_mig_state: u32,
    qkey: u32,
    rq_psn: u32,
    sq_psn: u32,
    dest_qp_num: u32,
    qp_access_flags: u32,
    cap: QpCap,
    ah_attr: AhAttr,
    alt_ah_attr: AhAttr,
    pkey_index: u16,
    alt_pkey_index: u16,
    en_sqd_async_notify: u8,
    sq_draining: u8,
    max_rd_atomic: u8,
    max_dest_rd_atomic: u8,
    min_rnr_timer: u8,
    port_num: u8,
    timeout: u8,
    retry_cnt: u8,
    rnr_retry: u8,
    alt_port_num: u8,
    alt_timeout: u8,
    rate_limit: u32,
    pad: u32,
}

#[repr(C)]
struct Sge {
    addr: u64,
    length: u32,
    lkey: u32,
}

#[repr(C)]
struct SendWr {
    wr_id: u64,
    next: *mut SendWr,
    sg_list: *mut Sge,
    num_sge: c_int,
    opcode: u32,
    send_flags: u32,
    imm: u32,
    remote_addr: u64,
    rkey: u32,
    tail: [u8; 76],
}

#[repr(C)]
struct RecvWr {
    wr_id: u64,
    next: *mut RecvWr,
    sg_list: *mut Sge,
    num_sge: c_int,
}

#[repr(C)]
#[derive(Default, Clone, Copy)]
struct Wc {
    wr_id: u64,
    status: u32,
    opcode: u32,
    vendor_err: u32,
    byte_len: u32,
    imm: u32,
    qp_num: u32,
    src_qp: u32,
    wc_flags: u32,
    pkey_index: u16,
    slid: u16,
    sl: u8,
    dlid_path_bits: u8,
}

const OFF_CTX_POLL_CQ: usize = 8 + 88;
const OFF_CTX_POST_SEND: usize = 8 + 200;
const OFF_CTX_POST_RECV: usize = 8 + 208;
const OFF_MR_LKEY: usize = 36;
const OFF_MR_RKEY: usize = 40;
const OFF_QP_NUM: usize = 52;

type PollCq = unsafe extern "C" fn(*mut c_void, c_int, *mut Wc) -> c_int;
type PostSend = unsafe extern "C" fn(*mut c_void, *mut SendWr, *mut *mut SendWr) -> c_int;
type PostRecv = unsafe extern "C" fn(*mut c_void, *mut RecvWr, *mut *mut RecvWr) -> c_int;

unsafe fn ctx_fn<T: Copy>(object: *mut c_void, offset: usize) -> T {
    unsafe {
        let ctx = *(object as *const *mut u8);
        *(ctx.add(offset) as *const T)
    }
}

unsafe fn field_u32(object: *mut c_void, offset: usize) -> u32 {
    unsafe { *((object as *const u8).add(offset) as *const u32) }
}

pub struct Verbs {
    _lib: Library,
    get_device_list: unsafe extern "C" fn(*mut c_int) -> *mut *mut c_void,
    free_device_list: unsafe extern "C" fn(*mut *mut c_void),
    get_device_name: unsafe extern "C" fn(*mut c_void) -> *const c_char,
    open_device: unsafe extern "C" fn(*mut c_void) -> *mut c_void,
    close_device: unsafe extern "C" fn(*mut c_void) -> c_int,
    alloc_pd: unsafe extern "C" fn(*mut c_void) -> *mut c_void,
    dealloc_pd: unsafe extern "C" fn(*mut c_void) -> c_int,
    reg_mr: unsafe extern "C" fn(*mut c_void, *mut c_void, usize, c_int) -> *mut c_void,
    dereg_mr: unsafe extern "C" fn(*mut c_void) -> c_int,
    create_cq: unsafe extern "C" fn(*mut c_void, c_int, *mut c_void, *mut c_void, c_int) -> *mut c_void,
    destroy_cq: unsafe extern "C" fn(*mut c_void) -> c_int,
    create_qp: unsafe extern "C" fn(*mut c_void, *mut QpInitAttr) -> *mut c_void,
    modify_qp: unsafe extern "C" fn(*mut c_void, *mut QpAttr, c_int) -> c_int,
    destroy_qp: unsafe extern "C" fn(*mut c_void) -> c_int,
    query_port: unsafe extern "C" fn(*mut c_void, u8, *mut [u8; 56]) -> c_int,
    query_gid: unsafe extern "C" fn(*mut c_void, u8, c_int, *mut [u8; 16]) -> c_int,
    reg_dmabuf_mr: Option<unsafe extern "C" fn(*mut c_void, u64, usize, u64, c_int, c_int) -> *mut c_void>,
}

impl Verbs {
    pub fn load(name: &str) -> Result<Arc<Verbs>> {
        unsafe {
            let lib = Library::new(name).with_context(|| format!("load {name}"))?;
            macro_rules! sym {
                ($n:literal) => {
                    *lib.get(concat!($n, "\0").as_bytes()).with_context(|| format!("{name} has no {}", $n))?
                };
            }
            let reg_dmabuf_mr = lib.get(b"ibv_reg_dmabuf_mr\0").ok().map(|s: libloading::Symbol<unsafe extern "C" fn(*mut c_void, u64, usize, u64, c_int, c_int) -> *mut c_void>| *s);
            let verbs = Verbs {
                get_device_list: sym!("ibv_get_device_list"),
                free_device_list: sym!("ibv_free_device_list"),
                get_device_name: sym!("ibv_get_device_name"),
                open_device: sym!("ibv_open_device"),
                close_device: sym!("ibv_close_device"),
                alloc_pd: sym!("ibv_alloc_pd"),
                dealloc_pd: sym!("ibv_dealloc_pd"),
                reg_mr: sym!("ibv_reg_mr"),
                dereg_mr: sym!("ibv_dereg_mr"),
                create_cq: sym!("ibv_create_cq"),
                destroy_cq: sym!("ibv_destroy_cq"),
                create_qp: sym!("ibv_create_qp"),
                modify_qp: sym!("ibv_modify_qp"),
                destroy_qp: sym!("ibv_destroy_qp"),
                query_port: sym!("ibv_query_port"),
                query_gid: sym!("ibv_query_gid"),
                reg_dmabuf_mr,
                _lib: lib,
            };
            Ok(Arc::new(verbs))
        }
    }

    pub fn supports_dmabuf(&self) -> bool {
        self.reg_dmabuf_mr.is_some()
    }
}

struct Selected {
    ctx: *mut c_void,
    name: String,
    port: u8,
    gid_index: i32,
    gid: [u8; 16],
    mtu: u32,
}

fn mapped_gid(ip: IpAddr) -> [u8; 16] {
    match ip {
        IpAddr::V4(v4) => v4.to_ipv6_mapped().octets(),
        IpAddr::V6(v6) => v6.octets(),
    }
}

fn is_ipv4_mapped(gid: &[u8; 16]) -> bool {
    gid[..10] == [0; 10] && gid[10] == 0xff && gid[11] == 0xff && gid[12] != 127
}

fn select_device(verbs: &Verbs, local: IpAddr) -> Result<Selected> {
    unsafe {
        let mut count: c_int = 0;
        let list = (verbs.get_device_list)(&mut count);
        ensure!(!list.is_null() && count > 0, "no rdma devices");
        let wanted = mapped_gid(local);
        let mut best: Option<(Selected, bool)> = None;
        for i in 0..count as usize {
            let dev = *list.add(i);
            let ctx = (verbs.open_device)(dev);
            if ctx.is_null() {
                continue;
            }
            let name = CStr::from_ptr((verbs.get_device_name)(dev)).to_string_lossy().into_owned();
            let mut keep = false;
            for port in 1..=2u8 {
                let mut attr = [0u8; 56];
                if (verbs.query_port)(ctx, port, &mut attr) != 0 {
                    break;
                }
                let state = u32::from_ne_bytes(attr[0..4].try_into().unwrap());
                let mtu = u32::from_ne_bytes(attr[8..12].try_into().unwrap());
                let gids = i32::from_ne_bytes(attr[12..16].try_into().unwrap());
                if state != 4 || attr[46] != 2 {
                    continue;
                }
                for index in 0..gids {
                    let mut gid = [0u8; 16];
                    if (verbs.query_gid)(ctx, port, index, &mut gid) != 0 {
                        continue;
                    }
                    let exact = gid == wanted;
                    if !exact && !is_ipv4_mapped(&gid) {
                        continue;
                    }
                    let better = match &best {
                        None => true,
                        Some((_, was_exact)) => exact || !was_exact,
                    };
                    if better && (exact || best.as_ref().is_none_or(|(_, e)| !e)) {
                        if let Some((old, _)) = best.take() {
                            if old.ctx != ctx {
                                (verbs.close_device)(old.ctx);
                            }
                        }
                        best = Some((Selected { ctx, name: name.clone(), port, gid_index: index, gid, mtu }, exact));
                        keep = true;
                    }
                }
            }
            if !keep {
                (verbs.close_device)(ctx);
            }
        }
        (verbs.free_device_list)(list);
        match best {
            Some((s, _)) => Ok(s),
            None => bail!("no active RoCE port with an IPv4 address"),
        }
    }
}

pub struct Endpoint {
    verbs: Arc<Verbs>,
    ctx: *mut c_void,
    pd: *mut c_void,
    cq: *mut c_void,
    qp: *mut c_void,
    send_mr: *mut c_void,
    recv_mr: *mut c_void,
    send_buf: Box<[u8]>,
    recv_buf: Box<[u8]>,
    port: u8,
    gid_index: i32,
    mtu: u32,
    pub device: String,
    local: ConnInfo,
}

unsafe impl Send for Endpoint {}

impl Endpoint {
    pub fn open(verbs: Arc<Verbs>, local_ip: IpAddr) -> Result<Endpoint> {
        let sel = select_device(&verbs, local_ip)?;
        unsafe {
            let mut ep = Endpoint {
                verbs: verbs.clone(),
                ctx: sel.ctx,
                pd: null_mut(),
                cq: null_mut(),
                qp: null_mut(),
                send_mr: null_mut(),
                recv_mr: null_mut(),
                send_buf: vec![0u8; CHUNK].into_boxed_slice(),
                recv_buf: vec![0u8; CHUNK].into_boxed_slice(),
                port: sel.port,
                gid_index: sel.gid_index,
                mtu: sel.mtu,
                device: sel.name,
                local: ConnInfo { qpn: 0, psn: 0, gid: sel.gid, addr: 0, rkey: 0 },
            };
            ep.pd = (verbs.alloc_pd)(ep.ctx);
            ensure!(!ep.pd.is_null(), "ibv_alloc_pd failed");
            ep.send_mr = (verbs.reg_mr)(ep.pd, ep.send_buf.as_mut_ptr() as *mut c_void, CHUNK, 1);
            ensure!(!ep.send_mr.is_null(), "ibv_reg_mr failed for the send buffer (is the memlock limit high enough?)");
            ep.recv_mr = (verbs.reg_mr)(ep.pd, ep.recv_buf.as_mut_ptr() as *mut c_void, CHUNK, 1 | 2);
            ensure!(!ep.recv_mr.is_null(), "ibv_reg_mr failed for the receive buffer (is the memlock limit high enough?)");
            ep.cq = (verbs.create_cq)(ep.ctx, CQ_DEPTH, null_mut(), null_mut(), 0);
            ensure!(!ep.cq.is_null(), "ibv_create_cq failed");
            let mut init = QpInitAttr {
                qp_context: null_mut(),
                send_cq: ep.cq,
                recv_cq: ep.cq,
                srq: null_mut(),
                cap: QpCap { max_send_wr: SEND_DEPTH as u32, max_recv_wr: RECV_DEPTH as u32, max_send_sge: 1, max_recv_sge: 1, max_inline_data: 0 },
                qp_type: 2,
                sq_sig_all: 0,
            };
            ep.qp = (verbs.create_qp)(ep.pd, &mut init);
            ensure!(!ep.qp.is_null(), "ibv_create_qp failed");
            let mut attr = QpAttr { qp_state: 1, pkey_index: 0, port_num: ep.port, qp_access_flags: 2, ..Default::default() };
            let rc = (verbs.modify_qp)(ep.qp, &mut attr, 1 | 16 | 32 | 8);
            ensure!(rc == 0, "modify qp to INIT failed ({rc})");
            for _ in 0..RECV_DEPTH {
                ep.post_recv()?;
            }
            let mut psn = [0u8; 4];
            psn.copy_from_slice(&crate::pairing::random(4)?);
            ep.local.qpn = field_u32(ep.qp, OFF_QP_NUM);
            ep.local.psn = u32::from_le_bytes(psn) & 0xff_ffff;
            ep.local.addr = ep.recv_buf.as_ptr() as u64;
            ep.local.rkey = field_u32(ep.recv_mr, OFF_MR_RKEY);
            Ok(ep)
        }
    }

    pub fn info(&self) -> ConnInfo {
        self.local.clone()
    }

    pub fn connect(&mut self, remote: &ConnInfo) -> Result<()> {
        unsafe {
            let mut attr = QpAttr {
                qp_state: 2,
                path_mtu: self.mtu.max(1),
                dest_qp_num: remote.qpn,
                rq_psn: remote.psn,
                max_dest_rd_atomic: 1,
                min_rnr_timer: 12,
                ah_attr: AhAttr { dgid: remote.gid, sgid_index: self.gid_index as u8, hop_limit: 64, is_global: 1, port_num: self.port, ..Default::default() },
                ..Default::default()
            };
            let rc = (self.verbs.modify_qp)(self.qp, &mut attr, 1 | 128 | 256 | 1048576 | 4096 | 131072 | 32768);
            ensure!(rc == 0, "modify qp to RTR failed ({rc})");
            let mut attr = QpAttr { qp_state: 3, sq_psn: self.local.psn, timeout: 14, retry_cnt: 7, rnr_retry: 7, max_rd_atomic: 1, ..Default::default() };
            let rc = (self.verbs.modify_qp)(self.qp, &mut attr, 1 | 512 | 1024 | 2048 | 65536 | 8192);
            ensure!(rc == 0, "modify qp to RTS failed ({rc})");
        }
        Ok(())
    }

    fn post_recv(&mut self) -> Result<()> {
        unsafe {
            let mut wr = RecvWr { wr_id: 3, next: null_mut(), sg_list: null_mut(), num_sge: 0 };
            let mut bad: *mut RecvWr = null_mut();
            let post: PostRecv = ctx_fn(self.qp, OFF_CTX_POST_RECV);
            let rc = post(self.qp, &mut wr, &mut bad);
            ensure!(rc == 0, "ibv_post_recv failed ({rc})");
        }
        Ok(())
    }

    fn post_write(&mut self, wr_id: u64, len: usize, imm: u32, remote: &ConnInfo) -> Result<()> {
        unsafe {
            let mut sge = Sge { addr: self.send_buf.as_ptr() as u64, length: len as u32, lkey: field_u32(self.send_mr, OFF_MR_LKEY) };
            let mut wr = SendWr {
                wr_id,
                next: null_mut(),
                sg_list: if len == 0 { null_mut() } else { &mut sge },
                num_sge: if len == 0 { 0 } else { 1 },
                opcode: 1,
                send_flags: 2,
                imm: imm.to_be(),
                remote_addr: remote.addr,
                rkey: remote.rkey,
                tail: [0; 76],
            };
            let mut bad: *mut SendWr = null_mut();
            let post: PostSend = ctx_fn(self.qp, OFF_CTX_POST_SEND);
            let rc = post(self.qp, &mut wr, &mut bad);
            ensure!(rc == 0, "ibv_post_send failed ({rc})");
        }
        Ok(())
    }

    fn poll(&mut self, out: &mut [Wc]) -> Result<usize> {
        unsafe {
            let poll: PollCq = ctx_fn(self.cq, OFF_CTX_POLL_CQ);
            let n = poll(self.cq, out.len() as c_int, out.as_mut_ptr());
            ensure!(n >= 0, "ibv_poll_cq failed ({n})");
            Ok(n as usize)
        }
    }

    pub fn register_dmabuf(&self, fd: i32, offset: u64, len: usize, iova: u64) -> Result<*mut c_void> {
        let f = self.verbs.reg_dmabuf_mr.context("this libibverbs has no ibv_reg_dmabuf_mr")?;
        let mr = unsafe { f(self.pd, offset, len, iova, fd, 1 | 2) };
        ensure!(!mr.is_null(), "ibv_reg_dmabuf_mr failed");
        Ok(mr)
    }
}

impl Drop for Endpoint {
    fn drop(&mut self) {
        unsafe {
            if !self.qp.is_null() {
                (self.verbs.destroy_qp)(self.qp);
            }
            if !self.cq.is_null() {
                (self.verbs.destroy_cq)(self.cq);
            }
            if !self.send_mr.is_null() {
                (self.verbs.dereg_mr)(self.send_mr);
            }
            if !self.recv_mr.is_null() {
                (self.verbs.dereg_mr)(self.recv_mr);
            }
            if !self.pd.is_null() {
                (self.verbs.dealloc_pd)(self.pd);
            }
            if !self.ctx.is_null() {
                (self.verbs.close_device)(self.ctx);
            }
        }
    }
}

pub struct RdmaChannel {
    ep: Endpoint,
    remote: ConnInfo,
    tcp: TcpStream,
    credits: u32,
    inflight: usize,
    partial: Vec<u8>,
    frames: VecDeque<Vec<u8>>,
    max_frame: usize,
}

impl RdmaChannel {
    pub fn new(ep: Endpoint, remote: ConnInfo, tcp: TcpStream, max_frame: usize) -> RdmaChannel {
        RdmaChannel { ep, remote, tcp, credits: 1, inflight: 0, partial: Vec::new(), frames: VecDeque::new(), max_frame }
    }

    pub fn device(&self) -> &str {
        &self.ep.device
    }

    fn peer_alive(&self) -> bool {
        let mut byte = 0u8;
        let n = unsafe { libc::recv(self.tcp.as_raw_fd(), &mut byte as *mut u8 as *mut c_void, 1, libc::MSG_PEEK | libc::MSG_DONTWAIT) };
        if n == 0 {
            return false;
        }
        if n < 0 {
            let e = std::io::Error::last_os_error();
            return matches!(e.kind(), std::io::ErrorKind::WouldBlock | std::io::ErrorKind::Interrupted);
        }
        true
    }

    fn drain(&mut self) -> Result<bool> {
        let mut wcs = [Wc::default(); 16];
        let n = self.ep.poll(&mut wcs)?;
        for wc in &wcs[..n] {
            ensure!(wc.status == 0, "rdma completion failed with status {} (vendor {}): the peer is unreachable", wc.status, wc.vendor_err);
            match wc.opcode {
                1 | 0 => self.inflight = self.inflight.saturating_sub(1),
                129 => {
                    self.ep.post_recv()?;
                    let imm = u32::from_be(wc.imm);
                    if imm & CREDIT != 0 {
                        self.credits += 1;
                        continue;
                    }
                    let len = (imm & LEN_MASK) as usize;
                    ensure!(len <= CHUNK, "rdma chunk of {len} bytes exceeds the buffer");
                    ensure!(self.partial.len() + len <= self.max_frame, "frame exceeds the {} byte cap", self.max_frame);
                    self.partial.extend_from_slice(&self.ep.recv_buf[..len]);
                    self.post_credit()?;
                    if imm & LAST != 0 {
                        self.frames.push_back(std::mem::take(&mut self.partial));
                    }
                }
                other => bail!("unexpected rdma completion opcode {other}"),
            }
        }
        Ok(n > 0)
    }

    fn post_credit(&mut self) -> Result<()> {
        while self.inflight >= SEND_DEPTH - 8 {
            self.drain()?;
        }
        let remote = self.remote.clone();
        self.ep.post_write(WR_CREDIT, 0, CREDIT, &remote)?;
        self.inflight += 1;
        Ok(())
    }

    fn wait<F: FnMut(&mut Self) -> bool>(&mut self, timeout: Duration, mut done: F) -> Result<bool> {
        let started = Instant::now();
        let mut last_check = started;
        loop {
            self.drain()?;
            if done(self) {
                return Ok(true);
            }
            let waited = started.elapsed();
            if waited >= timeout {
                return Ok(false);
            }
            if waited < Duration::from_micros(100) {
                std::hint::spin_loop();
                continue;
            }
            if last_check.elapsed() >= Duration::from_millis(20) {
                last_check = Instant::now();
                if !self.peer_alive() {
                    return Err(PeerClosed.into());
                }
            }
            std::thread::sleep(if waited < Duration::from_millis(100) { Duration::from_micros(20) } else { Duration::from_micros(200) });
        }
    }

    pub fn send(&mut self, body: &[u8], timeout: Duration) -> Result<()> {
        ensure!(body.len() <= self.max_frame, "frame of {} bytes exceeds the {} byte cap", body.len(), self.max_frame);
        let chunks: Vec<&[u8]> = if body.is_empty() { vec![&[][..]] } else { body.chunks(CHUNK).collect() };
        let last = chunks.len() - 1;
        for (i, chunk) in chunks.into_iter().enumerate() {
            let ready = self.wait(timeout, |c| c.credits > 0 && c.inflight == 0)?;
            ensure!(ready, "timed out waiting for the peer to take the previous chunk");
            self.credits -= 1;
            self.ep.send_buf[..chunk.len()].copy_from_slice(chunk);
            let imm = chunk.len() as u32 | if i == last { LAST } else { 0 };
            let remote = self.remote.clone();
            self.ep.post_write(WR_DATA, chunk.len(), imm, &remote)?;
            self.inflight += 1;
        }
        Ok(())
    }

    pub fn recv(&mut self, timeout: Duration) -> Result<Option<Vec<u8>>> {
        let got = self.wait(timeout, |c| !c.frames.is_empty())?;
        Ok(if got { self.frames.pop_front() } else { None })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{offset_of, size_of};

    #[test]
    fn abi_layout_matches_libibverbs() {
        assert_eq!(size_of::<SendWr>(), 128);
        assert_eq!(offset_of!(SendWr, imm), 36);
        assert_eq!(offset_of!(SendWr, remote_addr), 40);
        assert_eq!(offset_of!(SendWr, rkey), 48);
        assert_eq!(size_of::<RecvWr>(), 32);
        assert_eq!(size_of::<Wc>(), 48);
        assert_eq!(offset_of!(Wc, wc_flags), 36);
        assert_eq!(size_of::<QpInitAttr>(), 64);
        assert_eq!(size_of::<AhAttr>(), 32);
        assert_eq!(size_of::<QpAttr>(), 144);
        assert_eq!(offset_of!(QpAttr, ah_attr), 56);
        assert_eq!(offset_of!(QpAttr, pkey_index), 120);
        assert_eq!(offset_of!(QpAttr, max_rd_atomic), 126);
        assert_eq!(offset_of!(QpAttr, rate_limit), 136);
    }

    #[test]
    fn missing_library_is_an_error_not_a_panic() {
        assert!(Verbs::load("libdefinitely-not-ibverbs.so.9").is_err());
    }

    #[test]
    fn mapped_gids_match_ipv4() {
        let gid = mapped_gid("10.100.50.1".parse().unwrap());
        assert!(is_ipv4_mapped(&gid));
        assert_eq!(&gid[12..], &[10, 100, 50, 1]);
        assert!(!is_ipv4_mapped(&mapped_gid("127.0.0.1".parse().unwrap())));
    }
}

#[allow(dead_code)]
fn _unused(_: *const c_void) {
    let _ = null::<u8>();
}
