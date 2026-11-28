# Bare metal inference-bench (plan)

The plan behind #1525 to #1532: a tiny OS or microkernel that runs inference-bench straight on the hardware. It compares the options, says which one to build, and lists the slices; none is built yet. The QEMU numbers come from a throwaway spike with no GPU.

## 1. The constraint

specrt reaches the GPU like this (versions as on this desktop, 27 Nov 2026):

| Layer | What it is | Where it runs | Licence |
|---|---|---|---|
| `nvidia.ko`, `nvidia-uvm.ko` (open kernel modules 595.112.07) | resource manager in the kernel: PCI, BAR mapping, memory, channels, interrupts, and the RPC client to the GSP | Linux kernel | MIT, dual MIT/GPLv2 once linked into a Linux module (`COPYING` in open-gpu-kernel-modules) |
| `gsp_ga10x.bin` (72 MB), `gsp_tu10x.bin` (30 MB) | the resource manager proper, running on the GPU System Processor (a RISC-V core on the card) | the GPU | NVIDIA driver licence, binary |
| `libcuda.so` (91 MB), `libnvidia-ptxjitcompiler.so` (32 MB), `libnvidia-gpucomp.so` (94 MB) | CUDA driver API, PTX to SASS JIT | user space, opens `/dev/nvidia*` and issues RM ioctls | NVIDIA driver licence, binary |
| `libnvrtc.so` (115 MB) | CUDA C++ to PTX or cubin | user space | CUDA EULA, listed as redistributable |
| `inference-bench-server` (35 MB, static Rust, links only libc, libm, libgcc_s) | our code | user space | ours |

specrt loads `libcuda` and `libnvrtc` at run time through cudarc's `dynamic-loading` feature (`crates/specrt/Cargo.toml`). It compiles kir's CUDA source with NVRTC to PTX or cubin (`crates/specrt/src/jit.rs`, `compile_cubin`, `load_uncached`) and launches through `cuModuleLoad` and `cuLaunchKernel`. The RDMA cluster path dlopens `libibverbs.so.1` and registers device memory as dma-buf (`crates/specrt/src/rdma.rs`, `docs/design/distributed.md`). The NVMe tier reads with `O_DIRECT` (`crates/specrt/src/diskio.rs`, `loader.rs`). Host memory comes from `/proc/meminfo` (`hardware.rs`, `serve.rs`).

On Turing and later the kernel module is a thin client: almost every GPU operation is an RPC to the GSP firmware. In the 595 sources (`src/nvidia/inc/kernel/vgpu/rpc_global_enums.h`) there are 219 RM RPC functions, and the control calls they carry are spread over 2,365 `NV*_CTRL_CMD_*` defines and 315 class headers in `src/common/sdk/nvidia/inc`. The open module tree is about 2.2 million lines of C and headers (`/usr/src/nvidia-595.112.07`: `src/nvidia` 1.40 M, `src/common` 0.47 M, `kernel-open` 0.22 M). `libcuda` talks to that kernel interface through ioctls whose layout is private and changes with each driver release. Nothing about `libcuda` is portable to a kernel that is not Linux or Windows.

So the question is not "which small OS", it is "who drives the GPU".

## 2. Option A: appliance Linux

A Linux kernel plus an initramfs. A small Rust init is PID 1: it mounts `/proc`, `/sys`, `/dev`, loads the NVIDIA and NIC modules, brings up the network, mounts the model disk, starts `inference-bench-server`, and restarts it if it exits. No distro, no shell in production, no package manager, no systemd.

### What it takes

| Piece | Choice | Why |
|---|---|---|
| Kernel | a custom `defconfig` fragment on a longterm kernel (6.12 today): x86_64 and arm64, PCI, NVMe, the NICs we meet (virtio, e1000e, igb, ixgbe, ice, mlx5), `CONFIG_DRM` for nvidia-drm, IOMMU, dma-buf, RDMA core and `mlx5_ib` / `ib_uverbs` for cluster mode, no sound, no Bluetooth, no wireless, no graphics beyond simpledrm for a console | the GPU module only needs PCI, DMA and `CONFIG_DRM`; everything else is attack surface and boot time |
| NVIDIA modules | `open-gpu-kernel-modules` built against that exact kernel in CI | MIT/GPLv2, so building and shipping them is clean (section 2.4) |
| Firmware and user space | GSP firmware, `libcuda`, `libnvidia-ptxjitcompiler`, `libnvidia-gpucomp`, `libnvidia-ml` from the matching driver `.run` with `--no-kernel-modules`, `libnvrtc` from the CUDA toolkit | the module and firmware must match to the build number |
| Image builder | our own Rust tool (`xtask appliance` or a small crate) that copies the files listed in a TOML manifest into a cpio, zstd compresses it and writes a UKI | buildroot builds a whole toolchain and libc we do not need: our only dynamic binary is the NVIDIA user space, which needs glibc, and that we copy from the build host. mkosi or a distro builder brings systemd and packages back |
| Boot | a Unified Kernel Image (kernel, initramfs, command line in one PE file) for UEFI from USB or a disk partition, the same files over iPXE or HTTP boot for PXE, a qcow2 or raw disk for rented VMs | one artefact for every path, signable for Secure Boot |
| Root | the initramfs is the root, held in RAM, never written. Models and the prefix or session store live on an NVMe partition mounted at `/models` and `/var/lib/inference-bench` | read-only by construction |
| Config | `/etc/inference-bench/server.toml` inside the image, overridden by a file on the data partition, overridden by `ib.config=<url or path>` and a few `ib.*` keys on the kernel command line (bind address, API key reference, model list URL) | the project rule is config first, no env vars; the command line is the only input PXE gives us |
| Network | DHCP in the Rust init (or a static `ip=` from the command line, which the kernel already parses), NTP once at boot for TLS | |

### Measured in the spike

All on this desktop, QEMU with KVM, 2 vCPUs, Debian's stock 6.12.115 kernel (12.2 MB, not trimmed), busybox as a stand in for the Rust init:

| Image | Contents | Compressed size | Kernel start to event |
|---|---|---|---|
| minimal | busybox and its libc | 1.3 MB | init running at 0.18 to 0.31 s, whole QEMU run 0.77 to 0.99 s |
| server | busybox, `inference-bench-server` built from this branch, glibc, virtio-net modules, the tiny Whisper fixture, a `server.toml` | 15.6 MB (56 MB unpacked) | network up at 0.36 s, `GET /v1/models` answers at 0.43 s, from the host through a forwarded port too |

Inside that guest `/health`, `/v1/capabilities` and `/v1/audio/status` answered (`"device":"CPU"`). Two calls killed their request with a panic: `GET /metrics` and `POST /v1/audio/transcriptions`, both with cudarc's `Unable to dynamically load the "cuda" shared library`. The engine checks `hardware::cuda_present()` in some paths but not in these two: `crates/server/src/gpu.rs` `driver_used_bytes` calls `CudaContext::new(0)`, and `speech::service::open` calls `CudaContext::new` again in its fallback check. cudarc's `culib()` panics where it could return an error. On a desktop this never shows because the driver is installed. It is filed as a bug and is the first thing slice 1 needs.

Sizes of the GPU payload from this machine, zstd -12:

| File | Raw | Compressed |
|---|---|---|
| `gsp_ga10x.bin` (Ampere and later) | 72 MB | 61 MB |
| `gsp_tu10x.bin` (Turing) | 30 MB | 15 MB |
| `libnvrtc.so` 13.2 | 115 MB | 39 MB |
| `libnvidia-gpucomp.so` | 94 MB | 29 MB |
| `libnvidia-tileiras.so` (dlopened by libcuda) | 97 MB | 32 MB |
| `libnvidia-nvvm.so.4` (dlopened by libcuda) | 78 MB | 25 MB |
| `libnvidia-ptxjitcompiler.so` | 32 MB | 9 MB |
| `libcuda.so` | 91 MB | 8 MB |
| `nvidia.ko` + `nvidia-uvm.ko` | 30 MB | 7 MB |
| server binary | 35 MB | 9 MB |

So a GPU appliance image is about 235 MB compressed with both GSP blobs and about 220 MB with only `gsp_ga10x`, and about 680 MB unpacked in RAM. On a 96 GB card host with 64 GB or more of RAM, that is about 1% of host memory. Debian's own initrd on this machine is 100 MB before the root filesystem is even mounted.

### Boot time

The spike shows the software side is under half a second to a listening server. On real hardware the floor is set elsewhere:

| Phase | Typical | Who controls it |
|---|---|---|
| UEFI POST on a server board | 20 to 90 s (memory training, option ROMs, BMC) | the board vendor, not us |
| UKI load from NVMe or USB | under 1 s; PXE over 1 GbE for 200 MB about 2 s, over 10 GbE well under 1 s | us, by image size |
| Kernel to init | 0.2 to 0.4 s (spike) | us |
| `nvidia.ko` load and GSP boot | estimated 1 to 3 s per card (firmware upload and GSP init); not measured here because GPU0 is in use, slice 3 measures it | NVIDIA |
| Model load | bound by NVMe: 10 GB/s on the Samsung 9100 (`docs/design/moe-offload.md`), so 3 s for the 27B NVFP4 weights, plus the JIT cache (warm) | us |

Against a Debian server boot the win is the 10 to 30 s of distro init and the lack of drift, not the POST.

### Updates

The image is the release. An update is a new UKI. Two boot slots (A and B partitions or two EFI entries) with a boot counter: the new image boots once, the init marks it good after `/health` answers with the configured models loaded, otherwise the firmware falls back to the old slot on the next boot. PXE hosts take whatever the boot server serves, so a fleet update is a file swap on the boot server and a reboot. Spec updates keep working as they do today (`inference_bench_core::specs`, fetched to the data partition), so new models do not need a new image. Kernel and driver come only through a new image, built in CI from the pinned versions, the same way Talos ties `nvidia-open-gpu-kernel-modules` to a Talos release ("versioned as `<nvidia-driver-version>-<talos-release-version>`", Talos NVIDIA guide).

### Licence

- **Kernel modules:** open-gpu-kernel-modules `COPYING`: "Except where noted otherwise, the individual files within this package are licensed as MIT ... when linked together to form a Linux kernel module, the resulting Linux kernel module is dual licensed as MIT/GPLv2." Building them in CI and shipping the `.ko` is allowed; ship the source or a pointer to the exact tag beside the image.
- **GSP firmware and `libcuda` (driver user space):** NVIDIA Driver License for Linux, section 1.1(d): you may "Distribute the SOFTWARE provided for use with operating system kernels distributed under the terms of an OSI-approved open source license ... provided that (i) the binary files thereof are not modified in any way (except for uncompressing of compressed files) and (ii) this Agreement is provided to each SOFTWARE recipient." Linux is GPLv2, so an appliance image may carry the unmodified files with the licence text in the image and beside the download. Compressing the initramfs as a whole is fine: the recipient gets the files bit for bit after unpacking.
- **Limits in the same licence that bite:** 2.1, firmware may only run on NVIDIA hardware (fine). 2.7, no "commercial hosting services with the SOFTWARE" except as granted, and 1.1(c) grants use "on infrastructure you own or lease". Running it on our own or leased boxes is covered by 1.1(c); selling an image to third parties is covered by 1.1(d). 2.8: "GeForce or Titan SOFTWARE ... is not licensed for datacenter deployment." The RTX PRO 6000 is not GeForce, but an image deployed to rented GeForce boxes or LNVPS marketplace nodes with consumer cards falls under 2.8 in a datacentre. That is a user and legal decision, not a technical one.
- **NVRTC:** CUDA EULA Attachment A lists "NVIDIA Runtime Compilation Library and Header", Linux files `libnvrtc.so, libnvrtc-builtins.so, libnvrtc_static.a, libnvrtc-builtins_static.a`, as distributable, under 1.1.2: the application must have "material additional functionality" (it does) and the files must be unmodified (section 2.3).
- **Nouveau firmware** in linux-firmware is a separate package under its own WHENCE terms and is not needed for option A.

Conclusion: an appliance image with open modules, unmodified GSP firmware, driver user space and NVRTC can be redistributed if the licences travel with it. The only open question is GeForce in a datacentre.

### 2.5 Cheaper still: no NVRTC on the box

NVRTC is the largest single file and only turns CUDA C++ into PTX or cubin. kir already compiles cubins (`jit::cubin_for`) and keeps a disk cache keyed by source, arch and options (`jit::cache_key`). An appliance could ship a cubin set built in CI for each target arch (sm_80, sm_89, sm_90, sm_120) for the specs it serves, and fall back to NVRTC only when a spec or option misses the cache. That drops 39 MB compressed and the JIT time at first load. It is an optimisation for a later slice, not a requirement.

## 3. Option B: a Rust unikernel or microkernel

### Candidates

| OS | What it is | Devices today | Fit |
|---|---|---|---|
| Hermit (hermit-os) | unikernel, the app links the kernel as a static library; Rust std target `x86_64-unknown-hermit` is tier 3 "with std"; x86_64, aarch64, riscv64 | virtio devices and smoltcp networking, boots under QEMU or its own uhyve hypervisor | the closest to "the app is the OS", but it targets VMs; no PCI GPU, no NVMe driver of note |
| Theseus | research OS in Rust ("to experiment with novel OS structure"), boots from USB or PXE on NUCs, ThinkPads, Supermicro | a few NICs, PCI passthrough under QEMU | research grade, "certain features may be deprioritized" |
| Redox | microkernel, Unix-like, drivers are user-space daemons with schemes | storage, some NICs, framebuffer; no GPU compute driver | a general purpose OS, not a minimal one |
| seL4 with a Rust user land (rust-sel4) | formally verified microkernel, drivers in user space | whatever we write | the strongest isolation story, every driver is ours |

None of them has a GPU compute driver for any vendor. The work is the same on all four: write an NVIDIA driver.

### What driving an NVIDIA GPU without Linux means

On Turing and later the path is through the GSP:

1. **PCI and memory:** enumerate the device, map BAR0 (registers), BAR1 (VRAM aperture) and BAR2, set up IOMMU or identity DMA, MSI-X interrupts.
2. **Secure boot of the GSP:** parse the VBIOS, run devinit, then the FWSEC firmware on the Falcon security processor, then load the GSP boot loader, the booter and the 30 to 72 MB GSP image into the WPR2 region and start the RISC-V core. On Hopper and Blackwell there is an FSP step in front of this. The Linux nova-core docs have chapters for each (VBIOS, devinit, FSP and Secure Boot, FWSEC, Falcon, TLV tags in firmware images).
3. **GSP RPC:** a shared memory command and status queue, then the RM protocol on top: `GSP_RM_ALLOC` and `GSP_RM_CONTROL` carry the client, device, subdevice, memory and channel objects; `GSP_INIT_DONE` and other events come back. The message layouts are generated headers that change with each firmware build (`g_rpc-structures.h`), so the driver must be pinned to one firmware version, as nouveau is (the linux-firmware copy on this machine carries only `gsp-535.113.01.bin` for it).
4. **Memory management:** a VRAM allocator, GPU page tables (the GPU MMU), instance memory for page tables, mapping host pages for DMA.
5. **Channels:** a FIFO channel per stream with a pushbuffer and a GPFIFO, doorbells through the USERD area, semaphores or fences for completion.
6. **Compute class:** the compute engine class object for the architecture, the QMD (queue meta data) layout per generation, shared memory and register config, constant buffers for kernel parameters.
7. **Code:** SASS for the target. PTX is not executable; turning it into SASS is `ptxas` or the driver's JIT, both NVIDIA binaries that run only on Linux or Windows user space (`libnvidia-ptxjitcompiler` is a Linux ELF that expects glibc). Without them we would need our own SASS emitter for each architecture, which NVIDIA does not document. The open source alternative is Mesa's NAK compiler (NVK's shader compiler, written in Rust), which targets SASS from NIR for graphics and compute shaders up to Blackwell, but it takes SPIR-V, not PTX.

### What exists in the open

- **nova (nova-core and nova-drm), in Linux mainline since 6.15:** Rust, GSP only, Turing and later. As of the LPC 2026 talk (5 Oct 2026) "the GSP boots with its firmware on Turing through Blackwell hardware, with basic functionality coming online", and it is "not yet in a state for end users". The nova-core task list still has MMU and page table management, a VRAM allocator, instance memory, the FIFO engine (channels) and the client and device RPC interfaces open. NVIDIA says the long term plan is for nova to replace its own out-of-tree Open RM, with no timeline, and its user space "will need adapting" because nova is not ABI compatible with the current ioctls. Getting "Vulkan Compute and CUDA apps running on Nova" is listed as future work. vGPU starts landing in Linux 7.5.
- **nouveau plus GSP:** nouveau boots GSP firmware 535.113.01 on Turing and later and gives NVK a channel and memory model.
- **NVK (Mesa):** a conformant Vulkan 1.4 driver for Kepler through Ada and consumer Blackwell, on top of nouveau, with compute shaders. specrt already has a wgpu backend that runs on Vulkan (`docs/design/wgpu.md`), so wgpu on NVK on nouveau is a fully open stack that runs our dense kernels today, on Linux. It has no NVFP4 or FP8 tensor cores through Vulkan, no delta-net layers on wgpu yet, and none of the CUDA fast paths.

### Can CUDA user space or our own PTX run on top

- **libcuda on a non-Linux kernel:** no. It is a glibc ELF that issues private RM ioctls to `/dev/nvidiactl` and `/dev/nvidia*` and dlopens four more NVIDIA libraries. Reimplementing those ioctls on top of our own RM is reimplementing the open kernel module's whole user interface for one driver build, and the licence (2.2, no reverse engineering of the binary parts; 2.6, no circumventing technical limits) makes emulating the binary's expectations a legal risk as well.
- **Our own PTX launching:** kir emits CUDA C++, NVRTC turns it into PTX or cubin. A cubin built in CI (section 2.5) is SASS, so with a cubin set we do not need a JIT on the box. What we would still need is everything in steps 1 to 6: a GSP RM client, memory and channels, and a QMD writer that launches a cubin's kernel with its parameters. That is the CUDA driver's launch path, which is the part NVIDIA keeps closed.

### Effort, honestly

| Piece | Prior art | Our estimate |
|---|---|---|
| PCI, BAR, MSI-X, IOMMU on Hermit or seL4 | easy on Linux, ours on a new OS | 1 to 2 months |
| GSP secure boot (VBIOS, devinit, FWSEC, booter, FSP on Hopper and later), one firmware version, one or two architectures | nova-core and nouveau, both GPL; reading them is fine, copying GPL code into an MIT or proprietary kernel is not | 4 to 8 months |
| RM client objects over GSP RPC, memory, MMU, channels | nouveau has it, nova is still writing it after more than a year upstream | 6 to 12 months |
| Compute launch: QMD per architecture, constant buffers, semaphores | partly in nouveau and NVK, never documented by NVIDIA | 3 to 6 months |
| NVMe, a NIC with RDMA, TCP, TLS, a filesystem for the model store, on a unikernel | Hermit has virtio only; real NICs (mlx5, ice) are each a large driver | 4 to 8 months |
| Keeping up: every new GSP firmware and every new GPU generation changes the RPC structures | NVIDIA and Red Hat engineers full time | permanent |

That is two to three engineer years before the first token on one GPU generation, in a space where NVIDIA's own team has not reached compute since nova went upstream in 2025. It would also lose every CUDA fast path we have measured against vLLM, unless the cubins launched through our own QMD writer run identically, which nothing proves.

## 4. Option C: Linux as a driver host only

The same as option A but stricter: the kernel is only there to run `nvidia.ko`, NVMe, the NIC and RDMA, and our Rust code is the entire user space. PID 1 is our init, it execs (or links in) the server, there is no other process, no shell, no `nvidia-smi`, no getty.

Against option A this changes very little, because option A was already that. The real differences:

| | A (appliance) | C (driver host) |
|---|---|---|
| Processes | init, server, maybe a debug shell behind a command line flag | one binary, init is the server |
| Debugging | a busybox shell and `nvidia-smi` in a debug image | logs to the serial console and to `/metrics`; a debug build is a different image |
| Persistence daemon | NVIDIA's `nvidia-persistenced` not needed: the server holds the devices open all its life | same |
| Restart on crash | init restarts the server | the kernel panics or reboots on PID 1 exit, so a crash is a reboot (seconds plus POST) |
| glibc | needed by libcuda in both | same |

C is a build flag of A (init and server in one binary), not a separate design. Taking it further, a "Linux library OS" with Unikraft or a Linux built as a unikernel (UKL), gives little: the GPU still needs `nvidia.ko` in a real kernel, and once a real kernel is there the process boundary costs us nothing measurable at inference time (decode is GPU bound, the host side is one thread per step).

## 5. What each option gains and breaks

| | Stock distro (today) | A: appliance Linux | B: Rust microkernel | C: driver host |
|---|---|---|---|---|
| Boot to serving (after POST) | 10 to 30 s distro boot plus model load | 0.4 s plus GSP and model load (spike) | unknown, no GPU path exists | as A |
| Jitter | systemd timers, journald, updatedb, unattended upgrades, other users | none of those; isolate CPUs with `isolcpus`/`nohz_full` on the command line for the decode thread | lowest in theory | as A |
| Host memory for models and KV offload | distro uses 1 to 3 GB | image 0.65 GB in RAM, nothing else | lowest | as A |
| VRAM for models | the same in all; a display server or desktop app on the card costs VRAM (the user's app on GPU0) | no display, the whole card | same | same |
| Attack surface | full distro, sshd, package manager | kernel, NIC, our server only; read-only root; no shell | smallest, but every driver is new and unaudited code | smallest of the Linux ones |
| Rented GPU boxes (vast.ai and similar) | works, they hand out containers or VMs | does not apply: you get a container on their kernel. Ship the same files as a container image instead | does not apply | does not apply |
| Bare-metal rentals and LNVPS marketplace nodes (a Debian 13 libvirt host today) | install a .deb | boot the UKI by PXE or as the OS on the GPU node; a VM with VFIO passthrough of the GPU can boot the same image as a qcow2 | only on hardware we own | as A |
| RDMA cluster mode (`rdma.rs`, gpu-direct) | works | works if the image has `mlx5_ib`, `ib_uverbs`, `rdma_rxe`, `nvidia-peermem` (9 KB), dma-buf and `libibverbs` plus the provider (`libmlx5`, 0.5 MB) | breaks: needs an RDMA NIC driver and verbs on the new OS | as A |
| NVMe offload (`diskio.rs`, `O_DIRECT`) | works | works, NVMe and ext4 or xfs in the kernel config | breaks until we write NVMe and a filesystem | as A |
| Desktop app (`inference-bench-ui`) | the product | not on the appliance; the app connects to it over the network as a remote server, as it does with hosts today (`specrt::host`, pairing) | same | same |
| Spec auto updates | works | works, written to the data partition | needs TLS and a filesystem | as A |
| Model support | all | all | only what our own launch path runs | all |
| Main risk | drift between hosts | driver and kernel ABI: open modules must be rebuilt per kernel, firmware pinned per driver; GeForce clause 2.8 | years of work that NVIDIA may obsolete with nova; firmware churn; legal exposure if we emulate libcuda | a crash is a reboot |

## 6. Recommendation

Build option A, with C as a build profile of it. It keeps every engine feature, it is a few weeks of work instead of years, it can be checked in QEMU here without a GPU, and the licences allow shipping it. Do not start option B: nothing in the open drives an NVIDIA GPU for compute without Linux today, NVIDIA's own Rust driver (nova) has not reached channels or compute, and a CUDA-free path would lose the kernels that make specrt fast. Revisit B only if nova reaches compute and exposes an interface our own user space could drive, and then the right shape is still a Linux kernel with nova in it (option C with nova in place of `nvidia.ko`), not our own kernel.

Two things are worth doing on the way because they help every deployment, not only the appliance:

- the server must not panic when `libcuda` is missing (bug filed; seen in the spike);
- a prebuilt cubin set per arch removes NVRTC from the hot path at first load.

## 7. Slices

Filed as backlog issues with no goal; the owner promotes them. Each one names its check.

| # | Issue | Slice | Needs a GPU | Check |
|---|---|---|---|---|
| 0 | #1525 | Server and engine survive a host with no `libcuda` (bug) | no | a test that runs the server with no driver library on the path and calls `/metrics` and the CPU transcription path |
| 1 | #1526 | QEMU appliance with no GPU: Rust image builder (manifest TOML to cpio, zstd, UKI), Rust PID 1 init (mounts, modules, network from DHCP or `ip=`, config from file and `ib.*` command line keys, starts and restarts the server), serves the tiny Whisper fixture on CPU | no | a script boots the image in QEMU with KVM, curls `/v1/models`, posts the fixture WAV to `/v1/audio/transcriptions` and gets the fixture's greedy tokens, records boot to serving time; runs in CI |
| 2 | #1527 | Read-only root and the data partition: models, spec updates, session store and JIT cache on a virtio disk in QEMU (NVMe on hardware), A/B UKI slots with a boot counter and the `/health` good-mark | no | QEMU with two disks: an update to a broken image falls back to the old one on the next boot |
| 3 | #1528 | GPU payload: build open modules for the pinned kernel in CI, add GSP firmware, driver user space and NVRTC from the matching `.run` and toolkit, licence texts in the image; boot it on a real GPU box | yes, a spare machine or a GPU1 passthrough VM in a window the owner grants | `nvidia-smi` equivalent from `/metrics`, the tiny fixtures, then the 27B, with GSP boot time and model load time recorded |
| 4 | #1529 | Fleet boot: iPXE or UEFI HTTP boot script and a qcow2 for libvirt hosts (LNVPS marketplace style) with VFIO GPU passthrough | QEMU part no, passthrough yes | the same UKI boots by PXE in QEMU (no GPU) and as a qcow2 |
| 5 | #1531 | Cluster mode in the image: `mlx5_ib`, `ib_uverbs`, `rdma_rxe`, `nvidia-peermem`, `libibverbs` and providers | `rdma_rxe` part no, real RoCE yes | two QEMU guests on soft RoCE run `tests/rdma.rs` style frames between stages on the tiny fixture |
| 6 | #1532 | Driver host profile (option C): init and server in one binary, no shell, `isolcpus` for the decode thread, jitter measured | yes for the jitter numbers | step time p99 and ITL p99 on the 4B against the same box on Debian |
| 7 | #1530 | Prebuilt cubin set: `inference-bench warmup` style CI job that compiles the kir modules for the served specs for sm_80, sm_89, sm_90, sm_120, shipped in the image cache dir; NVRTC becomes optional | building no, proving the hit yes | a load with NVRTC absent from the image succeeds from the cubin set and gives the same logits |

Slice 1 depends on slice 0 (it transcribes, which panics today). Slices 2, 4 and 7 depend on 1. Slice 3 depends on 1. Slices 5 and 6 depend on 3.

## Sources

- open-gpu-kernel-modules, `COPYING` and README: github.com/NVIDIA/open-gpu-kernel-modules ("must be used with GSP firmware and user-space NVIDIA GPU driver components from a corresponding ... driver release", "any Turing or later GPU").
- NVIDIA Driver License for Linux: nvidia.com/en-us/drivers/nvidia-license/linux/ (sections 1.1(c), 1.1(d), 2.1, 2.2, 2.6, 2.7, 2.8).
- CUDA EULA and Attachment A: docs.nvidia.com/cuda/eula/ (sections 1.1.2, 2.2, 2.3; "NVIDIA Runtime Compilation Library and Header").
- nova: kernel.org/doc/html/next/gpu/nova/ and its core task list; LPC 2026 talk by John Hubbard and Danilo Krummrich, reported by Phoronix on 8 Oct 2026 and summarised on windowsforum.com (10 Oct 2026); KVM Forum 2026 vGPU talk, linuxmind.dev (14 Nov 2026).
- NVK: docs.mesa3d.org/drivers/nvk.html (Vulkan 1.4 conformant, Kepler to Ada and consumer Blackwell).
- Hermit: github.com/hermit-os/kernel; Rust platform support page for `*-unknown-hermit` (tier 3, std).
- Theseus: github.com/theseus-os/Theseus. Redox: doc.redox-os.org/book/drivers.html (user-space driver daemons).
- Talos NVIDIA guide: docs.siderolabs.com, "NVIDIA GPU" (`nvidia-open-gpu-kernel-modules` extension tied to the Talos release).
- Local: `/usr/src/nvidia-595.112.07` (line counts, `rpc_global_enums.h`), `/lib/firmware/nvidia` (GSP blobs, nouveau's `gsp-535.113.01.bin`), file sizes from `/usr/lib/x86_64-linux-gnu` and `/usr/local/cuda-13.2`, the QEMU spike (stock Debian 6.12.115 kernel, busybox init, this branch's `inference-bench-server`).
