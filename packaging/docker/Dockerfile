FROM nvidia/cuda:13.0.0-devel-ubuntu24.04 AS build
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates build-essential pkg-config && rm -rf /var/lib/apt/lists/*
RUN curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
ENV PATH=/root/.cargo/bin:$PATH
WORKDIR /src
COPY . .
RUN cargo build --release --locked -p inference-bench-server

FROM nvidia/cuda:13.0.0-runtime-ubuntu24.04
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=build /src/target/release/inference-bench-server /usr/local/bin/
COPY crates/specrt/specs /etc/inference-bench/specs
ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility
EXPOSE 8080
ENTRYPOINT ["inference-bench-server", "--bind", "0.0.0.0:8080", "--log-format", "json"]
