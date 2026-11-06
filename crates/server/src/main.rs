use clap::Parser;
use inference_bench_core::logging::LogArgs;
use inference_bench_server::cli::ServeArgs;

#[derive(Parser)]
#[command(name = "inference-bench-server", about = "Headless OpenAI-compatible inference server", version)]
struct Cli {
    #[command(flatten)]
    log: LogArgs,
    #[command(flatten)]
    serve: ServeArgs,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    cli.serve.run(&cli.log)
}
