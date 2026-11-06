fn main() -> anyhow::Result<()> {
    specrt::worker::run(std::io::stdin().lock(), std::io::stdout().lock())
}
