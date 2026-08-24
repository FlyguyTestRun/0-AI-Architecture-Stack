"""Command line interface.

``zerostack doctor`` is the command that matters most on a new machine: it reports
which backend is live in every layer, so the fallback behaviour is visible rather
than mysterious.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from zerostack.app import ZerostackApp

app = typer.Typer(
    name="zerostack",
    help="A zero cost AI architecture baseline.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()

_STATUS_STYLE = {True: "green", False: "yellow"}


@app.command()
def doctor() -> None:
    """Report which backend is active in every layer."""
    health = ZerostackApp().health()
    layers = health["layers"]

    table = Table(title="Zerostack layer status", show_lines=False)
    table.add_column("Layer", style="bold")
    table.add_column("Active backend")
    table.add_column("Detail", overflow="fold")

    orchestrator = layers["orchestrator"]
    installed = [name for name, ok in orchestrator["available"].items() if ok]
    table.add_row("2 Orchestrator", orchestrator["active"], f"installed: {', '.join(installed)}")

    rag = layers["rag"]
    table.add_row(
        "3 RAG",
        f"{rag['vector_store']} ({rag['vector_store_mode']})",
        f"embeddings: {rag['embeddings']} dim={rag['dimensions']} chunks={rag['indexed_chunks']}",
    )

    llm = layers["llm"]
    reachable = llm["ollama_reachable"]
    table.add_row(
        "4 LLM",
        f"[{_STATUS_STYLE[reachable]}]{llm['active_provider']}[/]",
        f"model: {llm['model']} | ollama reachable: {reachable}",
    )

    tools = layers["tools"]
    table.add_row("5 Tools", str(tools["count"]), ", ".join(tools["names"]) or "none")

    data = layers["data"]
    table.add_row("7 Data", "sqlite", f"{data['sqlite_path']} | runs: {data['runs']}")

    obs = layers["observability"]
    table.add_row(
        "Observability",
        "on" if obs["enabled"] else "off",
        f"export: {obs['export_traces']} | log: {obs['trace_log']}",
    )

    console.print(table)

    if not reachable:
        console.print(
            Panel(
                "Ollama is not reachable, so answers come from the offline extractive "
                "provider. Run `make up` then `make pull-model` for generated answers.",
                title="Fallback active",
                border_style="yellow",
            )
        )


@app.command()
def ingest(
    path: Path = typer.Argument(None, help="File or directory. Defaults to data/corpus."),
) -> None:
    """Index documents into the vector store."""
    try:
        report = ZerostackApp().ingest(path)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(code=1) from exc

    console.print(
        f"[green]Indexed[/] {report['chunks']} chunk(s) from {report['files']} file(s). "
        f"Collection now holds {report['indexed_chunks']} chunk(s)."
    )
    for skipped in report["skipped"]:
        console.print(f"[yellow]skipped[/] {skipped}")


@app.command()
def ask(
    question: str = typer.Argument(..., help="The question to answer."),
    show_steps: bool = typer.Option(False, "--steps", help="Print the agent steps."),
    as_json: bool = typer.Option(False, "--json", help="Emit the raw result as JSON."),
) -> None:
    """Ask the agent a question."""
    result = ZerostackApp().ask(question)

    if as_json:
        console.print_json(json.dumps(result.to_dict()))
        return

    console.print(Panel(result.answer, title="Answer", border_style="green"))

    if result.sources:
        console.print(f"[dim]sources: {', '.join(result.sources)}[/]")
    console.print(
        f"[dim]{result.orchestrator} | {result.llm_provider}/{result.llm_model} | "
        f"{result.latency_ms}ms[/]"
    )

    if show_steps:
        table = Table(title="Agent steps")
        table.add_column("Step", style="bold")
        table.add_column("Detail", overflow="fold")
        for step in result.steps:
            table.add_row(step.name, step.detail)
        console.print(table)


@app.command()
def demo() -> None:
    """Ingest the sample corpus and run a scripted set of questions.

    This is the fastest proof that every layer is wired correctly on a new machine.
    """
    instance = ZerostackApp()
    report = instance.ingest()
    console.print(
        f"[green]Ingested[/] {report['chunks']} chunk(s) from {report['files']} file(s)\n"
    )

    questions = [
        "How long is the probationary period and what happens if someone does not pass?",
        "How much can a support agent refund without approval?",
        "What must a postmortem contain?",
        "What is (120 * 3) / 4?",
    ]

    for question in questions:
        result = instance.ask(question)
        console.print(Panel(result.answer, title=question, border_style="cyan"))
        console.print(
            f"[dim]{' -> '.join(step.name for step in result.steps)} | "
            f"sources: {', '.join(result.sources) or 'none'} | {result.latency_ms}ms[/]\n"
        )

    console.print("[bold green]Demo complete.[/] Run `zerostack doctor` for layer status.")


@app.command()
def runs(limit: int = typer.Option(10, help="How many runs to show.")) -> None:
    """Show recent agent runs from the data layer."""
    records = ZerostackApp().recent_runs(limit=limit)
    if not records:
        console.print("[yellow]No runs recorded yet.[/]")
        return

    table = Table(title=f"Last {len(records)} run(s)")
    table.add_column("When", style="dim")
    table.add_column("Question", overflow="fold")
    table.add_column("Engine")
    table.add_column("ms", justify="right")
    for record in records:
        table.add_row(
            record["created_at"][:19],
            record["question"][:60],
            f"{record['orchestrator']}/{record['llm_provider']}",
            str(record["latency_ms"]),
        )
    console.print(table)


@app.command()
def analytics() -> None:
    """Aggregate run metrics through the data layer."""
    console.print_json(json.dumps(ZerostackApp().analytics()))


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Bind address."),
    port: int = typer.Option(8000, help="Port."),
    reload: bool = typer.Option(False, help="Reload on code changes."),
) -> None:
    """Run the FastAPI service."""
    import uvicorn

    uvicorn.run("zerostack.api.main:api", host=host, port=port, reload=reload)


if __name__ == "__main__":
    app()
