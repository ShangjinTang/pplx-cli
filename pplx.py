#!/usr/bin/env python3

# If you need to run this standalone without using in this project, you can run:
# uv run <script_name>, such as `uv run pplx.py`
# See: https://docs.astral.sh/uv/concepts/projects/run/#running-scripts
# /// script
# dependencies = [
#     "httpx",
#     "loguru",
#     "pydantic",
#     "requests",
#     "rich",
#     "typer",
# ]
# ///

"""
Enhanced Perplexity CLI Tool
A professional CLI interface for the Perplexity API with Rich output and flexible configuration.
"""

import json
import os
import sys
from enum import Enum
from typing import Any, Dict, Optional

import httpx
import typer
from loguru import logger
from pydantic import BaseModel, Field, ValidationError
from rich import print as rprint
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from typing_extensions import Annotated

# Initialize Rich console
console = Console()

# Default configurations
DEFAULT_BASE_URL = "https://api.perplexity.ai"
DEFAULT_MODEL_ROUTER = {
    "sonar-reasoning-pro": "sonar-reasoning-pro",
    "sonar-reasoning": "sonar-reasoning",
    "sonar-pro": "sonar-pro",
    "sonar": "sonar",
}
DEFAULT_MODEL = "sonar-pro"
AVAILABLE_MODELS = list(DEFAULT_MODEL_ROUTER.keys())


class OutputType(str, Enum):
    """Available output types"""

    DEFAULT = "default"
    PLAIN = "plain"
    JSON = "json"


class PerplexityConfig(BaseModel):
    """Configuration model for Perplexity API settings"""

    api_key: str = Field(..., description="Perplexity API key")
    base_url: str = Field(default=DEFAULT_BASE_URL, description="Base URL for API")
    model_router: Dict[str, str] = Field(
        default_factory=lambda: DEFAULT_MODEL_ROUTER.copy()
    )
    model: str = Field(default=DEFAULT_MODEL, description="Model to use")
    timeout: int = Field(default=30, description="Request timeout in seconds")


class PerplexityClient:
    """Enhanced Perplexity API client with Rich output support"""

    def __init__(self, config: PerplexityConfig):
        self.config = config
        self.client = httpx.Client(
            base_url=config.base_url,
            timeout=config.timeout,
            headers={
                "accept": "application/json",
                "content-type": "application/json",
                "Authorization": f"Bearer {config.api_key}",
            },
        )
        logger.debug(f"Initialized client with base URL: {config.base_url}")

    def get_model_name(self, model_key: str) -> str:
        """Map model key to full model name using router"""
        return self.config.model_router.get(model_key, model_key)

    def validate_model(self, model: str) -> bool:
        """Validate if model exists in available models"""
        return model in self.config.model_router.keys()

    def query(
        self,
        message: str,
        model: str,
        system_prompt: str = "Be precise and concise.",
        show_usage: bool = False,
        show_citations: bool = True,
    ) -> Dict[str, Any]:
        """Send query to Perplexity API"""

        if not self.validate_model(model):
            available_models = ", ".join(self.config.model_router.keys())
            raise ValueError(
                f"Invalid model: {model}. Available models: {available_models}"
            )

        full_model_name = self.get_model_name(model)
        logger.debug(f"Using model: {model} -> {full_model_name}")

        payload = {
            "model": full_model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message},
            ],
        }

        logger.debug(f"Request payload: {payload}")

        try:
            response = self.client.post("/chat/completions", json=payload)
            response.raise_for_status()

            result = response.json()
            logger.debug(f"Response status: {response.status_code}")

            return {
                "content": result["choices"][0]["message"]["content"],
                "usage": result.get("usage", {}),
                "citations": result.get("citations", []),
                "model_used": full_model_name,
            }

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                raise ValueError("Invalid API key. Please check your API key.")
            elif e.response.status_code == 429:
                raise ValueError("Rate limit exceeded. Please try again later.")
            else:
                raise ValueError(
                    f"API error ({e.response.status_code}): {e.response.text}"
                )

        except httpx.RequestError as e:
            raise ValueError(f"Network error: {str(e)}")

    def __del__(self):
        """Clean up client connection"""
        if hasattr(self, "client"):
            self.client.close()


class OutputRenderer:
    """Rich-based output renderer for different display modes"""

    @staticmethod
    def format_citations(citations: list) -> str:
        """Format citations as a numbered list string"""
        if not citations:
            return ""

        citation_lines = []
        for i, citation in enumerate(citations):
            citation_lines.append(f"{i + 1}. {citation}")

        return "\n\nCitations:\n" + "\n".join(citation_lines)

    @staticmethod
    def render_default(
        content: str, citations: list = None, usage: Dict[str, Any] = None
    ) -> None:
        """Render in default rich output type with citations at bottom, no boxes for easy copying"""
        # Main content without box - just markdown formatting
        console.print(Markdown(content))

        # Usage table if provided
        if usage:
            console.print()  # Add spacing
            OutputRenderer.render_usage(usage)

        # Citations at bottom without box
        if citations:
            console.print("\n[bold yellow]Citations:[/bold yellow]")
            for i, citation in enumerate(citations):
                console.print(f"[yellow]{i + 1}.[/yellow] {citation}")

    @staticmethod
    def render_plain(content: str, citations: list = None) -> str:
        """Return content in plain output type with citations appended"""
        result = content
        if citations:
            result += OutputRenderer.format_citations(citations)
        return result

    @staticmethod
    def render_json(
        content: str,
        citations: list = None,
        usage: Dict[str, Any] = None,
        model_used: str = None,
    ) -> str:
        """Return content in JSON output type with citations appended to content"""
        # Append citations to content if they exist
        full_content = content
        if citations:
            full_content += OutputRenderer.format_citations(citations)

        result = {"content": full_content, "model_used": model_used}

        if usage:
            result["usage"] = usage

        return json.dumps(result, indent=2)

    @staticmethod
    def render_usage(usage: Dict[str, Any]) -> None:
        """Render token usage information in a table"""
        if not usage:
            return

        table = Table(
            title="Token Usage", show_header=True, header_style="bold magenta"
        )
        table.add_column("Metric", style="cyan")
        table.add_column("Count", justify="right", style="green")

        for key, value in usage.items():
            table.add_row(key.replace("_", " ").title(), str(value))

        console.print(table)

    @staticmethod
    def render_error(message: str) -> None:
        """Render error message with red styling"""
        console.print(
            Panel(
                f"[red]{message}[/red]",
                title="[bold red]Error[/bold red]",
                border_style="red",
            )
        )

    @staticmethod
    def render_info(message: str) -> None:
        """Render info message with blue styling"""
        console.print(f"[blue]ℹ {message}[/blue]")


def parse_model_router(router_str: str) -> Dict[str, str]:
    """Parse model router from JSON string"""
    try:
        router = json.loads(router_str)
        if not isinstance(router, dict):
            raise ValueError("Model router must be a JSON object")
        return router
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON output type in model router")


def load_config(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    model_router: Optional[str] = None,
    disable_router: bool = False,
) -> PerplexityConfig:
    """Load configuration from command line args and environment variables"""

    # API Key - command line takes priority
    final_api_key = api_key or os.getenv("PERPLEXITY_API_KEY")
    if not final_api_key:
        raise ValueError(
            "API key is required. Provide it via --api-key argument or "
            "set PERPLEXITY_API_KEY environment variable"
        )

    # Base URL - command line takes priority, falls back to env, then default
    final_base_url = base_url or os.getenv("PERPLEXITY_BASE_URL", DEFAULT_BASE_URL)

    # Model router - start with defaults, then apply customizations
    final_model_router = DEFAULT_MODEL_ROUTER.copy()

    # Parse from environment if available
    env_router = os.getenv("PERPLEXITY_MODEL_ROUTER")
    if env_router:
        try:
            env_parsed = parse_model_router(env_router)
            final_model_router.update(env_parsed)
        except ValueError as e:
            logger.warning(f"Invalid PERPLEXITY_MODEL_ROUTER: {e}, using defaults")

    # Parse from command line if provided (takes priority)
    if model_router:
        try:
            cli_parsed = parse_model_router(model_router)
            final_model_router.update(cli_parsed)
        except ValueError as e:
            raise ValueError(f"Invalid --model-router: {e}")

    # Model - command line takes priority, falls back to env, then default
    final_model = model or os.getenv("PERPLEXITY_MODEL", DEFAULT_MODEL)

    # Router validation - only if not disabled
    if not disable_router:
        # Validate model exists in router
        if final_model not in final_model_router:
            available_models = ", ".join(final_model_router.keys())
            raise ValueError(
                f"Invalid model '{final_model}'. Available models: {available_models}. "
                f"Use --disable-router to bypass model validation."
            )
    else:
        # When router is disabled, add the model to router as-is if not present
        if final_model not in final_model_router:
            final_model_router[final_model] = final_model
            logger.debug(f"Router disabled: Added model '{final_model}' as-is")

    return PerplexityConfig(
        api_key=final_api_key,
        base_url=final_base_url,
        model_router=final_model_router,
        model=final_model,
    )


def setup_logging(verbose: bool = False) -> None:
    """Configure loguru logging"""
    logger.remove()  # Remove default handler

    if verbose:
        logger.add(
            sys.stderr,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level="DEBUG",
        )
        logger.debug("Debug logging enabled")
    else:
        logger.add(
            sys.stderr,
            format="<level>{level}</level>: <level>{message}</level>",
            level="WARNING",
        )


# Initialize Typer app
app = typer.Typer(
    name="perplexity-cli",
    help="Enhanced CLI tool for Perplexity AI API",
    rich_markup_mode="rich",
)


@app.callback(invoke_without_command=True)
def main(
    message: Annotated[
        Optional[str], typer.Argument(help="The query message to send to Perplexity")
    ] = None,
    api_key: Annotated[
        Optional[str], typer.Option("--api-key", help="Perplexity API key")
    ] = None,
    base_url: Annotated[
        Optional[str], typer.Option("--base-url", help="Custom base URL for API")
    ] = None,
    model: Annotated[
        Optional[str], typer.Option("--model", help="Model to use")
    ] = None,
    model_router: Annotated[
        Optional[str],
        typer.Option("--model-router", help="Custom model router (JSON format)"),
    ] = None,
    disable_router: Annotated[
        bool,
        typer.Option(
            "--disable-router", help="Disable model router validation (for custom APIs)"
        ),
    ] = False,
    output_type: Annotated[
        OutputType, typer.Option("--output-type", help="Output format")
    ] = OutputType.DEFAULT,
    usage: Annotated[
        bool,
        typer.Option(
            "--usage", help="Show token usage information (default/json only)"
        ),
    ] = False,
    no_citations: Annotated[
        bool,
        typer.Option("--no-citations", help="Disable citations (enabled by default)"),
    ] = False,
    system_prompt: Annotated[
        str, typer.Option("--system-prompt", help="Custom system prompt")
    ] = "Be precise and concise.",
    verbose: Annotated[
        bool, typer.Option("-v", "--verbose", help="Enable debug logging")
    ] = False,
    ctx: typer.Context = None,
) -> None:
    """
    Send a query to Perplexity AI and display the response with rich formatting.

    Output formats:
        default: Rich formatted output with citations at bottom (no box around citations)
        plain: Plain text format with citations appended
        json: JSON format with citations appended to content field

    Examples:
        # Default rich output type
        ./pplx.py "What is AI?"

        # Plain text output type
        ./pplx.py "What is AI?" --output-type plain | echo

        # JSON output type
        ./pplx.py "What is AI?" --output-type json | jq .

        # Custom API without routing
        ./pplx.py "Hello" --api-key "key" --base-url "https://api.example.com" --model "provider/model" --disable-router --output-type json
    """

    # If no message provided, show help
    if message is None:
        print(ctx.get_help())
        raise typer.Exit()

    setup_logging(verbose)
    renderer = OutputRenderer()

    try:
        # Load configuration
        config = load_config(
            api_key=api_key,
            base_url=base_url,
            model=model,
            model_router=model_router,
            disable_router=disable_router,
        )

        # Use provided model or fall back to config model
        selected_model = model or config.model

        # Initialize client
        client = PerplexityClient(config)

        # Show info about selected model (only for default output with verbose)
        if verbose and output_type == OutputType.DEFAULT:
            full_model = client.get_model_name(selected_model)
            renderer.render_info(f"Using model: {selected_model} -> {full_model}")
            renderer.render_info(f"Base URL: {config.base_url}")
            renderer.render_info(f"Router disabled: {disable_router}")

        # Send query (citations enabled by default unless explicitly disabled)
        show_citations = not no_citations
        result = client.query(
            message=message,
            model=selected_model,
            system_prompt=system_prompt,
            show_usage=usage,
            show_citations=show_citations,
        )

        # Render results based on output type
        citations = result["citations"] if show_citations else []

        if output_type == OutputType.DEFAULT:
            renderer.render_default(
                content=result["content"],
                citations=citations,
                usage=result["usage"] if usage else None,
            )
        elif output_type == OutputType.MARKDOWN:
            markdown_output = renderer.render_markdown(result["content"], citations)
            print(markdown_output)
        elif output_type == OutputType.PLAIN:
            plain_output = renderer.render_plain(result["content"], citations)
            print(plain_output)
        elif output_type == OutputType.JSON:
            json_output = renderer.render_json(
                content=result["content"],
                citations=citations,
                usage=result["usage"] if usage else None,
                model_used=result["model_used"],
            )
            print(json_output)

        logger.debug("Query completed successfully")

    except ValueError as e:
        if output_type == OutputType.JSON:
            error_json = json.dumps({"error": str(e)}, indent=2)
            print(error_json)
        else:
            renderer.render_error(str(e))
        raise typer.Exit(1)

    except Exception as e:
        if output_type == OutputType.JSON:
            error_json = json.dumps({"error": f"Unexpected error: {str(e)}"}, indent=2)
            print(error_json)
        else:
            renderer.render_error(f"Unexpected error: {str(e)}")
        logger.exception("Unexpected error occurred")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
