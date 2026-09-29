import os
import argparse
from faster_whisper import WhisperModel
import ollama
from fpdf import FPDF
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

console = Console()

def process_lecture(audio_path, output_pdf = None, ollama_model = "llama3.2:1b"):
    if(os.path.exists(audio_path) == False):
        console.print(f"[bold red]Error:[/bold red] File not found at '{audio_path}'")
        return

    if output_pdf is None:
        output_pdf = os.path.splitext(audio_path)[0] + "_Summary.pdf"

    console.print(f"\n[bold blue]Starting Local Audio Processing[/bold blue]")
    console.print(f"[dim]Target: {os.path.basename(audio_path)}[/dim]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        TimeElapsedColumn(),
        console=console
        ) as progress:

            task1 = progress.add_task("[yellow]Transcribing audio (Whisper)...", total = 100)
            model = WhisperModel("base", device = "cpu", compute_type="int8", cpu_threads = 6)
            segments, info = model.transcribe(audio_path, beam_size = 3)

            full_text = " ".join([segment.text for segment in segments])
            progress.update(task1, completed = 100, description = "[green]Transcription complete!")

            if not full_text.strip():
                 console.print("[bold red]Error:[/bold red] Generated transcript was completely empty.")
                 return

            task2 = progress.add_task("[cyan]Generating summary (Ollama)...", total = 100)

            prompt = (
                "You are an elite academic assistant. Summarize the following lecture transcript. "
                "Provide a title, an overall concise overview, and a bulleted list of core concepts with their definitions:\n\n"
                f"{full_text}"  
            )
            summary_response = ollama.generate(model=ollama_model, prompt=prompt)
            summary_text = summary_response.get("response", "")
            progress.update(task2, completed=100, description="[green]Summary generated!")

            task3 = progress.add_task("[magenta]Writing final PDF layout...", total = 100)

            pdf = FPDF()
            pdf.add_page()

            # Header banner style
            pdf.set_font("Helvetica", style="B", size=16)
            pdf.cell(0, 10, txt="Lecture Summary Notes", ln=True, align="C")
            pdf.ln(10)
        
        # Summary body content text
            pdf.set_font("Helvetica", size=11)
            clean_text = summary_text.encode('latin-1', 'replace').decode('latin-1')
            pdf.multi_cell(0, 7, txt=clean_text)
        
            pdf.output(output_pdf)
            progress.update(task3, completed=100, description="[green]PDF successfully compiled!")

    console.print(f"\n[bold green]Success[/bold green] Your summart is ready at:")
    console.print(f"[underline cyan]{os.path.abspath(output_pdf)}[/underline cyan]\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Completely offline lecture transcript audio summarizer.")
    parser.add_argument("audio", type=str, nargs="?", help="Path to your target audio file (.mp3, .wav, etc.)")
    parser.add_argument("-o", "--output", type=str, default=None, help="Optional custom name/path for output PDF")
    parser.add_argument("--model", type=str, default="llama3.2:1b", help="Installed Ollama model to use for summarization")
    
    args = parser.parse_args()
    audio_path = args.audio or input("Path to your audio file: ").strip()
    if not audio_path:
        parser.error("an audio file path is required")
    process_lecture(audio_path, args.output, args.model)