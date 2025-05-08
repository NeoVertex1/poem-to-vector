import re
import requests
import time
import numpy as np
import threading
from collections import Counter
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.live import Live
from rich.prompt import Prompt

console = Console()

emotions = ["joy", "sadness", "anger", "fear", "trust", "anticipation", "surprise"]

def parse_poem(poem_text):
    lines = [l.strip() for l in poem_text.strip().split('\n') if l.strip()]
    all_words = [re.findall(r'\b\w+\b', l.lower()) for l in lines]
    word_set = sorted(set(w for ws in all_words for w in ws))
    word_index = {w: i for i, w in enumerate(word_set)}
    return lines, all_words, word_index

def assign_emotion_by_word(word, word_index):
    idx = word_index[word]
    return emotions[idx % len(emotions)]
def word_to_hue(word, word_index): return word_index[word] % 12

def golden_ratio_analysis(word_counts):
    ratios = np.array([word_counts[i+1]/word_counts[i] if word_counts[i] else 0
                       for i in range(len(word_counts)-1)])
    phi = (1 + 5 ** 0.5) / 2
    closest_i = np.argmin(np.abs(ratios - phi)) if len(ratios) else 0
    return ratios, phi, closest_i

def binary_string_from_word(word): return ''.join('1' if c in "aeiou" else '0' for c in word.lower())
def binary_whispers(all_words):
    return [''.join(binary_string_from_word(w) for w in ws) for ws in all_words]

def recurrence_matrix(all_words):
    seq = [w for line in all_words for w in line]
    n = len(seq)
    mat = np.zeros((n, n), dtype=int)
    for i in range(n):
        mat[i] = (np.array(seq) == seq[i]).astype(int)
    return mat, seq
def zeros_embrace(all_words):
    flat_words = [w for ws in all_words for w in ws]
    counts = Counter(flat_words)
    rare = {w for w, c in counts.items() if c == 1}
    void_map = [[1 if w in rare else 0 for w in ws] for ws in all_words]
    return rare, void_map

def collect_poetry_data(poem_text):
    lines, all_words, word_index = parse_poem(poem_text)
    word_counts = np.array([len(ws) for ws in all_words])
    emotions_table = [[(w, assign_emotion_by_word(w, word_index)) for w in ws] for ws in all_words]
    ratios, phi, phi_idx = golden_ratio_analysis(word_counts)
    binary_lines = binary_whispers(all_words)
    all_bits = ''.join(binary_lines)
    rec_mat, rec_seq = recurrence_matrix(all_words)
    hues = [[(w, word_to_hue(w, word_index)) for w in ws] for ws in all_words]
    rare, void_map = zeros_embrace(all_words)
    return {
        "lines": lines,
        "all_words": all_words,
        "word_index": word_index,
        "word_counts": word_counts,
        "emotions_table": emotions_table,
        "ratios": ratios,
        "phi": phi,
        "phi_idx": phi_idx,
        "binary_lines": binary_lines,
        "all_bits": all_bits,
        "recurrence_matrix": rec_mat,
        "rec_sequence": rec_seq,
        "hues": hues,
        "rare_words": rare,
        "void_map": void_map
    }

def summarize_metrics(data, cycle):
    emo_counts = Counter([e for line in data['emotions_table'] for _, e in line])
    mean_ratio = np.mean(data['ratios']) if len(data['ratios']) > 0 else 0
    std_ratio = np.std(data['ratios']) if len(data['ratios']) > 0 else 0
    phi = data['phi']
    phi_val = data['ratios'][data['phi_idx']] if len(data['ratios']) > 0 else 0
    ones = data['all_bits'].count('1'); zeros = data['all_bits'].count('0')
    frac_density = data['recurrence_matrix'].sum() / data['recurrence_matrix'].size if data['recurrence_matrix'].size else 0
    all_hues = [h for line in data['hues'] for _, h in line]
    unique_hues = len(set(all_hues)); hue_count = dict(Counter(all_hues))
    rare = data['rare_words']
    metrics = (
        f"[cycle]{cycle}[/cycle]\n"
        f"<emotions>{emo_counts}</emotions>\n"
        f"<ratios>mean={mean_ratio:.3f} std={std_ratio:.3f}</ratios>\n"
        f"<phi>{phi:.3f}</phi>\n"
        f"<phi_val>{phi_val:.3f}</phi_val>\n"
        f"<binary>ones={ones} zeros={zeros}</binary>\n"
        f"<fractaldensity>{frac_density:.3f}</fractaldensity>\n"
        f"<hues>{unique_hues}/12 {hue_count}</hues>\n"
        f"<voids>{len(rare)} ex: {list(rare)[:5]}</voids>\n"
    )
    return metrics

def get_llm_response(prompt: str, model: str = "phi4:latest", ollama_url: str = 'http://localhost:11434/api/chat'):
    system_prompt = (
        "You are a mathematical textual analyst. For every prompt, you will:\n"
        "1. <data>: Summarize the mathematical and structural features of the provided poem and its computed metrics.\n"
        "2. <analysis>: Offer a deep analysis: discuss patterns, connections, creative process found in the math structure.\n"
        "3. <new_poem>: Write a new poem of any kind, but its structure (line lengths, ratios, use of voids, hue/emotional flow etc) should echo or remix the same mathematical ideas. "
        "Use plain text for the poem, surrounded by the tag.\n"
        "Return ONLY valid XML with those three tags. Example:\n"
        "<data>text here</data>\n<analysis>text</analysis>\n<new_poem>poem here</new_poem>"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 1}
    }
    start_time = time.time()
    try:
        response = requests.post(ollama_url, json=payload)
        response.raise_for_status()
        response_data = response.json()
        llm_text_response = response_data.get('message', {}).get('content', 'No response content.')
        duration = time.time() - start_time
        return llm_text_response, duration
    except Exception as e:
        duration = time.time() - start_time
        return f"[LLM ERROR] {e}", duration

def parse_llm_xml(text):
    d = {"data": "", "analysis": "", "new_poem": ""}
    try:
        text = text.replace("</data>", "</data>\n").replace("</analysis>", "</analysis>\n").replace("</new_poem>", "</new_poem>\n")
        for tag in d.keys():
            patt = re.compile(rf"<{tag}>(.*?)</{tag}>", re.DOTALL | re.IGNORECASE)
            m = patt.search(text)
            if m:
                d[tag] = m.group(1).strip()
    except Exception as e:
        d["analysis"] = f"Could not parse XML ({e})"
        d["new_poem"] = text
    return d

def append_to_poems_file(data, filename="poems.xml"):
    entry = (
        f"<entry>\n"
        f"  <timestamp>{datetime.now().isoformat()}</timestamp>\n"
        f"  <cycle>{data.get('cycle','')}</cycle>\n"
        f"  <data>{data.get('data','')}</data>\n"
        f"  <analysis>{data.get('analysis','')}</analysis>\n"
        f"  <new_poem>{data.get('new_poem','')}</new_poem>\n"
        f"</entry>\n"
    )
    with open(filename, "a") as f:
        f.write(entry)

def capture_keypress(state):
    # background thread to capture "p" or "q"
    import sys
    import termios
    import tty
    while not state["stop"]:
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            ch = sys.stdin.read(1)
            if ch.lower() == "p":
                state["paused"] = True
            elif ch.lower() == "q":
                state["quit"] = True
        except Exception:
            pass
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

def main(poem_file=None):
    if poem_file:
        with open(poem_file, "r") as f:
            poem = f.read()
    else:
        console.print("[bold yellow]Paste your poem, end with Ctrl-D (Ctrl-Z on Windows):[/bold yellow]")
        lines = []
        try:
            while True:
                lines.append(input())
        except EOFError:
            pass
        poem = "\n".join(lines)
    cycle = 0
    model = "phi4:latest"
    ollama_url = "http://localhost:11434/api/chat"
    state = {"paused": False, "stop": False, "quit": False}
    capture_thread = threading.Thread(target=capture_keypress, args=(state,), daemon=True)
    capture_thread.start()
    while True:
        cycle += 1
        console.clear()
        console.rule(f"[bold green]AI Poetry Evolution - Cycle {cycle}")
        console.print(Panel(poem, title=f"Current Poem for Analysis (Cycle {cycle})"))
        data = collect_poetry_data(poem)
        metrics = summarize_metrics(data, cycle)
        console.print(Panel(metrics, title="Mathematical Analysis"))
        prompt = f"[Cycle {cycle}] Poem so far:\n{poem}\n\n{metrics}\nRespond per your XML guide."
        with console.status("[bold green]Waiting for LLM / Ollama... Press 'p' to pause after this cycle, 'q' to quit."):
            llm_text, elapsed = get_llm_response(prompt, model=model, ollama_url=ollama_url)
        console.print(Panel(f"Ollama Elapsed: {elapsed:.2f} s", title="Status"))
        xml_data = parse_llm_xml(llm_text)
        append_to_poems_file({**xml_data, "cycle": str(cycle)}, filename="poems.xml")
        console.print(Panel(xml_data.get('data',""), title="[data]"))
        console.print(Panel(xml_data.get('analysis',""), title="[analysis]"))
        console.print(Panel(xml_data.get("new_poem",""), title="[new_poem]", subtitle="Press Enter to continue | p to pause | q to quit"))
        if not llm_text.startswith("[LLM ERROR]") and xml_data.get("new_poem", ""):
            poem = xml_data["new_poem"]
        # Pause handling
        if state.get("quit"):
            console.print("[bold red]User requested quit. Exiting...[/bold red]")
            break
        if state.get("paused"):
            console.print("[bold yellow]Cycle paused. Press Enter to continue, or q to quit.[/bold yellow]")
            state["paused"] = False
            while True:
                resp = Prompt.ask("[PAUSED] Press ENTER to continue, or q to quit", default="")
                if resp.strip().lower() == "q":
                    state["quit"] = True
                    break
                if resp.strip() == "":
                    break
            if state.get("quit"):
                break

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AI Poetry Rich CLI with pause/continue/quit, no Textual")
    parser.add_argument("--poem-file", default=None)
    args = parser.parse_args()
    main(args.poem_file)
