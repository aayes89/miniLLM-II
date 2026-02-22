import re
import unicodedata
import os
import ftfy
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

input_path = "corpus_completo.txt"
output_path = "corpus_clean.txt"

# =============================
# REGEX
# =============================
# <ref>...</ref> multilínea no greedy
ref_pattern = re.compile(r"<ref\b[^>]*?>.*?</ref>", re.IGNORECASE | re.DOTALL)

# <ref ... />
ref_selfclose = re.compile(r"<ref\b[^>]*/>", re.IGNORECASE)

# cualquier tag HTML
html_tag_pattern = re.compile(r"<[^>]+>")

# {{ ... }}  (no anidado)
template_pattern = re.compile(r"\{\{[^{}]*\}\}")

# {| ... |}
table_pattern = re.compile(r"\{\|[^{}]*?\|\}", re.DOTALL)

# [[Categoría:...]]
category_pattern = re.compile(r"\[\[\s*Categor[ií]a:[^\]]+\]\]", re.IGNORECASE)

# [[Archivo:...]] o [[File:...]]
file_pattern = re.compile(r"\[\[\s*(Archivo|File):[^\]]+\]\]", re.IGNORECASE)

# [[link|texto]] → texto
wikilink_pattern = re.compile(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]")

# URLs
url_pattern = re.compile(r"https?://[^\s]+")

# entidades HTML (&nbsp; &amp; etc)
entity_pattern = re.compile(r"&[a-zA-Z0-9#]+;")

# espacios múltiples
multiple_spaces = re.compile(r"\s+")

# caracteres fuera de rango razonable
weird_bytes = re.compile(r"[^\x09\x0A\x0D\x20-\x7E\u00A0-\uFFFF]")

mojibake_pattern = re.compile(r"(?:Ã.|Â.|â.|Ô.|├.|┬.|�|ÔÇ.|Ôûü.)")


def fix_encoding(text):
    try:
        return text.encode("latin1").decode("utf-8")
    except:
        return text


def aggressive_clean(text):

    text = fix_encoding(text)
    text = ftfy.fix_text(text)
    text = unicodedata.normalize("NFKC", text)

    text = "".join(
        ch for ch in text
        if unicodedata.category(ch)[0] != "C"
    )

    text = mojibake_pattern.sub("", text)

    text = ref_pattern.sub("", text)
    text = ref_selfclose.sub("", text)
    text = table_pattern.sub("", text)
    text = template_pattern.sub("", text)

    text = category_pattern.sub("", text)
    text = file_pattern.sub("", text)

    text = wikilink_pattern.sub(r"\1", text)

    text = html_tag_pattern.sub("", text)
    text = entity_pattern.sub("", text)
    text = url_pattern.sub("", text)

    stripped = text.strip()

    if stripped.startswith(("==", "*", "#", ";", ":", "|", "!")):
        return ""

    if sum(c.isalpha() for c in stripped) < 25:
        return ""

    stripped = weird_bytes.sub("", stripped)
    stripped = multiple_spaces.sub(" ", stripped)

    return stripped.strip()


def process_block(block):

    output = []
    total_bytes = 0

    for raw_line in block:
        total_bytes += len(raw_line)

        try:
            line = raw_line.decode("utf-8")
        except:
            line = raw_line.decode("latin1", errors="ignore")

        clean = aggressive_clean(line)
        if clean:
            output.append(clean)

    return "\n".join(output), total_bytes


def block_reader(file, block_lines=50000):
    block = []
    for raw_line in file:
        block.append(raw_line)
        if len(block) >= block_lines:
            yield block
            block = []
    if block:
        yield block


if __name__ == "__main__":

    total_size = os.path.getsize(input_path)
    workers = min(12, cpu_count())

    with open(input_path, "rb") as fin, \
         open(output_path, "w", encoding="utf-8") as fout, \
         Pool(workers) as pool, \
         tqdm(total=total_size, unit="B", unit_scale=True, desc="Limpiando") as pbar:

        for cleaned, processed_bytes in pool.imap_unordered(
                process_block,
                block_reader(fin),
                chunksize=1):

            if cleaned:
                fout.write(cleaned + "\n")

            pbar.update(processed_bytes)
