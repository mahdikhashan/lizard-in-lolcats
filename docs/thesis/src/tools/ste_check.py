"""Check the prose of the thesis against the ASD-STE100 linter of this repository.

The script removes the LaTeX markup from each .tex file and writes plain Markdown text.
Then it runs .claude/skills/asd-ste100/scripts/ste-lint.py on that text.

- Headings become Markdown headings.
- Tables, equations, TikZ pictures and figure files are removed. Captions stay as sentences.
- Inline math becomes the word "X". Citations disappear. References become "1".

Usage: python3 tools/ste_check.py [files...]   (default: all chapters, front matter and appendices)
"""
import os
import pathlib
import re
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent.parent
LINT = HERE.parents[2] / '.claude' / 'skills' / 'asd-ste100' / 'scripts' / 'ste-lint.py'
DROP_ENVS = ('tabular', 'tabularx', 'longtable', 'equation', 'equation*', 'align', 'align*',
             'tikzpicture', 'displaymath', 'gather', 'gather*', 'multline', 'verbatim')


def strip_env(text, env):
    pattern = re.compile(r'\\begin\{' + re.escape(env) + r'\}.*?\\end\{' + re.escape(env) + r'\}', re.S)
    return pattern.sub('\n\n', text)


def braced(text, start):
    """Return (content, end) of the {...} group that starts at text[start] == '{'."""
    depth = 0
    for i in range(start, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                return text[start + 1:i], i + 1
    return text[start + 1:], len(text)


def keep_command(text, name, fmt):
    """Replace \\name{arg} with fmt.format(arg), with nested braces."""
    out, i = [], 0
    token = '\\' + name
    while True:
        j = text.find(token, i)
        if j < 0:
            out.append(text[i:])
            return ''.join(out)
        k = j + len(token)
        if k < len(text) and (text[k].isalpha()):
            out.append(text[i:k])
            i = k
            continue
        while k < len(text) and text[k] in '*':
            k += 1
        if k < len(text) and text[k] == '[':
            k = text.find(']', k) + 1
        if k >= len(text) or text[k] != '{':
            out.append(text[i:k])
            i = k
            continue
        arg, end = braced(text, k)
        out.append(text[i:j])
        out.append(fmt.format(arg))
        i = end


def detex(text):
    text = re.sub(r'(?<!\\)%.*', '', text)
    text = re.sub(r'\\\[.*?\\\]', '\n\n', text, flags=re.S)
    for env in DROP_ENVS:
        text = strip_env(text, env)
    text = re.sub(r'\$[^$]*\$', 'X', text)
    for name in ('chapter', 'addchap', 'section', 'subsection', 'subsubsection', 'paragraph'):
        level = {'chapter': '#', 'addchap': '#', 'section': '##', 'subsection': '###'}.get(name, '####')
        text = keep_command(text, name, '\n\n' + level + ' {}\n\n')
    text = keep_command(text, 'caption', '\n\n{}\n\n')
    for name in ('textbf', 'emph', 'textit', 'textsc', 'mbox', 'text', 'enquote', 'underline', 'textrm'):
        text = keep_command(text, name, '{}')
    text = keep_command(text, 'code', '`{}`')
    text = keep_command(text, 'texttt', '`{}`')
    text = keep_command(text, 'url', 'URL')
    text = keep_command(text, 'href', 'link')
    for name in ('cite', 'citep', 'citet', 'citealp', 'label', 'includegraphics', 'input', 'vspace', 'hspace',
                 'addcontentsline', 'markboth', 'setlength', 'thispagestyle', 'phantomsection', 'captionsetup'):
        text = keep_command(text, name, '')
    for name in ('cref', 'Cref', 'ref', 'eqref', 'autoref', 'pageref'):
        text = keep_command(text, name, '1')
    text = re.sub(r'\\item\s*(\[[^\]]*\])?', '\n- ', text)
    text = re.sub(r'\\begin\{[^}]*\}(\[[^\]]*\])?(\{[^}]*\})?', '\n\n', text)
    text = re.sub(r'\\end\{[^}]*\}', '\n\n', text)
    text = text.replace('~', ' ').replace('\\%', '%').replace('\\&', '&').replace('\\_', '_')
    text = text.replace('``', '"').replace("''", '"').replace('---', ' - ').replace('--', '-')
    text = re.sub(r'\\[a-zA-Z]+\*?', ' ', text)
    text = text.replace('{', '').replace('}', '')
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n\s*\n+', '\n\n', text)
    return text.strip() + '\n'


def main(argv):
    files = [pathlib.Path(a) for a in argv] or sorted(
        list((HERE / 'frontmatter').glob('*.tex')) + list((HERE / 'chapters').glob('*.tex'))
        + list((HERE / 'appendices').glob('*.tex')))
    failed = 0
    with tempfile.TemporaryDirectory() as tmp:
        for f in files:
            md = pathlib.Path(tmp) / (f.stem + '.md')
            md.write_text(detex(f.read_text()))
            if os.environ.get('STE_DUMP'):
                pathlib.Path(os.environ['STE_DUMP']).mkdir(parents=True, exist_ok=True)
                (pathlib.Path(os.environ['STE_DUMP']) / md.name).write_text(md.read_text())
            result = subprocess.run([sys.executable, str(LINT), str(md)], capture_output=True, text=True)
            status = 'ok' if result.returncode == 0 else 'FAIL'
            failed += result.returncode != 0
            lines = [l for l in result.stdout.splitlines() if l.strip()]
            print(f'== {f.name}: {status}')
            for line in lines:
                if line.startswith(str(md)):
                    print('   ' + line.replace(str(md), f.stem))
                elif 'violations' in line:
                    print('   ' + line)
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
