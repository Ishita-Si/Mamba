import nbformat as nbf
nb = nbf.v4.new_notebook(); c = []
c.append(nbf.v4.new_markdown_cell("# Mamba (selective SSM) from scratch — DL-061\nRuns on a free Colab T4. Set Runtime > Change runtime type > T4 GPU.\nCore code: `src/model.py` (written from scratch; official `state-spaces/mamba` repo only used as a reference for init choices)."))
c.append(nbf.v4.new_code_cell("!git clone https://github.com/<YOUR_USERNAME>/mamba-dl.git\n%cd mamba-dl\n!pip -q install matplotlib"))
c.append(nbf.v4.new_code_cell("!python tests_scan.py   # parallel scan == sequential scan, block is causal"))
c.append(nbf.v4.new_markdown_cell("## 1. Selective copying (Colab-scale: L=256, K=16)\nThree models of equal width/depth: Mamba (selective), Mamba-LTI (ablation, no selection), Transformer."))
c.append(nbf.v4.new_code_cell("""for arch in ['mamba','mamba_lti','transformer']:
    !python train.py --task copy --arch {arch} --d_model 64 --layers 2 --d_state 16 --L 256 --K 16 --bs 64 --steps 3000 --seed 0"""))
c.append(nbf.v4.new_markdown_cell("## 2. Character LM on TinyShakespeare"))
c.append(nbf.v4.new_code_cell("""for arch in ['mamba','transformer','mamba_lti']:
    !python train.py --task lm --arch {arch} --d_model 128 --layers 4 --d_state 16 --L 256 --bs 64 --steps 5000 --lr 2e-3"""))
c.append(nbf.v4.new_markdown_cell("## 3. Results table & plots"))
c.append(nbf.v4.new_code_cell("!python make_report_assets.py"))
c.append(nbf.v4.new_code_cell("from IPython.display import Image, display\nfor f in ['results/copy_curves.png','results/lm_curves.png']: display(Image(f))\nprint(open('results/results_table.md').read())"))
nb.cells = c; nbf.write(nb, "mamba_colab.ipynb")
