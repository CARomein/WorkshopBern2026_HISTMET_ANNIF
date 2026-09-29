# HISTMET & Annif: From Historical Text to Subject Metadata

Workshop materials in two parts, the first drawing on the NWO-funded project HISTMET (HIStorical Themes via METadata), each a self-contained Google Colab notebook. Both use the same corpus: 4,930 early modern normative texts (*Policeyordnungen*) from the City-Republic of Bern (1528–1798), labelled by hand following the classification of the *Repertorium der Policeyordnungen*.

| Part | Notebook | Question | Runtime | |
|---|---|---|---|---|
| One | `01_HISTMET_text_length.ipynb` (Kamyab Karimi, C. Annemieke Romein) | Can unsupervised clustering recover subjects, and does text length explain why it fails on short text regions? | T4 GPU, c. 5 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CARomein/WorkshopBern2026_HISTMET_ANNIF/blob/master/01_HISTMET_text_length.ipynb) |
| Two | `02_Annif_subject_indexing.ipynb` (C. Annemieke Romein, Jona M. Hassenbach) | How well does supervised subject indexing with Annif perform, and what matters more: transcription quality or configuration? | CPU | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CARomein/WorkshopBern2026_HISTMET_ANNIF/blob/master/02_Annif_subject_indexing.ipynb) |

Open the notebooks in the order given; each runs in its own Colab session, so no installation on your own computer is required. To keep your changes, choose *File → Save a copy in Drive* after opening a notebook.

## Repository contents

The notebooks sit in the root of the repository. The folder `histmet/` contains the data for Part One (`bern_prepared.csv.gz`) and the expected result figures. The remaining files serve Part Two: `projects.cfg` (the Annif project definitions), `data/` (the Annif vocabularies), the folders `trans_<year>_level<n>/` (transcriptions and hand-assigned labels, split into training and test sets), `results_default/` and `results_optimal/` (reference results), and `visualize_simple.py` (a plotting helper).

## Citation

Please cite this repository via its Zenodo DOI (*to be added*); see also `CITATION.cff`.

## Licence

*To be added.*
