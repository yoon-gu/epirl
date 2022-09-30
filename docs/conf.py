# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Optimal Control of Mathematical Biology Models'
copyright = '2022, Yoon-gu Hwang'
author = 'Yoon-gu Hwang'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

language = 'ko'

today_fmt = '%Y년 %B %d일'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

html_last_updated_fmt = '%Y년 %B %d일'

# -- LaTeX configuration ---------------------------------------------------
latex_engine = 'xelatex'
latex_elements = {
	'papersize':'a4paper',
    'fontpkg': r'''
\setmainfont[Kerning=On,Mapping=tex-text]{나눔명조}
\setsansfont[Kerning=On,Mapping=tex-text]{나눔명조}
\setmonofont{JetBrains Mono}
''',
    'preamble': r'''
\usepackage{kotex}
'''
}
latex_show_urls = 'footnote'