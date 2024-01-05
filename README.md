# Disability Progression in Multiple Sclerosis
This repository is a collection of tools for assessing disability progression in Multiple Sclerosis based on changes in Expanded Disability Status Scale (EDSS) scores over time.

Explore the definitions here: https://multiple-sclerosis-disability-progression.streamlit.app/

See ``tutorial.ipynb`` for a quick introduction and usage examples of our code. Can't load/render the notebook? Try https://nbviewer.org/.

## Contents
### Definitions
Definitions of fixed baseline, roving reference [1], and EDSS progression. The module ``baselines.py`` contains a function to annotate a fixed or roving reference to a follow-up. The module ``progression.py`` contains a function to annotate the first progression event in a follow-up.
### Evaluation
This directory contains helper functions for cohort-level evaluations and survival analysis with the lifelines [2] package.
### Data
This directory contains an example dataset of 200 follow-ups. Use this dataset as a formatting reference and for testing.
### Webapp toolbox
Frontend elements and visualization helpers for the streamlit app.

## Authors
Gabriel Bsteh<sup>1, 2</sup>, Stefanie Marti<sup>3</sup>, Robert Hoepner<sup>3</sup>

<sup>1</sup>Department of Neurology, Medical University of Vienna, Vienna, Austria\
<sup>2</sup>Comprehensive Center for Clinical Neurosciences and Mental Health, Medical University of Vienna, Vienna, Austria\
<sup>3</sup>Department of Neurology, Inselspital, Bern University Hospital and University of Bern, Switzerland

## References

[1] Kappos L, Butzkueven H, Wiendl H, et al. **Greater sensitivity to multiple sclerosis disability 
worsening and progression events using a roving versus a fixed reference value in a prospective cohort 
study**. Mult Scler. 2018;24(7):963-973. doi:10.1177/1352458517709619

[2] https://lifelines.readthedocs.io/en/latest/index.html
