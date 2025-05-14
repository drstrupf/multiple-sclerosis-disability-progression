# Disability Progression in Multiple Sclerosis
This repository is a collection of tools for assessing disability progression in Multiple Sclerosis based on changes in Expanded Disability Status Scale (EDSS) scores over time. The event annotation algorithm differentiates between **Progression Independent of Relapse Activity (PIRA)** and **Relapse Associated Worsening (RAW)**, and the additional worsening types **PIRA with relapse during confirmation** and **Undefined Worsening** that are required for supporting non-standardized real-world data from observational cohorts.

![Annotated follow-up with PIRA, RAW, and undefined worsening](images/example_with_all_types.png?raw=true "Annotated follow-up with PIRA, RAW, and undefined worsening")
<sup>**Figure 1** Example of a follow-up annotated with our code. Minimal required EDSS increase + 1, next-confirmed roving reference, events confirmed at the next assessment, RAW window 30 days pre-/post-relapse, default option for undefined worsening (re-baselining only), no minimal distance requirement, no event merging.</sup>

Version [1.1.0](https://github.com/drstrupf/multiple-sclerosis-disability-progression/releases/tag/v1.1.0) is the code we used for the publication [Disability progression is a question of definition-A methodological reappraisal by example of primary progressive multiple sclerosis](https://pubmed.ncbi.nlm.nih.gov/39662164/) [1]. This version does not yet support relapses or annotating multiple events. Versions 2.0.0 and later fully cover the annotation functionality of version 1.1.0 and thus reproduce the results generated with this version. The implementation was completely refactored, and the release with relapse support should be used for any further analyses.

## How to use the resources in this repository

See [tutorial.ipynb](https://github.com/drstrupf/multiple-sclerosis-disability-progression/blob/acdbcf836ebb5f03ac7d4c114d1280f12a7b8e41/tutorial.ipynb) for a quick introduction and usage examples of our code. Can't load/render the notebook? Try https://nbviewer.org/.

See [methods.ipynb](https://github.com/drstrupf/multiple-sclerosis-disability-progression/blob/acdbcf836ebb5f03ac7d4c114d1280f12a7b8e41/methods.ipynb) for examples for each definition option and combinations of options. Can't load/render the notebook? Try https://nbviewer.org/.

Explore the definitions here: https://multiple-sclerosis-disability-progression.streamlit.app/

## Authors
Gabriel Bsteh<sup>1, 2</sup>, Stefanie Marti<sup>3</sup>, Robert Hoepner<sup>3</sup>

<sup>1</sup>Department of Neurology, Medical University of Vienna, Vienna, Austria\
<sup>2</sup>Comprehensive Center for Clinical Neurosciences and Mental Health, Medical University of Vienna, Vienna, Austria\
<sup>3</sup>Department of Neurology, Inselspital, Bern University Hospital and University of Bern, Switzerland

**Interested in using our code?** Please contact the authors. 
* [Gabriel Bsteh](https://www.meduniwien.ac.at/web/forschung/researcher-profiles/researcher-profiles/detail/?res=gabriel_bsteh&cHash=0896fd3f091c51c7c5c37b55b83d8def)
* [Robert Hoepner](http://www.neurologie.insel.ch/de/ueber-uns/teams/details/person/detail/robert-hoepner)

## References

[1] Bsteh G, Marti S, Krajnc N, Traxler G, Salmen A, Hammer H, Leutmezer F, Rommer P, Di Pauli F, Chan A, 
Berger T, Hegen H, Hoepner R. **Disability progression is a question of definition-A methodological reappraisal 
by example of primary progressive multiple sclerosis**. *Mult Scler Relat Disord*. 2025 Jan;93:106215. 
doi: 10.1016/j.msard.2024.106215. Epub 2024 Dec 6. PMID: 39662164.

[2] Kappos L, Butzkueven H, Wiendl H, et al. **Greater sensitivity to multiple sclerosis disability 
worsening and progression events using a roving versus a fixed reference value in a prospective cohort 
study**. *Mult Scler*. 2018;24(7):963-973. doi:10.1177/1352458517709619

[3] https://lifelines.readthedocs.io/en/latest/index.html
