# Disability Accrual and Improvement in Multiple Sclerosis
This repository is a collection of tools for assessing disability accrual and improvement in Multiple Sclerosis based on changes in Expanded Disability Status Scale (EDSS) scores over time. The event annotation algorithm differentiates between **Progression Independent of Relapse Activity (PIRA)** and **Relapse Associated Worsening (RAW)**, and the additional event types **PIRA with relapse during confirmation** and **Undefined Worsening** that are required for supporting non-standardized real-world data from observational cohorts. It also detects **Improvement** events.

![Annotated follow-up with all event types](images/example_follow_up.png?raw=true "Annotated follow-up with all event types")
<sup>**Figure 1** Example of a follow-up annotated with our code. Symmetric annotation mode (accrual and improvement), minimal required EDSS increase/decrease + 1, events confirmed at the next assessment, RAW window 30 days pre-/post-relapse, undefined worsening possible at all assessments, no minimal distance requirement, no event merging.</sup>

Version **3.0.0** supports relapses, improvement, and multiple event annotation. In addition, it contains a few minor bug fixes, and the implementation was fully refacored, so this version should be used for all further analyses.

## How to use the resources in this repository

See [tutorial.ipynb](https://github.com/drstrupf/multiple-sclerosis-disability-progression/blob/main/tutorial.ipynb) for a quick introduction and usage examples of our code. Can't load/render the notebook? Try https://nbviewer.org/.

See [methods.ipynb](https://github.com/drstrupf/multiple-sclerosis-disability-progression/blob/main/methods.ipynb) for examples for each definition option and combinations of options. Can't load/render the notebook? Try https://nbviewer.org/.

Explore the definitions here: https://multiple-sclerosis-disability-progression.streamlit.app/


### Event types
![Event types](images/six_types.png?raw=true "Event types")
<sup>**Figure 2** Event types. **A)** Relapse-associated worsening (RAW). **B)** Progression Independent of Relapse Activity (PIRA). **C)** Improvement. **D)** PIRA with relapse during confirmation. **E)** Undefined worsening, event detected at the post-relapse re-baselining assessment. **F)** Undefined worsening, mixed type.</sup>


## Authors
Gabriel Bsteh<sup>1, 2</sup>, Stefanie Marti<sup>3</sup>, Robert Hoepner<sup>3</sup>

<sup>1</sup>Department of Neurology, Medical University of Vienna, Vienna, Austria\
<sup>2</sup>Comprehensive Center for Clinical Neurosciences and Mental Health, Medical University of Vienna, Vienna, Austria\
<sup>3</sup>Department of Neurology, Inselspital, Bern University Hospital and University of Bern, Switzerland

**Interested in using our code?** Please contact the authors. 
* [Gabriel Bsteh](https://www.meduniwien.ac.at/web/forschung/researcher-profiles/researcher-profiles/detail/?res=gabriel_bsteh&cHash=0896fd3f091c51c7c5c37b55b83d8def)
* [Robert Hoepner](http://www.neurologie.insel.ch/de/ueber-uns/teams/details/person/detail/robert-hoepner)

Found a **bug**? Do you have a **feature request**? We would appreciate your feedback!
Please [**open an issue**](https://github.com/drstrupf/multiple-sclerosis-disability-progression/issues).

## Archived versions

Version [2.1.0](https://github.com/drstrupf/multiple-sclerosis-disability-progression/releases/tag/v2.1.0) is the code we used for the publication [Dissecting definitions of disability accrual in relapsing multiple sclerosis-Have we reached standardization yet?](https://pubmed.ncbi.nlm.nih.gov/41351456/) [4]. Versions 2.0.0 and later fully cover the annotation functionality of version 1.1.0 and thus reproduce the results generated with this version.

Version [1.1.0](https://github.com/drstrupf/multiple-sclerosis-disability-progression/releases/tag/v1.1.0) is the code we used for the publication [Disability progression is a question of definition-A methodological reappraisal by example of primary progressive multiple sclerosis](https://pubmed.ncbi.nlm.nih.gov/39662164/) [1]. This version does not yet support relapses or annotating multiple events. 

## References

[1] Bsteh G, Marti S, Krajnc N, Traxler G, Salmen A, Hammer H, Leutmezer F, Rommer P, Di Pauli F, Chan A, 
Berger T, Hegen H, Hoepner R. **Disability progression is a question of definition-A methodological reappraisal 
by example of primary progressive multiple sclerosis**. *Mult Scler Relat Disord*. 2025 Jan;93:106215. 
doi: 10.1016/j.msard.2024.106215. Epub 2024 Dec 6. PMID: 39662164.

[2] Kappos L, Butzkueven H, Wiendl H, et al. **Greater sensitivity to multiple sclerosis disability 
worsening and progression events using a roving versus a fixed reference value in a prospective cohort 
study**. *Mult Scler*. 2018;24(7):963-973. doi:10.1177/1352458517709619

[3] https://lifelines.readthedocs.io/en/latest/index.html

[4] Bsteh G, Marti S, Hammer H, Krajnc N, Guger M, Di Pauli F, Kraus J, Enzinger C, Chan A, Berger T, Hegen H, Hoepner R. **Dissecting definitions of disability accrual in relapsing multiple sclerosis-Have we reached standardization yet?** *Mult Scler*. 2026 Feb;32(2):179-191. doi: 10.1177/13524585251396283. Epub 2025 Dec 6. PMID: 41351456; PMCID: PMC12916871.
