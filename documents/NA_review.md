# Variables review

###Categories of variables we have : 
- **Demographics** (e.g. age, sex, race etc)
- **Physical measures/descriptions** (e.g. height, weight, BMI)
- **Medical History and chronic conditions**
- **Lifestyle - behaviours, habits**
- **Healthcare access - coverage and checkups**
- **Perceived/self-assessed health**
- **Contextual/social** (e.g. employment)
- **Derived or redundant, metadata**

### no_columns -- keep for now
Columns I would keep for now (not necessarily in final model) from the no_columns list of Henrik. We could still end up
removing them once we check their weights in the log regression : 
- `SCNTMEL1`- how often were people stressed ab not having money for nutritios meals. imbalanced but the people that were stressed could be correlated w the target. to check!
- `SCNTWRK1` - how many hours they work per week in all jobs combined.
- `CVDINFR4` - ever diagnosed w heart attack, again keep if there isnt another problem with this variable. 
Imbalanced distribution but thats natural
- `CVDCRHD4`- ever had CHD, same as above. imbalance is expected.

**NB** : CVINF4 and CVDCRHD4 and were not in x_indices, so actually we cannot add them.

### not_interesting -- keep for now
- `MARITAL` - sociodemocratic factor, can correlate w mental health, stress etc. keep for now, remove later if needed.
- `EDUCA`
- `RENTHOM1`
- `CHILDREN`
- `_CRACE1`


#### Notes
- Disease-related variables (`CVDINFR4`, `CVDCRHD4`, `_MICHD`) are naturally imbalanced - this is expected. 
- Keep socioeconomic, sociodemocratic variables like marital status for now. We can exclude later after checking 
feature coefficients in log regression or using Lasso regularization for log regression.
- These variables will be re-evaluated after running the log regression and feature correlation checks.





