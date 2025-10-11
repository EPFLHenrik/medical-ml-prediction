# Qualitative conclusions got from EDA - Andrea

## Boxplots

1. _AGE80 : the positive are super shifted up - pretty logic. In general the dataset is composed of middle aged people;
2. _BMI5 : the positives are shifted up - obese BMI is >=30, so positives are in the range of ~obese;
3. _DRDXAR1 : seems to be super related to having arthritis;
4. _DRNKWEK : at least counterintuitive;
5. _FRUTSUM : not worth;
6. -> _LTASTH1 : no asthmatics smh;
7. -> _RACE : i think i shouldn't have removed outliers here;
8. _RFCHOL : high chol => cardiac diseases (wow);
9. _SMOKER3 : former smokers more prone;
10. -> ADDEPEV2 : try w/ outliers;
11. ALCDAY5 : counterintuitive but could have quit drinking due to health + consider age;
12. BPHIGH4 : high blood pressure seems a factor;
13. -> CHCCOPD1;
14. -> CHCKIDNY;
15. -> CHCONCR;
16. -> CIMEMLOS;
17. -> CVDSTRK3;
18. -> DIABETE3;
19. DRADVISE : diagnosis <-> salt reduction;
20. GENHLTH : self-consciousness;
21. HAREHAB : just people with MICHD get rehab ofc;
22. HAVART3: again chronic arthritis evidence;
23. HIVTST6 : generational bias prolly;
24. -> HLTHPLN1;
25. HTIN4 : over 6.5 for no reason + get a stroke man...;
26. HTM4 : same;
27. INCOME2 : salary inequalities;
28. -> MEDCOST;
29. PA1MIN_ : little up shift for diagnosed;
30. SEX : more evidence for males despite being super-less in the dataset;
31. SMOKE100 : obv evidence;
32. TOLDHI2 : other evidence for cholesterol;
33. -> USENOW;
34. -> VETRAN3;

## Histograms for w/ outliers variables

1. _LTASTH1 : asthmatics more prone;
2. _RACE : watch out hispanics ->_HISPANIC;
3. ADDEPEV2 : depression!;
4. CHCCOPD1, CHCKIDNY, CHCOCNCR, CIMEMLOS, CVDSTRK3 : evidence;
5. DIABETE3 : people with diabetes...;
6. MEDCOST : weak correlation i guess;
7. VETERAN3 : veterans don't have a healthy past.

## Heatmaps

1. TOLDHI2 + BPHIGH4 = corr bw chol and bp;
2. SEX + HAVARTH3 = corr bw sex and arthritis;
3. EMPLOY1 + SEX;
4. EMPLOY1 + EDUCA;
5. EMPLOY1 + INCOME2;
6. INCOME2 + SEX;
7. INCOME2 + EDUCA;
8. EMPLOY1 + WEIGHT2 + HEIGHT3;
9. smh no corr bw SMOKE100 and SMODAY2;
10. FRUIT1 + SMOKDAY2;
11. corr bw EX_ and green consumption;
12. DRADVISE + WTCHSALT : ofc you control sodium if a doctor told you to;
13. DRADVISE + REHABS : generally followed;
14. _DRDXAR1 + RFCHOL : chol and arthritis;
15. _DRDXAR1 + HEIGHT;
16. _DRDXAR + AGE80;
17. BMI with DRNKWEK + FRUITSUM;
18. PA1MIN with FRUIT and VEGGIE sum;