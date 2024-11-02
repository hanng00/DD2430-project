For all
- gpt-4o-mini
- hp: N=50, w = 24 * 30 * 3 (three months)
- Dataset: ICEWS18
------
result_01
* GenTKG - Base Case - 2000 samples
* Hits@{1,3,5,10} - 0.24, 0.39, 0.44, 0.49

result_02
* GenTKG - ZR Case - 1 000 samples (but its all zeros so can just as well use that)
* Correct hits are hallucinations.
* Hits@{1,3,5,10} - 0.02, 0.02, 0.02, 0.03

result_03
* GenTKG-cosine - ZR Case - 2000 samples
* Hits@{1,3,5,10} - 0.1, 0.2, 0.23, 0.26

result_03b
* GenTKG-cosine - Base Case - 2000 samples
* Hits@{1,3,5,10} - 0.24, 0.39, 0.44, 0.49

result_04
* NaiveTKG - Base Case - 2000 samples
* Hits@{1,3,5,10} - 0.26, 0.38, 0.42, 0.48

result_05
* NaiveTKG - ZR Case - 2000 samples
* Hits@{1,3,5,10} - 0.09, 0.16, 0.19, 0.25