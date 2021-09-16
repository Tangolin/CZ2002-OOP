Lab 1
---
In this lab, the individual time complexity analysis of merge-insertion sort and mergesort is required, it will then be followed by a cross-analysis. Both files can compare by key comparisons or CPU time by a few simple changes.
<br/>
<br/>
### Pre-requisites
---
Run ```pip install -r requirements.txt```
<br/>
<br/>

### Files
---
```hybrid.py```
plots the optimal S value (in terms of CPU time) against various lengths and types of the lists.
<br/>
<br/>
**usage**: ```python hybrid.py --max <choice_of_max> --int <choice_of_int> --cutoff <choice_of_cutoff> --type <choice_of_type>```
<br/>
<br/>

run ```python hybrid.py --help``` for explanation of variables.

##### sample output

![alt text](https://github.com/Tangolin/CZ2101-Algorithm-Design-and-Analysis/blob/main/lab-1/output_10_1_random.jpg)<br/><br/><br/><br/>

```compare.py```
compares the performance between merge-insertion and hybrid sort in terms of CPU time or key comparisons.
<br/>
<br/>
**usage**: ```python compare.py --max <choice_of_max> --cutoff <choice_of_cutoff> --type <choice_of_type> ```
<br/>
<br/>

run ```python compare.py --help``` for explanation of variables.

##### sample output

![alt text](https://github.com/Tangolin/CZ2101-Algorithm-Design-and-Analysis/blob/main/lab-1/10000_CPU_Time_reverse.png)
