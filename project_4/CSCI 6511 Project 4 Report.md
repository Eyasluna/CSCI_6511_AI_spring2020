# CSCI 6511 Project 4 Report

### Yibo Fu

## G25190736

## **Algorithm Choice**

I implemented Viterbi algorithm to solve the  Customer Journey problem in HMM model. Viterbi is a kind of dynamic programming Viterbi algorithm that makes uses of a dynamic programming trellis. Viterbi algorithm for finding optimal sequence of hidden states. Given an observation sequence and an HMM λ = (A,B), the algorithm returns the state path through the HMM that assigns maximum likelihood to the observation sequence.

![](https://github.com/Eyasluna/CSCI_6511_AI_spring2020/blob/master/project_4/HMM_1.PNG?raw=true)

The Viterbi algorithm is identical to the forward algorithm except that it takes the max over the previous path probabilities whereas the forward algorithm takes the sum. Note also that the Viterbi algorithm has one component that the forward algorithm doesn’t have: **backpointers**. The reason is that while the forward algorithm needs to produce an observation likelihood, the Viterbi algorithm must produce a probability and also the most likely state sequence. We compute this best state sequence by keeping track of the path of hidden states that led to each state.

Finally, we can give a formal definition of the Viterbi recursion as follows:

![HMM_2](https://github.com/Eyasluna/CSCI_6511_AI_spring2020/blob/master/project_4/HMM_2.PNG?raw=true)

## **Problems We Have**

Give an HMM for this situation. Specify the states, the transition probabilities and the emission probabilities. Output the most likely explanation of the state given the observations.

## **How to run script**

``` shell
$ python HMM_customer.py [file]
```

While the [file] is the test file path. 

![](https://github.com/Eyasluna/CSCI_6511_AI_spring2020/blob/master/project_4/HMM_3.PNG?raw=true)

