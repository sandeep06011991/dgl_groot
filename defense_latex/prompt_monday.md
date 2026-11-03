Section 1:

Lets freeze the table and show a tentative prompt with expectated text in latex before we do more runs. 

Gsplit (0% cached) + DGL comparision is the only comparision we will use. 
Quiver allows caching but its causing feat loading is efficient. 
To simulate the case where caching exists which would make the feat loading efficient, we compare the training time. 
So e2e epoch time matters , if cachine exists feat loading would be efficient, so we compare only the training times. 

We structured too many tables which is hard to put in the latex. 
The structure for section 1 should contain 2 tables (one per graph).
Within each graph, we have two horizontal sections GAT and SAGE. 
The vertical sections contain two the pcie and nvlink  
We report the epoch_time (training_time) in brackets. 

Delta is 2 values epoch time outside and trainin time in brackets. 
Further Delta should be positive when gsplit gains. % improvement. 

You dont have to ablate both hidden and depth together. Each axis must be ablataed invidually. delete this clutter. I think 512 hidden size is good enough for us to ablate depth. Depth go up till 6 layers. 


I think we already have a set of configurations. Lets break our setup into two parts. 
1. Generate a settings file, Why we do this as the some of the runs we need might already be there. 
Therefore iterating bash for loops is incosistent as some wwholes mught be present. 
2. The run file iterates over the settings and does the runs. 

Create a seperate branch, for this task

The key conclusion of this the gains are either not there or they are diminishing both are actual problems we can attack over with hybrid situations.

Finally section 1 contains 2 tables, which can be generated from the main list of all results. 
Keeping all runs in the same place ensures there is no inconsitency between ablation results and the main results. 

Lets also adjust the tone. this is a research so restrict yourself to stating points without overdrawing conclusion. 
total epoch time neednt be netiuoned, you have to mention per epoch time only. 

Cross check that all scripts for automated population of latex tables is created. 

Remove the mention of quiver. 

Section 2:
The hybrid parallelisms tables are little cobbled with the text can you fix it.
Section 3: 

Lets fix the fidelity of the simulator to ensure its closer to predicted values.
Fidelity is found if we can predict gsplit values for 1 hidden row variant and depth variant correctly. 

Section 5:

Lets use one or two  of the sage hidden sizes we already found with max potential and one sitatuon with depth  Lets run all the hybrid parallelisms combinations and its actually not htat many values we can do exhaustive exploration.  This is still raw 