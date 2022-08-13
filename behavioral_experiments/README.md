
# Behavioral Experiments

Here, we provide the results of behavioral experiments conducted to provide a human baseline on CVR. We used 20 problem samples for each rule, which corresponds to the lowest data regime. We recruited 21 participants from \href{www.prolific.co}{Prolific}. Participants were instructed to identify the odd stimulus violating the rule they had to infer over a series of trials. Prior to the practice phase, they were quizzed on their understanding of the task. Participants practiced the task on a separate set of visual stimuli different from the benchmark. During the experiment, participants were informed about the start of each block as well as the concomitant rule switch. For each trial, they were presented with 4 choices on the screen and instructed to choose the image which seemed to be different according to the rule that they had to learn. They rated confidence in their choice and received feedback after each trial. In addition, they were asked to describe the rule at the end of each block.

The csv file contains the results of each block peformed by the participants. The columns of the csv file correspond to:

worker_id: a participant identifier (modified to anonymize participants).
age: the participant's age.
sex: the participant's sex.
task_idx: the index of the task.
task_name: the name of the task. It corresponds to the function used to generate the problem samples.
accuracy: the proportion of correct responses.
confidence_min: the minimum confidence value reported by the participant from 0 to 100. 
confidence_max: the maximum confidence value reported by the participant from 0 to 100.
confidence_average: the average confidence value.
rt_avg: the average reaction time.
task_description: the task description provided by participants.

