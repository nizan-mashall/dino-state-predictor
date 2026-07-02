# TinyBC

1. Transfer the training data. 
2. train the model.
3. load and evaluate.

1. watch GPU works in progress
watch -n 5 "squeue -u $USER"
2. demonstare the model performence
sbatch ~/dino-state-predictor/eval.sh
3. source myenv/bin/activate
Insights:
1. Hand_in_eyes config provide much better results than agentview.
2. training with both views seems to better converge, the success rate doesn't improved but the manipulation seems robust.
3. Initalize the manipulator position in constant state improve the model to 92% success rate instead of 70%.
4. Determent how the demos amount effects the results. 
5. adding also patches spatial info improved from 69-72% success rate, adding dropout and additonal model depth caused 75% success rate.
6. use vitb14 instead of vits14 incearse performence by 2%.
7. eliminating normilization and using adamW provide consistet better results than
8. CKP 65 with transformer achived 100% on multiple noise stages.
