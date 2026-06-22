# TinyBC

1. Transfer the training data. 
2. train the model.
3. load and evaluate.

1. watch GPU works in progress
watch -n 5 "squeue -u $USER"
2. demonstare the model performence
sbatch ~/dino-state-predictor/eval.sh

Insights:
1. Hand_in_eyes config provide much better results than agentview
2. training with both views seems to better converge but the results are still not clear.