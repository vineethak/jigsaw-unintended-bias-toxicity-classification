# jigsaw-unintended-bias-toxicity-classification
Detect toxicity in online comments. Reduce bias to non-toxic words commonly occurring in toxic comments 

When we train a model for toxic comment classification it relates toxicity of the comment to coomonly occuring words eg : gay. 
A sentence containing 'gay' could be a toxic or non-toxic sentence depending on the context. The goal of this challengence is to reduce classification bias on such kind  of words. 

The main target, with special weights that highlites data points that mention identities
the main taregt, again, but without any extra weights
5 toxicity types
9 main identites
max value for any out of 9 identities columns
binary column that indecates whether at least one identity was mentioned
All targets, except the last one, were used as soft targets - flaot values from 0 to 1, not hard binary 0/1 targets.

Used common weights for the first loss. The toxicity subtypes were trained without any special weights. And the identites loss (last 12 targets) were trained with 0 weight for the NA identities. 

Ref: https://github.com/iezepov/combat-wombat-bias-in-toxicity/blob/master/README.md

Train Bi-Directional LSTM with sample weights. Sample weights are formed giving more weightage to toxic words. 
weights = np.ones(len(train))
weights += (iden >= 0.5).any(1)
weights += (train["target"].values >= 0.5) & (iden < 0.5).any(1)
weights += (train["target"].values < 0.5) & (iden >= 0.5).any(1)
weights /= weights.mean()  # Don't need to downscale the loss
