## BPR Model with Attribute-to-Feature Mappings

*Warning : This repository is being developed, and not ready for use*

This is a simple implementation for paper [Learning Attribute-to-Feature Mappings for Cold-Start Recommendations](http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=5693971&tag=1), which can be applied to cold-start situations in recommendation systems.

In detail, Bayesian Personalized Ranking (BPR) model is a matrix factorization model for item recommendation. However, when new item comes with no rating information (i.e. cold-start), it's better to use the private information (i.e. attributes) to infer probable ratings of the items, or to infer feature of the underlying BPR model for better prediction for new items.

The paper offered four simple mapping functions (and a random prediction as baseline), which make it possible to predict under cold-start situation with underlying BPR model. Please refer to the paper for further detail.

Code of BPR model are modified from [This repository](https://github.com/gamboviol/bpr). Many thanks to gamboviol.

