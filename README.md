# Code repository for GNN_Ranking

This repository contains the code for the Master's thesis "Aproximando el cálculo de métricas relacionales utilizando GNN´s", a work carried out by Andrés Domínguez Ruiz in 2025. This code contains the framework for running the betweenness, closeness, local clustering, and page rank models.

Code is written in Python and the proposed model is implemented using Pytorch.

**Main package Requirements**  
Pytorch, NetworKit, NetworkX and SciPy.  
Use of conda environment is recommended, but not necessary.
Experiments in paper used PyTorch (0.4.1) and Python (3.7).

**Running the model code**:  
There are four variants of our model: Betweenness Centrality variant, Closeness Centrality variant, Local Clustering variant, and Page Rank variant. These can be run via four scripts `betweenness.py`, `closeness.py`, `local_clustering.py` and `page_rank.py`. To test the model, we have provided some synthetic graph datasets. These datasets are based on three different directed graph types: Scale-free (SF), Erdos-Renyi (ER) and Gaussian Random Partition graphs (GRP). 

To run betweenness variant (on scale-free datasets) :
```
python betweenness.py --g SF
```

For closeness variant (on scale-free datasets):
```
python closeness.py --g SF
```

For local clustering variant (on scale-free datasets):
```
python local_clustering.py --g SF
```

For page rank variant (on scale-free datasets):
```
python page_rank.py --g SF
```

"SF" can be replaced by "ER" or "GRP".

**Dataset Details**:
Graphs are created using script `./datasets/generate_graph.py`, which generates graphs with number of nodes varying from 5,000 to 10,000 (randomly chosen). Graph generation parameters are also randomly selected. After generation of graphs, betweenness and closeness centrality are calculated and stored in pickle file in folder `./datasets/graphs/`.

Training and test splits are created using `./datasets/create_dataset.py`. 40 graphs are used for training and 10 graphs for testing. Training samples are created by permuting node sequence while generating adjacency matrices from the graphs (training). Training and testing samples are stored as `training.pickle` and `test.pickle` in folders named corresponding to graph-type at `./datasets/data_splits/`.


 
**Note (PyTorch 1.0 or higher)**:  
This code was written and tested on PyTorch (0.4.1), so it has some incompatibilities with newer versions. With PyTorch versions (1.0 or higher), this code may give inconsistence performance. This is because of some of the changes in newer versions cause problems with this code. One reason is dropout not acting as intended in the code (See [https://discuss.pytorch.org/t/moving-from-pytorch-0-4-1-to-1-0-changes-model-output/41760/3](https://discuss.pytorch.org/t/moving-from-pytorch-0-4-1-to-1-0-changes-model-output/41760/3)).
For example, changing the dropout code (at two locations) in MLP layer in `layer.py`,
```
score_temp = F.dropout(score_temp,self.dropout)
```
to (for newer PyTorch versions)
```
score_temp = F.dropout(score_temp,self.dropout,self.training)
```
improves the performance similar to original results. In addition to this fix, I found results varying a bit (between older and newer versions) even with same random seed. I will look into it and provide a patch for newer PyTorch versions.

**Note about package requirements**

In case there are conflicts while installing the packages, please check the following [issue](https://github.com/sunilkmaurya/GNN_Ranking/issues/2) for matched versions of the packages. Thanks @natema for the information. 

**Execution Note:**

It appears that on Windows, the dependencies are misconfigured, so it was necessary to migrate the project to an Ubuntu system. To create a virtual environment: python3.10 env/bin/activate, And then activate it with: source env/bin/activate. 

Additionally, a **modification to the MLP layer is required when running on GPU**; otherwise, the training process is ignored.


**Hyperbolic Graphs**
For these graphs, since we are currently only interested in the nodes and edges (not their properties), it's sufficient to read the `.edgelist` file from the generated graphs.
