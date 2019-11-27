### Missing functions with respect to R version
- AlterStructure
    - RemoveNodesbyIDs, CollapseCliques, RewireBranches
- boot
- DataVSEdges
- dim.selection
- evaluate.structure
- InferTrimRadius
- Interactive
- OverDispersion
- plotting
    - missing all except PlotPG (for basic functionality)
- pseudotime

- computeElasticGraphWithGrammars :
	- missing ParallelRep option

- GetSubGraph 'branches' does not handle the presence of loops yet


### Corrected R version bugs
Several bugs of the R version were corrected. 
If you downloaded the original R package and notice a difference with this one, please make sure you are using the latest R version


- typos in function and parameter names: 
	- e.g., CollapseBrances -> CollapseBranches, ExtendLeaves('WeigthedCentroid')  -> ExtendLeaves('WeightedCentroid')
- f_reattach_edges :
	- average mistake -> add parentheses
- Bisectedge :
	- mean function mistake -> concatenate the two arguments
- ExtendLeaves :
	- assignment Mus = Lambdas -> assignment Mus = Mus
- CollapseBranches :
	- two successive return statements -> deleted second return statement
	- "if x: false else: false" condition -> if x: true else: false
- ShiftBranching : 
	- BrIds = intersect(BrIds, BrIds) -> BrIds = intersect(BrIds, BrPoints)
- getPrimitiveGraphStructureBarCode :
	- barcode error : N[names(N)>=3] -> N[as.integer(names(N))>=3]

### R version bugs found but not corrected yet
- PrimitiveElasticGraphEmbdment : 
	- prob points option bugged
- ReportOnPrimitiveGraphEmbdement :
	- nStars wrong (only 3-stars are counted)
	- nRays always 0
	- energy always non penalized

- AdjustElasticMatrix_initial :
	- does not update AdjustVect in addition to ElasticMatrix ?