# Install/load package
if (!requireNamespace("BiocManager", quietly = TRUE))
install.packages("BiocManager")
BiocManager::install("pathifier")
library(pathifier)

# Load expression data for PATHIFIER
exp.matrix <- read.delim(file =file.choose(), as.is = T, row.names = 1)

# Load Pathway genesets 
gene_sets <- as.matrix(read.delim(file = file.choose(), header = F, sep = "\t", as.is = T))

#  Generate a list that contains genes in genesets
gs <- list()
for (i in 1:nrow(gene_sets)){
  a <- as.vector(gene_sets[i,3:ncol(gene_sets)])
  a <- na.omit(a)
  a <- a[a != ""]
  a <- matrix(a, ncol = 1)
  gs[[length(gs)+1]] <- a
  rm(a,i)
}

# Generate a list that contains the names of the genesets used
pathwaynames <- as.list(gene_sets[,1])

# Generate a list that contains the previous two lists: genesets and their names
PATHWAYS <- list(); 
PATHWAYS$gs <- gs; 
PATHWAYS$pathwaynames <- pathwaynames

# Prepare data and parameters 
# Extract information from binary phenotypes and assign 1 = Normal, 0 = Tumor
normals <- as.vector(as.logical(exp.matrix[1,]))
exp.matrix <- as.matrix(exp.matrix[-1, ])

# Calculate MIN_STD
N.exp.matrix <- exp.matrix[,as.logical(normals)]
rsd <- apply(N.exp.matrix, 1, sd)
min_std <- quantile(rsd, 0.25)

# Calculate MIN_EXP 
min_exp <- quantile(as.vector(exp.matrix), 0.1) # Percentile 10 of data

# Filter low value genes. At least 10% of samples with values over min_exp
# Set expression levels < MIN_EXP to MIN_EXP
over <- apply(exp.matrix, 1, function(x) x > min_exp)
G.over <- apply(over, 2, mean)
G.over <- names(G.over)[G.over > 0.1]
exp.matrix <- exp.matrix[G.over,]
exp.matrix[exp.matrix < min_exp] <- min_exp

# Set N as the number of genes in your corresponding expression file
V <- names(sort(apply(exp.matrix, 1, var), decreasing = T))[1:N]
V <- V[!is.na(V)]
exp.matrix <- exp.matrix[V,]
genes <- rownames(exp.matrix) # Checking genes
allgenes <- as.vector(rownames(exp.matrix))

# Generate a list that contains previous data: gene expression, normal status, and name of genes
DATASET <- list(); 
DATASET$allgenes <- allgenes; 
DATASET$normals <- normals
DATASET$data <- exp.matrix

# Run Pathifier
PDS <- quantify_pathways_deregulation(DATASET$data, DATASET$allgenes, PATHWAYS$gs, PATHWAYS$pathwaynames, DATASET$normals, min_std = min_std, min_exp = min_exp)

# Save PDS scores 
write.table(PDS$scores, file = "PDS_Result.txt", quote = FALSE, sep = "\t", row.names = FALSE, col.names = TRUE)
