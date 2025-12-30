
for graph in arxiv products papers orkut; do
#for graph in papers; do	
  sbatch -J ${graph} download.sh ${graph} 
done 
#sbatch download.sh products
