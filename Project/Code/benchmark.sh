# 300 and 600

for sample_size in 1000 2000 3000 4000 5000 6000 7000 8000 9000
{
    echo "FFT:" >> ../Results/{$sample_size}_stats.txt
    time ./memusg.sh Rscript fft.r $sample_size >> ../Results/{$sample_size}_stats.txt
    echo "\nRandom Forest:" >> ../Results/{$sample_size}_stats.txt
    time ./memusg.sh Rscript randomForest.r $sample_size >> ../Results/{$sample_size}_stats.txt
}