for sample_size in 250 500 1000 2000 4000 8000
{
    echo "FFT:" >> ../Results/{$sample_size}_stats.txt
    time ./memusg.sh Rscript fft.r $sample_size >> ../Results/{$sample_size}_stats.txt
    echo "\nRandom Forest:" >> ../Results/{$sample_size}_stats.txt
    time ./memusg.sh Rscript randomForest.r $sample_size >> ../Results/{$sample_size}_stats.txt
}