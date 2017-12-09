# for each dataset
for dataset_name in "velocity" "synapse" "tomcat"
{
    #  1000 2000 3000 4000 5000 6000 7000 8000 9000
    # for each data size
    for sample_size in 300 600
    {
        echo "\nFFT:" >> ../Results/${dataset_name}_${sample_size}_stats.txt
        time ./memusg.sh Rscript fft.r $sample_size $dataset_name >> ../Results/${dataset_name}_${sample_size}_stats.txt
        echo "\nRandom Forest:" >> ../Results/${dataset_name}_${sample_size}_stats.txt
        time ./memusg.sh Rscript randomForest.r $sample_size $dataset_name >> ../Results/${dataset_name}_${sample_size}_stats.txt
    }
}