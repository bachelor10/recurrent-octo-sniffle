# This is an attempt to get our code running with RNN



### Current issues
- You need to remove last newline of each file in the DATA directory, this is because each line is expected to be valid json in ndjson
- Number of classes is highly different, well known json error expects everyone to have at least FLAGS.num_per_class or something samples. Which we don't


