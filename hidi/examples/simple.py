from hidi import inout, clean, matrix, pipeline


# CSV file with link_id and item_id columns
in_files = ['hidi/examples/data/user-item.csv']

# File to write output data to
outfile = 'embeddings.csv'

transforms = [
    inout.ReadTransform(in_files),      # Read data from disk
    clean.DedupeTransform(),            # Dedupe it
    matrix.SparseTransform(),           # Make a sparse user*item matrix
    matrix.SimilarityTransform(),       # To item*item similarity matrix
    matrix.SVDTransform(),              # Perform SVD dimensionality reduction
    matrix.ItemsMatrixToDFTransform(),  # Make a DataFrame with an index
    inout.WriteTransform(outfile)       # Write results to csv
]

pl = pipeline.Pipeline(transforms)
pl.run()
