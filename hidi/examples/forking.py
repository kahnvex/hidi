import numpy as np

from hidi import pipeline, inout, matrix, forking


def to_float32(df, **kwargs):
    return df.astype(np.int32).astype(np.float32)


def create_pipeline(infiles):
    pl = pipeline.Pipeline([
        inout.ReadTransform(infiles),
        matrix.SparseTransform(),
        matrix.SimilarityTransform(),
        matrix.ApplyTransform(fn=to_float32),
        matrix.ScalarTransform(fn='log1p')
    ])

    left = pipeline.Pipeline([
        matrix.SNMFTransform(rank=32, max_iter=2),
        matrix.DenseTransform(),
        matrix.ItemsMatrixToDFTransform(),
        inout.WriteTransform('snmf-embeddings.csv')
    ])

    right = pipeline.Pipeline([
        matrix.SVDTransform(n_components=32, n_iter=2),
        matrix.ItemsMatrixToDFTransform(),
        inout.WriteTransform('svd-embeddings.csv')
    ])

    pl.add(forking.ProcessForkTransform([left, right], progress=False))

    return pl


def run_pipeline():
    pl = create_pipeline(['hidi/examples/data/user-item.csv'])

    return pl.run(progress=False)


if __name__ == '__main__':
    run_pipeline()
