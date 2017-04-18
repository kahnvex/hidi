from hidi.transform import Transform


class DedupeTransform(Transform):
    """
    Deduplicate link-item tall skinny DataFrame
    """
    def __init__(self, skip_dedupe=False):
        self.skip_dedupe = skip_dedupe

    def transform(self, df, **kwargs):
        """
        Takes a :code:`df` that has :code:`link_id` and :code:`item_id`
        columns, and deduplicates them so that each pair is represented
        at most once.
        """
        if self.skip_dedupe:
            return df, kwargs

        return df.drop_duplicates(subset=['link_id', 'item_id']), kwargs
