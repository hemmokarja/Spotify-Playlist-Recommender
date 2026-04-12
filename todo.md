TENSORISER:
* Add seq_lens

PlaylistRecommenderInference
* Add seq_lens to last_step_probs as input -> take the last valid embedding -> pass to head (needed during evaluation when batching is used)


* Add artist dropout into TrackEmbedder to allow simulating inference where artist is not always present
* Remove LayerNorm from TrackEmbedder output - previously that hasn't worked well

Head
* give tracks as input to head -> adjust sampler
* give config as input to head -> loss kwargs and sampler kwargs as regular config params
* cache all_item_embs for inference somehow? Or static item embedder sufficient?

Masking test-only items during training
* they're never in loss if we ensure that in sampler (only sample train tracks)
* add ability to mask them out with allowed_mask (need logprobs)
